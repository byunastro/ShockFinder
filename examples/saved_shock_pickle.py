"""Read selected numeric arrays from the repository's large protocol-4 pickles.

This is deliberately narrow: it recognizes the ShockResult/DissipationResult
layout written by this repository and maps raw ndarray payloads read-only. It
does not unpickle or execute objects, and rejects unexpected byte counts.
"""

from __future__ import annotations

import pickletools
from pathlib import Path

import numpy as np


_RESULT_FIELDS = {
    "mach": ("<f8", 1), "shock": ("|b1", 1),
    "center_index": ("<i4", 1), "upstream_index": ("<i4", 1),
    "downstream_index": ("<i4", 1), "selected_indices": ("<i4", 1),
    "pos": ("<f8", 3), "dx": ("<f8", 1), "normal": ("<f8", 3),
    "level": ("<i4", 1), "zone_width": ("<f8", 1),
}
_DISSIPATION_FIELDS = {name: ("<f8", 1) for name in ("flux", "total", "area")}
_SAMPLING_SHAPES = {
    'rows': (), 'temperature': (2,), 'density': (2,), 'pressure': (2,),
    'distance': (2,), 'pos': (2, 3), 'validation_pressure': (2,),
    'relative_temperature_change': (2,), 'relative_ratio_change': (), 'converged': (),
}


class _PickleMetadata:
    """Inert pickle metadata; no referenced class or callable is imported."""
    def __init__(self, kind, value=None):
        self.kind, self.value, self.state = kind, value, None


def _sampling_state(path):
    """Read only the nested sampling dictionary's ndarray metadata.

    The small stack below models containers and NumPy's numeric ndarray state;
    GLOBAL/REDUCE/BUILD are represented by inert records and never executed.
    Raw byte spans are skipped, including the large dense-array payloads. Memo
    references remain explicit because NumPy shares dtype objects and Python
    shares the nested ``pos`` key with the dense result's ``pos`` key.
    """
    opcodes = {ord(op.code): op for op in pickletools.opcodes}
    stack, memo, marker = [], {}, object()

    def marked():
        for i in range(len(stack)-1, -1, -1):
            if stack[i] is marker:
                values = stack[i+1:]
                del stack[i:]
                return values
        raise ValueError('sampling pickle metadata has an unmatched MARK')

    with Path(path).open('rb') as stream:
        if stream.read(2) != b'\x80\x04':
            raise ValueError('sampling requires a protocol-4 pickle')
        stream.seek(0)
        while True:
            code = stream.read(1)
            if not code:
                raise ValueError('sampling pickle metadata has no STOP')
            op = opcodes.get(code[0])
            if op is None:
                raise ValueError('invalid sampling pickle opcode')
            name = op.name
            if name in ('BINBYTES', 'BINBYTES8', 'SHORT_BINBYTES'):
                width = {'BINBYTES': 4, 'BINBYTES8': 8, 'SHORT_BINBYTES': 1}[name]
                length = int.from_bytes(stream.read(width), 'little')
                offset = stream.tell()
                if offset+length > Path(path).stat().st_size:
                    raise ValueError('sampling array payload extends beyond the pickle')
                stack.append(_PickleMetadata('bytes', (offset, length)))
                stream.seek(length, 1)
                continue
            arg = op.arg.reader(stream) if op.arg else None
            if name in ('PROTO', 'FRAME'):
                continue
            if name == 'MEMOIZE':
                memo[len(memo)] = stack[-1]
            elif name in ('BINPUT', 'LONG_BINPUT', 'PUT'):
                memo[int(arg)] = stack[-1]
            elif name in ('BINGET', 'LONG_BINGET', 'GET'):
                stack.append(memo[int(arg)])
            elif name == 'MARK':
                stack.append(marker)
            elif name in ('SHORT_BINUNICODE', 'BINUNICODE', 'BINUNICODE8',
                          'BININT', 'BININT1', 'BININT2', 'LONG1', 'LONG4', 'BINFLOAT'):
                stack.append(arg)
            elif name in ('NONE', 'NEWTRUE', 'NEWFALSE'):
                stack.append({'NONE': None, 'NEWTRUE': True, 'NEWFALSE': False}[name])
            elif name in ('EMPTY_DICT', 'EMPTY_LIST', 'EMPTY_TUPLE'):
                stack.append({'EMPTY_DICT': dict, 'EMPTY_LIST': list, 'EMPTY_TUPLE': tuple}[name]())
            elif name == 'TUPLE':
                stack.append(tuple(marked()))
            elif name in ('TUPLE1', 'TUPLE2', 'TUPLE3'):
                n = int(name[-1])
                values = tuple(stack[-n:])
                del stack[-n:]
                stack.append(values)
            elif name == 'SETITEMS':
                values = marked()
                if len(values) % 2 or not isinstance(stack[-1], dict):
                    raise ValueError('invalid sampling pickle dictionary')
                stack[-1].update(zip(values[::2], values[1::2]))
            elif name == 'SETITEM':
                value, key = stack.pop(), stack.pop()
                stack[-1][key] = value
            elif name == 'APPENDS':
                values = marked()
                stack[-1].extend(values)
            elif name == 'APPEND':
                value = stack.pop()
                stack[-1].append(value)
            elif name == 'STACK_GLOBAL':
                attribute, module = stack.pop(), stack.pop()
                stack.append(_PickleMetadata('global', (module, attribute)))
            elif name == 'GLOBAL':
                stack.append(_PickleMetadata('global', tuple(arg.split(' ', 1))))
            elif name in ('NEWOBJ', 'REDUCE'):
                args, callable_ref = stack.pop(), stack.pop()
                kind = 'object'
                if isinstance(callable_ref, _PickleMetadata) and callable_ref.kind == 'global':
                    module, attribute = callable_ref.value
                    if module == 'numpy' and attribute == 'dtype':
                        kind = 'dtype'
                    elif module in ('numpy.core.multiarray', 'numpy._core.multiarray') and attribute == '_reconstruct':
                        kind = 'ndarray'
                stack.append(_PickleMetadata(kind, (callable_ref, args)))
            elif name == 'BUILD':
                state = stack.pop()
                if not isinstance(stack[-1], _PickleMetadata):
                    raise ValueError('unexpected sampling pickle BUILD target')
                stack[-1].state = state
            elif name == 'STOP':
                if len(stack) != 1 or not isinstance(stack[0], _PickleMetadata):
                    raise ValueError('unexpected sampling pickle result layout')
                state = stack[0].state
                state = state[1] if isinstance(state, tuple) and len(state) == 2 else state
                if not isinstance(state, dict):
                    raise ValueError('unexpected ShockResult pickle state')
                units = {name: state.get(name, default) for name, default in (
                    ('temperature_unit', 'K'), ('density_unit', 'Msol/kpc3'))}
                if not all(isinstance(value, str) for value in units.values()):
                    raise ValueError('unexpected sampled-state unit metadata')
                if 'sampling_tolerance' in state:
                    tolerance = state['sampling_tolerance']
                    if (isinstance(tolerance, bool) or not isinstance(tolerance, (float, int))
                            or not np.isfinite(tolerance) or not 0. <= tolerance < 1.):
                        raise ValueError('invalid sampled-state tolerance')
                    units['sampling_tolerance'] = float(tolerance)
                return state.get('sampling'), units
            else:
                raise ValueError(f'unsupported sampling pickle opcode {name}')


def _map_sampling(path, count, metadata=None):
    sampling, units = _sampling_state(path)
    if metadata is not None:
        metadata.update(units)
    if sampling is None:
        return None
    required = {'rows', 'temperature', 'density', 'pressure', 'distance', 'pos'}
    if not isinstance(sampling, dict) or not required.issubset(sampling) or set(sampling)-set(_SAMPLING_SHAPES):
        raise ValueError('unexpected sampled-state pickle fields')
    arrays, nsample = {}, None
    for name in ('rows', *[key for key in sampling if key != 'rows']):
        array = sampling[name]
        if not isinstance(array, _PickleMetadata) or array.kind != 'ndarray':
            raise ValueError(f'sampling {name} must be a numeric ndarray')
        state = array.state
        if not isinstance(state, tuple) or len(state) != 5 or state[0] != 1:
            raise ValueError(f'unsupported sampling {name} ndarray state')
        _, shape, dtype_state, fortran, payload = state
        if (not isinstance(shape, tuple) or any(type(n) is not int or n < 0 for n in shape)
                or not isinstance(dtype_state, _PickleMetadata) or dtype_state.kind != 'dtype'
                or not isinstance(payload, _PickleMetadata) or payload.kind != 'bytes'
                or fortran is not False):
            raise ValueError(f'unsupported sampling {name} shape, dtype, or array order')
        descriptor = dtype_state.value[1]
        dtype = '<i8' if name == 'rows' else ('|b1' if name == 'converged' else '<f8')
        if (not isinstance(descriptor, tuple) or not descriptor
                or descriptor[0] not in (dtype, dtype[1:])
                or not isinstance(dtype_state.state, tuple) or len(dtype_state.state) < 2
                or dtype_state.state[1] not in (('|',) if name == 'converged' else ('<', '='))):
            raise ValueError(f'sampling {name} must have dtype {dtype}')
        if name == 'rows':
            if len(shape) != 1:
                raise ValueError('sampling rows must be one-dimensional')
            nsample = shape[0]
        if shape != (nsample,)+_SAMPLING_SHAPES[name]:
            raise ValueError(f'sampling {name} has unexpected shape')
        offset, length = payload.value
        if length != int(np.prod(shape))*np.dtype(dtype).itemsize:
            raise ValueError(f'sampling {name} has unexpected byte count')
        arrays[name] = np.memmap(path, mode='r', dtype=dtype, offset=offset, shape=shape, order='C')
    rows = arrays['rows']
    if np.any(rows < 0) or np.any(rows >= count) or np.any(rows[1:] <= rows[:-1]):
        raise ValueError('sampling rows must be sorted unique result-row indices')
    return arrays


class _LazyTotal:
    def __init__(self, flux, area):
        self.flux, self.area = flux, area

    def __len__(self):
        return len(self.flux)

    def __getitem__(self, index):
        return self.flux[index]*self.area[index]


def _array_payloads(path, expected):
    opcodes = {ord(op.code): op for op in pickletools.opcodes}
    found = {}
    field = None
    path = Path(path)
    with path.open("rb") as stream:
        if stream.read(2) != b"\x80\x04":
            raise ValueError(f"{path}: expected protocol-4 pickle")
        stream.seek(0)
        while True:
            code = stream.read(1)
            if not code:
                break
            op = opcodes.get(code[0])
            if op is None:
                raise ValueError(f"{path}: invalid pickle opcode at {stream.tell()-1}")
            if op.name in ("BINBYTES", "BINBYTES8", "BYTEARRAY8"):
                width = 4 if op.name == "BINBYTES" else 8
                length = int.from_bytes(stream.read(width), "little")
                offset = stream.tell()
                if field in expected and field not in found:
                    found[field] = (offset, length)
                stream.seek(length, 1)
                field = None
            elif op.name == "SHORT_BINBYTES":
                length = int.from_bytes(stream.read(1), "little")
                offset = stream.tell()
                expected_size = (np.dtype(expected[field][0]).itemsize*expected[field][1]
                                 if field in expected else None)
                # NumPy's ndarray reconstruction contains a one-byte b'b'
                # marker before the actual payload. Never map that marker.
                if (field in expected and field not in found and length > 1
                        and length % expected_size == 0):
                    found[field] = (offset, length)
                stream.seek(length, 1)
                if length > 1:
                    field = None
            else:
                arg = op.arg.reader(stream) if op.arg else None
                if op.name in ("SHORT_BINUNICODE", "BINUNICODE") and arg in expected:
                    field = arg
                if op.name == "STOP":
                    break
    missing = set(expected)-set(found)
    if missing:
        raise ValueError(f"{path}: missing array payloads {sorted(missing)}")
    return found


def _map_fields(path, schema, count=None):
    payloads = _array_payloads(path, schema)
    if count is None:
        length = payloads[next(iter(schema))][1]
        dtype, width = schema[next(iter(schema))]
        count, remainder = divmod(length, np.dtype(dtype).itemsize*width)
        if remainder:
            raise ValueError("first array payload has unexpected byte count")
    arrays = {}
    for name, (dtype, width) in schema.items():
        offset, length = payloads[name]
        shape = (count,) if width == 1 else (count, width)
        if length != int(np.prod(shape))*np.dtype(dtype).itemsize:
            raise ValueError(f"{name} payload has unexpected byte count")
        # ShockResult.pos/normal are saved Fortran-contiguous (pickle ndarray
        # state has NEWTRUE before the payload); mapping as C silently mixes
        # coordinates from unrelated rows.
        order = "F" if width == 3 else "C"
        arrays[name] = np.memmap(path, mode="r", dtype=dtype, offset=offset, shape=shape, order=order)
    return arrays, count


def open_saved_shock_snapshot(directory, snapshot, *, position_unit):
    """Return read-only mapped result, dissipation, and labels for one output.

    The caller should verify sampled fields against its saved front catalog;
    this reader is only valid for the known protocol-4 numeric pickle layout.
    Mapping is cheap and does not materialize the 33 GB result in memory.
    """
    from types import SimpleNamespace

    directory = Path(directory)
    if position_unit not in ("km", "kpc", "Mpc"):
        raise ValueError("position_unit must be declared as km, kpc, or Mpc")
    stem = f"{int(snapshot):05d}.pkl"
    result_path = directory/f"result_{stem}"
    dissipation_path = directory/f"dissipation_{stem}"
    labels_path = directory/f"labels_{stem}"
    result, count = _map_fields(result_path, _RESULT_FIELDS)
    result['sampling'] = _map_sampling(result_path, count, result)
    try:
        dissipation, _ = _map_fields(dissipation_path, _DISSIPATION_FIELDS, count)
    except ValueError as exc:
        if "missing array payloads ['total']" not in str(exc):
            raise
        dissipation, _ = _map_fields(dissipation_path,
                                     {name: _DISSIPATION_FIELDS[name] for name in ("flux", "area")}, count)
        dissipation["total"] = _LazyTotal(dissipation["flux"], dissipation["area"])
    # Labels are a single top-level ndarray rather than a named state field.
    opcodes = {ord(op.code): op for op in pickletools.opcodes}
    with labels_path.open("rb") as stream:
        if stream.read(2) != b"\x80\x04":
            raise ValueError("labels require protocol-4 pickle")
        stream.seek(0)
        while True:
            code = stream.read(1)
            if not code:
                raise ValueError("labels pickle has no ndarray payload")
            op = opcodes[code[0]]
            if op.name in ("BINBYTES", "BINBYTES8", "BYTEARRAY8"):
                width = 4 if op.name == "BINBYTES" else 8
                length = int.from_bytes(stream.read(width), "little")
                offset = stream.tell()
                break
            if op.name == "SHORT_BINBYTES":
                length = int.from_bytes(stream.read(1), "little")
                offset = stream.tell()
                if length == count*4:
                    break
                stream.seek(length, 1)
                continue
            if op.arg:
                op.arg.reader(stream)
    if length != count*np.dtype("<i4").itemsize:
        raise ValueError("labels do not align with result rows")
    labels = np.memmap(labels_path, mode="r", dtype="<i4", offset=offset, shape=(count,))
    return (SimpleNamespace(**result, position_unit=position_unit),
            SimpleNamespace(**dissipation), labels)

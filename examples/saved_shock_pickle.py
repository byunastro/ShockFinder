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

"""Inspect NumPy pickles by replacing binary payloads with offset descriptors.

No ShockFinder classes, algorithms, or arbitrary pickle globals are executed.
The supported globals are NumPy dtype/array reconstruction and the two known
ShockFinder record classes. Array shapes, dtypes, byte order and layout come
from their actual pickle state rather than inference from file size.

FRAME opcodes are removed because skipping payloads changes frame lengths.
The metadata stream retains the original memo numbering. Inserted payload
descriptors contain no MEMOIZE or PUT opcodes. Actual array data remain on disk.
"""

from __future__ import annotations

import io
import math
import pickle
import pickletools
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ArrayMetadata:
    shape: tuple[int, ...]
    dtype: np.dtype
    fortran_order: bool
    offset: int
    nbytes: int

    def map(self, path: Path):
        if not self.nbytes:
            return np.empty(self.shape, dtype=self.dtype)
        return np.memmap(path, mode="r", dtype=self.dtype, offset=self.offset,
                         shape=self.shape, order="F" if self.fortran_order else "C")


class _ArrayProxy:
    def __setstate__(self, state):
        if not isinstance(state, tuple) or len(state) != 5 or state[0] != 1:
            raise ValueError("unsupported NumPy array pickle state")
        _, shape, dtype, order, payload = state
        shape = tuple(shape)
        dtype = np.dtype(dtype)
        if dtype.hasobject:
            raise ValueError("object arrays require an explicit adapter")
        if any(not isinstance(x, int) or x < 0 for x in shape):
            raise ValueError("invalid array shape")
        if not isinstance(payload, tuple) or len(payload) != 3 or payload[0] != "payload":
            raise ValueError("array data do not use the supported binary payload layout")
        _, offset, size = payload
        if size != math.prod(shape) * dtype.itemsize:
            raise ValueError("array payload length disagrees with shape/dtype")
        self.array = ArrayMetadata(shape, dtype, bool(order), offset, size)


def _reconstruct(*args):
    return _ArrayProxy()


class _RecordProxy:
    object_type = ""

    def __setstate__(self, state):
        if isinstance(state, tuple) and len(state) == 2 and state[0] is None:
            state = state[1]
        if not isinstance(state, dict):
            raise ValueError("unsupported ShockFinder record state")
        self.fields = state


class _ResultProxy(_RecordProxy):
    object_type = "shocktest.core.ShockResult"


class _DissipationProxy(_RecordProxy):
    object_type = "shocktest.pyShockFinder.DissipationResult"


class _MetadataUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module in {"numpy.core.multiarray", "numpy._core.multiarray"} and name == "_reconstruct":
            return _reconstruct
        allowed = {
            ("numpy", "ndarray"): _ArrayProxy,
            ("numpy", "dtype"): np.dtype,
            ("shocktest.core", "ShockResult"): _ResultProxy,
            ("shocktest.pyShockFinder", "DissipationResult"): _DissipationProxy,
        }
        if (module, name) not in allowed:
            raise ValueError(f"unsupported pickle global: {module}.{name}; explicit adapter required")
        return allowed[module, name]

    def persistent_load(self, pid):
        raise ValueError("persistent pickle IDs require an explicit adapter")


def _descriptor(offset, size):
    # LONG1 encodes positive offsets beyond 2 GiB. No new memo entries are added.
    output = bytearray(b"(\x8c\x07payload")
    for value in (offset, size):
        width = max(1, (value.bit_length() + 8) // 8)
        raw = value.to_bytes(width, "little", signed=True)
        output.extend(b"\x8a" + bytes([width]) + raw)
    output.extend(b"t")
    return output


def _read_exact(stream, count):
    raw = stream.read(count)
    if len(raw) != count:
        raise ValueError("truncated pickle metadata")
    return raw


def read_metadata(path: str | Path, max_metadata_bytes=8 * 1024 * 1024):
    """Return a metadata-only object after verifying a complete, supported pickle."""
    path = Path(path)
    size = path.stat().st_size
    before = path.stat().st_mtime_ns
    opcodes = {ord(op.code): op for op in pickletools.opcodes}
    output = bytearray()
    stopped = False
    with path.open("rb") as stream:
        while stream.tell() < size:
            code = _read_exact(stream, 1)
            op = opcodes.get(code[0])
            if op is None:
                raise ValueError(f"unknown pickle opcode at {stream.tell() - 1}")
            kind = op.arg.name if op.arg else None
            if op.name == "FRAME":
                length = int.from_bytes(_read_exact(stream, 8), "little")
                if stream.tell() + length > size:
                    raise ValueError("truncated pickle frame")
                continue
            if op.name in {"NEXT_BUFFER", "READONLY_BUFFER"}:
                raise ValueError("out-of-band NumPy buffers require an explicit adapter")
            output.extend(code)
            if kind in {"uint1", "uint2", "uint4", "uint8", "int4", "float8"}:
                width = {"uint1": 1, "uint2": 2, "uint4": 4, "uint8": 8, "int4": 4, "float8": 8}[kind]
                output.extend(_read_exact(stream, width))
            elif kind in {"bytes1", "bytes4", "bytes8", "bytearray8", "string1", "string4",
                          "unicodestring1", "unicodestring4", "unicodestring8", "long1", "long4"}:
                width = 1 if kind.endswith("1") else 4 if kind.endswith("4") else 8
                length_raw = _read_exact(stream, width)
                length = int.from_bytes(length_raw, "little")
                offset = stream.tell()
                if offset + length > size:
                    raise ValueError(f"truncated pickle payload at {offset}")
                if kind.startswith(("bytes", "bytearray")):
                    output.pop()
                    output.extend(_descriptor(offset, length))
                    stream.seek(length, 1)
                else:
                    if len(output) + width + length > max_metadata_bytes:
                        raise ValueError("pickle metadata exceed configured bound")
                    output.extend(length_raw)
                    output.extend(_read_exact(stream, length))
            elif kind in {"decimalnl_short", "decimalnl_long", "floatnl", "stringnl",
                          "stringnl_noescape", "unicodestringnl", "stringnl_noescape_pair"}:
                for _ in range(2 if kind == "stringnl_noescape_pair" else 1):
                    line = stream.readline(max_metadata_bytes + 1)
                    if not line.endswith(b"\n"):
                        raise ValueError("truncated or oversized pickle line")
                    output.extend(line)
            elif kind is not None:
                raise ValueError(f"unsupported pickle argument {kind}")
            if len(output) > max_metadata_bytes:
                raise ValueError("pickle metadata exceed configured bound")
            if op.name == "STOP":
                if stream.tell() != size:
                    raise ValueError("pickle contains trailing bytes")
                stopped = True
                break
    if not stopped:
        raise ValueError("pickle has no STOP opcode")
    if path.stat().st_size != size or path.stat().st_mtime_ns != before:
        raise ValueError("input changed during inspection")
    value = _MetadataUnpickler(io.BytesIO(output)).load()
    return value


def object_fields(value):
    if isinstance(value, _RecordProxy):
        return value.object_type, value.fields
    if isinstance(value, dict):
        return "builtins.dict", value
    if value is None:
        return "builtins.NoneType", {}
    if isinstance(value, _ArrayProxy):
        return "numpy.ndarray", {"array": value}
    raise ValueError(f"unsupported root object type: {type(value).__name__}")


def array_metadata(value):
    if not isinstance(value, _ArrayProxy) or not hasattr(value, "array"):
        raise ValueError("field is not a supported NumPy ndarray")
    return value.array

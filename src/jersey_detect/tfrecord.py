"""Minimal TFRecord and tf.train.Example reader and writer.

Reads the Roboflow TFRecord exports used by the original notebook without
depending on TensorFlow. CRC checksums are skipped on read, so use this only
on files you trust.
"""

from __future__ import annotations

import struct
from collections.abc import Iterator

# ---------------------------------------------------------------- TFRecord framing


def iter_records(path: str) -> Iterator[bytes]:
    """Yield the raw payload of every record in a TFRecord file."""
    with open(path, "rb") as fh:
        while True:
            header = fh.read(8)
            if not header:
                return
            if len(header) < 8:
                raise ValueError(f"Truncated TFRecord header in {path}")
            (length,) = struct.unpack("<Q", header)
            fh.read(4)  # masked CRC of the length (not verified)
            data = fh.read(length)
            if len(data) != length:
                raise ValueError(f"Truncated TFRecord payload in {path}")
            fh.read(4)  # masked CRC of the data (not verified)
            yield data


def write_record(fh, payload: bytes) -> None:
    """Write one record with zeroed CRCs (readable by this module, not by TensorFlow)."""
    fh.write(struct.pack("<Q", len(payload)))
    fh.write(b"\x00\x00\x00\x00")
    fh.write(payload)
    fh.write(b"\x00\x00\x00\x00")


# ---------------------------------------------------------------- protobuf wire format


def _read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7


def _iter_fields(buf: bytes) -> Iterator[tuple[int, int, object]]:
    """Yield (field_number, wire_type, value) for each field of a protobuf message."""
    pos = 0
    while pos < len(buf):
        key, pos = _read_varint(buf, pos)
        field, wire = key >> 3, key & 0x7
        if wire == 0:
            value, pos = _read_varint(buf, pos)
        elif wire == 1:
            value, pos = buf[pos : pos + 8], pos + 8
        elif wire == 2:
            length, pos = _read_varint(buf, pos)
            value, pos = buf[pos : pos + length], pos + length
        elif wire == 5:
            value, pos = buf[pos : pos + 4], pos + 4
        else:
            raise ValueError(f"Unsupported protobuf wire type {wire}")
        yield field, wire, value


def _decode_feature(buf: bytes) -> tuple[str, list]:
    """Decode a tf.train.Feature into (kind, values) where kind is bytes, float or int64."""
    for field, _, payload in _iter_fields(buf):
        if field == 1:  # BytesList
            return "bytes", [v for f, _, v in _iter_fields(payload) if f == 1]
        if field == 2:  # FloatList, packed or unpacked
            values: list[float] = []
            for f, wire, v in _iter_fields(payload):
                if f != 1:
                    continue
                if wire == 2:
                    values.extend(struct.unpack(f"<{len(v) // 4}f", v))
                else:
                    values.append(struct.unpack("<f", v)[0])
            return "float", values
        if field == 3:  # Int64List, packed or unpacked
            ints: list[int] = []
            for f, wire, v in _iter_fields(payload):
                if f != 1:
                    continue
                if wire == 2:
                    pos = 0
                    while pos < len(v):
                        n, pos = _read_varint(v, pos)
                        ints.append(n)
                else:
                    ints.append(v)
            return "int64", ints
    return "empty", []


def decode_example(payload: bytes) -> dict[str, tuple[str, list]]:
    """Decode a serialized tf.train.Example into {feature_name: (kind, values)}."""
    features: dict[str, tuple[str, list]] = {}
    for field, _, features_msg in _iter_fields(payload):
        if field != 1:  # Example.features
            continue
        for f, _, entry in _iter_fields(features_msg):
            if f != 1:  # Features.feature map entries
                continue
            name = ""
            value = b""
            for ef, _, ev in _iter_fields(entry):
                if ef == 1:
                    name = ev.decode("utf-8")
                elif ef == 2:
                    value = ev
            features[name] = _decode_feature(value)
    return features


# ---------------------------------------------------------------- encoding (for tests and tools)


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        byte = n & 0x7F
        n >>= 7
        if n:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _ld(field: int, payload: bytes) -> bytes:
    return _varint((field << 3) | 2) + _varint(len(payload)) + payload


def encode_example(features: dict[str, tuple[str, list]]) -> bytes:
    """Encode {name: (kind, values)} as a serialized tf.train.Example (packed lists)."""
    entries = b""
    for name, (kind, values) in features.items():
        if kind == "bytes":
            body = b"".join(_ld(1, v) for v in values)
            feature = _ld(1, body)
        elif kind == "float":
            feature = _ld(2, _ld(1, struct.pack(f"<{len(values)}f", *values)))
        elif kind == "int64":
            feature = _ld(3, _ld(1, b"".join(_varint(v) for v in values)))
        else:
            raise ValueError(f"Unknown kind {kind}")
        entry = _ld(1, name.encode("utf-8")) + _ld(2, feature)
        entries += _ld(1, entry)
    return _ld(1, entries)

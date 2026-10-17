import io
import struct

import pytest

from jersey_detect.tfrecord import decode_example, encode_example, iter_records, write_record


def test_example_roundtrip_all_kinds():
    features = {
        "image/encoded": ("bytes", [b"\x89PNG-fake"]),
        "image/object/bbox/xmin": ("float", [0.1, 0.25]),
        "image/object/class/label": ("int64", [1, 300]),
    }
    decoded = decode_example(encode_example(features))
    assert decoded["image/encoded"] == ("bytes", [b"\x89PNG-fake"])
    kind, values = decoded["image/object/bbox/xmin"]
    assert kind == "float"
    assert [round(v, 5) for v in values] == [0.1, 0.25]
    assert decoded["image/object/class/label"] == ("int64", [1, 300])


def test_missing_feature_is_empty():
    decoded = decode_example(encode_example({"a": ("int64", [7])}))
    assert "b" not in decoded


def test_tfrecord_file_roundtrip(tmp_path):
    path = tmp_path / "data.tfrecord"
    payloads = [encode_example({"n": ("int64", [i])}) for i in range(3)]
    with open(path, "wb") as fh:
        for p in payloads:
            write_record(fh, p)
    read_back = list(iter_records(str(path)))
    assert read_back == payloads


def test_truncated_file_raises(tmp_path):
    path = tmp_path / "bad.tfrecord"
    path.write_bytes(struct.pack("<Q", 100) + b"\x00" * 4 + b"short")
    with pytest.raises(ValueError):
        list(iter_records(str(path)))

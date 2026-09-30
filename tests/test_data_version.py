import hashlib

import pytest

from src.config import ROOT, load_config
from src.make_dataset import csv_bytes, write_dataset
from src.utils import detect_dataset_version, md5_file, read_dvc_md5


def test_generation_is_byte_identical(tmp_path):
    a = write_dataset("v1", tmp_path / "a.csv").read_bytes()
    b = write_dataset("v1", tmp_path / "b.csv").read_bytes()
    assert a == b
    assert write_dataset("v2", tmp_path / "c.csv").read_bytes() == csv_bytes("v2")


def test_versions_differ():
    assert csv_bytes("v1") != csv_bytes("v2")
    assert len(csv_bytes("v1").splitlines()) == 179  # header + 178 rows
    assert len(csv_bytes("v2").splitlines()) == 179 + 120


def test_detect_version_roundtrip(tmp_path):
    for v in ("v1", "v2"):
        path = write_dataset(v, tmp_path / f"{v}.csv")
        assert detect_dataset_version(md5_file(path)) == v
    assert detect_dataset_version("deadbeef") == "unknown"


def test_committed_pointer_hash_matches_generator():
    pointer = ROOT / load_config()["paths"]["raw_dvc"]
    if not pointer.exists():
        pytest.skip("no data/raw/wine.csv.dvc yet (run `make data VERSION=v1`)")
    md5 = read_dvc_md5(pointer)
    version = detect_dataset_version(md5)
    assert version in ("v1", "v2")
    assert hashlib.md5(csv_bytes(version)).hexdigest() == md5

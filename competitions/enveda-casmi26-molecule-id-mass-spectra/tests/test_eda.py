"""Tests for the local EDA application's data and HTTP contracts."""

import hashlib
import importlib.util
import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
pytest.importorskip("rdkit")

server_path = Path(__file__).resolve().parents[1] / "eda" / "server.py"
spec = importlib.util.spec_from_file_location("casmi_eda_server", server_path)
eda = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eda)


def test_parquet_load_limit_search_and_unlabelled(tmp_path):
    rows = [
        {
            "spectrum_id": "s1",
            "molecule_id": "m1",
            "ms2_mzs": [42.0, 84.0],
            "ms2_normalized_intensities": [1.0, 0.5],
            "precursor_mz": 100.0,
        },
        {
            "spectrum_id": "s2",
            "molecule_id": "m2",
            "ms2_mzs": [50.0],
            "ms2_normalized_intensities": [1.0],
            "precursor_mz": 101.0,
        },
    ]
    source = tmp_path / "test.parquet"
    pq.write_table(pa.Table.from_pylist(rows), source)
    dataset = eda.Dataset(source, max_rows=1)
    assert dataset.summary()["truncated"] is True
    assert dataset.summary()["total_rows"] == 2
    assert dataset.summary()["has_structures"] is False
    assert dataset.search("m1")["total"] == 1
    assert dataset.search("m2")["total"] == 0
    assert dataset.by_id["s1"]["peaks"] == [[42.0, 1.0], [84.0, 0.5]]
    assert dataset.by_id["s1"]["smiles"] is None


def test_structure_similarity_for_manual_candidates_and_demo():
    dataset = eda.Dataset(None, demo=True)
    assert dataset.summary()["demo"] is True
    result = dataset.related("CCO", ["CCO", "CCN"], 10)
    assert result["items"][0]["similarity"] == pytest.approx(1)
    assert result["items"][0]["spectrum_id"] is None
    assert result["items"][0]["spectra"] == []
    assert "<svg" in result["items"][0]["svg"]
    assert dataset.related("Cn1c(=O)c2c(ncn2C)n(C)c1=O")["items"][0]["spectrum_id"] == "demo-1"
    with pytest.raises(ValueError):
        dataset.related("not smiles")


def test_peak_array_mismatch_and_invalid_values_are_reported(tmp_path):
    source = tmp_path / "invalid.parquet"
    pq.write_table(pa.Table.from_pylist([{
        "spectrum_id": "broken", "ms2_mzs": [42.0, float("nan"), 44.0],
        "ms2_normalized_intensities": [1.0, 0.5],
    }]), source)
    record = eda.Dataset(source).by_id["broken"]
    assert record["peaks"] == [[42.0, 1.0]]
    assert record["peak_diagnostics"] == {
        "mz_count": 3, "intensity_count": 2, "paired_count": 2,
        "invalid_pair_count": 1, "unpaired_count": 1,
    }


def test_labelled_train_schema_and_provenance(tmp_path):
    rows = [
        {"spectrum_id": "s1", "inchikey14": "AAA", "normalized_smiles": "CCO",
         "ms2_mzs": [42.0], "ms2_normalized_intensities": [1.0], "adduct": "[M+H]+",
         "ionization_mode": "positive", "collision_energy_ev": [20.0]},
        {"spectrum_id": "s2", "inchikey14": "AAA", "normalized_smiles": "CCO",
         "ms2_mzs": [43.0], "ms2_normalized_intensities": [1.0], "adduct": "[M-H]-",
         "ionization_mode": "negative", "collision_energy_ev": [40.0]},
        {"spectrum_id": "s3", "inchikey14": "BBB", "normalized_smiles": "CCN",
         "ms2_mzs": [44.0], "ms2_normalized_intensities": [1.0]},
    ]
    source = tmp_path / "train.parquet"
    pq.write_table(pa.Table.from_pylist(rows), source)
    sidecar = source.with_suffix(".provenance.json")
    sidecar.write_text(json.dumps({
        "source_kind": "unverified mirror",
        "parquet_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }), encoding="utf-8")
    dataset = eda.Dataset(source)
    assert dataset.summary()["molecules"] == 2
    assert dataset.summary()["has_structures"] is True
    assert dataset.summary()["provenance"]["checksum_status"] == "verified"
    assert dataset.search("AAA")["total"] == 2
    assert dataset.by_id["s1"]["smiles"] == "CCO"
    assert dataset.related("CCO")["items"][0]["molecule_id"] == "AAA"
    spectra = dataset.related("CCO")["items"][0]["spectra"]
    assert [entry["id"] for entry in spectra] == ["s1", "s2"]
    assert [entry["adduct"] for entry in spectra] == ["[M+H]+", "[M-H]-"]
    assert [entry["collision_energy"] for entry in spectra] == [[20.0], [40.0]]
    sidecar.write_text(json.dumps({"parquet_sha256": "0" * 64}), encoding="utf-8")
    assert eda.Dataset(source).summary()["provenance"]["checksum_status"] == "mismatch"
    sidecar.write_text("not json", encoding="utf-8")
    assert eda.Dataset(source).summary()["provenance"]["checksum_status"] == "invalid_sidecar"


def test_http_api_roundtrip():
    dataset = eda.Dataset(None, demo=True)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), eda.make_handler(dataset))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_port}"
    try:
        with urlopen(base + "/api/spectra/demo-1") as response:
            assert json.load(response)["peak_count"] == 5
        request = Request(
            base + "/api/related",
            data=json.dumps({"smiles": "CCO", "candidates": ["CCN"]}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request) as response:
            assert any(item["smiles"] == "CCN" for item in json.load(response)["items"])
        with pytest.raises(HTTPError) as error:
            urlopen(base + "/api/spectra/does-not-exist")
        assert error.value.code == 404
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join()

"""Local CASMI spectra and structure explorer. Requires pyarrow and RDKit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import mimetypes
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

import pyarrow.parquet as pq
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Draw import rdMolDraw2D

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT.parent / "data" / "input" / "test.parquet"
GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
MAX_BODY = 100_000


def read_provenance(path):
    """Read a matching optional sidecar without treating its contents as trusted."""
    sidecar = path.with_suffix(".provenance.json")
    if not sidecar.exists():
        return None
    fields = (
        "source_kind",
        "dataset",
        "dataset_page",
        "revision_note",
        "declared_mirror_license",
        "retrieved_at_utc",
    )
    try:
        if sidecar.stat().st_size > 65536:
            raise ValueError("Sidecar is too large")
        raw = json.loads(sidecar.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Sidecar must contain an object")
        result = {key: raw[key][:2000] for key in fields if isinstance(raw.get(key), str)}
        expected = raw.get("parquet_sha256")
        result["checksum_status"] = "unavailable"
        # Avoid an unexpected full scan of a very large dataset at startup.
        if (
            isinstance(expected, str)
            and re.fullmatch(r"[a-fA-F0-9]{64}", expected)
            and path.stat().st_size <= 512 * 1024 * 1024
        ):
            digest = hashlib.sha256()
            with path.open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
            result["checksum_status"] = (
                "verified" if digest.hexdigest() == expected.lower() else "mismatch"
            )
        return result
    except (OSError, ValueError, UnicodeError):
        return {"checksum_status": "invalid_sidecar"}


def molecule(smiles):
    if not isinstance(smiles, str) or not smiles.strip() or len(smiles) > 2000:
        raise ValueError("SMILES must be a non-empty string of at most 2000 characters")
    params = Chem.SmilesParserParams()
    params.allowCXSMILES = False
    params.parseName = False
    mol = Chem.MolFromSmiles(smiles.strip(), params)
    if mol is None or mol.GetNumAtoms() > 300:
        raise ValueError("Invalid SMILES or molecule has more than 300 atoms")
    return mol


def svg_for(mol):
    drawer = rdMolDraw2D.MolDraw2DSVG(260, 150)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    return value


def demo_rows():
    # Hand-created illustration, not measured spectra or competition data.
    return [
        {
            "molecule_id": "demo-caffeine",
            "spectrum_id": "demo-1",
            "smiles": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
            "precursor_mz": 195.088,
            "adduct": "[M+H]+",
            "ionization_mode": "positive",
            "instrument_type": "synthetic",
            "collision_energy_ev": [20],
            "ms2_mzs": [55.03, 67.04, 110.07, 138.07, 195.09],
            "ms2_normalized_intensities": [0.2, 0.42, 0.65, 1.0, 0.12],
        },
        {
            "molecule_id": "demo-theobromine",
            "spectrum_id": "demo-2",
            "smiles": "Cn1c(=O)c2[nH]cnc2n(C)c1=O",
            "precursor_mz": 181.072,
            "adduct": "[M+H]+",
            "ionization_mode": "positive",
            "instrument_type": "synthetic",
            "collision_energy_ev": [20],
            "ms2_mzs": [55.03, 69.04, 96.06, 138.07, 181.07],
            "ms2_normalized_intensities": [0.25, 0.3, 0.6, 1.0, 0.15],
        },
        {
            "molecule_id": "demo-theophylline",
            "spectrum_id": "demo-3",
            "smiles": "Cn1c(=O)c2c(ncn2C)[nH]c1=O",
            "precursor_mz": 181.072,
            "adduct": "[M+H]+",
            "ionization_mode": "positive",
            "instrument_type": "synthetic",
            "collision_energy_ev": [20],
            "ms2_mzs": [55.03, 69.04, 96.06, 124.05, 181.07],
            "ms2_normalized_intensities": [0.2, 0.35, 0.7, 1.0, 0.13],
        },
    ]


class Dataset:
    def __init__(self, path, max_rows=5000, demo=False):
        if max_rows < 1:
            raise ValueError("max_rows must be positive")
        self.demo = demo
        self.provenance = None
        if demo:
            rows = demo_rows()
            self.total_rows = len(rows)
            self.source = "synthetic demo"
            self.columns = list(rows[0])
        else:
            path = Path(path)
            if not path.is_file():
                raise FileNotFoundError(
                    f"Data file not found: {path}; use --demo for synthetic examples"
                )
            pf = pq.ParquetFile(path)
            self.columns = pf.schema_arrow.names
            self.total_rows = pf.metadata.num_rows
            self.source = str(path.resolve())
            self.provenance = read_provenance(path)
            rows = []
            for batch in pf.iter_batches(batch_size=min(max_rows, 512)):
                rows.extend(batch.slice(0, max_rows - len(rows)).to_pylist())
                if len(rows) >= max_rows:
                    break
        self.records = []
        self.by_id = {}
        for index, row in enumerate(rows):
            row = safe(row)
            spec_id = str(row.get("spectrum_id") or f"row-{index}")
            if spec_id in self.by_id:
                base_id = f"{spec_id}__row_{index}"
                spec_id = base_id
                suffix = 1
                while spec_id in self.by_id:
                    spec_id = f"{base_id}_{suffix}"
                    suffix += 1
            mzs = row.get("ms2_mzs") or []
            intensities = row.get("ms2_normalized_intensities") or []
            peaks = [
                [mz, intensity]
                for mz, intensity in zip(mzs, intensities, strict=False)
                if isinstance(mz, (int, float)) and isinstance(intensity, (int, float))
            ]
            paired_count = min(len(mzs), len(intensities))
            peak_diagnostics = {
                "mz_count": len(mzs),
                "intensity_count": len(intensities),
                "paired_count": paired_count,
                "invalid_pair_count": paired_count - len(peaks),
                "unpaired_count": abs(len(mzs) - len(intensities)),
            }
            item = {
                "id": spec_id,
                "molecule_id": (
                    row.get("molecule_id") or row.get("inchikey14") or row.get("inchikey")
                ),
                "smiles": (
                    row.get("smiles") or row.get("normalized_smiles") or row.get("canonical_smiles")
                ),
                "precursor_mz": row.get("precursor_mz"),
                "adduct": row.get("adduct"),
                "ion_mode": row.get("ionization_mode"),
                "collision_energy": row.get("collision_energy_ev"),
                "instrument": row.get("instrument_type"),
                "peak_count": len(peaks),
            }
            metadata = {
                k: v for k, v in row.items() if k not in {"ms2_mzs", "ms2_normalized_intensities"}
            }
            detail = {**item, "peaks": peaks, "peak_diagnostics": peak_diagnostics,
                      "metadata": metadata}
            self.records.append(detail)
            self.by_id[spec_id] = detail
        self._structures = None

    def summary(self):
        return {
            "source": self.source,
            "label": "Synthetic demo — not measured data" if self.demo else Path(self.source).name,
            "rows": len(self.records),
            "total_rows": self.total_rows,
            "truncated": len(self.records) < self.total_rows,
            "molecules": len(
                {str(r["molecule_id"]) for r in self.records if r["molecule_id"] is not None}
            ),
            "columns": self.columns,
            "has_structures": any(r["smiles"] for r in self.records),
            "demo": self.demo,
            "provenance": self.provenance,
        }

    def search(self, query="", offset=0, limit=50):
        q = query.casefold().strip()
        matches = [
            r
            for r in self.records
            if not q
            or q in json.dumps(r["metadata"], ensure_ascii=False).casefold()
            or q in r["id"].casefold()
        ]
        return {
            "items": [
                {k: v for k, v in r.items() if k not in {"peaks", "peak_diagnostics", "metadata"}}
                for r in matches[offset : offset + limit]
            ],
            "total": len(matches),
            "offset": offset,
            "limit": limit,
        }

    def structures(self):
        if self._structures is None:
            grouped = {}
            for record in self.records:
                smiles = record["smiles"]
                if not isinstance(smiles, str):
                    continue
                try:
                    mol = molecule(smiles)
                except ValueError:
                    continue
                canonical = Chem.MolToSmiles(mol)
                if canonical not in grouped:
                    grouped[canonical] = (GENERATOR.GetFingerprint(mol), [])
                grouped[canonical][1].append(record)
            self._structures = [
                (canonical, fingerprint, records)
                for canonical, (fingerprint, records) in grouped.items()
            ]
        return self._structures

    def related(self, smiles, candidates=None, limit=12):
        query = molecule(smiles)
        fingerprint = GENERATOR.GetFingerprint(query)
        if candidates is None:
            candidates = []
        if not isinstance(candidates, list) or len(candidates) > 100:
            raise ValueError("candidates must be a list of at most 100 SMILES")
        pool = list(self.structures())
        seen = {canonical for canonical, _, _ in pool}
        for candidate in candidates:
            mol = molecule(candidate)
            canonical = Chem.MolToSmiles(mol)
            if canonical not in seen:
                seen.add(canonical)
                pool.append((canonical, GENERATOR.GetFingerprint(mol), []))
        ranked = sorted(
            (
                (DataStructs.TanimotoSimilarity(fingerprint, fp), canonical, records)
                for canonical, fp, records in pool
            ),
            key=lambda x: (-x[0], x[1]),
        )[:limit]
        return {
            "query_smiles": Chem.MolToSmiles(query),
            "items": [
                {
                    "smiles": canonical,
                    "similarity": score,
                    "molecule_id": records[0]["molecule_id"] if records else None,
                    "spectrum_id": records[0]["id"] if records else None,
                    "spectra": [
                        {
                            key: record[key]
                            for key in (
                                "id", "adduct", "ion_mode", "instrument", "collision_energy",
                                "precursor_mz",
                            )
                        }
                        for record in records
                    ],
                    "svg": svg_for(molecule(canonical)),
                }
                for score, canonical, records in ranked
            ],
        }


def make_handler(dataset):
    class Handler(BaseHTTPRequestHandler):
        def respond(self, data, status=200):
            body = json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlsplit(self.path)
            try:
                if url.path == "/api/dataset":
                    return self.respond(dataset.summary())
                if url.path == "/api/spectra":
                    params = parse_qs(url.query)
                    offset = max(0, int(params.get("offset", [0])[0]))
                    limit = min(100, max(1, int(params.get("limit", [50])[0])))
                    return self.respond(dataset.search(params.get("q", [""])[0], offset, limit))
                if url.path.startswith("/api/spectra/"):
                    item = dataset.by_id.get(unquote(url.path[len("/api/spectra/") :]))
                    return self.respond(
                        item if item else {"error": "Spectrum not found"}, 200 if item else 404
                    )
                if url.path.startswith("/api/"):
                    return self.respond({"error": "Not found"}, 404)
                root = (ROOT / "static").resolve()
                path = (root / unquote(url.path).lstrip("/")).resolve()
                if url.path == "/":
                    path = root / "index.html"
                if not path.is_relative_to(root) or not path.is_file():
                    return self.respond({"error": "Not found"}, 404)
                content = path.read_bytes()
                self.send_response(200)
                self.send_header(
                    "Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                )
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            except ValueError as exc:
                self.respond({"error": str(exc)}, 400)

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > MAX_BODY:
                    raise ValueError("Invalid request size")
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict):
                    raise ValueError("Expected a JSON object")
                if self.path == "/api/structure":
                    mol = molecule(data.get("smiles"))
                    return self.respond({"smiles": Chem.MolToSmiles(mol), "svg": svg_for(mol)})
                if self.path == "/api/related":
                    limit = min(50, max(1, int(data.get("limit", 12))))
                    return self.respond(
                        dataset.related(data.get("smiles"), data.get("candidates"), limit)
                    )
                self.respond({"error": "Not found"}, 404)
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self.respond({"error": str(exc)}, 400)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=DEFAULT_DATA, help="Local CASMI-compatible Parquet file"
    )
    parser.add_argument("--demo", action="store_true", help="Use explicitly synthetic demo spectra")
    parser.add_argument(
        "--max-rows", type=int, default=5000, help="Maximum rows loaded into memory (default 5000)"
    )
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    data = Dataset(args.data, max_rows=args.max_rows, demo=args.demo)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(data))
    print(
        f"CASMI EDA: http://127.0.0.1:{server.server_port} | {data.source} | "
        f"loaded {len(data.records)}/{data.total_rows} spectra",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

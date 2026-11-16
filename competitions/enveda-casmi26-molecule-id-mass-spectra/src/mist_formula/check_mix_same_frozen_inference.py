"""Verify a mixup checkpoint with the frozen public-0.479 inference source."""

import argparse
import importlib.util
import json
from pathlib import Path

import pandas as pd
import torch
from train_mix_same import Cache, evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--precise-mzs", type=Path, required=True)
    parser.add_argument("--source-metadata", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    frozen_spec = importlib.util.spec_from_file_location("frozen_mist_model", args.frozen_model)
    frozen = importlib.util.module_from_spec(frozen_spec)
    frozen_spec.loader.exec_module(frozen)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = frozen.FormulaTransformer(**checkpoint["model_config"]).cuda()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    catalog = frozen.FormulaCatalog(
        checkpoint["formula_counts"], max_peaks=checkpoint["max_peaks"],
        ppm=checkpoint["ppm"], k=checkpoint["candidate_formulas"],
    )
    data = Cache(args.cache, args.precise_mzs, args.source_metadata)
    panel = pd.read_parquet(args.panel)
    expected = json.loads(args.manifest.read_text())["validation"]
    actual = evaluate(model, data, catalog, panel, "cuda", len(expected["ranks"]))
    if actual["query_mols"] != expected["query_mols"] or actual["ranks"] != expected["ranks"]:
        raise AssertionError("Frozen inference model disagrees with trainer validation ranks")
    if actual["mrr25"] != expected["mrr25"] or actual["top1"] != expected["top1"]:
        raise AssertionError("Frozen inference model disagrees with trainer summary")
    result = {"parity": "exact", "queries": actual["queries"],
              "mrr25": actual["mrr25"], "top1": actual["top1"]}
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()

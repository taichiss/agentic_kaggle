from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NOTEBOOK = REPO_ROOT / "notebook.ipynb"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "kaggle_kernel" / "ver8_train_audio_mlp_submit"
DEFAULT_SLUG = "birdclef-2026-full-fit-probe-ver8-submit"
DEFAULT_TITLE = "BirdCLEF 2026 Full-Fit Probe Ver8 Submit"
DEFAULT_MODEL_SOURCE = "google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1"
DEFAULT_DATASET_SOURCES = (
    "suzukitaichi/birdclef-2026-perch-probe-cv-models",
    "rishikeshjani/perch-onnx-for-birdclef-2026",
)
DEFAULT_KERNEL_SOURCES = ("ashok205/tf-wheels",)
DEFAULT_COMPETITION_SOURCES = ("birdclef-2026",)

EXTRA_LOADER_CODE = """

TRAIN_AUDIO_MLP_HEAD_BUNDLE_NAME = "train_audio_mlp_head_bundle.joblib"
TRAIN_AUDIO_MLP_HEAD_DIRS = [
    Path("/kaggle/input/birdclef-2026-perch-probe-cv-models"),
    Path("/kaggle/input/datasets/suzukitaichi/birdclef-2026-perch-probe-cv-models"),
]

def _load_train_audio_mlp_head_bundle():
    checked = []
    for root in TRAIN_AUDIO_MLP_HEAD_DIRS:
        direct = root / TRAIN_AUDIO_MLP_HEAD_BUNDLE_NAME
        checked.append(str(direct))
        if direct.exists():
            return joblib.load(direct), str(direct)

        zip_candidates = sorted(root.glob("*.zip"))
        for zip_path in zip_candidates:
            with zipfile.ZipFile(zip_path) as zf:
                for member in zf.namelist():
                    if member.lstrip("./").endswith(TRAIN_AUDIO_MLP_HEAD_BUNDLE_NAME):
                        with zf.open(member) as fp:
                            return joblib.load(io.BytesIO(fp.read())), f"{zip_path}!{member}"

    for path in INPUT_ROOT.rglob(TRAIN_AUDIO_MLP_HEAD_BUNDLE_NAME):
        return joblib.load(path), str(path)

    print(
        f"train_audio MLP head bundle not found. Checked direct paths: {checked}. "
        "Proceeding with probe-only submit."
    )
    return None, None

def _gelu_np(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * (x ** 3))))

def _apply_layer_norm_np(x, gamma, beta, epsilon):
    mean = x.mean(axis=1, keepdims=True)
    variance = ((x - mean) ** 2).mean(axis=1, keepdims=True)
    normalized = (x - mean) / np.sqrt(variance + epsilon)
    return normalized * gamma + beta

def predict_train_audio_mlp_head_logits(emb_test, bundle):
    x = emb_test.astype(np.float32, copy=False)
    x = _apply_layer_norm_np(
        x,
        np.asarray(bundle["layer_norm_gamma"], dtype=np.float32),
        np.asarray(bundle["layer_norm_beta"], dtype=np.float32),
        float(bundle.get("layer_norm_epsilon", 1e-6)),
    )
    x = x @ np.asarray(bundle["hidden_kernel"], dtype=np.float32) + np.asarray(
        bundle["hidden_bias"], dtype=np.float32
    )
    x = _gelu_np(x).astype(np.float32)
    logits = x @ np.asarray(bundle["output_kernel"], dtype=np.float32) + np.asarray(
        bundle["output_bias"], dtype=np.float32
    )
    return logits.astype(np.float32)
"""

NEW_SUBMIT_CELL_SOURCE = """# ── Cell 10: Submit with probe + continued-head blend ─────────────────

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

assert SUBMIT_BACKEND == "full_fit_probe", (
    "This notebook variant is configured for saved full-fit probe submit. "
    "Set SUBMIT_BACKEND='full_fit_probe'."
)

train_audio_head_bundle, train_audio_head_path = _load_train_audio_mlp_head_bundle()
if train_audio_head_bundle is None:
    raise FileNotFoundError("train_audio MLP head bundle is required for ver8 submit")

probe_bundle, probe_bundle_path = _load_full_fit_probe_bundle(FULL_FIT_PROBE_PATTERN)
print(f"Using saved full-fit probe bundle: {probe_bundle_path}")
print(f"  trained classes: {len(probe_bundle['classifiers'])}")
print(f"Using final MLP head bundle: {train_audio_head_path}")
print(f"  hidden dim: {train_audio_head_bundle.get('hidden_dim')}")
print(f"  soundscape dataset: {train_audio_head_bundle.get('soundscape_dataset')}")

t0 = time.time()
probe_scores = apply_full_fit_probe_bundle(emb_te, sc_te, probe_bundle)
head_scores = predict_train_audio_mlp_head_logits(emb_te, train_audio_head_bundle)
alpha = np.asarray(
    train_audio_head_bundle.get("probe_blend_alpha_by_class", np.zeros(len(PRIMARY_LABELS))),
    dtype=np.float32,
)
alpha = np.clip(alpha, 0.0, 1.0)
final_scores = (
    (1.0 - alpha[None, :]) * probe_scores + alpha[None, :] * head_scores
).astype(np.float32)
print(f"Probe + head blend inference: {time.time()-t0:.1f}s")
print(f"Score range [{final_scores.min():.3f}, {final_scores.max():.3f}]")
print(f"Alpha summary mean={alpha.mean():.4f} nonzero={(alpha > 0).sum()}")

probs = sigmoid(final_scores)
probs = np.clip(probs, 0.0, 1.0)

sub = pd.DataFrame(probs.astype(np.float32), columns=PRIMARY_LABELS)
sub.insert(0, "row_id", meta_te["row_id"].values)
assert list(sub.columns) == ["row_id"] + PRIMARY_LABELS
assert len(sub) == len(test_paths) * N_WINDOWS
assert not sub.isna().any().any()
sub.to_csv("submission.csv", index=False)

print(f"\\nsubmission.csv saved — shape {sub.shape}")
print(f"Total wall time: {(time.time() - _WALL_START)/60:.1f} min")
"""


def load_kaggle_username(config_path: Path) -> str:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    username = payload.get("username")
    if not isinstance(username, str) or not username:
        raise ValueError(f"username missing in {config_path}")
    return username


def patch_notebook(notebook_path: Path) -> dict[str, object]:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    patched_loader = False
    patched_submit = False

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if "def apply_full_fit_probe_bundle(emb_test, scores_test, bundle):" in source:
            if "def _load_train_audio_mlp_head_bundle():" not in source:
                source = source.rstrip() + EXTRA_LOADER_CODE + "\n"
                cell["source"] = source.splitlines(keepends=True)
            patched_loader = True
            continue

        if (
            'assert SUBMIT_BACKEND == "full_fit_probe"' in source
            and 'sub.to_csv("submission.csv", index=False)' in source
        ):
            cell["source"] = NEW_SUBMIT_CELL_SOURCE.splitlines(keepends=True)
            patched_submit = True

    if not patched_loader:
        raise ValueError("failed to locate full-fit probe loader cell")
    if not patched_submit:
        raise ValueError("failed to locate submit cell")
    return notebook


def build_metadata(
    owner: str,
    slug: str,
    title: str,
    notebook_name: str,
) -> dict[str, object]:
    return {
        "id": f"{owner}/{slug}",
        "title": title,
        "code_file": notebook_name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "false",
        "enable_tpu": "false",
        "enable_internet": "false",
        "competition_sources": list(DEFAULT_COMPETITION_SOURCES),
        "dataset_sources": list(DEFAULT_DATASET_SOURCES),
        "kernel_sources": list(DEFAULT_KERNEL_SOURCES),
        "model_sources": [DEFAULT_MODEL_SOURCE],
    }


def prepare_bundle(
    notebook_path: Path,
    output_dir: Path,
    owner: str,
    slug: str,
    title: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    patched_notebook = patch_notebook(notebook_path)

    notebook_name = notebook_path.name
    target_notebook = output_dir / notebook_name
    target_notebook.write_text(
        json.dumps(patched_notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )

    metadata = build_metadata(
        owner=owner,
        slug=slug,
        title=title,
        notebook_name=notebook_name,
    )
    metadata_path = output_dir / "kernel-metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return metadata_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--kaggle-config", type=Path, default=REPO_ROOT / "kaggle.json")
    parser.add_argument("--owner", type=str, default=None)
    parser.add_argument("--slug", type=str, default=DEFAULT_SLUG)
    parser.add_argument("--title", type=str, default=DEFAULT_TITLE)
    args = parser.parse_args()

    owner = args.owner or load_kaggle_username(args.kaggle_config)
    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    metadata_path = prepare_bundle(
        notebook_path=args.notebook,
        output_dir=args.output_dir,
        owner=owner,
        slug=args.slug,
        title=args.title,
    )
    print(metadata_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

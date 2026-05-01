from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path

KNOWN_LAYERS = (
    "core",
    "providers",
    "data",
    "features",
    "models",
    "training",
    "inference",
)

ALLOWED_IMPORTS = {
    "core": {"core"},
    "providers": {"core", "providers"},
    "data": {"core", "providers", "data"},
    "features": {"core", "data", "features"},
    "models": {"core", "features", "models"},
    "training": {"core", "providers", "data", "features", "models", "training"},
    "inference": {"core", "providers", "data", "features", "models", "inference"},
}


@dataclass(frozen=True)
class Violation:
    path: Path
    line: int
    message: str


def iter_python_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts and ".venv" not in path.parts
    )


def file_layer(src_root: Path, path: Path) -> str | None:
    rel = path.relative_to(src_root)
    if not rel.parts:
        return None
    layer = rel.parts[0]
    if layer in KNOWN_LAYERS and len(rel.parts) >= 2:
        return layer
    return None


def imported_layers(tree: ast.AST) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                layer = alias.name.split(".", 1)[0]
                if layer in KNOWN_LAYERS:
                    found.append((node.lineno, layer))
        elif isinstance(node, ast.ImportFrom):
            if node.level != 0 or node.module is None:
                continue
            layer = node.module.split(".", 1)[0]
            if layer in KNOWN_LAYERS:
                found.append((node.lineno, layer))
    return found


def validate_repo(repo_root: Path) -> list[Violation]:
    src_root = repo_root / "src"
    if not src_root.exists():
        return []

    violations: list[Violation] = []
    for path in iter_python_files(src_root):
        layer = file_layer(src_root, path)
        if layer is None:
            continue

        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        allowed = ALLOWED_IMPORTS[layer]
        for line, imported_layer in imported_layers(tree):
            if imported_layer not in allowed:
                violations.append(
                    Violation(
                        path=path.relative_to(repo_root),
                        line=line,
                        message=(
                            f"{layer} layer cannot import {imported_layer}; "
                            f"allowed={sorted(allowed)}"
                        ),
                    )
                )
    return violations


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    violations = validate_repo(repo_root)
    if not (repo_root / "src").exists():
        print("archgate OK: src/ is not present yet; nothing to validate.")
        return 0

    if violations:
        for violation in violations:
            print(f"{violation.path}:{violation.line}: {violation.message}", file=sys.stderr)
        return 1

    checked_files = len(iter_python_files(repo_root / "src"))
    print(f"archgate OK: validated {checked_files} Python files under src/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

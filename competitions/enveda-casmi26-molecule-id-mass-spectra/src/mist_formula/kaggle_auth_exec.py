"""Execute a Kaggle command with the workspace token, without echoing it."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    source = Path("/home/dev/Biohub/.env")
    values = {
        key.strip(): value.strip().strip('"').strip("'")
        for line in source.read_text().splitlines()
        if "=" in line and not line.lstrip().startswith("#")
        for key, value in [line.split("=", 1)]
    }
    token = values.get("KAGGLE_API")
    if not token:
        raise RuntimeError("workspace Kaggle token missing")
    if len(sys.argv) < 2:
        raise ValueError("command missing")
    env = dict(os.environ, KAGGLE_API_TOKEN=token)
    os.execvpe(sys.argv[1], sys.argv[1:], env)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Feed information from git to the Nix store.
"""
import json
import subprocess
from pathlib import Path

OUT = Path(__file__).parent / "last-updated.json"

timestamps = {}

include = lambda ext: list(Path("src").rglob(f"*.{ext}"))  # noqa: E731
for file in include("md") + include("html"):
    # https://www.git-scm.com/docs/git-log#_pretty_formats
    timestamps[str(file)] = int(
        subprocess.run(
            ["git", "log", "--pretty=%at", "-1", file], capture_output=True
        ).stdout.decode()
    )

with OUT.open("w") as f:
    json.dump(timestamps, f, indent=4, sort_keys=True)

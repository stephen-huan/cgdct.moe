"""
Render a markdown document to html.

See https://pandoc.org/MANUAL.html.

Filters

- https://github.com/xu-cheng/pandoc-katex
- https://github.com/pandoc-ext/section-bibliographies
- https://codeberg.org/NomisIV/pandoc-tree-sitter-filter
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

source = Path("src")

build = Path("build")

# `pandoc --print-default-template=html`
template = Path("pandoc-templates/default.html5")
metadata = Path("metadata.json")

# `pandoc --list-extensions=markdown`
with Path("extensions.json").open() as f:
    extensions = "".join(sorted(json.load(f)))

katex = Path("lib/katex")
katex_config = Path("katex.toml")
highlight = Path("atom-one-light.theme")

# TODO: https://pandoc.org/filters.html
# https://github.com/pandoc-ext/section-bibliographies

# TODO: https://tree-sitter.github.io/tree-sitter/syntax-highlighting
# https://codeberg.org/NomisIV/pandoc-tree-sitter-filter


def render(
    path: Path,
    out: Path = build / "out.html",
    table_of_contents: bool = True,
    toc_depth: int = 3,
) -> int:
    """Render a path to html."""
    return subprocess.run(
        [
            "pandoc",
            f"--from=markdown{extensions}",
            "--to=html",
            "--standalone",
            f"--template={template}",
            f"--output={out}",
            f"--table-of-contents={table_of_contents}",
            f"--toc-depth={toc_depth}",
            f"--metadata-file={metadata}",
            f"--highlight-style={highlight}",
            "--filter=pandoc-katex",
            f"--css={katex / 'katex.min.css'}",
            path,
        ],
        env=os.environ
        | {
            "PANDOC_KATEX_CONFIG_FILE": katex_config,
        },
    ).returncode


def hash_file(path: Path) -> bytes:
    """Hash the file."""
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").digest()


def process_file(path: Path) -> None:
    """Process the file."""
    if path.is_file():
        dst = Path(str(path).replace(str(source), str(build)))
        if path.suffix == ".md":
            dst.parent.mkdir(exist_ok=True, parents=True)
            render(path, out=dst.with_suffix(".html"))
        else:
            if not dst.exists() or hash_file(dst) != hash_file(path):
                dst.parent.mkdir(exist_ok=True, parents=True)
                shutil.copy(path, dst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a static site")
    parser.add_argument(
        "-w", "--watch", action="store_true", help="watch for changes"
    )

    args = parser.parse_args()

    # TODO: Path.walk
    for path in source.glob("**/*"):
        process_file(path)

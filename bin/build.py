"""
Render a markdown document to html.

See https://pandoc.org/MANUAL.html.

Filters

- https://github.com/xu-cheng/pandoc-katex
- https://github.com/pandoc-ext/section-bibliographies
- https://codeberg.org/NomisIV/pandoc-tree-sitter-filter
"""

import json
import os
import subprocess
from pathlib import Path

build = Path("build")
build.mkdir(exist_ok=True)

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


if __name__ == "__main__":
    render(Path("src/blog/tomato-egg-noodle.md"))
    # render(Path("src/blog/algorithm-library.md"))
    # render(Path("src/blog/far.md"))

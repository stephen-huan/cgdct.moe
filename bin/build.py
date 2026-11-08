#!/usr/bin/env python3
"""
Render a markdown document to html.

See https://pandoc.org/MANUAL.html.

Filters

- https://github.com/xu-cheng/pandoc-katex
- https://github.com/lierdakil/pandoc-crossref
- https://github.com/pandoc-ext/section-bibliographies
- https://codeberg.org/NomisIV/pandoc-tree-sitter-filter
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

import yaml
from watchfiles import watch

# TODO: https://pandoc.org/filters.html
# https://github.com/pandoc-ext/section-bibliographies

# TODO: https://tree-sitter.github.io/tree-sitter/syntax-highlighting
# https://codeberg.org/NomisIV/pandoc-tree-sitter-filter

# TODO: fix katex suppress equation numbers (align*)

source = Path("src")
config = Path("config")
build = Path("build")

# `pandoc --print-default-template=html`
template = Path("pandoc-templates/default.html5")
templates = Path("templates")

katex_config = config / "katex.toml"
highlight = config / "atom-one-light.theme"

metadata_name = "metadata"
spec = spec_from_file_location(metadata_name, config / f"{metadata_name}.py")
if spec is not None and spec.loader is not None:
    metadata = module_from_spec(spec)
    sys.modules[metadata_name] = metadata
    spec.loader.exec_module(metadata)
    from metadata import get_metadata  # pyright: ignore

    JSONEncoder = (
        metadata.JSONEncoder
        if "JSONEncoder" in metadata.__dict__
        else json.JSONEncoder
    )
else:

    def get_metadata(
        path: Path, dst: Path, is_template: bool = False
    ) -> dict[str, Any]:
        """Return the metadata for the path."""
        (path, dst, is_template)  # pyright: ignore
        return {}

    from json import JSONEncoder


metadata_path = config / "metadata.json"
with metadata_path.open() as f:
    metadata = json.load(f)

# `pandoc --list-extensions=markdown`
with (config / "extensions.json").open() as f:
    extensions = "".join(sorted(json.load(f)))


def get_frontmatter(path: Path) -> dict[str, Any]:
    """Parse the yaml metadata header of the path."""
    return (
        next(yaml.safe_load_all(path.open()))
        if path.suffix == ".md" and path.open().readline() == "---\n"
        else {}
    )


def render(
    path: Path, dst: Path = build / "out.html", src: Path | None = None
) -> int:
    """Render a path to html."""
    path_metadata = get_frontmatter(path)
    # evaluate file as template and re-evaluate
    if path_metadata.get("is-template", False) and src is None:
        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".md"
        ) as tmpfile, tempfile.NamedTemporaryFile(
            mode="w+", suffix=".json"
        ) as metadata_tmpfile:
            json.dump(
                get_metadata(path, dst, is_template=True),
                metadata_tmpfile,
                cls=JSONEncoder,
            )
            metadata_tmpfile.flush()
            # https://github.com/jgm/pandoc/issues/1950#issuecomment-74613325
            subprocess.run(
                [
                    "pandoc",
                    f"--from=markdown{extensions}",
                    "--to=html",
                    f"--template={path}",
                    f"--output={tmpfile.name}",
                    f"--metadata-file={metadata_path}",
                    f"--metadata-file={metadata_tmpfile.name}",
                    path,
                ],
                check=True,
            )
            return render(path, dst, Path(tmpfile.name))
    if src is None:
        src = path
    table_of_contents = path_metadata.get("toc", metadata.get("toc", False))
    toc_depth = path_metadata.get("toc-depth", metadata.get("toc-depth", 4))
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".json") as tmpfile:
        json.dump(get_metadata(path, dst), tmpfile, cls=JSONEncoder)
        tmpfile.flush()
        return subprocess.run(
            [
                "pandoc",
                f"--from=markdown{extensions}",
                "--to=html",
                "--standalone",
                f"--template={template}",
                f"--include-in-header={templates / 'favicon.html'}",
                f"--output={dst}",
                f"--table-of-contents={table_of_contents}",
                f"--toc-depth={toc_depth}",
                f"--metadata-file={metadata_path}",
                f"--metadata-file={tmpfile.name}",
                f"--highlight-style={highlight}",
                "--filter=pandoc-crossref",
                "--citeproc",
                "--filter=pandoc-katex",
                src,
            ],
            env=os.environ
            | {
                "PANDOC_KATEX_CONFIG_FILE": katex_config,
            },
            check=True,
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
            if path.name not in ["index.md", "404.md"]:
                dst = dst.parent / path.stem / "index.md"
            dst.parent.mkdir(exist_ok=True, parents=True)
            render(path, dst=dst.with_suffix(".html"))
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
    if args.watch:
        print("Watching for changes...")
        for changes in watch(source, raise_interrupt=False):
            _, paths = zip(*changes)
            for path in paths:
                process_file(Path(path).relative_to(source.parent.absolute()))

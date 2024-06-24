"""
Render a markdown document to html.

See https://pandoc.org/MANUAL.html.

Filters

- https://github.com/xu-cheng/pandoc-katex
- https://github.com/pandoc-ext/section-bibliographies
- https://codeberg.org/NomisIV/pandoc-tree-sitter-filter
"""

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from watchfiles import watch

source = Path("src")
config = Path("config")
build = Path("build")

# `pandoc --print-default-template=html`
template = Path("pandoc-templates/default.html5")
templates = Path("templates")

metadata_path = config / "metadata.json"
with metadata_path.open() as f:
    metadata = json.load(f)

# `pandoc --list-extensions=markdown`
with (config / "extensions.json").open() as f:
    extensions = "".join(sorted(json.load(f)))

katex_config = config / "katex.toml"
highlight = config / "atom-one-light.theme"

with (Path(__file__).parent / "last-updated.json").open() as f:
    last_updated = json.load(f)
    tzinfo = ZoneInfo(metadata["timezone"])
    last_updated = {
        Path(path): datetime.fromtimestamp(timestamp, tz=tzinfo)
        for path, timestamp in last_updated.items()
    }

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
    if path in last_updated and path not in metadata["footer-exclude"]:
        dt = last_updated[path]
        date = dt.strftime(metadata["date-format"])
        timestamp = f'<time datetime="{dt.isoformat()}">{date}</time>'
        lastupdated = f"Last updated: {timestamp}."
    else:
        lastupdated = ""
    url = f"{metadata['git-repo']}{path}"
    space = " " if len(lastupdated) > 0 else ""
    pagesource = f'<a href="{url}">Page source</a>.{space}'
    headers = copy.deepcopy(metadata["headers"])
    for header in headers:
        if header["url"] == "/":
            header["active"] = path == source / "index.md"
        elif header["url"] in str(path):
            header["active"] = True
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".json") as tmpfile:
        json.dump({"headers": headers}, tmpfile)
        tmpfile.flush()
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
                f"--metadata-file={metadata_path}",
                f"--metadata-file={tmpfile.name}",
                f"--highlight-style={highlight}",
                "--filter=pandoc-katex",
                f"--variable=git-pagesource:{pagesource}",
                f"--variable=git-lastupdated:{lastupdated}",
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
            if path.name not in ["index.md", "404.md"]:
                dst = dst.parent / path.stem / "index.md"
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
    if args.watch:
        print("Watching for changes...")
        for changes in watch(source, raise_interrupt=False):
            _, paths = zip(*changes)
            for path in paths:
                process_file(Path(path).relative_to(source.parent.absolute()))

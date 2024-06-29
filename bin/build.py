#!/usr/bin/env python3
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

import yaml
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
    tzinfo = ZoneInfo(metadata.get("timezone", "UTC"))
    last_updated = {
        Path(path): datetime.fromtimestamp(timestamp, tz=tzinfo)
        for path, timestamp in last_updated.items()
    }

with (config / "news.json").open() as f:
    news = json.load(f)

# TODO: https://pandoc.org/filters.html
# https://github.com/pandoc-ext/section-bibliographies

# TODO: https://tree-sitter.github.io/tree-sitter/syntax-highlighting
# https://codeberg.org/NomisIV/pandoc-tree-sitter-filter

# TODO: generate from stdin


def get_metadata(path: Path) -> dict:
    """Parse the yaml metadata of the path."""
    return (
        next(yaml.safe_load_all(path.open()))
        if path.suffix == ".md" and path.open().readline() == "---\n"
        else {}
    )


def render(
    path: Path, out: Path = build / "out.html", src: Path | None = None
) -> int:
    """Render a path to html."""
    path_metadata = get_metadata(path)
    # evaluate file as template and re-evaluate
    # https://github.com/jgm/pandoc/issues/1950#issuecomment-74613325
    if path_metadata.get("is-template", False) and src is None:
        metadata_tmp = {
            "news": news[: path_metadata.get("news-entries", len(news))]
        }
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".md") as tmpfile:
            with tempfile.NamedTemporaryFile(
                mode="w+", suffix=".json"
            ) as metadata_tmpfile:
                json.dump(metadata_tmp, metadata_tmpfile)
                metadata_tmpfile.flush()
                subprocess.run(
                    [
                        "pandoc",
                        f"--template={path}",
                        f"--output={tmpfile.name}",
                        f"--metadata-file={metadata_path}",
                        f"--metadata-file={metadata_tmpfile.name}",
                        path,
                    ]
                )
                return render(path, out, Path(tmpfile.name))
    if src is None:
        src = path
    table_of_contents = path_metadata.get("toc", metadata.get("toc", False))
    toc_depth = path_metadata.get("toc-depth", metadata.get("toc-depth", 4))
    if path in last_updated and str(path) not in metadata.get(
        "footer-exclude", []
    ):
        dt = last_updated[path]
        date = dt.strftime(metadata.get("date-format", "%Y-%m-%d"))
        timestamp = f'<time datetime="{dt.isoformat()}">{date}</time>'
        lastupdated = f"Last updated: {timestamp}."
    else:
        lastupdated = ""
    url = f"{metadata.get('git-repo', '')}{path}"
    space = " " if len(lastupdated) > 0 else ""
    pagesource = f'<a href="{url}">Page source</a>.{space}'
    headers = copy.deepcopy(metadata.get("headers", []))
    for header in headers:
        if header["url"] == "/":
            header["active"] = path == source / "index.md"
        elif header["url"] in str(path):
            header["active"] = True
    css = ["/css/main.css"]
    css_path = Path(
        str(path).replace(str(source), "src/css/root")
    ).with_suffix(".css")
    if css_path.exists():
        css.append(f"/{css_path.relative_to(source)}")
    metadata_tmp = {
        "headers": headers,
        "css": css,
    }
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".json") as tmpfile:
        json.dump(metadata_tmp, tmpfile)
        tmpfile.flush()
        return subprocess.run(
            [
                "pandoc",
                f"--from=markdown{extensions}",
                "--to=html",
                "--standalone",
                f"--template={template}",
                f"--include-in-header={templates / 'favicon.html'}",
                f"--output={out}",
                f"--table-of-contents={table_of_contents}",
                f"--toc-depth={toc_depth}",
                f"--metadata-file={metadata_path}",
                f"--metadata-file={tmpfile.name}",
                f"--highlight-style={highlight}",
                "--filter=pandoc-katex",
                f"--variable=git-pagesource:{pagesource}",
                f"--variable=git-lastupdated:{lastupdated}",
                src,
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

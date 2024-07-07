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
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from zoneinfo import ZoneInfo

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

metadata_path = config / "metadata.json"
with metadata_path.open() as f:
    metadata = json.load(f)

date_format = metadata.get("date-format", "%Y-%m-%d")
tzinfo = ZoneInfo(metadata.get("timezone", "UTC"))

# `pandoc --list-extensions=markdown`
with (config / "extensions.json").open() as f:
    extensions = "".join(sorted(json.load(f)))

with (config / "news.json").open() as f:
    news = json.load(f)

with (config / "publications.json").open() as f:
    publications = json.load(f)

with (Path(__file__).parent / "last-updated.json").open() as f:
    last_updated = {
        Path(path): datetime.datetime.fromtimestamp(timestamp, tz=tzinfo)
        for path, timestamp in json.load(f).items()
    }


class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder."""

    def default(self, o: object) -> object:
        """Support serializing date objects."""
        if isinstance(o, datetime.date):
            return o.strftime(date_format)
        return super().default(o)


def get_metadata(path: Path) -> dict:
    """Parse the yaml metadata of the path."""
    return (
        next(yaml.safe_load_all(path.open()))
        if path.suffix == ".md" and path.open().readline() == "---\n"
        else {}
    )


def render(
    path: Path, dst: Path = build / "out.html", src: Path | None = None
) -> int:
    """Render a path to html."""
    path_metadata = get_metadata(path)
    # evaluate file as template and re-evaluate
    if path_metadata.get("is-template", False) and src is None:
        selected_publications = publications
        if path_metadata.get("publications-selected"):
            selected_publications = sorted(
                (
                    publication
                    for publication in publications
                    if publication["selected"] is not None
                ),
                key=lambda publication: publication["selected"],
            )
        elif path_metadata.get("publications-by-year"):
            selected_publications = {}
            for publication in publications:
                year = publication["year"]
                if year not in selected_publications:
                    selected_publications[year] = []
                selected_publications[year].append(publication)
            selected_publications = [
                {"year": year, "publications": selected_publications[year]}
                for year in sorted(selected_publications, reverse=True)
            ]
        posts = (
            {
                "url": f"/{post.relative_to(source).parent / post.stem}/",
                "title": (post_metadata := get_metadata(post))["title"],
                "subtitle": post_metadata.get(
                    "subtitle", post_metadata.get("description", None)
                ),
                "date": post_metadata["date"],
                "keywords": post_metadata.get("keywords", []),
            }
            for post in (source / "blog").glob("*.md")
            if post != source / "blog" / "index.md"
        )
        metadata_tmp = {
            "news": news[: path_metadata.get("news-entries", len(news))],
            "publications": selected_publications,
            "blog-posts": sorted(
                posts, key=lambda post: post["date"], reverse=True
            ),
        }
        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".md"
        ) as tmpfile, tempfile.NamedTemporaryFile(
            mode="w+", suffix=".json"
        ) as metadata_tmpfile:
            json.dump(metadata_tmp, metadata_tmpfile, cls=JSONEncoder)
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
    if path in last_updated and str(path) not in metadata.get(
        "footer-exclude", []
    ):
        dt = last_updated[path]
        date = dt.strftime(date_format)
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
        elif header["url"] in str(dst):
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
        json.dump(metadata_tmp, tmpfile, cls=JSONEncoder)
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
                "--filter=pandoc-katex",
                f"--variable=git-pagesource:{pagesource}",
                f"--variable=git-lastupdated:{lastupdated}",
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

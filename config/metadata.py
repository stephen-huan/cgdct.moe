import copy
import datetime
import json
from functools import cache
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml

# TODO: package build.py

source = Path("src")
config = Path("config")

with (config / "metadata.json").open() as f:
    static_metadata = json.load(f)

date_format = static_metadata.get("date-format", "%Y-%m-%d")
tzinfo = ZoneInfo(static_metadata.get("timezone", "UTC"))

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


@cache
def get_frontmatter(path: Path) -> dict[str, Any]:
    """Parse the yaml metadata header of the path."""
    return (
        next(yaml.safe_load_all(path.open()))
        if path.suffix == ".md" and path.open().readline() == "---\n"
        else {}
    )


def get_css(path: Path) -> list[str]:
    """Return the stylesheets for the path."""
    css = ["/css/main.css"]
    css_path = Path(
        str(path).replace(str(source), "src/css/root")
    ).with_suffix(".css")
    if css_path.exists():
        css.append(f"/{css_path.relative_to(source)}")
    return css


def get_headers(path: Path, dst: Path) -> list[dict[str, str | bool]]:
    """Get the active header for the path."""
    headers = copy.deepcopy(static_metadata.get("headers", []))
    for header in headers:
        if header["url"] == "/":
            header["active"] = path == source / "index.md"
        elif header["url"] in str(dst):  # noqa: SIM114
            header["active"] = True
        elif header["name"] == "blog" and path.parents[0] == source / "tag":
            header["active"] = True
    return headers


def get_pagesource(path: Path) -> str:
    """Get the source of the path."""
    url = f"{static_metadata.get('git-repo', '')}{path}"
    return f"<a href='{url}'>Page source</a>."


def get_lastupdated(path: Path) -> str:
    """Return the last updated date for the path."""
    if (
        str(path) in static_metadata.get("footer-exclude", [])
        or path not in last_updated
    ):
        return ""
    else:
        dt = last_updated[path]
        format = get_frontmatter(path).get("date-format", date_format)
        date = dt.strftime(format)
        timestamp = f'<time datetime="{dt.isoformat()}">{date}</time>'
        return f"&nbsp;Last updated: {timestamp}."


@cache
def get_posts() -> list[dict[str, Any]]:
    """Get blog posts."""
    return [
        {
            "url": f"/{post.relative_to(source).parent / post.stem}/",
            "title": (post_metadata := get_frontmatter(post))["title"],
            "subtitle": post_metadata.get(
                "subtitle", post_metadata.get("description", None)
            ),
            "date": post_metadata["date"],
            "keywords": post_metadata.get("keywords", []),
        }
        for post in (source / "blog").glob("*.md")
        if post != source / "blog" / "index.md"
    ]


@cache
def get_projects() -> list[dict[str, Any]]:
    """Get project pages."""
    return [
        {
            "url": f"/{post.relative_to(source).parent / post.stem}/",
            "title": (
                post_metadata["title"]
                if (post_metadata := get_frontmatter(post))["title"]
                else post_metadata["pagetitle"]
            ),
            "subtitle": post_metadata.get(
                "subtitle", post_metadata.get("description", None)
            ),
            "date": post_metadata["date"],
            "keywords": post_metadata.get("keywords", []),
        }
        for post in (source / "projects").glob("*.md")
        if post != source / "projects" / "index.md"
    ]


def get_publications(path_metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Get publications for the path."""
    if path_metadata.get("publications-selected"):
        return sorted(
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
        return [
            {"year": year, "publications": selected_publications[year]}
            for year in sorted(selected_publications, reverse=True)
        ]
    else:
        return publications


def get_metadata(
    path: Path, dst: Path, is_template: bool = False
) -> dict[str, Any]:
    """Return the metadata for the path."""
    if is_template:
        path_metadata = get_frontmatter(path)
        posts = get_posts()
        projects = get_projects()
        keywords = get_frontmatter(source / "blog" / "index.md")[
            "display-keywords"
        ]
        return {
            "news": news[: path_metadata.get("news-entries", len(news))],
            "publications": get_publications(path_metadata),
            "blog-posts": sorted(
                posts, key=lambda post: post["date"], reverse=True
            ),
            "posts-keyword": {
                keyword: {
                    "posts": sorted(
                        (
                            post
                            for post in posts + projects
                            if keyword in post["keywords"]
                        ),
                        key=lambda post: post["date"],
                        reverse=True,
                    )
                }
                for keyword in keywords
            },
        }
    else:
        return {
            "headers": get_headers(path, dst),
            "css": get_css(path),
            "git-pagesource": get_pagesource(path),
            "git-lastupdated": get_lastupdated(path),
            "self": get_frontmatter(path),
        }

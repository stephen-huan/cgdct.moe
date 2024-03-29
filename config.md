<!--
Add here global page variables to use throughout your website.
-->
+++
author = "Stephen Huan"
pgp = "EA6E27948C7DBF5D0DF085A10FBC2E3BA99DD60E"
date_format= "yyyy-mm-dd"
mintoclevel = 2

# Add here files or directories that should be ignored by Franklin, otherwise
# these files might be copied and, if markdown, processed by Franklin which
# you might not want. Indicate directories by ending the name with a `/`.
# Base files such as LICENSE.md and README.md are ignored by default.
ignore = [
    "Project.toml",
    "Manifest.toml",
    ".JuliaFormatter.toml",
    "node_modules/",
    "package-lock.json",
    "package.json",
    ".prettierrc.json",
    ".prettierignore",
    "pyproject.toml",
    "__pycache__/",
    "flake.nix",
    "flake.lock",
    ".direnv/",
    ".envrc",
    "default.nix",
    "node-env.nix",
    "node-packages.nix",
    ".editorconfig",
    ".vnurc",
    ".lycheecache",
    "lychee.toml",
    "bin/",
    "pkgs/",
    "src/",
    "utils/",
    "_css/",
]

# RSS (the website_{title, descr, url} must be defined to get RSS)
generate_rss = true
website_title = "Stephen Huan"
website_descr = "Stephen Huan's personal website"
website_url   = "https://cgdct.moe/"

# doesn't seem to have a default value for tag pages
div_content = "franklin-content"

auto_code_path = true

# header structure (url, display name)
headers = [
    ("/", "about"),
    ("/blog/", "blog"),
    ("/projects/", "projects"),
    ("/publications/", "publications"),
    ("/assets/pdf/cv.pdf", "cv"),
]

# git repo for page source
git_repo = "https://github.com/stephen-huan/cgdct.moe/blob/master"

footer_text = "さみしいも、たのしい。"

# footer exclude
footer_exclude = Set(
    ["/404/", "/blog/", "/projects/", "/publications/"]
)

# path to publications data
publications = "_assets/publications.json"

# path to last updated data
last_updated = "src/last-updated.json"
+++

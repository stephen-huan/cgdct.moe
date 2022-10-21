<!--
Add here global page variables to use throughout your website.
-->
+++
author = "Stephen Huan"
date_format= "yyyy-mm-dd"
mintoclevel = 2

# Add here files or directories that should be ignored by Franklin, otherwise
# these files might be copied and, if markdown, processed by Franklin which
# you might not want. Indicate directories by ending the name with a `/`.
# Base files such as LICENSE.md and README.md are ignored by default.
ignore = ["Project.toml", "Manifest.toml",
          "node_modules/", "package-lock.json", "package.json"]

# RSS (the website_{title, descr, url} must be defined to get RSS)
generate_rss = true
website_title = "Stephen Huan"
website_descr = "Stephen Huan's personal website"
website_url   = "https://stephen-huan.github.io/"

auto_code_path = true

# character for the icon
icon = "浣"

# header structure (url, display name)
headers = [("/", "about"),
           ("/menu1/", "blog"),
           ("/menu2/", "projects"),
           ("/menu3/", "cv"),
          ]

pgp = "EA6E27948C7DBF5D0DF085A10FBC2E3BA99DD60E"
+++

<!--
Add here global latex commands to use throughout your pages.
-->
\newcommand{\R}{\mathbb R}
\newcommand{\scal}[1]{\langle #1 \rangle}

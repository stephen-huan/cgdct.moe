#!/usr/bin/env python3
"""
Generate tag pages from the blog.
"""

from build import get_metadata, source

# TODO: relative_to(walk_up=True)

template = """\
---
title: "#{keyword}"
description: "All blog posts tagged with #{keyword}."
is-template: true
---

:::{{.tagged-posts .clean-table .no-header}}

```{{=html}}
$posts-keyword.{keyword}:../../templates/tag.html()$

```

:::
"""

tag = source / "tag"
tag.mkdir(parents=True, exist_ok=True)

for keyword in get_metadata(source / "blog" / "index.md")["display-keywords"]:
    with (tag / f"{keyword}.md").open("w") as f:
        f.write(template.format(keyword=keyword))

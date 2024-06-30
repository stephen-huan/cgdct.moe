---
title: false
pagetitle: "blog"
description: "convex gradient descent convergence theorems"
is-template: true
display-keywords: ["math", "cs", "anime", "misc"]
---

# cgdct stands for... {.blog-title}

## ...convex gradient descent convergence theorems {.blog-title}

:::blog-tags

```{=html}
<ul>
  $for(display-keywords/allbutlast)$
  <li>
    <a href="/tag/$it$"><b>#</b> $it$</a>
    <p>&nbsp; &middot; &nbsp;</p>
  </li>
  $endfor$
  <li>
    <a href="/tag/$display-keywords/last$"><b>#</b> $display-keywords/last$</a>
  </li>
</ul>
```

:::

:::blog-posts

```{=html}
<ul>
  $for(blog-posts)$
  <li>
    <h3><a href="$it.url$">$it.title$</a></h3>
    $it:../../templates/post.html()$
  </li>
  $endfor$
</ul>
```

:::

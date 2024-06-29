---
title: false
pagetitle: "publications"
description: "A list of all my publications."
is-template: true
publications-by-year: true
---

:::title

# publications

:::

:::publications

```{=html}
$for(publications)$
<div class="year">
  <h2 id="$it.year$">
    <a href="#$it.year$" class="header-anchor">$it.year$</a>
  </h2>
</div>
<ol>
  $for(it.publications)$
  $it:../templates/publications.html()$
  $endfor$
</ol>
$endfor$

```

:::

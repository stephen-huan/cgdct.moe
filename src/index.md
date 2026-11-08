---
title: false
pagetitle: "Stephen Huan"
description: "Stephen Huan's personal website."
css:
  - "/css/main.css"
  - "/css/root/news.css"
  - "/css/root/publications.css"
  - "/css/root/index.css"
disable-blog-name: true
is-template: true
news-entries: 5
publications-selected: true
---

# **Stephen** Huan [(浣苇杭)]{lang="zh-Hans"} {.top-title}

:::pgp
PGP:\ [`<samp>0xEA6E27948C7DBF5D0DF085A10FBC2E3BA99DD60E</samp>`{=html}
](/assets/misc/EA6E27948C7DBF5D0DF085A10FBC2E3BA99DD60E.asc)
:::

:::columns-start
::::{.column-right .index-column-right}
:::::profile
[![profile picture](/assets/profile_picture.webp){width=530 height=749}
](/assets/profile_picture.jpg)
:::::
:::::address
| GHC 6221
| 5000 Forbes Ave
| Pittsburgh, PA 15213
:::::
::::

I'm a PhD student in the [Computer Science Department](https://csd.cmu.edu/)
at [Carnegie Mellon University](https://www.cmu.edu/), advised by [Andrej
Risteski](https://www.andrew.cmu.edu/user/aristesk/). Prior to CMU, I was an
undergraduate at [Georgia Tech](https://www.gatech.edu/), where I worked on
computation and inference with [Florian Schäfer](https://f-t-s.github.io/).

My research interests are in problems spanning machine learning, statistics,
and applied mathematics. I'm particularly interested in efficient numerical
algorithms for generative modeling, sampling, and statistical inference. See
these [high-level summaries](/projects/) for more details about my work.

I am best reached via [email](mailto:slhuan@cs.cmu.edu).

## external links {.links}

- GitHub: [stephen-huan](https://github.com/stephen-huan)
- ORCID: [0009-0009-6309-4415](https://orcid.org/0009-0009-6309-4415)
- [Google Scholar](https://scholar.google.com/citations?user=CKjDUWwAAAAJ)

::::columns-end
::::
:::

## [news](/news/) {.showhover}

:::{.news .clean-table .no-header}

```{=html}
$../templates/news.html()$

```

:::

## [selected publications](/publications/) {.showhover}

:::publications

```{=html}
<ol>
  $for(publications)$
  $it:../templates/publications.html()$
  $endfor$
</ol>
```

:::

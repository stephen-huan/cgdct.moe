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

:::title

# **Stephen** Huan [(浣苇杭)]{lang="cmn"}

:::

:::pgp
PGP:&nbsp;[<samp>0xEA6E27948C7DBF5D0DF085A10FBC2E3BA99DD60E</samp>](/assets/misc/EA6E27948C7DBF5D0DF085A10FBC2E3BA99DD60E.asc)
:::

:::columns-start
::::{.column-right .index-column-right}
:::::profile
[![profile picture](/assets/profile_picture.webp){width=530 height=749}](/assets/profile_picture.jpg)
:::::
:::::address
S1320 CODA

756 W Peachtree St

Atlanta, GA 30332
:::::
::::

I'm an undergraduate student studying computer science at [Georgia
Tech](https://www.gatech.edu/), currently advised by [Florian
Schäfer](https://f-t-s.github.io/) in the [School of Computational
Science and Engineering](https://cse.gatech.edu/). Prior to Georgia
Tech, I was at [TJHSST](https://tjhsst.edu/).

My research exploits the interplay between numerical
computation and statistical inference for faster algorithms.
For example, my [previous work](/projects/) includes computing
sparse Cholesky factors of dense covariance matrices using
ideas from optimal experimental design and compressed sensing.

I am best reached via [email](mailto:shuan@gatech.edu).

::::links

## external links

::::

- GitHub: [stephen-huan](https://github.com/stephen-huan)
- ORCID: [0009-0009-6309-4415](https://orcid.org/0009-0009-6309-4415)
- [Google Scholar](https://scholar.google.com/citations?user=CKjDUWwAAAAJ)

::::columns-end
::::
:::

:::showhover

## [news](/news/)

:::

:::{.news .clean-table .no-header}

```{=html}
$../templates/news.html()$

```

:::

:::showhover

## [selected publications](/publications/)

:::

:::publications

```{=html}
<ol>
  $for(publications)$
  $it:../templates/publications.html()$
  $endfor$
</ol>
```

:::

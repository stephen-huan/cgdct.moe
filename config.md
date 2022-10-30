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
ignore = ["Project.toml", "Manifest.toml",
          "node_modules/", "package-lock.json", "package.json",
          "bin/",
         ]

# RSS (the website_{title, descr, url} must be defined to get RSS)
generate_rss = true
website_title = "Stephen Huan"
website_descr = "Stephen Huan's personal website"
website_url   = "https://stephen-huan.github.io/"

# doesn't seem to have a default value for tag pages
div_content = "franklin-content"

auto_code_path = true

# character for the icon
icon = "浣"

# header structure (url, display name)
headers = [("/", "about"),
           ("/blog/", "blog"),
           ("/projects/", "projects"),
           ("/publications/", "publications"),
           ("/assets/pdf/cv.pdf", "cv"),
          ]
+++

<!--
Add here global LaTeX commands to use throughout your pages.
-->
\newcommand{\url}[1]{[#1](!#1)}
<!-- images -->
\newenvironment{figure}{\begin{wrap}{figure}}{\end{wrap}}
\newcommand{\caption}[1]{\begin{wrap}{figcaption}#1\end{wrap}}
\newcommand{\figpreview}[3]{
  \begin{wrap}{a href="!#3"}
    \figalt{#1}{#2}
  \end{wrap}
}
<!-- columns -->
\newenvironment{columns}{
  \begin{wrap}{div class="row"}
}{
  ~~~<div style="clear: both"></div>~~~\end{wrap}
}
\newenvironment{column}[2]{
  \begin{wrap}{div class="#1" style="width: #2;"}
}{
  \end{wrap}
}

<!--
LaTeX-like math macros.
n.b.: these are parsed by Franklin, *not* KaTeX.
KaTeX has its own macro system: https://katex.org/docs/supported.html#macros
-->

<!-- colors -->
\newcommand{\silver       }{#9e9997}
\newcommand{\lightblue    }{#a1b4c7}
\newcommand{\seagreen     }{#23553c}
\newcommand{\orange       }{#ea8810}
\newcommand{\rust         }{#b8420f}

\newcommand{\lightsilver  }{#e7e6e5}

\newcommand{\darksilver   }{#96918f}
\newcommand{\darklightblue}{#8999a9}
\newcommand{\darkseagreen }{#1e4833}
\newcommand{\darkorange   }{#c7740e}
\newcommand{\darkrust     }{#9c380d}

<!-- general -->
\newcommand{\defeq}{\coloneqq}
\newcommand{\BigO}{\mathcal{O}}
\newcommand{\Id}{\text{Id}}
\newcommand{\vec}[1]{\bm{#1}} <!-- \renewcommmand -->
\newcommand{\T}{\top}
\newcommand{\dd}{\, \text{d}}
\newcommand{\Reverse}{\updownarrow}

<!-- paired delimiters -->
\newcommand{\norm}[1]{\lVert #1 \rVert}
\newcommand{\fro}[1]{\lVert #1 \rVert_{\operatorname{FRO}}}
\newcommand{\card}[1]{\lvert #1 \rvert}
\newcommand{\abs}[1]{\lvert #1 \rvert}
\newcommand{\inner}[2]{\langle #1, #2 \rangle}

<!-- operators -->
\newcommand{\argmin}{\operatorname*{argmin}}
\newcommand{\argmax}{\operatorname*{argmax}}
\newcommand{\curl}{\operatorname{curl}}
\newcommand{\diag}{\operatorname{diag}}
\newcommand{\trace}{\operatorname{trace}}
\newcommand{\logdet}{\operatorname{logdet}}
\newcommand{\chol}{\operatorname{chol}}

<!-- probability -->
\newcommand{\p}{\pi}
\newcommand{\E}[1]{\mathbb{E}[#1]}
\newcommand{\Var}[1]{\mathbb{V}\text{ar}[#1]}
\newcommand{\Cov}[2]{\mathbb{C}\text{ov}[#1, #2]}
\newcommand{\Corr}[2]{\mathbb{C}\text{orr}[#1, #2]}

\newcommand{\Entropy}[1]{\mathbb{H}[#1]}
\newcommand{\MI}[2]{\mathbb{I}[#1; #2]}
<!--
the extra level of indirection is necessary to parse as a KaTeX macro. see:
- https://latexref.xyz/_005c_0040ifstar.html
- https://katex.org/docs/supported.html#macros
-->
\newcommand{\KL}{
  \newcommand{\KLhelper}{\@ifstar{\KLstar}{\KLnostar}}
  \newcommand{\KLnostar}[2]{
    \mathbb{D}_{\operatorname{KL}}       ( #1 \;         \| \; #2        )
  }
  \newcommand{\KLstar  }[2]{
    \mathbb{D}_{\operatorname{KL}} \left ( #1 \; \middle \| \; #2 \right )
  }
  \KLhelper
}

\newcommand{\N}{\mathcal{N}}
\newcommand{\mean}{\mu}
\newcommand{\var}{\sigma^2}
\newcommand{\std}{\sigma}


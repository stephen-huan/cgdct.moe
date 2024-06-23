+++
title = "projects"
meta_description = "A list of all my projects."
+++

<!-- \card{url}{preview image}{width}{height}{title}{description} -->
\newcommand{\card}[6]{
  ~~~
  <div class="card">
    <a href="/projects/!#1/">
      <img src="!#2" alt="!#5" width="!#3" height="!#4">
      <div class="card-body">
        <h2>#5</h2>
        <p>#6</p>
      </div>
    </a>
  </div>
  ~~~
}

@@title
\chapter{projects}
@@

@@category
## inference and computation
@@

@@inference
\makecard{cholesky}
@@


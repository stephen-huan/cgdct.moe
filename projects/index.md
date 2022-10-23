+++
title = "projects"
+++

<!-- \card{url}{preview image}{title}{description} -->
\newcommand{\card}[4]{
  ~~~
  <div class="card">
    <a href="/projects/!#1/">
      <img src="!#2" alt="!#3" width="auto" height="auto"></img>
      <div class="card-body">
        <h2>#3</h2>
        <p>#4</p>
      </div>
    </a>
  </div>
  ~~~
}

@@title
# projects
@@

@@category
## inference and computation
@@

@@inference
\makecard{cholesky}
@@


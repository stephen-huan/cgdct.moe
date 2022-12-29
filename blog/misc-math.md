+++
title = "Miscellaneous mathematical observations"
date = Date(2022, 9, 23)
rss = "Self-contained observations that aren't long enough for their own post."

tags = ["math"]
mintoclevel = 1
+++

\newcommand{\date}[1]{
  ~~~<span class="post-meta"><time datetime="!#1">!#1</time></span>~~~
}

{{maketitle}}

\tableofcontents

# Gradient is orthogonal to contours

\date{2022-09-23}

_Geometric intuition for the gradient._

\begin{definition}
For a scalar function $ f: \R^n \to \R $ we say a \emph{contour} or \emph{level
set} of $ f $ is a parameterized curve $ \vec{r}(t): \R \to \R^n $ such that
for every point on the curve, the function value is the same, that is, there is
a fixed constant $ c \in \R $ such that
\begin{align}
  \label{eq:contour}
  f(\vec{r}(t)) = c
\end{align}
\end{definition}

**Claim**. The gradient $ \nabla f $ is
orthogonal to the contour at every point.

\begin{proof}
Take the derivative of both sides of \cref{eq:contour} with respect to $ t $.

<!-- Align distinct align*s: https://tex.stackexchange.com/a/282052 -->
\begin{align*}
  \hphantom{
    \frac{df}{dt} =
      \frac{df}{d r_1} \frac{d r_1}{dt} + \dotsb +
      \frac{df}{d r_n} \frac{d r_n}{dt}
  }
  \mathllap{f(\vec{r}(t))} &= c \\
  \frac{df}{dt} &= 0
\end{align*}

Expanding using the multivariate chain rule,

\begin{align*}
  \frac{df}{dt} =
    \frac{df}{d r_1} \frac{d r_1}{dt} + \dotsb +
    \frac{df}{d r_n} \frac{d r_n}{dt} &= 0 \\
  \begin{pmatrix}
    \frac{df}{d r_1} & \cdots & \frac{df}{d r_n}
  \end{pmatrix}
  \begin{pmatrix}
    \frac{d r_1}{dt} \\
    \vdots \\
    \frac{d r_n}{dt}
  \end{pmatrix} &= 0 \\
  \inner{\nabla f(t)}{\vec{r}'(t)} &= 0
\end{align*}

which is what we wanted to show.
\end{proof}

Thus the gradient is orthogonal to every tangent of the contour, so
moving along the contour moves orthogonally to the gradient. This
should make intuitive sense since the change in the function value
after moving in the direction $ \Delta \vec{x} $ is approximately
(to first order Taylor series approximation)

\begin{align*}
  f(\vec{x} + \Delta \vec{x}) \approx
    f(\vec{x}) + \nabla f(t)^\T \Delta \vec{x}
\end{align*}

Therefore if we want the function value to not change (which is the
definition of a contour), we want the direction of movement to be
orthogonal to the gradient, which is precisely what we have shown happens.

# Positive definite implies Cauchy–Schwarz

\date{2022-09-23}

_A quick derivation of the Cauchy–Schwarz inequality._

\begin{theorem}
(Cauchy–Schwarz inequality).
Every pair of vectors $ u, v $ satisfies
\begin{align*}
  \inner{u}{u} \inner{v}{v} &\geq \inner{u}{v}^2
\end{align*}
\end{theorem}

\begin{proof}
For vectors $ u, v \in \R^n $, let $ V \defeq
\begin{pmatrix} u & v \end{pmatrix} \in \R^{n \times 2}
$ be the matrix with $ u $ and $ v $ as columns.
\begin{align*}
  \Theta \defeq V^{\T} V =
    \begin{pmatrix}
      \inner{u}{u} & \inner{u}{v} \\
      \inner{v}{u} & \inner{v}{v}
    \end{pmatrix} & \\
  \det(\Theta) = \inner{u}{u} \inner{v}{v} - \inner{u}{v}^2 &\geq 0 \\
  \implies \inner{u}{u} \inner{v}{v} &\geq \inner{u}{v}^2
\end{align*}
where we use that $ \Theta $ is symmetric
positive-definite so its determinant is positive.
\end{proof}

A similar line of reasoning can be used to show that the variance is positive.
\begin{theorem}
For any random variable $ X $, define its variance as
\begin{align*}
  \Var{X} \defeq \E{X^2} - \E{X}^2
\end{align*}
Then $ \Var{X} \geq 0 $.
\end{theorem}

The usual approach is to show that $ \Var{X} = \E{(X - \E{X})^2} $
and then argue that the expectation of a nonnegative quantity must be
nonnegative. Alternatively, it follows directly from
[Jensen's
inequality](https://lectures.cgdct.moe/misc/bessel-correction/bessel.pdf#page=19)
on the convex function $ f(x) = x^2 $. We will take a different approach.

\begin{lemma}
For any random variable $ X $, let $ \mu_i \defeq \E{X^i} $ be its $ i
$-th moment. Then the matrices collecting its moments up to order $ n $
\begin{align}
  \label{eq:moment_matrix}
  M_r \defeq
    \begin{pmatrix}
      1 & \mu_1 & \mu_2 & \cdots & \mu_r \\
      \mu_1 & \mu_2 & \mu_3 & \cdots & \mu_{r + 1} \\
      \mu_2 & \mu_3 & \mu_4 & \cdots & \mu_{r + 2} \\
      \vdots & \vdots & \vdots & \ddots & \vdots \\
      \mu_r & \mu_{r + 1} & \mu_{r + 2} & \cdots & \mu_{2 r} \\
    \end{pmatrix}
\end{align}
are all positive semi-definite for $ r = 1, 2, \dotsc, \floor{n/2} $.
\end{lemma}

\begin{proof}
Take the expectation of the squared polynomial
with coefficients $ a_0, \dotsc, a_r $.
\begin{align*}
  \E{(a_0 + a_1 X + a_2 X^2 + \dotsb + a_r X^r)^2} \geq 0
\end{align*}
Expanding the square directly, we have
\begin{align*}
  \E*{\sum_{0 \leq i, j \leq r} a_i a_j X^{i + j}}
  = \sum_{0 \leq i, j \leq r} a_i a_j \E{X^{i + j}}
  = \sum_{0 \leq i, j \leq r} a_i a_j \mu_{i + j}
\end{align*}
where we use the linearity of expectation and the definition of moments.
But this is precisely the quadratic form $ \vec{a}^{\top} M \vec{a} $ for
$ \vec{a} \defeq (a_0, \dotsc, a_r) $ and $ M_{i, j} \defeq \mu_{i + j} $,
matching the definition in \cref{eq:moment_matrix}. Since $ \vec{a}^{\top}
M \vec{a} \geq 0 $ holds for any $ a \in \R^{r + 1} $, $ M $ is positive
semi-definite by definition.
\end{proof}

Now to prove the nonnegativity of variance, we
need only to apply the lemma for $ r = 1 $.
\begin{proof}
  Take $ M_2 = \left (
    \begin{smallmatrix}
      1 & \mu_1 \\
      \mu_1 & \mu_2
    \end{smallmatrix}
  \right ) = \left (
    \begin{smallmatrix}
      1 & \E{X} \\
      \E{X} & \E{X^2}
    \end{smallmatrix}
  \right )$.
  Since $ M_2 $ is positive semi-definite its determinant is nonnegative, so we
  have $ \det(M_2) = \E{X^2} - \E{X}^2 \geq 0 $, showing $ \Var{X} \geq 0 $.
\end{proof}



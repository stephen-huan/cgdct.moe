+++
title = "Multiplicative and additive functions"
date = Date(2022, 11, 16)
rss = "Very few exist."

tags = ["math"]
+++

\newcommand{\strang}{\operatorname{strang}}

{{maketitle}}

\tableofcontents

## Introduction

[Multiplicative
structure](https://en.wikipedia.org/wiki/Multiplicative_function) often
shows up in number theory when one considers common arithmetic functions
(functions whose domain are the integers).

\begin{definition}
We say an arithmetic function $ f: \Z \to \R
$ is \emph{multiplicative} if it satisfies
\begin{align}
  \label{eq:multiplicative}
  f(xy) = f(x) f(y)
\end{align}
for all coprime integers $ x, y $, that is, $ \gcd(x, y) = 1 $.
An example is [Euler's totient
function](https://en.wikipedia.org/wiki/Euler%27s_totient_function) $
\varphi(n) $ which is defined as the number of integers between $ 1 $
and $ n $ relatively prime to $ n $.
\end{definition}

In addition, we say that $ f $ is \emph{completely multiplicative} if it
satisfies \cref{eq:multiplicative} for \emph{all} integers $ x, y $, not
just those which are relatively prime. From now on we will use the term
"multiplicative" to refer to completely multiplicative functions and take
our domain to be the reals.

In contrast, some functions have \emph{additive}
structure. For example, for any real $ x, y $, the line
$ f(x) = k x$ for some fixed constant $ k $ satisfies
\begin{align}
  \label{eq:additive}
  f(x + y) = f(x) + f(y)
\end{align}
since $ f(x + y) = k(x + y) = k x + k y = f(x) + f(y) $.

### Examples

Some functions turn multiplicative structure into
additive structure; for example, for any real $ x, y >
0 $, the logarithm function (base irrelevant) satisfies
\begin{align}
  \log(xy) = \log(x) + \log(y)
\end{align}
which is an interesting and often quite useful property turning
multiplications into additions. However, there is no easy formula
(that I know of) for $ \log(x + y) $, which is quite unfortunate.

Every symmetric positive-definite (s.p.d.) matrix $ \Theta $ has a lower
triangular \emph{Cholesky factor} $ L $ satisfying $ \Theta = L L^{\top} $.
In some sense all matrix factorizations are product structures since they
write a complicated matrix as the product of matrices that are hopefully
esimpler or easier to work with. Expanding $ L L^{\top} $ reveals the hidden
\emph{additive} structure of the Cholesky factorization, owing to the fact
that the lower triangularity of $ L $ implies a sort of decreasing or nested
structure. Indeed, we can write $ \Theta $ as the contributions of rank-one
matrices given by the outer product of the columns of $ L $ \cref{eq:chol}
(which is generally true for any matrix product) where each outer product
affects smaller and smaller submatrices of $ \Theta $ (from the triangularity
of $ L $).
\begin{align}
  \label{eq:chol}
  \Theta = L L^{\top} =
    L_{:, 1} L_{:, 1}^{\top} + \dotsb + L_{:, n} L_{:, n}^{\top}
\end{align}

Although the Cholesky factor is both an additive and multiplicative
\emph{factorization}, the $ \chol(\cdot) $ operator itself does not play
well with addition or multiplication. For multiplication the product of two
s.p.d. matrices is not even necessarily s.p.d. and for addition even adding
a multiple of the identity, that is, trying to factor $ \Theta + \sigma^2
\Id $ requires somewhat sophisticated tricks \cite{schafer2021sparse},
even if one is able to factor $ \Theta $ efficiently.

### Problem statement

Throughout these examples it seems although multiplicative and
additive structure can be converted between, they are hard
to mix in some sense. In order to make this notion precise,
\begin{definition}
Suppose some function $ T: \R \to \R $ satisfies both
\begin{align}
  \label{eq:mult_cond}
  \hphantom{T(x + y)}
  \mathllap{T(x   y)} &= \mathrlap{T(x)   T(y)}
    \hphantom{T(x) + T(y)} && \text{[multiplicative structure]}
\end{align}
\begin{align}
  \label{eq:add_cond}
  T(x + y) &= T(x) + T(y) &&
    \hphantom{\text{[multiplicative structure]}}
    \mathllap{\text{[additive structure]}}
\end{align}
for all $ x, y \in \R $. What can we infer
about $ T $ just from these conditions?
\end{definition}

## Solution

Our strategy will to identify a few specific observations and then work our
way up to the naturals $ \Nat $, the integers $ \Z $, the rationals $ \Q $,
and then finally, the reals $ \R $. Extension to the complex numbers $ \C $
is left as an exercise for the reader.

### Naturals

First, we observe that if we fix $ y = 1 $, then the multiplicative
condition \cref{eq:mult_cond} implies $ \forall x \in \R $
\begin{align*}
  T(1 \cdot x) &= T(1) \cdot T(x) = T(x) \\
\end{align*}
Moving $ T(x) $ to the left side and factoring, we have
\begin{align*}
  \hphantom{T(1 \cdot x) = T(1) \cdot T(x)}
  \mathllap{(T(1) - 1) T(x)} &= \mathrlap{0} \hphantom{T(x)}
\end{align*}
implying either $ T(1) = 1 $ or $ T(x) = 0 $ for all $ x $. So immediately
the constant function $ T(x) = 0 $ is a candidate solution and indeed, it
also satisfies the additive condition \cref{eq:add_cond} trivially. For
the more interesting situation where $ T(1) = 1 $, now using the additive
condition \cref{eq:add_cond}, we have
\begin{align*}
  T(x + 1) &= T(x) + T(1) = T(x) + 1 \\
\end{align*}
Therefore $ T(2) = T(1) + 1 = 2 $, $ T(3) = T(2) + 1 = 3 $,
and so on. By induction $ T(x) = x $ for all natural numbers
$ x $ (where we exclude 0 as a natural number for now).

### Integers

To extend this result to the integers, first we fill
in the hole at 0 by making use of \cref{eq:add_cond},
\begin{align*}
  T(0 + 0) &= 2 T(0) = T(0) \\
  \implies T(0) &= 0
\end{align*}
Then we observe for a natural number $ x $ and a
negative number $ y = -x $, using \cref{eq:add_cond},
\begin{align*}
  T(x + (-x)) &= T(x) + T(-x) = T(0) = 0 \\
  \implies T(-x) &= -T(x)
\end{align*}
where we use that $ T(x) = x $ for natural $ x $,
concluding that $ T(x) = x $ for integer $ x $.

### Rationals

To extend this result to the rationals: let $ x = p/q $ for
integers $ p, q $ and let $ y = q $. From \cref{eq:mult_cond},
\begin{align*}
  T((p/q) q) &= T(p/q) T(q) = T(p) = p \\
  \implies T(p/q) &= p/T(q) = p/q
\end{align*}
where we use $ T(p) = p $ and $ T(q) = q $ for integers
$ p, q $, concluding that $ T(x) = x $ for rational $ x $.

### Reals

Throughout all these examples it seems that $ T(x) = x $, but it is hard to
extend to the reals without additional structure; we've sort of hit the limit
on what our assumptions tell us. We would like our function to commute with
limits, that is, we would like for any sequence $ (x_n) $,
\begin{align*}
  T \left (\lim_{n \to \infty} x_n \right ) &=
    \lim_{n \to \infty} T(x_n)
\end{align*}

This is precisely condition (iii) of Theorem 4.3.2. of
\cite{abbott2015understanding} defining the continuity of a function:
> **Theorem 4.3.2 (Characterizations of
> Continuity).** Let $ f: A \to \bm{R} $, and let
>
> $ c \in A $. The function $ f $ is continuous at
> $ c $ if and only if any one of the following
>
> three conditions is met:
>
> ... \newline
>
> (iii) For all $ (x_n) \to c $ (with $ x_n \in A $),
> it follows that $ f(x_n) \to f(c) $.

If we enforce that $ T(x) $ must be continuous, then every
real number $ x $ is a Cauchy sequence of rational numbers, so
assume $ (x_n) \to x $ for rational $ x_n $ and observe that
\begin{align*}
  T(x) = T \left (\lim_{n \to \infty} x_n \right)
    = \lim_{n \to \infty} T(x_n) = \lim_{n \to \infty} x_n = x
\end{align*}
where we use that $ T(x_n) = x_n $ from the fact that $ T $
is identity on the rationals so we can conclude $ T(x) = x $
for all real $ x $. In summary, we have the following theorem:
\begin{theorem}
Suppose some \emph{continuous} function $ T: \R \to \R $ satisfies both

\begin{align*}
  T(x   y) &= T(x)   T(y) && \text{[multiplicative structure]} \\
  T(x + y) &= T(x) + T(y) &&       \text{[additive structure]}
\end{align*}

for all $ x, y \in \R $. Then either
1. $ T(x) = 0 $ for all $ x \in \R $ or
2. $ T(x) = x $ for all $ x \in \R $.
\end{theorem}

## Conclusion

This is just one of many reasons why the identity operator is
perhaps the easiest possible operator to work with and analyze;
for more details, see our soon-to-be released GitHub repository
[brownie-in-motion/identity](https://github.com/brownie-in-motion/identity)
which is joint work with [Daniel Lu](https://blog.danielclu.com/)
([brownie-in-motion](https://github.com/brownie-in-motion/)) and [Eshan
Ramesh](https://esrh.me/) ([eshrh](https://github.com/eshrh/)).

It's well known \cite{bezanson2017julia} that [Gilbert
Strang](https://math.mit.edu/~gs/)'s favorite matrix is the following -1,
2, -1 tridiagonal matrix \cref{eq:strang_matrix} which forms a second-order
[finite difference](https://en.wikipedia.org/wiki/Finite_difference)
approximation for the derivative.
\begin{align}
  \label{eq:strang_matrix}
  \strang(n) &=
    \begin{pmatrix}
       2 & -1 &        &        &    &    \\
      -1 &  2 &     -1 &        &    &    \\
         & -1 & \ddots & \ddots &    &    \\
         &    & \ddots & \ddots & -1 &    \\
         &    &        &     -1 &  2 & -1 \\
         &    &        &        & -1 &  2 \\
    \end{pmatrix}
\end{align}

@@img-large
\begin{figure}
  \figpreview{
      121 cupcakes with Gilbert Strang's favorite -1, 2, -1 matrix.
  }{cupcakematrix.webp}{640}{480}{
    https://math.mit.edu/~gs/PIX/cupcakematrix.jpg
  }
  \caption{
    121 cupcakes with Gilbert Strang's favorite -1, 2, -1 matrix.
  }
\end{figure}
@@

If I was asked upfront I'm not sure what my favorite linear
operator would be, but the identity function/matrix/operator
is definitely pretty high on the list after this discussion.

Lastly, this article is similar to Exercise 4.3.13 of
\cite{abbott2015understanding} but I only discovered
the connection after I already had the idea. A similar
line of reasoning holds for the textbook exercise.

\bibliography{references}


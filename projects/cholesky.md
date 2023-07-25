+++
title = "Cholesky factorization by greedy conditional selection"
date = Date(2023, 7, 25)
rss = "inference with conditional <em>k</em>-nearest neighbors?"
rss_category = "inference and computation"
preview_image = "/assets/projects/cholesky/oikura.webp"
maxtoclevel = 3

tags = ["math", "cs"]
+++

\newcommand{\GP}{\mathcal{GP}}
\newcommand{\K}{K}
\newcommand{\CM}{\Theta}
\newcommand{\PM}{\Theta^{-1}}

\newcommand{\Train}{\text{Tr}}
\newcommand{\Pred}{\text{Pr}}

\newcommand{\Loss}{\mathcal{L}}
\newcommand{\SpSet}{\mathcal{S}}
\newcommand{\Order}{\mathcal{I}}
\newcommand{\I}{I}
\newcommand{\J}{J}
\newcommand{\V}{V}

# **Sparse Cholesky factorization \\ by greedy conditional selection**

_This post summarizes joint work with [Joseph
Guinness](https://guinness.cals.cornell.edu), [Matthias
Katzfuss](https://sites.google.com/view/katzfuss/), [Houman
Owhadi](http://users.cms.caltech.edu/~owhadi/index.htm), and
[Florian Schäfer](https://f-t-s.github.io). For additional
details, see the [paper](https://arxiv.org/abs/2307.11648)
or [source code](https://github.com/stephen-huan/cholesky-by-selection)_.

_Thanks [Seulgi](https://www.dagmarsmithart.com/) for
drawing the preview image! The full resolution image may be
found [here](https://misc.cgdct.moe/oikura/sodachi.png)._

<!-- \tableofcontents -->

## The problem

Our motivating example is a regression problem
that we model with Gaussian processes.

In our setting, we have $ N $ points whose
corresponding covariance matrix $ \CM $ is formed by

\begin{align*}
  \CM_{i, j} \defeq \K(\vec{x}_i, \vec{x}_j),
\end{align*}

or pairwise evaluation of a kernel function $
\K(\cdot, \cdot) $. Commonly used kernels include the
[Matérn](https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function) family
of covariance functions, which include the exponential kernel and squared
exponential kernel (also known as the Gaussian or RBF kernel) as special cases.
The kernel function along with a mean function $ \mu(\cdot) $ define a Gaussian
process $ f \sim \GP(\mu, K) $ such that for any set of points $ X $, $
f(X) $ is jointly distributed according to a multivariate Gaussian.

If we partition our points into training points for which we know the value
of $ f $ and testing points at which we wish to predict the value of $ f
$, then the Gaussian process modeling of $ f $ enables us to compute the
posterior distribution of the testing points in closed form as
\begin{align}
  \label{eq:cond}
  \E{\vec{y}_\Pred \mid \vec{y}_\Train} &=
    \vec{\mu}_\Pred +
    \CM_{\Pred, \Train} \CM_{\Train, \Train}^{-1}
    (\vec{y}_\Train - \vec{\mean}_\Train) \\
  \Covv{\vec{y}_\Pred \mid \vec{y}_\Train} &=
    \CM_{\Pred, \Pred} -
    \CM_{\Pred, \Train} \CM_{\Train, \Train}^{-1}
    \CM_{\Train, \Pred}.
\end{align}

Note that our predictions naturally come equipped with uncertainty estimates
from the posterior covariance. Due to the ability to measure uncertainty and to
encode prior information in the kernel function, Gaussian processes and kernel
methods enjoy widespread usage in geostatistics, machine learning, and optimal
experimental design.

However, computing \cref{eq:cond} requires inverting the covariance matrix of
the training points. Additional statistical inference often requires computing
quantities like the likelihood or log determinant. For a dense covariance
matrix $ \CM $, directly computing these quantities has a computational cost
of $ \BigO(N^3) $ and a memory cost of $ \BigO(N^2) $, which is prohibitively
expensive for large $ N $. Can we efficiently approximate \cref{eq:cond} while
maintaining predictive accuracy?

## The screening effect

Spatial statisticians have long observed the \emph{screening
effect}, or the observation that conditional on nearby points,
more distant points have very little influence on the prediction.

@@img-half,centering
\begin{figure}
  \figpreview{Unconditional covariance}{figures/screen1.svg}{203.811}{157.984}{
    {{assets}}/figures/screen1.pdf
  }
  \figpreview{Conditional covariance}{figures/screen2.svg}{203.811}{157.984}{
    {{assets}}/figures/screen2.pdf
  }
  \caption{
    Screening effect in a Matérn kernel with $ \ell = 1 $
    and $ \nu = 1/2 $. Unconditional correlation with $ (0,
    0) $ is shown $ \textcolor{\lightblue}{\text{before}} $
    and $ \textcolor{\seagreen}{\text{after}} $ conditioning
    on four $ \textcolor{\orange}{\text{nearby points}} $.
  }
\end{figure}
@@

In other words, after conditioning on a few points, the conditional correlation
localizes around the target point, implying approximate conditional sparsity.

### The Vecchia approximation

We can exploit this conditional sparsity by looking at the joint distribution.

\begin{align*}
  \p(\vec{y}) &= \p(y_1) \p(y_2 \mid y_1) \p(y_3 \mid y_1, y_2) \dotsm
    \p(y_N \mid y_1, y_2, \dotsc, y_{N - 1}).
\end{align*}

The Vecchia approximation proposes to approximate the joint distribution by
\begin{align}
  \label{eq:vecchia}
  \p(\vec{y}) &\approx \p(y_{i_1}) \p(y_{i_2} \mid y_{s_2})
    \p(y_{i_3} \mid y_{s_3}) \dotsm \p(y_{i_N} \mid y_{s_N})
\end{align}
for some ordering $ i_1, \dotsc, i_N $ and sparsity sets $ s_k \subseteq
\{ i_1, \dotsc, i_{k - 1} \} $ with $ \card{s_k} \ll k $. The quality of
the approximation depends significantly on the chosen ordering and sparsity
pattern. In order to exploit the screening effect as much as possible, it
is natural to take the nearest neighbors as the sparsity pattern. For the
ordering, we would like to select points that have good coverage of the
entire space. Towards that end, we form the reverse-maximin ordering by
first selecting the last index $ i_N $ arbitrarily and then choosing for
$ k = N - 1, \dotsc, 1 $ the index
\begin{align*}
  i_k = \argmax_{i \in -\Order_{k + 1}} \; \min_{j \in \Order_{k + 1}}
    \norm{\vec{x}_i - \vec{x}_j}
\end{align*}
where $ -\Order \defeq \{ 1, \dotsc, N \} \setminus \Order $ and $
\Order_n \defeq \{ i_n, i_{n + 1}, \dotsc, i_N \} $, that is, select
the point furthest from previously selected points every iteration.
This ordering and sparsity pattern can be computed efficiently with
geometric data structures like the $ k $-d tree or ball tree.

### Sparse Cholesky factors

From another perspective, the Vecchia approximation implicitly forms
an approximate Cholesky factor of the covariance matrix (a lower
triangular matrix $ L $ such that $ L L^{\top} \approx \CM $).

We can hope for (approximate) sparsity in the Cholesky factor as Cholesky
factorization can be viewed as recursively computing the block decomposition
\begin{align*}
  \chol(\CM) &=
  \begin{pmatrix}
    \Id & 0 \\
    \textcolor{\darkorange}{\CM_{2, 1} \CM_{1, 1}^{-1}} & \Id
  \end{pmatrix}
  \begin{pmatrix}
    \chol(\CM_{1, 1}) & 0 \\
    0 & \chol(\textcolor{\lightblue}{
      \CM_{2, 2} - \CM_{2, 1} \CM_{1, 1}^{-1} \CM_{1, 2}
    })
  \end{pmatrix} \\
\end{align*}
where $ \CM_{2, 2} - \CM_{2, 1} \CM_{1, 1}^{-1} \CM_{1, 2} $ is the conditional
covariance in \cref{eq:cond}. Thus Cholesky factorization is a process of
iteratively conditioning the underlying Gaussian process, which we hope will
lead to approximate sparsity from the screening effect.

#### KL minimization

We can use the same ordering and sparsity pattern as the Vecchia approximation
\cref{eq:vecchia}. In order to compute the entries of $ L $, we can minimize
the Kullback-Leibler (KL) divergence between two centered Gaussians with the
true and approximate covariance matrices over the space $ \SpSet $ of lower
triangular matrices satisfying the specified sparsity pattern,

\begin{align*}
  L &\defeq \argmin_{\hat{L} \in \SpSet} \,
    \KL*{\N(\vec{0}, \CM)}{\N(\vec{0}, (\hat{L} \hat{L}^{\top})^{-1})}.
\end{align*}

Note that we want $ L L^{\top} \approx \PM $, the inverse of the covariance
matrix (sometimes called the precision matrix). While $ \CM $ encodes marginal
relationships between the variables, $ \PM $ encodes conditional relationships
and therefore often leads to sparser factors.

Luckily this optimization problem has a closed-form explicit solution
\begin{align*}
  L_{s_i, i} &= \frac{\CM_{s_i, s_i}^{-1} \vec{e}_1}
    {\sqrt{\vec{e}_1^{\top} \CM_{s_i, s_i}^{-1} \vec{e}_1}}
\end{align*}
which computes the nonzero entries for the $ i $-th
column of $ L $ in time $ \BigO(\card{s_i}^3) $.

If we were to directly use a sparse Cholesky factor to approximate $
\CM_{\Train, \Train}^{-1} $ in \cref{eq:cond}, we would still need to
compute $ \CM_{\Pred, \Train} $ which could be prohibitive. Instead, we
put "prediction points first" by forming the joint covariance matrix
of training and prediction points
\begin{align*}
  L = \begin{pmatrix}
    L_{\Pred, \Pred} & 0 \\
    L_{\Train, \Pred} & L_{\Train, \Train}
  \end{pmatrix}
  \qquad
  L L^{\top} \approx \begin{pmatrix}
    \CM_{\Pred, \Pred} & \CM_{\Pred, \Train} \\
    \CM_{\Train, \Pred} & \CM_{\Train, \Train}
  \end{pmatrix}^{-1}
\end{align*}

Given the sparse approximate Cholesky factor $ L $,
\cref{eq:cond} can be approximated efficiently with
\begin{align*}
  \E{\vec{y}_\Pred \mid \vec{y}_\Train} &=
    -L_{\Pred, \Pred}^{-\top} L_{\Train, \Pred}^{\top} \vec{y}_\Train \\
  \Covv{\vec{y}_\Pred \mid \vec{y}_\Train} &=
    L_{\Pred, \Pred}^{-\top} L_{\Pred, \Pred}^{-1}.
\end{align*}

This can be viewed as a generalization of
the Vecchia approximation \cref{eq:vecchia}.

#### Decomposing the KL divergence

Plugging the optimal $ L $ back into the KL
divergence yields the interesting decomposition

\begin{align*}
  2 \KL*{\N(\vec{0}, \CM)}{\N(\vec{0}, (L L^{\top})^{-1})} &=
    \sum_{i = 1}^N
      \left [
        \log \left ( \CM_{i, i \mid s_i \setminus \{ i \}} \right ) -
        \log \left ( \CM_{i, i \mid i + 1:} \right )
      \right ].
\end{align*}

This sum is the accumulated \emph{difference} in posterior log variance for
a series of independent regression problems: each to predict the $ i $-th
variable given a subset of the variables after it in the ordering. The term $
\log \left ( \CM_{i, i \mid s_i \setminus \{ i \}} \right ) $ obtained when
restricted to the variables in the sparsity pattern $ s_i $ is compared to
the ground truth $ \log \left ( \CM_{i, i \mid i + 1:} \right ) $.

We can generalize this formula if multiple columns share the same sparsity
pattern, a setting called aggregated or supernodal Cholesky factorization
which improves computational efficiency. If $ \tilde{i} $ is a list of
aggregated columns, its contribution to the KL divergence is
\begin{align*}
  \sum_{i \in \tilde{i}}
    \log \left (\CM_{i, i \mid s_i \setminus \{ i \} } \right ) &=
    \logdet(\CM_{\tilde{i}, \tilde{i} \mid \tilde{s}})
\end{align*}
or the posterior log determinant of the group's covariance matrix. However,
this formula assumes the aggregated columns are adjacent in the ordering,
which may not be the case.

In the case of non-adjacent columns, a sparsity entry can be below one column
but above another, thus only \emph{partially} conditioning the group when it
is added to the sparsity pattern. Writing $ \vec{y}_{\parallel k} $ for the
conditioning of the variables in $ \vec{y} $ by $ k $ but skipping the first
$ p $, we have

\begin{align*}
  \CM_{\tilde{i}, \tilde{i} \parallel k} &\defeq
  \Covv{\vec{y}_{\parallel k}} =
  \begin{pmatrix}
    L_{:p} L_{:p}^{\top} &
    L_{:p} {L'}_{p + 1:}^{\top} \\
    {L'}_{p + 1:} L_{:p}^{\top} &
    {L'}_{p + 1:} {L'}_{p + 1:}^{\top}
  \end{pmatrix} =
  \begin{pmatrix}
    L_{:p} \\
    {L'}_{p + 1:}
  \end{pmatrix}
  \begin{pmatrix}
    L_{:p} \\
    {L'}_{p + 1:}
  \end{pmatrix}^{\top}
\end{align*}

where $ L $ is the fully unconditional Cholesky
factor and $ L' $ is the fully conditional factor.

@@img-large
\begin{figure}
  \figpreview{Partial Cholesky factor}{figures/partial.svg}{221.103}{85.039}{
    {{assets}}/figures/partial.pdf
  }
  \caption{
    Decomposition of the covariance matrix of partially conditioned variables
    into a fully pure Cholesky factor and a fully conditional Cholesky factor.
  }
\end{figure}
@@

Armed with this representation, the contribution to the KL divergence is simply
\begin{align*}
  \sum_{i \in \tilde{i}}
    \log \left (\CM_{i, i \mid s_i \setminus \{ i \} } \right ) &=
    \logdet(\CM_{\tilde{i}, \tilde{i} \parallel k}).
\end{align*}

## Information-theoretic objectives

The previous formulas all imply that the sparsity selection should minimize
the posterior variance (or log determinant) of the target which are measures
of the uncertainty of the target. In other words, the selected entries should
be \emph{maximally informative} to the target.

To make this notion precise, we can define
the mutual information or information gain as
\begin{align*}
  \MI{\vec{y}_\Pred}{\vec{y}_\Train} &\defeq
    \Entropy{\vec{y}_\Pred} - \Entropy{\vec{y}_\Pred \mid \vec{y}_\Train}
\end{align*}
where $ \Entropy{X} $ is the entropy of the random variable $ X $. Since the
entropy of $ \vec{y}_\Pred $ is constant, maximizing the mutual information is
equivalent to minimizing the posterior entropy. For multivariate Gaussians the
entropy is monotonically increasing with the log determinant, so maximizing
mutual information is indeed equivalent to minimizing posterior variance!

In addition, we observe an information-theoretic analogue of the EV-VE identity
\begin{align*}
  \textcolor{\orange}{\Var{\vec{y}_\Pred}} &=
    \textcolor{\lightblue}{\E{\Var{\vec{y}_\Pred \mid \vec{y}_\Train}}} +
      \textcolor{\rust}{\Var{\E{\vec{y}_\Pred \mid \vec{y}_\Train}}} \\
  \textcolor{\orange}{\Entropy{\vec{y}_\Pred}} &=
    \textcolor{\lightblue}{\Entropy{\vec{y}_\Pred \mid \vec{y}_\Train}} +
      \textcolor{\rust}{\MI{\vec{y}_\Pred}{\vec{y}_\Train}}
\end{align*}
where the posterior entropy can be identified with the posterior
variance and the mutual information corresponds to the variance of
the optimal estimator $ \E{\vec{y}_\Pred \mid \vec{y}_\Train} $
\cref{eq:cond}, which can be intuitively understood as the extent
to which the estimator actually varies with data.

Finally, the expected mean squared error of the estimator $
\E{\vec{y}_\Pred \mid \vec{y}_\Train} $ which is defined as $ \text{MSE}
\defeq \E{(\vec{y}_\Pred - \E{\vec{y}_\Pred \mid \vec{y}_\Train})^2}
$ is simply the posterior variance because
\begin{align}
  \text{MSE}
  = \E{\E{(\vec{y}_\Pred - \E{\vec{y}_\Pred \mid \vec{y}_\Train})^2
          \mid \vec{y}_\Train}}
  = \E{\Var{\vec{y}_\Pred \mid \vec{y}_\Train}}
  = \Var{\vec{y}_\Pred \mid \vec{y}_\Train},
\end{align}
from the quirk that the posterior covariance \cref{eq:cond}
does not depend on observing $ \vec{y}_\Train $.

## Greedy selection

Rather than use distance as a geometric proxy for importance,
why not use information directly? The following toy example
illustrates the benefits of selection by information.

In this 1-d regression problem, the goal is to predict the $ y $-value of the
$ \textcolor{\orange}{\text{target}} $ point, given the locations of many $
\textcolor{\lightblue}{\text{training}} $ points. However, the predictor can
only use the $ y $-values of two training points, so which points it ends up
$ \textcolor{\seagreen}{\text{selecting}} $ will determine the prediction.

@@img-half,centering
\begin{figure}
  \figpreview{Nearest neighbors}{figures/select1.svg}{177.143}{152.792}{
    {{assets}}/figures/select1.pdf
  }
  \figpreview{Conditional selection}{figures/select2.svg}{177.143}{152.792}{
    {{assets}}/figures/select2.pdf
  }
  \caption{
    The $ \textcolor{\darkrust}{\text{red}} $ line is the $
    \textcolor{\darkrust}{\text{conditional mean}} $ $ \mu $, conditional
    on the $ \textcolor{\seagreen}{\text{selected}} $ points, and the $ \pm
    2 \sigma $ $ \textcolor{\rust}{\text{confidence interval}} $ is shaded
    for the $ \textcolor{\rust}{\text{conditional variance}} $ $ \sigma^2 $.
  }
\end{figure}
@@

If one selects the nearest neighbors (left panel), both points
are to the right of the target and quite close to each other.
This results in an inaccurate prediction with high uncertainty.

A more balanced view of the situation is obtained when picking the slightly
farther but ultimately more informative point to the left (right panel),
improving the prediction and reducing the uncertainty. In higher dimensions
this phenomena is only exacerbated as there are more "sides" of the target,
and so more reason to spread out the selected points.

The resulting selections on a 2-d grid with Matérn kernels
of different smoothnesses are visualized in this [YouTube
video](https://youtu.be/lyJf3S5ThjQ) (I recommend putting
it on double time; it's quite satisfying).

It is intractable (and probably NP hard) to select the optimal $ k
$ points out of $ N $ points total, so we greedily select the most
informative point, conditional on all points previously selected. The
naive time complexity is $ \mathcal{O}(N k^4) $, but by maintaining
a partial Cholesky factor we can reduce this to $ \mathcal{O}(N
k^2) $. For multiple points, application of the [matrix determinant
lemma](https://en.wikipedia.org/wiki/Matrix_determinant_lemma)
\begin{align*}
  \logdet(\CM_{\Pred, \Pred \mid \I, k}) - \logdet(\CM_{\Pred, \Pred \mid \I})
  &= \log(\CM_{k, k \mid \I, \Pred}) - \log(\CM_{k, k \mid \I})
\end{align*}
switches the roles of the prediction points and the training points, allowing
the change in log determinant to be computed much more efficiently as the
change in variance of the \emph{training} point after conditioning on the
prediction points. For $ m $ targets the time complexity is $ \mathcal{O}(N k^2
+ N m^2 + m^3) $, gracefully reducing to the complexity of the single-point
algorithm when $ m = 1 $. Finally, efficient application of
[rank-one downdating](https://en.wikipedia.org/wiki/Cholesky_decomposition#Rank-one_downdate)
accounts for partially conditioned points
at no additional asymptotic time cost.

### Sparsity pattern

We can directly apply our algorithm to select
the sparsity pattern of Cholesky factors.

@@img-half,centering
\begin{figure}
  \figpreview{Cholesky factor}{figures/chol.svg}{113.784}{113.784}{
    {{assets}}/figures/chol.pdf
  }
  \figpreview{Point selection}{figures/points.svg}{113.386}{113.382}{
    {{assets}}/figures/points.pdf
  }
  \caption{
    Equivalence of sparsity selection in
    Cholesky factors and point selection in GPs.
  }
\end{figure}
@@

For a column of a Cholesky factor in isolation, the
$ \textcolor{orange}{\text{target}} $ point is the
$ \textcolor{orange}{\text{diagonal}} $ entry, $
\textcolor{lightblue}{\text{candidates}} $ are $
\textcolor{lightblue}{\text{below}} $ it, and the $
\textcolor{seagreen}{\text{selected}} $ entries are added to the $
\textcolor{seagreen}{\text{sparsity pattern}} $. Points violating lower
triangularity are not shown. Thus, sparsity selection in Cholesky
factorization (left panel) is analogous to training point selection
in directed Gaussian process regression (right panel).

## Numerical experiments

Here I give, in my opinion, the most
thought-provoking experiments from the paper.

### Nearest neighbors classification

We perform image classification on the MNIST dataset of handwritten digits
(greyscale images of digits from 0-9) using a very simple $ k $-nearest
neighbors approach: select $ k $ images, either with nearest neighbors ($ k
$-NN) or with conditional selection (C$ k $-NN). Take the most frequently
occurring label in the selected images, and use that as the predicted label.

@@img-small
\begin{figure}
  \figpreview{MNIST classification}{figures/mnist.svg}{179.784}{143.356}{
    {{assets}}/figures/mnist.pdf
  }
  \caption{Classification accuracy of $ k $-NN variants on the MNIST dataset.}
\end{figure}
@@

The $ k $-NN algorithm (shown in blue) linearly decreases in accuracy with
increasing $ k $ as the bias-variance trade-off would predict. However,
not only does C$ k $-NN maintain a higher accuracy for all $ k $, but its
accuracy actually \emph{increases} until around $ k \approx 10 $.

This suggests conditioning able to somehow "robustify" the selection procedure.

### Sparse signal recovery

Our selection algorithm can be viewed as the covariance equivalent of
orthogonal matching pursuit. To see this, let $ \CM $ be a symmetric
positive-definite (s.p.d.) matrix, so it admits a factorization $ \CM
= F^{\top} F $ for feature vectors $ F $. Perform a QR factorization
on $ F = QR $ so that
\begin{align*}
  \CM = (QR)^{\top} (QR) = R^{\top} (Q^{\top} Q) R = R^{\top} R.
\end{align*}
But $ R $ is an upper triangular matrix, so by the uniqueness of the Cholesky
factorization, $ R $ is a Cholesky factor of $ \CM $. As the QR factorization
performs iterative orthogonalization in the feature space $ F $ through the
Gram–Schmidt process, this corresponds to conditioning in $ \CM $.

Motivated by this connection, we consider an experiment in which we randomly
generate an _a priori_ sparse Cholesky factor $ L $ and attempt to recover it
given only the measurements $ Q \defeq L L^{\top} $. We report the intersection
over union (IOU) of the sparsity patterns.

@@img-half,centering
\begin{figure}
  \figpreview{Nearest neighbors}{figures/omp1.svg}{175.697}{144.856}{
    {{assets}}/figures/omp1.pdf
  }
  \figpreview{Conditional selection}{figures/omp2.svg}{161.911}{144.197}{
    {{assets}}/figures/omp2.pdf
  }
  \caption{
    Accuracy over varying matrix size $ N $
    and nonzero entries per column $ s $.
  }
\end{figure}
@@

Conditional selection maintains a near-prefect accuracy
with increasing problem size and factor density, while
the other naive methods perform significantly worse.

### Preconditioning the conjugate gradient

The conjugate gradient is a classical iterative algorithm for solving
linear systems in s.p.d. matrices. In this experiment, we use sparse
Cholesky factors as a preconditioner for the conjugate gradient and
track how many nonzeros per column we need to converge to a specified
precision within a fixed number of iterations (here, 50).

@@img-small
\begin{figure}
  \figpreview{Conjugate gradient}{figures/cg.svg}{172.514}{143.264}{
    {{assets}}/figures/cg.pdf
  }
  \caption{Number of nonzeros per column to converge within 50 iterations.}
\end{figure}
@@

Our conditional selection methods (shown in orange and red) need a nearly
constant number of nonzeros with increasing problem size $ N $. The nearest
neighbor-based methods, however, start sharply increasing. This could hint at
a qualitatively different regime in which the conditional methods discover
hidden structure nearest neighbors cannot access.

## Conclusion

Replacing nearest neighbors with conditional selection has the potential to
significantly improve the accuracy of sparsity patterns and the statistical
efficiency of the resulting approximate sparse Cholesky factor, resulting in
faster and more accurate computation of statistical quantities of interest
such as the posterior mean, posterior variance, likelihood, log determinant,
and inversion of the covariance matrix.

Beyond Gaussian processes, efficient computation with large structured
positive-definite matrices could accelerate optimization routines (through
Newton's method and its variants), solving linear systems with iterative
methods like the conjugate gradient by preconditioning, and more. Optimal
transport, mirror descent, and the natural gradient are all possibilities.

By replacing geometric criteria hand-crafted towards a specific family of
covariance functions with relatively general information-theoretic criteria,
this work represents an important step in automating the discovery of
exploitable structure in complex positive-definite matrices. Importantly,
the selection methods only require access to the entries of the matrix,
\emph{not} the underlying points, allowing it to be applied in principle
to any s.p.d. matrix.

## Postscript

\begin{align*}
  \text{Conditioning in Gaussian processes} \iff
  \text{Gaussian elimination} \iff
  \text{Sodachi Oikura's favorite}
\end{align*}

@@img-large,border,nonumber
\begin{figure}
  \figpreview{
    Two portraits with Euler on the left and Gauss on the right
  }{gauss.jpg}{2560}{1440}{
    https://misc.cgdct.moe/anime/gauss.png
  }
  > …けれど私が１等尊敬してるのはガウス先生だよ。

  > ...but the [mathematician] I respect the most is Gauss.
  \caption{
    ~~~---~~~from
    [Zoku Owarimonogatari]
    (https://www.monogatari-series.com/zokuowarimonogatari/)
    episode 2, at roughly 13:00.
  }
\end{figure}
@@

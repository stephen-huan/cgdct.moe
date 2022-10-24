+++
title = "far: a (f)ast re-write of the Unix utility p(ar)"
date = Date(2020, 12, 16)
rss = "Optimal variance text reflow for aesthetically pleasing borders."

tags = ["cs"]
+++

{{maketitle}}

[[Video]](https://youtu.be/H3Agto3ZSnk)
[[Source Code (C++)]]({{assets}}/far.cpp)
[[Source Code (Python)]]({{assets}}/far.py)

[`par`](http://www.nicemice.net/par/) is a formatting tool that inserts line
breaks to make the length of each line less than a set number of characters,
usually 79 (terminals historically have 80 width). Unfortunately, `par` is
incredibly complicated and introduces random whitespace. So I made my own.

For `far` to make the paragraphs look good, it minimizes the variance of
each line. However, it's constrained to use the fewest number of lines
possible, so it doesn't generate really short lines. Finally, it ignores
the last line when minimizing variance and tries to make the last line
shorter than average, because a typical paragraph usually looks better
if the last line is shorter. To summarize,
1. Minimize the variance of the lengths of each line...
2. ...subject to the constraint that the number of lines is smallest
3. Ignore the last line, while making sure it's shorter than average

`far` uses dynamic programming to minimize variance. It
tokenizes the paragraph by splitting on whitespace, and each
subproblem in the dynamic program is a suffix of this token list.
If $ X $ is a vector that represents the length of each
line in a given paragraph with $ \card{X} = n $ lines,
then the variance of line lengths is defined as
\begin{align*}
  \Var{X} &= \E{(X - \E{X})^2} = \frac{1}{n} \sum_{x \in X}
    \left ( x - \frac{1}{n} \sum_{x \in X} x \right )^2
\end{align*}
The fewest number of lines constraint means $ n $ is constant and so is the
sum of line lengths $ \sum_{x \in X} x $ because it is determined by two
things: the characters in the tokens and the number of spaces introduced
by merging two tokens (combining the words "hello" and "world" onto the
same line gives "hello world", with an additional space). The characters
stay the same, and the number of spaces is fixed if the number of lines is
fixed. Each token starts off as its own line, and each merge reduces the
number of lines by 1, so if two solutions have the same number of lines,
they must have done the same number of merges.

Thus, minimizing $ \Var{X} $ is equivalent to minimizing the sum of squares $
\sum_{x \in X} x^2 $ if the number of lines is fixed. Recall that we are trying
to minimize variance over the entire paragraph. The overall paragraph has some
mean line length $ \mu \defeq \E{X} $. Each line will contribute $ (x - \mu)^2
$ to the overall paragraph's variance. So we want to minimize
\begin{align*}
  (x_1 - \mu)^2 + (x_2 - \mu)^2 + \dotsb + (x_n - \mu)^2
\end{align*}
where $ x_i $ is the length of the $ i $-th line. Expanding,
\begin{align*}
  [x_1^2 - 2 x_1 \mu + \mu^2] + \dotsb + [x_n^2 - 2 x_n \mu + \mu^2]
\end{align*}
We can ignore the $ \mu^2 $ terms since $ \mu = \frac{1}{n} \sum_{x
\in X} x $ is constant from the aforementioned logic (both $ n $
and the total number of characters is constant) and reorganize into
\begin{align*}
  [x_1^2 + \dotsb + x_n^2] - 2 \mu (x_1 + \dotsb + x_n)
\end{align*}
The last term is a constant, so minimizing the variance of the overall
paragraph is equivalent to minimizing the variance for a suffix of
the paragraph (both are minimizing the sum of squares). This is just
the variance of the subproblem, so the dynamic programming is valid
since optimal substructure holds. In practice, I skip calculating
variance entirely and simply compute the sum of squares. I also do
dynamic programming on the variance of each _prefix_ instead of each
suffix so it's easier to ignore the contribution of the last line.

Note that the reduction could have been quickly
observed by using the variance identity
\begin{align*}
  \Var{X} &= \E{X^2} - \E{X}^2
\end{align*}
along with the fact that $ \E{X} $ is constant.
Indeed, we essentially repeated this derivation.

That's it! The algorithm runs in $ \BigO(nK) $ where $ n $ is the number of
characters in the input text and $ K $ is the desired width. Since $ K $ is
usually fixed to some small constant (79, 72, etc.), this is essentially linear
in $ n $ and I suspect most of the running time is bottlenecked by just I/O
(reading the input text and printing out the formatted text). Running with a
width of 79 on a 1 MB file with over 20,000 lines takes under 200 milliseconds.
For 100 MB, `fmt` takes around 11.9 seconds, `par` takes 15.7, and `far` takes
16.6. So `far` is slightly slower than the others, but certainly not enough to
be noticeable for "reasonable" inputs, especially if output is redirected into
a file rather than displayed to terminal.

## Examples

original paragraph:
```plaintext
xxxxx xxx xxx xxxx xxxxxxxxx xx x xxxxxxxxx x xxxx xxxx xxxxxxx xxxxxxxx xxx
xxxxxxxxx xxxxxxxx xx xx xxxxx xxxxx xxxx xx x xxxx xx xxxxxxxx xxxxxxxx xxxx
xxx xxxx xxxx xxx xxxxxxxxxxxxxxxxxxx xxxxxxxxxxxxx xxx xxxxx xx xxxx x xxxx
xxxxxxxx xxxx xxxx xx xxxxx xxxx xxxxx xxxx xxxxxxxxx xxx xxxxxxxxxxx xxxxxx
xxx xxxxxxxxx xxxx xxxx xx x xx xxxx xxx xxxx xx xxx xxx xxxxxxxxxxx xxxx xxxxx
x xxxxx xxxxxxx xxxxxxx xx xx xxxxxx xx xxxxx
```

`fmt -w 72` (greedy algorithm):
```plaintext
xxxxx xxx xxx xxxx xxxxxxxxx xx x xxxxxxxxx x xxxx xxxx xxxxxxx xxxxxxxx
xxx xxxxxxxxx xxxxxxxx xx xx xxxxx xxxxx xxxx xx x xxxx xx xxxxxxxx
xxxxxxxx xxxx xxx xxxx xxxx xxx xxxxxxxxxxxxxxxxxxx xxxxxxxxxxxxx xxx
xxxxx xx xxxx x xxxx xxxxxxxx xxxx xxxx xx xxxxx xxxx xxxxx xxxx
xxxxxxxxx xxx xxxxxxxxxxx xxxxxx xxx xxxxxxxxx xxxx xxxx xx x xx xxxx
xxx xxxx xx xxx xxx xxxxxxxxxxx xxxx xxxxx x xxxxx xxxxxxx xxxxxxx xx xx
xxxxxx xx xxxxx
```

Looking at the output of the greedy algorithm, because it always forms a line
if it fits, it creates highly variable line lengths. For example, there are
many "valleys" where a line is shorter than the lines adjacent to it, like
lines 2 and lines 4, giving the paragraph a jagged appearance overall.

`par 72` (with `PARINIT` set to `rTbgqR B=.,?'_A_a_@ Q=_s>|`):
```plaintext
xxxxx xxx xxx xxxx xxxxxxxxx xx x xxxxxxxxx x xxxx xxxx xxxxxxx xxxxxxxx
xxx xxxxxxxxx xxxxxxxx xx xx xxxxx xxxxx xxxx xx x xxxx xx xxxxxxxx
xxxxxxxx xxxx xxx xxxx xxxx xxx xxxxxxxxxxxxxxxxxxx xxxxxxxxxxxxx
xxx xxxxx xx xxxx x xxxx xxxxxxxx xxxx xxxx xx xxxxx xxxx xxxxx xxxx
xxxxxxxxx xxx xxxxxxxxxxx xxxxxx xxx xxxxxxxxx xxxx xxxx xx x xx xxxx
xxx xxxx xx xxx xxx xxxxxxxxxxx xxxx xxxxx x xxxxx xxxxxxx xxxxxxx xx xx
xxxxxx xx xxxxx
```

`par` improves on `fmt`, but still creates a single large valley.

`far 72`:
```plaintext
xxxxx xxx xxx xxxx xxxxxxxxx xx x xxxxxxxxx x xxxx xxxx xxxxxxx
xxxxxxxx xxx xxxxxxxxx xxxxxxxx xx xx xxxxx xxxxx xxxx xx x xxxx
xx xxxxxxxx xxxxxxxx xxxx xxx xxxx xxxx xxx xxxxxxxxxxxxxxxxxxx
xxxxxxxxxxxxx xxx xxxxx xx xxxx x xxxx xxxxxxxx xxxx xxxx xx xxxxx
xxxx xxxxx xxxx xxxxxxxxx xxx xxxxxxxxxxx xxxxxx xxx xxxxxxxxx
xxxx xxxx xx x xx xxxx xxx xxxx xx xxx xxx xxxxxxxxxxx xxxx xxxxx
x xxxxx xxxxxxx xxxxxxx xx xx xxxxxx xx xxxxx
```

Finally, I would argue that `far` creates the most
aesthetically pleasing paragraph because it minimizes
the variance, creating the smoothest paragraph edge.

It's probably possible to modify `PARINIT` for `par` to work properly on this
example, and in general `par` works quite well, but it's hard to work through
the documentation to find precisely what to do and the recommended `PARINIT`
in the man page should work well. In contrast `far` works well "out of the
box" and for better or for worse, only has a single configuration option ---
the desired line width.

## Uses

This program is useful whenever writing plaintext in a monospace text editor,
e.g. when editing LaTeX, markdown files, college essays, and emails. It's
especially useful in `vim`, which lets you set the option `'formatprg'` so
the operator `gq` formats using the external program.

## Appendix

The paragraph reformatting problem can be phrased
in terms of three different [$ \ell^p $-norms
](https://en.wikipedia.org/wiki/Lp_space#The_p-norm_in_finite_dimensions).

If $ X $ is a vector that represents the length of each line for a given
paragraph, we want the every line in the paragraph to be shorter than $ K $
characters, so $ \max(X) = \norm{X}_\infty \leq K $. We want to only reformat
the paragraph in such a way that the number of lines is the shortest possible
over all paragraphs $ P $ that obey the length constraint, that is,
\begin{align*}
  \norm{X}_0 = \min_{P: \, \norm{P}_\infty \leq K} \; \norm{P}_0
\end{align*}
where we abuse the [$ \ell^0
$-"norm"](https://en.wikipedia.org/wiki/Lp_space#When_p_=_0).
Finally, the goal of minimizing variance is equivalent to minimizing the
$ \ell^2 $ norm of $ X $ since the $ \ell^2 $ norm is the sum of squares.
In summary, we have the optimization problem taken over $ \mathcal{P} \defeq
\{ \text{valid paragraphs $ X $ such that $ \norm{X}_\infty \leq K $ } \} $ of
\begin{align*}
  \min_{\substack{X \in \mathcal{P} \\ \norm{X}_0 = N}} \; \norm{X}_2 \\
  N = \min_{P \in \mathcal{P}} \; \norm{P}_0
\end{align*}
where we have written the problem entirely in terms of $
\ell^p $-norms. This perspective doesn't really help when
coming up with an efficient algorithm, but it's kind of cute.


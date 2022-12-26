+++
title = "Algorithms for competitive programming"
date = Date(2019, 12, 7)
rss = "Implementations of some useful algorithms."
maxtoclevel = 3

tags = ["cs"]
+++

{{maketitle}}

\tableofcontents

## Basics

Running: `python`.

Profiling: `python -m cProfile -s tottime`.

Fast execution: [`pypy`](https://www.pypy.org/).

### Input

#### USACO

```python
import sys
input = lambda: sys.stdin.readline()[:-1] # fast cin

with open("test.in") as f:
    N = int(f.readline())
    N, M = map(int, f.readline().split())
    l = list(map(int, f.readline().split()))
    m = [list(map(int, line.split())) for line in f]

with open("test.out", "w") as f:
    f.write(str(ans) + "\n")
```

#### Fast input/output

[Verification: SPOJ INTEST](https://www.spoj.com/problems/INTEST/),
[SPOJ INOUTEST](https://www.spoj.com/problems/INOUTEST/),
testing with [timeit](https://docs.python.org/3/library/timeit.html)
```python
import sys
input = sys.stdin.readline # fast cin

__lines, __print = [], print
print = lambda s: __lines.append(s) # fast cout
cout = lambda: __print("\n".join(map(str, __lines)))
```

#### Codeforces

```python
import sys
input = sys.stdin.readline # fast cin

N = int(input())
l = list(map(int, input().split()))
```

#### Graphs

##### Unweighted

```python
graph = {i: [] for i in range(N)}
for i in range(N):
    a, b = map(lambda x: int(x) - 1, f.readline().split())
    graph[a].append(b)
    graph[b].append(a)
```

##### Weighted

```python
graph = {i: {} for i in range(N)}
for i in range(N):
    a, b, w = map(int, f.readline().split())
    a -= 1
    b -= 1
    # will overwrite if same edge is repeated - pick min edge (usually)
    if b not in graph[a] or w < graph[a][b]:
        graph[a][b] = w
    if a not in graph[b] or w < graph[b][a]:
        graph[b][a] = w
```

## Numerical Algorithms

### Number Theory

[Euclidean algorithm](https://en.wikipedia.org/wiki/Euclidean_algorithm) -
[Verification]() -
Complexity: $ \BigO(\log b) $
```python
def gcd(a: int, b: int) -> int:
    while b != 0:
        a, b = b, a % b
    return a

def lcm(a: int, b: int) -> int: return (a//gcd(a, b))*b
```

floor
```python
x//1
```

ceiling
```python
-(-x//1)
```

#### Primes

```python
def prime(n: int) -> bool:
    """ Checks whether a number is prime or not with trial divison. """
    if n == 1 or (n % 2 == 0 and n != 2):
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True
```

[Sieve of Eratosthenes](https://en.wikipedia.org/wiki/Sieve_of_Eratosthenes) -
[Verification](https://codeforces.com/group/M4wsRWBHyZ/contest/259141/problem/G) -
Complexity: $ \BigO(n \log \log n) $
```python
def sieve(n: int) -> list:
    """ Generates a look up table of primality up to n. """
    l = [True]*(n + 1)
    l[0] = l[1] = False
    for i in range(2, int(n**0.5) + 1):
        if l[i]:
            for j in range(i*i, len(l), i):
                l[j] = False
    return l
```

```python
def primes(n: int) -> list:
    """ Return a list of all primes less than or equal to n. """
    s = sieve(n)
    return [i for i in range(n + 1) if s[i]]
```

#### Modulo

```python
# common
m = 1000000007

(a + b) % m == ((a % m) + (b % m)) % m
(a - b) % m == ((a % m) - (b % m)) % m
(a * b) % m == ((a % m) * (b % m)) % m
```
tl;dr spam modulo if the problem asks you to return the answer mod `m`.

If asking for count (which is often) must be $ \geq 0 $
```python
x if x >= 0 else x + m
```

[Modular exponentiation](https://en.wikipedia.org/wiki/Modular_exponentiation) -
[Verification](https://codeforces.com/group/M4wsRWBHyZ/contest/238084/problem/E) -
Complexity: $ \BigO(\log e) $
```python
def mod_exp(b: int, e: int, m: int) -> int:
    """ Returns b^e % m """
    if m == 1: return 0
    rtn = 1
    b %= m
    while e > 0:
        # bit on in the binary representation of the exponent
        if e & 1 == 1:
            rtn = (rtn*b) % m
        e >>= 1
        b = (b*b) % m
    return rtn
```

[Extended Euclidean algorithm](https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm) -
[Verification](https://codeforces.com/group/M4wsRWBHyZ/contest/238084/problem/G) -
Complexity: $ \BigO(\log b) $
```python
def extended_gcd(a: int, b: int) -> tuple:
    """ Returns (gcd(a, b), x, y) such that ax + by = gcd(a, b). """
    x, xp = 0, 1
    y, yp = 1, 0
    r, rp = b, a

    while r != 0:
        q = rp//r
        rp, r = r, rp - q*r
        xp, x = x, xp - q*x
        yp, y = y, yp - q*y

    return rp, xp, yp
```

[Modular multiplicative inverse](https://en.wikipedia.org/wiki/Modular_multiplicative_inverse) -
[Verification](https://codeforces.com/group/M4wsRWBHyZ/contest/238084/problem/G) -
Complexity: same as the Euclidean algorithm
```python
def inv(x: int, m: int) -> int:
    """ Returns the inverse y such that xy mod m = 1. """
    return extended_gcd(x, m)[1] % m

(a/b) % m == (a*inv(b, m)) % m
```

#### Factorization

```python
def factor(n: int) -> list:
    """ Factors a number with trial division. """
    l = {1, n}
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            l.add(i)
            l.add(n//i)
    return list(l)
```

[Prime Factorization](https://en.wikipedia.org/wiki/Trial_division) -
[Verification](https://codeforces.com/group/M4wsRWBHyZ/contest/259141/problem/G) -
Complexity: $ \BigO(\sqrt{x}/\log{x}) $
```python
def prime_factor(n: int, primes: list) -> list:
    """ Prime factorizes a number, given a precomputed list of primes. """
    l = set()
    rt = int(n**0.5) + 1
    for p in primes:
        if p > rt:
            break
        while n % p == 0:
            n = n//p
            l.add(p)
    if n != 1:
        l.add(n)
    return list(l)
```

### Binary

[Most Significant Bit](http://web.stanford.edu/class/cs166/lectures/16/Slides16.pdf) -
[Verification]() -
Complexity: Can be done in $ \BigO(1) $ with some work
```python
def msb(n: int) -> int:
    """ Returns the index of the most significant bit of n. """
    return n.bit_length() - 1
```

[Brian Kernighan's set bit count](https://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetKernighan) -
[Verification: Othello]() -
Complexity: $ \BigO(\log x) $
```python
def set_pos(x: int) -> list:
    """ Finds the number of 1's in a number. """
    l = []
    while x:
        xp = x & (x - 1)
        l.append((x - xp).bit_length() - 1)
        x = xp
    return l
```

[Binary Counter]() -
[Verification]() -
Complexity: amortized $ \BigO(1) $ over $ n $ operations
```python
def increment(c: int, m: int) -> int:
    """ Increments a binary counter. """
    i = 1
    l = 1 << m
    while i < l and c & i > 0:
        c ^= i
        i <<= 1
    if i < l:
        c ^= i
    return c
```

### Matrices

Miscellaneous operations
```python
def dotp(u: list, v: list) -> float:
    """ Dot product. """
    return sum(u[i]*v[i] for i in range(len(u)))

def col(m: list, i: int) -> list:
    """ Column of a matrix. """
    return [m[j][i] for j in range(len(m))]

def mat_mult(a: list, b: list) -> list:
    """ Matrix multiplication. """
    return [[dotp(a[i], col(b, j)) for j in range(len(b[0]))]
            for i in range(len(a))]

def identity(n: int) -> list:
    """ Identity matrix of size n x n. """
    return [[int(i == j) for j in range(n)] for i in range(n)]

def print_mat(m: list) -> list:
    """ Pretty prints a matrix. """
    for row in m:
        print(row)

def mat_exp(A: list, k: int) -> list:
    """ Returns A^k. """
    v = identity(len(A))
    while k > 0:
        if k & 1 == 1:
            v = mat_mult(v, A)
        k >>= 1
        A = mat_mult(A, A)
    return v
```

[Gauss–Jordan elimination](https://en.wikipedia.org/wiki/Gaussian_elimination) -
[Verification: Google Foobar](https://foobar.withgoogle.com/) -
Complexity: $ \BigO(n^3) $
```python
def inv(m):
    """ Inverts a matrix. """
    N = len(m)
    # augment matrix with identity
    for i in range(N):
        m[i] += [F(i == j) for j in range(N)]

    for c in range(N):
        # get first nonzero entry
        pivot = [i for i in range(c, N) if m[i][c] != F(0)][0]
        m[c], m[pivot] = m[pivot], m[c]
        v = m[c][c]
        # set pivot value to 1
        for i in range(len(m[c])):
            m[c][i] /= v
        # make all zeros
        for r in range(N):
            if r != c:
                v = m[r][c]
                for i in range(len(m[r])):
                    m[r][i] -= v*m[c][i]

    return [row[N:] for row in m]
```

### Linear Programming

[Simplex Algorithm](https://en.wikipedia.org/wiki/Simplex_algorithm) -
[Verification]() -
Complexity: $ \BigO(?) $ (see [smoothed
analysis](https://en.wikipedia.org/wiki/Smoothed_analysis))

```python
def list_dict(l: list) -> dict:
    """ Converts a list into a dictionary based off indexes. """
    return {i + 1: l[i] for i in range(len(l))}

def prog_dict(A, b, c, contypes, t, nonneg) -> tuple:
    """
    Converts the easier list representation to a more general
    dictionary representation used by the simplex algorithm.
    """
    return (
        {v + 1: {i + 1: A[v][i] for i in range(len(A[v]))}
         for v in range(len(A))
        },
        list_dict(b), list_dict(c), list_dict(contypes),
        t, {x + 1 for x in nonneg}
    )

def negated(d: dict) -> dict:
    """ Negates a row. """
    return {k: -v for k, v in d.items()}

def negate(l: list) -> list:
    """ Negates a row vector. """
    return [-x for x in l]

def general_standard(A, b, c, contypes, t, nonneg) -> tuple:
    """ Converts a general linear program into standard form. """
    # 1. The objective function is a minimization rather than a maximization
    if t == MIN:
        c = negate(c)
    # 2. Variables without nonnegativity constraints
    v = 0
    while v < len(c):
        if v not in nonneg:
            c.insert(v + 1, -c[v])
            for i in range(len(A)):
                A[i].insert(v + 1, -A[i][v])
            v += 1
        v += 1
    # 3. Replace equality constraints with inequalities
    tA, tb, tcontypes = [], [], []
    for i in range(len(A)):
        if contypes[i] == EQ:
            tA += [A[i]]*2
            tb += [b[i]]*2
            tcontypes += [LEQ, GEQ]
        else:
            tA.append(A[i])
            tb.append(b[i])
            tcontypes.append(contypes[i])
    A, b, contypes = tA, tb, tcontypes
    # 4. Replace greater than equal to with less than or equal to
    A = [negate(A[i]) if contypes[i] == GEQ else A[i] for i in range(len(A))]
    b = [-b[i] if contypes[i] == GEQ else b[i] for i in range(len(b))]

    return A, b, c

def standard_slack(A, b, c) -> tuple:
    """ Converts a linear program in standard form to one in slack form. """
    n = len(A[0]) + 1
    A = {i + n: list_dict(A[i]) for i in range(len(A))}
    return (set(range(1, n)), set(A.keys()), A,
            {i + n: b[i] for i in range(len(b))}, list_dict(c), 0)

def general_slack(prog) -> tuple:
    """ Converts a general linear program into slack form. """
    return standard_slack(*general_standard(*prog))

def pivot(N, B, A, b, c, v, l, e) -> tuple:
    """ Returns the new linear program after replacing x_l with x_e. """
    # Compute coefficients for the equation with the new basic variable x_e.
    bp, cp, Ap = {}, {}, {v: {} for v in B - {l} | {e}}
    bp[e] = b[l]/A[l][e]
    for j in N - {e}:
        Ap[e][j] = A[l][j]/A[l][e]
    Ap[e][l] = 1/A[l][e]
    # Compute coefficients of the remaining constraints.
    for i in B - {l}:
        bp[i] = b[i] - A[i][e]*bp[e]
        for j in N - {e}:
            Ap[i][j] = A[i][j] - A[i][e]*Ap[e][j]
        Ap[i][l] = -A[i][e]*Ap[e][l]
    # Compute objective function
    vp = v + c[e]*bp[e]
    for j in N - {e}:
        cp[j] = c[j] - c[e]*Ap[e][j]
    cp[l] = -c[e]*Ap[e][l]
    # Compute new sets of basic and nonbasic variables.
    Np = N - {e} | {l}
    Bp = B - {l} | {e}

    return Np, Bp, Ap, bp, cp, vp

def solve(N, B, A, b, c, v) -> tuple:
    """
    Solves a linear program in slack form
    whose initial basic solution is feasible.
    """
    while any(x > EPSILON for x in c.values()):
        # Bland's rule for both e and l
        e = sorted(i for i in N if c[i] > EPSILON)[0]
        l = min(B, key=lambda i:
                (b[i]/A[i][e] if A[i][e] > EPSILON else float("inf"), i)
               )
        if A[l][e] <= EPSILON:
            return UNBOUNDED + "$"
        N, B, A, b, c, v = pivot(N, B, A, b, c, v, l, e)
    return (tuple(b.get(i, 0) for i in range(1, len(N) + 1)),
            v, (N, B, A, b, c, v))

def initialize_simplex(A, b, c) -> tuple:
    """
    Determines whether or not a linear program is feasible, and if it
    is, returns a slack form whose initial basic solution is feasible.
    """
    k = min(range(len(b)), key=lambda i: b[i])
    # initial basic solution feasible
    if b[k] >= EPSILON:
        return standard_slack(A, b, c)
    # objective function -x_0
    cp = {i: 0 for i in range(1, len(c) + 1)}
    cp[0] = -1
    N, B, A, b, c, v = standard_slack(A, b, c)
    # add -x_0 to each equation in the linear program
    for i in A:
        A[i][0] = -1
    N |= {0}
    N, B, A, b, cp, v = pivot(N, B, A, b, cp, v, len(N) + k, 0)
    x, v, (N, B, A, b, cp, v) = solve(N, B, A, b, cp, v)
    if abs(v) <= EPSILON:
        N -= {0}
        # remove x_0 from constraints
        for i in A:
            del A[i][0]
        # substitute to form objective function
        cp = {i: 0 for i in N}
        for i in c:
            if i in A:
                for j in A[i]:
                    cp[j] -= c[i]*A[i][j]
                v += c[i]*b[i]
            else:
                cp[i] += c[i]
        return N, B, A, b, cp, v
    return INFEASIBLE

def simplex(prog) -> tuple:
    """ Solves an arbitrary linear program in standard form. """
    slack = initialize_simplex(*prog)
    if slack == INFEASIBLE:
        return INFEASIBLE
    return solve(*slack)[:-1]
```

### Fast Fourier Transform
[Cooley–Tukey Recursive FFT](https://cp-algorithms.com/algebra/fft.html#toc-tgt-2) -
[Verification]() -
Complexity: $ \BigO(n \log n) $
```python
import cmath, math

def recur_fft(a: list, inv: bool=False) -> list:
    """ Computes the DFT of a with recursion and complex roots of unity. """
    n = len(a)
    if n == 1:
        return a
    wn = cmath.exp((-1 if inv else 1)*2*cmath.pi*1j/n)
    w = 1
    y0, y1 = recur_fft(a[::2], inv), recur_fft(a[1::2], inv)
    y = [0]*n
    for k in range(n >> 1):
        t = w*y1[k]
        y[k] = y0[k] + t
        y[k + (n >> 1)] = y0[k] - t
        w *= wn
    return y

def inv_recur_fft(a: list) -> list:
    """ Computes the inverse DFT of a. """
    return [x/len(a) for x in recur_fft(a, True)]
```

[Cooley–Tukey Iterative FFT](https://cp-algorithms.com/algebra/fft.html#toc-tgt-5) -
[Verification]() -
Complexity: $ \BigO(n \log n) $
```python
def rev_increment(c: int, m: int) -> int:
    """ Increments a reverse binary counter. """
    i = 1 << (m - 1)
    while c & i > 0:
        c ^= i
        i >>= 1
    return c ^ i

def bit_rev_copy(a: list) -> list:
    """ Constructs an initial order from a by reversing the bits of the index. """
    n, m = len(a), len(a).bit_length() - 1
    A = [0]*n
    c = 0
    for i in range(n):
        A[c] = a[i]
        c = rev_increment(c, m)
    return A

def iter_fft(a: list, inv: bool=False) -> list:
    """ Computes the DFT iteratively. """
    n = len(a)
    A = bit_rev_copy(a)
    for s in range(1, n.bit_length()):
        m = 1 << s
        wm = cmath.exp((-1 if inv else 1)*2*cmath.pi*1j/m)
        for k in range(0, n, m):
            w = 1
            for j in range(m >> 1):
                t = w*A[k + j + (m >> 1)]
                u = A[k + j]
                A[k + j] = u + t
                A[k + j + (m >> 1)] = u - t
                w *= wm
    return A

def inv_iter_fft(a: list) -> list:
    """ Computes the inverse DFT of a. """
    return [x/len(a) for x in iter_fft(a, True)]
```

Readings for the following number theoretic FFT algorithms:
 - [Euler's totient function](https://en.wikipedia.org/wiki/Euler%27s_totient_function)
 - [Primitive root modulo n](https://en.wikipedia.org/wiki/Primitive_root_modulo_n)
 - [Application of FFT to polynomial multiplication](https://cs170.org/assets/pdf/hw03-sol.pdf)
 - [The Number Theoretic Transform (NTT)](https://www.nayuki.io/page/number-theoretic-transform-integer-dft)

[Iterative FFT with modulo (NTT)](https://cp-algorithms.com/algebra/fft.html#toc-tgt-7) -
[Verification: SPOJ MAXMATCH](https://www.spoj.com/problems/MAXMATCH/) -
Complexity: O(n log n)
```python
def find_kp(n: int) -> tuple:
    """
    Finds the smallest k such that p = kn + 1 is prime.
    pi(n) = n/log n, so the probability of a random number
    being prime is 1/log n, expect to try log n numbers
    until finding a prime - expected O((sqrt n)(log n)).
    """
    k, p = 1, n + 1
    while not prime(p):
        k += 1
        p += n
    return k, p

def find_generator(n: int, k: int, p: int) -> int:
    """
    Euler's totient function phi(n) gives the
    order of a modulo multiplicative group mod p.
    phi(p) = p - 1 = kn
    prime factors of kn are 2, maybe k
    expected O(log^8 ish n)
    """
    prime_factors = [2] + ([k] if prime(k) else [])
    for i in range(p):
        # coprime and thus in the group
        if gcd(i, p) == 1:
            if all(mod_exp(i, k*n//factor, p) != 1
                   for factor in prime_factors):
                return i

def find_wp(n: int) -> tuple:
    """
    Returns w, the principal nth root of unity
    and p, the prime determining the mod.
    """
    k, p = find_kp(n)
    g = find_generator(n, k, p)
    return mod_exp(g, k, p), p

def get_w(w: int, N: int, n: int, p: int) -> int:
    """ Returns w, the principal nth root of unity for n. """
    return mod_exp(w, 1 << (N - n + 1), p)

def int_fft(a: list, wn: int, p: int) -> list:
    """ Compute the DFT iteratively with modulo instead of complex numbers. """
    n, lgn = len(a), len(a).bit_length()
    A = bit_rev_copy(a)
    for s in range(1, lgn):
        m = 1 << s
        wm = mod_exp(wn, 1 << (lgn - s - 1), p)
        for k in range(0, n, m):
            w = 1
            for j in range(m >> 1):
                t = w*A[k + j + (m >> 1)]
                u = A[k + j]
                A[k + j] = (u + t) % p
                A[k + j + (m >> 1)] = (u - t) % p
                w = (w*wm) % p
    return A

def inv_int_fft(a: list, wn: int, p: int) -> list:
    """ Computes the inverse DFT of a. """
    wn, n1 = inv(wn, p), inv(len(a), p)
    return [(x*n1) % p for x in int_fft(a, wn, p)]
```

[Recursive FFT with modulo (NTT)](https://cp-algorithms.com/algebra/fft.html#toc-tgt-7) -
[Verification]() -
Complexity: $ \BigO(n \log n) $
```python
def recur_int_fft(a: list, wn: int, p: int) -> list:
    """ Computes the DFT of a with recursion and modulo. """
    n = len(a)
    if n == 1:
        return a
    w = 1
    y0, y1 = int_fft(a[::2], (wn*wn) % p, p), int_fft(a[1::2], (wn*wn) % p, p)
    y = [0]*n
    for k in range(n >> 1):
        t = (w*y1[k]) % p
        y[k] = (y0[k] + t) % p
        y[k + (n >> 1)] = (y0[k] - t) % p
        w = (w*wn) % p
    return y
```

Polynomial Multiplication with the FFT -
[Verification: SPOJ POLYMUL](https://www.spoj.com/problems/POLYMUL/) -
Complexity: $ \BigO(n \log n) $

Note: make sure $ p $ is bigger than $ m^2 n $, where $ m $ is the
biggest number in the array and $ n $ is the length of the array.
```python
def mirror(a: list) -> list:
    """
    Mirrors a such that the resulting list has a length which is a power of 2.
    """
    n, np = len(a), 1 << math.ceil(math.log2(len(a)))
    a += [0]*(np - n)
    # for i in range(np - n):
    #     a[n + i] = a[n - i - 2]
    return a

def poly_mult(a: list, b: list, w: int, p: int) -> list:
    """ Multiplies two polynomials via the modular FFT. """
    m = len(a) + len(b) - 1
    n = max(len(a), len(b))
    # make both lists the same size and degree bound 2n instead of n
    ap = mirror(a + [0]*(n - len(a)) + [0]*n)
    bp = mirror(b + [0]*(n - len(b)) + [0]*n)
    w = get_w(w, N, len(ap).bit_length(), p)
    ap, bp = int_fft(ap, w, p), int_fft(bp, w, p)
    return inv_int_fft([ap[i]*bp[i] for i in range(len(ap))], w, p)[:m]

def ntt_sign(l: list, p: int) -> list:
    """ Assumes that substantial numbers are signed and therefore negative. """
    return [x if x < (p >> 1) else x - p for x in l]

if __name__ == "__main__":
    N = 20
    w, p = find_wp(1 << N)
```

[2D Convolution](https://en.wikipedia.org/wiki/Multidimensional_discrete_convolution) -
[Verification]() -
Complexity: $ \BigO(nm \log{nm}) $
```python
def flatten(m: list, pad=0) -> list:
    """ Flattens a matrix into a list. """
    return [x for row in m for x in row + [0]*pad]

def reshape(l: list, m: int, n: int) -> list:
    """ Shapes a list into a M x N matrix."""
    return [[l[r*n + c] for c in range(n)] for r in range(m)]

def conv(h: list, x: list):
    """ Computes the 2D convolution. """
    M, N, H, W = len(x), len(x[0]), len(h), len(h[0])
    # need to pad the columns to the final size
    h, x = flatten(h, N - 1), flatten(x, W - 1)
    return reshape(fft(h, x), M + H - 1, N + W - 1)

def prune(h: list, x: list) -> list:
    """ Prunes a convolution for the specific K x K filter case. """
    m, k = conv(h, x), min(len(h), len(x))
    pad = (k - 1)>>1
    return [row[pad:-pad] for row in m[pad:-pad]]
```

## Strings

### Suffix Structures

#### Suffix Array

Reference paper:
[_Linear Suffix Array Construction
by Almost Pure Induced-Sorting_](https://doi.org/10.1109/DCC.2009.42).

[Suffix Array by Induced Sorting](http://web.stanford.edu/class/cs166/lectures/04/Small04.pdf) -
[Verification: SPOJ SARRAY](https://www.spoj.com/problems/SARRAY/) -
Complexity: $ \BigO(m) $
```python
def lms_block(s: list, t: list, i: int) -> str:
    """ Label each character of s with L (False) or S (True). """
    j, l = i + 1, False
    while j < len(s) and not (l and t[j]):
        if not t[j]:
            l = True
        j += 1
    return s[i:j]

def induced_sort(s: list, t: list, blocks: list=None) -> list:
    """ Induced sort of s. """
    n = len(s)
    c = {}
    for i in range(n):
        c[s[i]] = c.get(s[i], 0) + 1

    # modified bucket sort -
    # the size of the alphabet is bounded by the length of the string
    order = [i for i in range(min(c), max(c) + 1) if i in c]
    b = {order[0]: [0, c[order[0]] - 1]}
    for i in range(1, len(order)):
        b[order[i]] = [b[order[i - 1]][1] + 1,
                       b[order[i - 1]][1] + c[order[i]]]

    sa = [-1]*n
    if blocks is None:
        for i in range(n - 1, 0, -1):
            # LMS index
            if not t[i - 1] and t[i]:
                sa[b[s[i]][1]] = i
                b[s[i]][1] -= 1

    # given sorted order from recursive call
    else:
        for i in reversed(blocks):
            sa[b[s[i]][1]] = i
            b[s[i]][1] -= 1

    # put L types from the front
    for i in range(n):
        j = sa[i] - 1
        if sa[i] > 0 and not t[j]:
            sa[b[s[j]][0]] = j
            b[s[j]][0] += 1

    # reset right borders
    for i in range(1, len(order)):
        b[order[i]] = [b[order[i]][0],
                       c[order[i]] + (b[order[i - 1]][1] if i > 0 else -1)]

    # put S types from the back
    for i in range(n - 1, -1, -1):
        j = sa[i] - 1
        if sa[i] > 0 and t[j]:
            sa[b[s[j]][1]] = j
            b[s[j]][1] -= 1

    return sa

def suffix_array(s: list) -> list:
    """ Construct the suffix array for the string s. """
    # convert a string to an array of integer ordinal values
    if isinstance(s, str):
        s = list(map(ord, s))

    n = len(s)
    # True is "S" type and "L" is False
    t = [True]*n
    for i in range(n - 2, -1, -1):
        if s[i] < s[i + 1]:
            t[i] = True
        elif s[i] > s[i + 1]:
            t[i] = False
        else:
            t[i] = t[i + 1]

    sa = induced_sort(s, t)

    # LMS blocks
    blocks = [sa[i] for i in range(n)
              if sa[i] > 0 and t[sa[i]] and not t[sa[i] - 1]]

    # name blocks
    names = {}
    for i in range(len(blocks)):
        names[blocks[i]] = names.get(blocks[i - 1], -1) + \
            (i == 0 or \
             lms_block(s, t, blocks[i]) != lms_block(s, t, blocks[i - 1])
            )
    blocks = [i for i in range(n) if i in names]

    # reduced list guaranteed to be < n/2 of the original
    reduced = [names[block] for block in blocks]

    # all distinct characters - base case
    m = len(reduced)
    if len(set(reduced)) == m:
        sa1 = [0]*(m + 1)
        for i in range(m):
            sa1[reduced[i] + 1] = i
    else:
        sa1 = suffix_array(reduced + [-1])

    # sort blocks by suffix array of reduced string
    temp = [0]*len(sa1)
    for i in range(1, len(sa1)):
        temp[i] = blocks[sa1[i]]
    blocks = temp

    sa = induced_sort(s, t, blocks)
    return sa
```

##### LCP Array

Reference paper:
[_Linear-Time Longest-Common-Prefix Computation in Suffix Arrays
and Its Applications_](https://doi.org/10.1007/3-540-48194-X_17)

[Kasai](http://web.stanford.edu/class/cs166/lectures/03/Slides03.pdf) -
[Verification: USACO Standing Out](http://www.usaco.org/index.php?page=viewproblem2&cpid=768) -
Complexity: $ \BigO(m) $
```python
def lcp_array(s: str, sa: list) -> list:
    """" Construct the LCP array given a string s and its suffix array sa. """
    n = len(s)
    rank = [0]*n
    for i in range(n):
        rank[sa[i]] = i

    lcp = [0]*(n - 1)
    h = 0
    for i in range(n):
        if rank[i] > 1:
            # suffix before rank[i] in the suffix array
            k = sa[rank[i] - 1]
            while s[i + h] == s[k + h]:
                h += 1
            lcp[rank[i] - 1] = h
            if h > 0:
                h -= 1

    return lcp
```

##### Generalized Suffix Arrays

[Generalized Suffix Arrays](http://web.stanford.edu/class/cs166/lectures/02/Slides02.pdf) -
[Verification: SPOJ LPS](http://www.spoj.com/problems/LPS/) -
Complexity: O(m)
```python
def generalized_suffix_array(words: list) -> tuple:
    """ Build a single suffix array on the concatenation of multiple words. """
    n = [v for i, word in enumerate(words)
         for v in (list(map(ord, word)) + [i - len(words)])]
    return n, suffix_array(n)
```

#### Suffix Tree

[Suffix Trees](http://web.stanford.edu/class/cs166/lectures/04/Small04.pdf) -
[Verification: SPOJ STAMMER](http://www.spoj.com/problems/STAMMER/) -
Complexity: O(m)
```python
class SuffixTree:

    def __init__(self, key, parent=None, left=None, right=None, *,
                 start: int=None, end: int=None) -> None:
        # cartesian tree variables
        # stores the LCP value for internal nodes
        # and the suffix index for leaf nodes
        self.key = key
        self.parent = parent
        self.child = [left, right]

        # suffix tree variables
        # keyed by first character (distinct)
        self.children = {}
        # start and end indexes in the string
        self.start, self.end = start, end
        # root contains a copy of the original string
        self.s = None

    def __str__(self, n=None, s="", d=0) -> str:
        """ Fancy tree printing (uses the original string to draw edges). """
        if self.s is not None:
            rtn = []

            # precompute heights
            h = {None: -1, self: 0}
            q = [self]
            while len(q) > 0:
                n = q.pop()
                for child in n.children.values():
                    q.append(child)
                    h[child] = h[n] + 1

            # actually compute string representation
            q = [self]
            while len(q) > 0:
                n = q.pop()
            while len(q) > 0:
                n = q.pop()
                edge = self.s[n.start: n.end + 1] \
                    if n.start is not None and n.end is not None else ""
                # edge = (n.start, n.end)
                rtn.append("{}{} {}\n".format(" "*4*h[n], n.key, edge))
                for child in sorted(n.children, reverse=True):
                    q.append(n.children[child])
            return "".join(rtn)
        return "{}{}".format(self.key, "$" if len(self.children) == 0 else "")

    def __contains__(self, s: str) -> bool:
        """ Whether s is a substring of the suffix tree. """
        return self.match(s) is not None

    def match(self, s: str) -> "SuffixTree":
        """
        Returns the subtree matching the string s, None if it does not exist.
        """
        i = 0
        t = self
        while i < len(s):
            if s[i] not in t.children:
                return None
            t = t.children[s[i]]
            j = t.start
            while j <= t.end and i < len(s):
                if s[i] != self.s[j]:
                    return None
                j += 1
                i += 1
        return t

    def count(self, s: str) -> list:
        """ Reports the index of each occurrence of s in the string. """
        t = self.match(s)
        l = []
        if t is None:
            return l

        stk = [t]
        while len(stk) > 0:
            n = stk.pop()
            # leaf node
            if len(n.children) == 0:
                l.append(n.key)
            for child in n.children.values():
                stk.append(child)

        return l

    def suffix_array(self) -> list:
        """ Inorder traversal yields a suffix array. """
        sa = []
        stk = [self]
        while len(stk) > 0:
            n = stk.pop()
            if len(n.children) == 0:
                sa.append(n.key)
            # s log s where s is the size of the alphabet
            for k in sorted(n.children, reverse=True):
                stk.append(n.children[k])

        return sa

def cartesian_tree(l: list) -> SuffixTree:
    """ Constructs a Cartesian tree for a LCP array in O(n). """
    stk = []
    for i in range(len(l)):
        c = None
        while len(stk) > 0 and stk[-1].key > l[i]:
            c = stk.pop()
        stk.append(SuffixTree(l[i], stk[-1] if len(stk) > 0 else None, c))
        # add right child
        if len(stk) > 1:
            stk[-2].child[1] = stk[-1]
            # merge same values
            # if stk[-1].key == stk[-2].key:
            #     stk[-2].child[0] = stk[-1].child[0]
            #     stk.pop()
    return stk[0]

def process(n: SuffixTree, sa: list, sword: list,
            wstart: dict, c: int, i: int) -> None:
    """ Adds a suffix of the string if the node is missing a child. """
    if n.child[c] is None:
        end = wstart[sword[sa[i]] + 1] - 1 \
            if sword[sa[i]] + 1 < len(wstart) else len(sa) - 1
        n.child[c] = SuffixTree(sa[i], n, start=sa[i] + n.key, end=end)
        return i + 1
    return i

def dfs(s: str, sa: list, t: SuffixTree,
        sword: list, wstart: dict) -> SuffixTree:
    """ Construct a suffix tree from the string s given its suffix array. """
    stk = [(t, 0)]
    i = 0
    while len(stk) > 0:
        n, p = stk[-1]
        i = process(n, sa, sword, wstart, 0, i)
        # either leaf or done with children
        if p == 2:
            stk.pop()
            i = process(n, sa, sword, wstart, 1, i)
            # merge same values
            if n.parent is not None and n.key == n.parent.key:
                n.parent.child = [
                    child for child in n.parent.child + n.child
                    if child.start is not None or child.key != n.key
                ]
            # label nodes with start and end values
            for child in n.child:
                if child.start is None:
                    # min start index of children minus 1
                    child.end = min(child.child,
                                    key=lambda x: x.start).start - 1
                    # difference in LCP values
                    child.start = child.end - (child.key - n.key) + 1
                # key by first character of edge
                n.children[s[child.start]] = child
                child.parent = n

        # has children left to process
        else:
            child = n.child[p]
            stk[-1] = (n, p + 1)
            if child is not None and child.start is None:
                stk.append((child, 0))
    # copy original string to convert (start, end) into substrings
    t.s = s
    # additional state information
    t.sword, t.wstart = sword, wstart
    return t

def suffix_tree(s: str, sa: list, lcp: list,
                word: list=[], start: dict={}) -> SuffixTree:
    """ Constructs a suffix tree for a string in O(n). """
    return dfs(s, sa, cartesian_tree(lcp),
               word if len(word) > 0 else [0]*len(s),
               start if len(start) > 0 else {s: 0})
```

#### TODO

Ukkonen's algorithm (direct construction)

Lectures:
 - [Duke](https://www2.cs.duke.edu/courses/fall14/compsci260/resources/suffix.trees.in.detail.pdf)
 - [CMU](https://www.cs.cmu.edu/~ckingsf/bioinfo-lectures/suffixtrees.pdf)
 - [Stanford](https://web.stanford.edu/~mjkay/gusfield.pdf)

##### Generalized Suffix Tree

[Generalized Suffix Tree](http://web.stanford.edu/class/archive/cs/cs166/cs166.1186/lectures/03/Small03.pdf) -
[Verification]() -
Complexity: $ \BigO(n) $
```python
def generalized_suffix_tree(words: list) -> SuffixTree:
    """ Construct a suffix tree on the concatenation of multiple words. """
    s, sword, wstart = [], [], {}
    j = 0
    for i, word in enumerate(words):
        wstart[i] = j
        for k in range(len(word)):
            s.append(word[k])
            sword.append(i)
            j += 1
        s.append(chr(128 + i))
        sword.append(i)
        j += 1
    sword.append(len(words))

    s = "".join(s) + "$"
    sa = suffix_array(s)
    lcp = lcp_array(s, sa)

    t = suffix_tree(s, sa, lcp, sword, wstart)
    return t
```

##### Suffix Tree to DAG

Suffix Tree to DAG -
[Verification]() -
Complexity: $ \BigO(n) $
```python
def suffix_tree_dag(t: SuffixTree) -> dict:
    """ Converts a suffix tree to a numeric DAG. """
    ids, graph = {t: 0}, {}
    stk = [t]
    i = 1
    while len(stk) > 0:
        n = stk.pop()
        if ids[n] not in graph:
            graph[ids[n]] = []
        for child in n.children.values():
            ids[child] = i
            i += 1
            graph[ids[n]].append(ids[child])
            stk.append(child)
    return ids, graph
```

### Matching

#### Aho-Corasick

[Aho-Corasick](http://web.stanford.edu/class/archive/cs/cs166/cs166.1166/lectures/02/Small02.pdf) -
[Verification: USACO Censoring](http://www.usaco.org/index.php?page=viewproblem2&cpid=533) -
Complexity: $ \langle \BigO(n), \BigO(m + z) \rangle $
```python
from collections import deque

class Trie:

    def __init__(self, ch=None) -> None:
        self.children = {}                           # pointers to children
        self.ch = ch if isinstance(ch, str) else ""  # character
        self.end = []                                # represents a pattern
        self.suffix = self.output = None             # aho-corasick

        # create trie from list of patterns
        if isinstance(ch, list):
            for i, pattern in enumerate(ch):
                self.add(pattern, i)

    def __str__(self) -> str:
        if self.ch == "":
            rtn = []

            # precompute heights
            h = {None: -1, self: 0}
            q = [self]
            while len(q) > 0:
                n = q.pop()
                for child in n.children.values():
                    q.append(child)
                    h[child] = h[n] + 1

            # actually compute string representation
            q = [self]
            while len(q) > 0:
                n = q.pop()
                rtn.append("{}{}{}:{} {}\n".format(" "*h[n], n.ch,
                                                   "$" if n.end else "",
                                                   h[n.suffix], h[n.output]))
                for child in sorted(n.children.values(),
                                    reverse=True, key=lambda x: x.ch):
                    q.append(child)
            return "".join(rtn)
        return "{}{}".format(self.ch, "$" if self.end is not None else "")

    def add(self, s: str, id: int=None) -> None:
        """ Add s to the trie. """
        for ch in s:
            if ch not in self.children:
                self.children[ch] = Trie(ch)
            self = self.children[ch]
        self.end.append(id)

def link(root: Trie) -> None:
    """ Computes the suffix links and output links for a given trie. """
    q = deque([(None, root)])
    while len(q) > 0:
        prev, n = q.popleft()
        if prev is not None:
            # suffix links
            x = prev.suffix
            while x is not None:
                if n.ch in x.children:
                    n.suffix = x.children[n.ch]
                    break
                x = x.suffix
            else:
                n.suffix = root

            # output links
            n.output = n.suffix if len(n.suffix.end) > 0 else n.suffix.output
        for child in n.children.values():
            q.append((n, child))

def aho_corasick(s: str, patterns: list, t: Trie=None) -> list:
    """ Return all matches of patterns in s with the Aho-Corasick automata. """
    # create automata or use precomputed
    if t is None:
        root = Trie(patterns)
        link(root)
        t = root

    # find matches
    matches = [0]*len(patterns)
    for ch in s:
        while ch not in t.children and t.ch != "":
            t = t.suffix
        if ch in t.children:
            t = t.children[ch]
        if len(t.end) > 0:
            matches[t.end[0]] += 1
        word = t.output
        while word is not None:
            matches[word.end[0]] += 1
            word = word.output

    # reuse values for duplicated patterns
    stk = [root]
    while len(stk) > 0:
        n = stk.pop()
        for i in range(1, len(n.end)):
            matches[n.end[i]] = matches[n.end[0]]
        for child in n.children.values():
            stk.append(child)

    return matches
```

#### Knuth-Morris-Pratt

[Knuth-Morris-Pratt](https://web.stanford.edu/class/cs97si/10-string-algorithms.pdf) -
[Verification: SPOJ Rotations](https://www.spoj.com/problems/EC_WORLD/) -
Complexity: $ \langle \BigO(n), \BigO(m) \rangle $
```python
def prefix_function(s: str) -> list:
    pi = [-1]*len(s)
    k = -1
    for i in range(1, len(s)):
        while k >= 0 and s[k + 1] != s[i]:
            k = pi[k]
        if s[k + 1] == s[i]:
            k += 1
        pi[i] = k
    return pi

def kmp(s: str, p: str) -> list:
    pi = prefix_function(p)
    k = -1
    matches = []
    for i in range(len(s)):
        while k >= 0 and p[k + 1] != s[i]:
            k = pi[k]
        if p[k + 1] == s[i]:
            k += 1
        if k == len(p) - 1:
            matches.append(i - len(p) + 1)
            k = pi[k]
    return matches
```

## Dynamic Programming

### Memoization

```python
import sys
from functools import lru_cache

sys.setrecursionlimit(10**5)

@lru_cache(maxsize=None)
def recur(*args, **kwargs):
    ...

print(recur.cache_info())
```

## Graph Algorithms
### Traversals

[Breadth-first search](https://en.wikipedia.org/wiki/Breadth-first_search) -
[Verification: AI]() -
Complexity: $ \BigO(V + E) $
```python
from collections import deque

def bfs(graph, start):
    """ Breadth-first search on graph from start. """
    seen = {start}
    q = deque([start])
    while len(q) > 0:
        n = q.popleft()
        for child in graph[n]:
            if child not in seen:
                seen.add(child)
                q.append(child)
```

[Depth-first search](https://en.wikipedia.org/wiki/Depth-first_search) -
[Verification: AI]() -
Complexity: $ \BigO(V + E) $
```python
def dfs(graph, start):
    """ Depth-first search on graph from start. """
    seen = {start}
    stk = [start]
    while len(stk) > 0:
        n = stk.pop()
        for child in graph[n]:
            if child not in seen:
                seen.add(child)
                stk.append(child)
```

[Iterative post-order depth-first search](https://en.wikipedia.org/wiki/Tree_traversal) -
[Verification](https://codeforces.com/group/M4wsRWBHyZ/contest/259141/problem/A) -
Complexity: $ \BigO(V + E) $
```python
def dfs(graph, u):
    """ Iterative post-order depth-first search. """
    seen = {u}
    stk = [(u, 0)]
    while len(stk) > 0:
        n, p = stk[-1]
        # either leaf or done with children
        if len(graph[n]) == p:
            stk.pop()
        # has children left to process
        else:
            child = graph[n][p]
            stk[-1] = (n, p + 1)
            if child not in seen:
                stk.append((child, 0))
                seen.add(child)
```

[Iterative post-order DFS (without indexing children)](https://en.wikipedia.org/wiki/Tree_traversal) -
[Verification](https://codeforces.com/group/M4wsRWBHyZ/contest/259141/problem/A) -
Complexity: $ \BigO(V + E) $
```python
def dfs(graph, u):
    """ Iterative post-order depth-first search. """
    done, seen = set(), set()
    stk = [u]
    while len(stk) > 0:
        n = stk[-1]
        # done with children
        if n in seen:
            if n not in done:
                done.add(n)
            stk.pop()
        # has children left to process
        else:
            seen.add(n)
            for child in graph[n]:
                if child not in seen:
                    stk.append(child)
```

#### Eulerian Tour

[Eulerian Tour](https://en.wikipedia.org/wiki/Eulerian_path) -
[Verification: USACO Training Riding the Fences](https://train.usaco.org/usacoprob2?a=TpH8RHs6Taa&S=fence) -
Complexity: $ \BigO(V + E) $
```python
def eulerian_tour(graph: dict) -> list:
    """ Finds a Eulerian tour or walk of the graph, whichever is possible. """
    # remove nodes with no edges
    graph = {k: v for k, v in graph.items() if len(v) > 0}
    count = sum(len(graph[v]) % 2 for v in graph)
    if count != 0 and count != 2:
        return # no such tour
    start = 0 if count == 0 else \
        next((v for v in graph if len(graph[v]) % 2 == 1))

    stk = [(start, 0)]
    tour = []
    seen = set()
    while len(stk) > 0:
        n, p = stk[-1]
        # done with children
        if len(graph[n]) == p:
            tour.append(n)
            stk.pop()
        # has children left to process
        else:
            child = graph[n][p]
            stk[-1] = (n, p + 1)
            # edges have a unique id - (vertex, id)
            if (child[1] not in seen):
                stk.append((child[0], 0))
                seen.add(child[1])
    return tour[::-1]
```

#### Topological Sort

[Topological Sort](https://en.wikipedia.org/wiki/Topological_sorting) -
[Verification](https://leetcode.com/problems/course-schedule-ii/) -
Complexity: $ \BigO(V + E) $
```python
def dfs(graph: dict, u: int, order: list, done: set) -> list:
    """ Depth-first search on graph from start. """
    stk = [u]
    seen = set()
    while len(stk) > 0:
        n = stk[-1]
        # done with children
        if n in seen:
            if n not in done:
                done.add(n)
                order.append(n)
            stk.pop()
        # has children left to process
        else:
            seen.add(n)
            for child in graph[n]:
                if child not in done:
                    if child not in seen:
                        stk.append(child)
                    # cycle, child processed before
                    else:
                        return True

def topological_sort(graph: dict) -> list:
    """ Toplogical ordering on the directed acylic graph. """
    order, done = [], set()
    for n in graph:
        if n not in done:
            if dfs(graph, n, order, done):
                return []
    return order[::-1]
```

### Shortest Path

[Dijkstra](https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm) -
[Verification: USACO Training Sweet Butter](https://train.usaco.org/usacoprob2?a=TpH8RHs6Taa&S=butter) -
Complexity: $ \BigO((E + V) \log V) $ (I think)
```python
import heapq

def dijkstra(graph, i):
    """ Single-source shortest path for the graph starting at i. """
    dists = {i: float("inf") for i in range(len(graph))}
    paths = {i: 0 for i in range(len(graph))}
    dists[i] = 0
    pq = [(dists[i], i)]
    seen = set()
    while len(pq) > 0:
        dist, n = heapq.heappop(pq)
        if n in seen: continue
        seen.add(n)
        for c, w in graph[n].items():
            if dists[n] + w < dists[c]:
                dists[c] = dists[n] + w
                paths[c] = n
                if c not in seen:
                    heapq.heappush(pq, (dists[c], c))
    return dists, paths
```

Dijkstra assuming the priority queue provides a `.update(key, value)` function
```python
def heap_dijkstra(graph, i):
    """ Single-source shortest path for the graph starting at i. """
    dists = {i: float("inf") for i in range(len(graph))}
    paths = {i: 0 for i in range(len(graph))}
    dists[i] = 0
    pq = BST()
    for v in graph:
        pq.add((float("inf") if v != i else 0, v), v)

    seen = set()
    while len(pq) > 0:
        dist, n = pq.pop()
        dist = dist[0]
        seen.add(n)
        for c, w in graph[n].items():
            if dists[n] + w < dists[c] and c not in seen:
                temp = dists[c]
                dists[c] = dists[n] + w
                paths[c] = n
                pq.update((temp, c), c, (dists[c], c))
    return dists, paths
```

[Floyd-Warshall](https://en.wikipedia.org/wiki/Floyd%E2%80%93Warshall_algorithm) -
[Verification: USACO Training Bessie Come Home](https://train.usaco.org/usacoprob2?a=TpH8RHs6Taa&S=comehome) -
Complexity: $ \BigO(V^3) $
```python
def floyd_warshall():
    """ All-pairs shortest path for the graph. """
    m = [[float("inf")]*N for i in range(N)]

    for i in range(N):
        for j in range(N):
            m[i][j] = 0 if i == j else \
                (graph[i][j] if j in graph[i] else m[i][j])

    for k in range(N):
        for i in range(N):
            for j in range(N):
                if m[i][k] + m[k][j] < m[i][j]:
                     m[i][j] = m[i][k] + m[k][j]
```

Floyd-Warshall for sparse graphs -
[Verification: USACO Training Cow Tours](https://train.usaco.org/usacoprob2?a=TpH8RHs6Taa&S=cowtour)
```python
def floyd_warshall():
    """ All-pairs shortest path for the graph. """
    m = [[float("inf")]*N for i in range(N)]

    for i in range(N):
        for j in range(N):
            m[i][j] = 0 if i == j else \
                (graph[i][j] if j in graph[i] else m[i][j])

    poss = [set([i for i in range(N) if row[i] != float("inf")]) for row in m]

    for k in range(N):
        for i in range(N):
            if k not in poss[i]: continue
            for j in poss[k]:
                if m[i][k] + m[k][j] < m[i][j]:
                     m[i][j] = m[i][k] + m[k][j]
                     poss[i].add(j)
```

[Manhattan Distance](https://en.wikipedia.org/wiki/Taxicab_geometry)
```python
manhat = lambda i, j, x, y: abs(i - x) + abs(j - y)
```

[A*](https://en.wikipedia.org/wiki/A*_search_algorithm) -
[Verification: AI]() -
Complexity: ???
```python
def Astar(start):
    """ Single-source shortest path for the graph. """
    seen = set()
    pq = [(dist(start), start, 0)]
    while len(pq) > 0:
        dis, n, moves = heapq.heappop(pq)
        if n == goal:
            return moves
        if n in seen: continue
        for child in get_children(n):
            heapq.heappush(pq,
                           (heuristic(child) + moves + 1, child, moves + 1)
                          )
        seen.add(n)
```

### Union-find

[Union-find (or disjoint-set)](https://activities.tjhsst.edu/sct/lectures/1920/2019_10_18_Union_Find_and_MST.pdf) -
[Verification: USACO Closing](http://usaco.org/index.php?page=viewproblem2&cpid=646) -
Complexity: $ \BigO(\alpha(N)) $

where $ \alpha $ is the inverse [Ackermann
function](https://en.wikipedia.org/wiki/Ackermann_function).

@@centering
\begin{figure}
  \figpreview{Mikasa Ackerman}{358414.webp}{225}{350}{
    https://cdn.myanimelist.net/images/characters/15/358414.jpg
  }
  \caption{
    [Mikasa Ackerman](https://myanimelist.net/character/40881/Mikasa_Ackerman).
  }
\end{figure}
@@

```python
def union_init(n: int) -> tuple:
    """ Initialize the union-find data structure. """
    return {i: i for i in range(n)}, {i: 1 for i in range(n)}

def find(parent: dict, u: int) -> int:
    """ Find the root of u. """
    if parent[u] == u:
        return u
    parent[u] = find(parent, parent[u])
    return parent[u]

def union(parent: dict, size: dict, u: int, v: int) -> bool:
    """ Union the components of u and v. """
    ur, vr = find(parent, u), find(parent, v)
    if ur == vr:
        return False

    x, y = (ur, vr) if size[ur] < size[vr] else (vr, ur)
    parent[x] = y
    size[y] += size[x]
    return True
```

Iterative variant
```python
def find(parent: dict, u: int) -> int:
    """ Find the root of u. """
    i = u
    while parent[i] != i:
        i = parent[i]
    while parent[u] != u:
        p = parent[u]
        parent[u] = i
        u = p
    return i
```

### Minimum Spanning Tree

[Kruskal](https://activities.tjhsst.edu/sct/lectures/1920/2019_10_18_Union_Find_and_MST.pdf) -
[Verification: USACO Simplify](http://www.usaco.org/index.php?page=viewproblem2&cpid=101) -
Complexity: $ \BigO(E \log E) = \BigO(E \log V) $
```python
def kruskal(graph: dict) -> list:
    """ Find the minimum spanning tree of the graph with Kruskal's. """
    parent, size = {u: u for u in graph}, {u: 1 for u in graph}
    span = []
    for e in sorted((graph[u][v], u, v)
                    for u in graph for v in graph[u] if v > u):
        w, u, v = e
        if find(parent, u) != find(parent, v):
            span.append((w, u, v))
            union(parent, size, u, v)
    return span
```

[Prim](https://en.wikipedia.org/wiki/Prim%27s_algorithm) -
[Verification](https://train.usaco.org/usacoprob2?a=TpH8RHs6Taa&S=agrinet) -
Complexity: $ \BigO(V^2) $
```python
def prim(m: list) -> float:
    """ Find the minimum spanning tree of the graph with Prim's. """
    distances = {i: float("inf") for i in range(len(m))}
    paths = {i: 0 for i in range(len(m))}
    tree = {i: False for i in range(len(m))}
    size, cost = 1, 0
    tree[0] = True
    for i in range(len(m)):
        distances[i] = m[0][i]
        paths[i] = 0
    while size < len(m):
        i = min(distances,
                key=lambda x: distances[x] if not tree[x] else float("inf"))
        size += 1
        cost += distances[i]
        tree[i] = True
        for j in range(len(m)):
            if distances[j] > m[i][j]:
                distances[j] = m[i][j]
                paths[j] = i
    return cost
```

### Connected Components

Important if using recursion
```python
import sys
sys.setrecursionlimit(10**6)
```

#### Undirected

Connected components -
[Verification](https://codeforces.com/group/M4wsRWBHyZ/contest/238084/problem/I) -
Complexity: $ \BigO(V + E) $
```python
def assign(u, num, ids={}):
    """ Assign u and its children to the component num. """
    stk = [u]
    while len(stk) > 0:
        n = stk.pop()
        ids[n] = num
        for child in graph[n]:
            if child not in ids:
                ids[child] = num
                stk.append(child)

def connected(graph):
    """ Find the connected components of the graph. """
    ids, num = {}, 0
    for u in graph:
        if u not in ids:
            assign(u, num, ids)
            num += 1
    return ids, num
```

#### Directed (Strongly connected components)

Recursion bad - see iterative post-order/assign
```python
def visit(u, seen=set(), l=deque([])):
    """ Visit u and its children. """
    if u not in seen:
        seen.add(u)
        for v in reverse[u]:
            visit(v, seen, l)
        l.appendleft(u)
    return seen, l

def assign(u, num, ids={}):
    """ Assign u and its children to the component num. """
    if u not in ids:
        ids[u] = num
        for v in graph[u]:
            assign(v, num, ids)
    return ids
```

[Kosaraju-Sharir](https://activities.tjhsst.edu/sct/lectures/1920/2019_11_01_Strongly_Connected_Components.pdf) -
[Verification](https://codeforces.com/group/M4wsRWBHyZ/contest/259141/problem/A) -
Complexity: O(V + E)
```python
def assign(u, num, ids={}):
    """ Assign u and its children to the component num. """
    stk = [u]
    while len(stk) > 0:
        n = stk.pop()
        ids[n] = num
        for child in graph[n]:
            if child not in ids:
                ids[child] = num
                stk.append(child)

def visit(u, seen=set(), l=deque([])):
    """ Visit u and its children. """
    if u in seen: return
    stk = [u]
    done = set()
    while len(stk) > 0:
        n = stk[-1]
        # done with children
        if n in seen:
            if n not in done:
                l.appendleft(n)
                done.add(n)
            stk.pop()
        # has children left to process
        else:
            seen.add(n)
            for child in reverse[n]:
                if child not in seen:
                    stk.append(child)

def kosaraju_sharir(graph):
    """ Find the strongly connected components with Kosaraju-Sharir. """
    seen, l = set(), deque([])
    for u in graph:
        visit(u, seen, l)
    ids, num = {}, 0
    for u in l:
        if u not in ids:
            assign(u, num, ids)
            num += 1
    return ids, num
```

Recover components from ids
```python
comps = {i: [] for i in range(N)}
for key, value in ids.items():
    comps[value].append(key)
```

### Tree

#### LCA

[Lowest Common Ancestor](https://en.wikipedia.org/wiki/Lowest_common_ancestor) -
[Verification: PClassic]() -
Complexity: $ \langle 0, \BigO(n) \rangle $
```python
def bfs(graph, start):
    """ Breadth-first search on graph from start. """
    heights = {}
    parents = {start: start}
    q = deque([(start, 0)])
    while len(q) > 0:
        n, h = q.popleft()
        heights[n] = h
        for child in graph[n]:
            if child not in heights:
                heights[child] = h + 1
                parents[child] = n
                q.append((child, h + 1))
    return heights, parents

def lca(heights, parents, u, v):
    h1, h2 = heights[u], heights[v]
    x, y = (u, v) if h1 > h2 else (v, u)
    while h1 != h2:
        x = parents[x]
        if h1 > h2:
            h1 -= 1
        else:
            h2 -= 1
    while x != y:
        x = parents[x]
        y = parents[y]
    return x
```

##### 2^n Jump Pointers

[Jump Pointers](https://activities.tjhsst.edu/sct/lectures/1920/2019_10_25_LCA.pdf) -
[Verification: USACO Max Flow](http://www.usaco.org/index.php?page=viewproblem2&cpid=576) -
Complexity: $ \langle \BigO(n \log n), \BigO(\log n) \rangle $
```python
import math

def build_table(parents: dict) -> list:
    """ Builds a 2^n jump pointer table in O(n log n) """
    n, m = len(parents), math.ceil(math.log2(len(parents)))
    dp = [[0]*m for i in range(n)]

    for i in range(n):
        dp[i][0] = parents[i]

    for j in range(m - 1):
        for i in range(n):
            dp[i][j + 1] = dp[dp[i][j]][j]

    return dp

def jump(table: list, u: int, d: int) -> int:
    """ Returns the ancestor d height above a node u in O(log d). """
    i = 0
    while d > 0:
        if d & 1 == 1:
            u = table[u][i]
        d >>= 1
        i += 1
    return u

def lca(heights: dict, table: list, u: int, v: int) -> int:
    """ Returns the Lowest Common Ancestor (LCA) in O(log n) """
    h1, h2 = heights[u], heights[v]
    x, y = (u, v) if h1 > h2 else (v, u)
    x = jump(table, x, abs(h1 - h2))
    if x == y: return x

    i = len(table[x]) - 1
    while i >= 0:
        if table[x][i] != table[y][i]:
            x, y = table[x][i], table[y][i]
        i -= 1

    return table[x][0]
```

##### LCA with Range Minimum Query

[Euler Tour](https://activities.tjhsst.edu/sct/lectures/1819/2019_2_11_LCA.pdf) -
[Verification: USACO Max Flow](http://www.usaco.org/index.php?page=viewproblem2&cpid=576) -
Complexity: $ \langle \BigO(n), \BigO(1) \rangle $
```python
def traversal(u) -> tuple:
    """ Euler tour on the tree rooted at u. """
    stk = [(u, 0, 0)]
    seen = set()
    seen.add(u)
    left, right, l = {}, [], []
    while len(stk) > 0:
        n, p, h = stk[-1]
        # either leaf or done with children
        if len(graph[n]) == p:
            stk.pop()

            left[n] = len(l)
            right.append(n)
            l.append(h)
        # has children left to process
        else:
            child = graph[n][p]
            stk[-1] = (n, p + 1, h)
            if child not in seen:
                stk.append((child, 0, h + 1))
                seen.add(child)

                left[n] = len(l)
                right.append(n)
                l.append(h)
    return left, right, l

indexes, inv, array = traversal(0)
fh = fischer_heun(array)
l = inv[rmq(array, *fh, indexes[a], indexes[b])]
```

#### Heavy-Light Decomposition

[Heavy-Light Decomposition](https://activities.tjhsst.edu/sct/lectures/1819/2019_3_15_HLD.pdf) -
[Verification: USACO Milk Visits](http://usaco.org/index.php?page=viewproblem2&cpid=970) -
Complexity: $ \BigO(n) $
```python
def hld(graph: dict, start: int) -> tuple:
    """ Heavy-light decomposition of the tree. """
    parents = {start: start}
    heights = {start: 0}
    heavy = {}
    size = {u: 1 for u in range(len(graph))}
    largest = {u: 0 for u in range(len(graph))}
    stk = [(start, 0)]
    while len(stk) > 0:
        n, p = stk[-1]
        # either leaf or done with children
        if len(graph[n]) == p:
            for child in graph[n]:
                size[n] += size[child]
                # heavy edge is the largest subtree out of a node's children
                if child != parents[n] and size[child] > largest[n]:
                    largest[n] = size[child]
                    heavy[n] = child
            stk.pop()
        # has children left to process
        else:
            child = graph[n][p]
              stk[-1] = (n, p + 1)
            if child not in parents:
                parents[child] = n
                heights[child] = heights[n] + 1
                stk.append((child, 0))
    return parents, heights, heavy

def hld_decomp(graph: dict, heavy: dict, start: int) -> tuple:
    """ Heavy-light decomposition of the tree. """
    cur = 0
    head, indexes = {}, {}
    array = []
    seen = set()
    seen.add(start)
    stk = [(start, start)]
    while len(stk) > 0:
        n, h = stk.pop()
        head[n] = h
        indexes[n] = cur
        array.append(n)
        cur += 1
        for child in graph[n]:
            if child not in seen and child != heavy.get(n, None):
                seen.add(child)
                stk.append((child, child))
        # traverse heavy edges first, since it's a stack it goes after
        if n in heavy:
            seen.add(heavy[n])
            stk.append((heavy[n], h))
    return head, indexes, array

def query(parents: dict, heights: dict, head: dict,
          array: dict, indexes: dict, u: int, v: int) -> float:
    ans = 0
    while head[u] != head[v]:
        u, v = (u, v) if heights[head[u]] > heights[head[v]] else (v, u)
        mx = 0
        # mx = rmq(array, indexes[head[u]], indexes[v])
        ans = ans if ans > mx else mx
        u = parents[head[u]]
    u, v = (u, v) if heights[u] < heights[v] else (v, u)
    mx = 0
    # mx = rmq(array, indexes[u], indexes[v])
    ans = ans if ans > mx else mx
    return ans
```

#### Tree to Array

[Verification](https://codeforces.com/group/M4wsRWBHyZ/contest/259141/problem/B) -
Complexity: $ \BigO(V + E) $
```python
def tree_to_dag(u, seen=set()):
    """ Turn a tree into a DAG. """
    if u in seen: return
    stk = [(u, 0)]
    seen.add(u)
    new = {i: [] for i in range(len(graph))}
    while len(stk) > 0:
        n, p = stk[-1]
        # either leaf or done with children
        if len(graph[n]) == p:
            stk.pop()
        # has children left to process
        else:
            child = graph[n][p]
            stk[-1] = (n, p + 1)
            if child not in seen:
                stk.append((child, 0))
                seen.add(child)
                new[n].append(child)
    return new

def dfs(u, seen=set()):
    """ Depth-first search on graph from start. """
    if u in seen: return
    stk = [(u, 0)]
    seen.add(u)
    indexes = {}
    used = -1
    while len(stk) > 0:
        n, p = stk[-1]
        # either leaf or done with children
        if len(graph[n]) == p:
            if p == 0:
                used += 1
                indexes[n] = (used, used)
            else:
                l, r = indexes[graph[n][0]][0], indexes[graph[n][-1]][1] + 1
                indexes[n] = (l, r)
                used = r
            stk.pop()
        # has children left to process
        else:
            child = graph[n][p]
            stk[-1] = (n, p + 1)
            if child not in seen:
                stk.append((child, 0))
                seen.add(child)
    return indexes
```

### Flow

[Edmonds-Karp](https://en.wikipedia.org/wiki/Edmonds%E2%80%93Karp_algorithm) -
[Verification: SPOJ MTOTALF](https://www.spoj.com/problems/MTOTALF/) -
Complexity: $ \BigO(V E^2) $
```python
def bfs(c, f, s, t):
    """ Breadth-first search on graph from start. """
    q = deque([(s, float("inf"))])
    paths = {s: None}
    while len(q) > 0:
        n, flow = q.popleft()
        if n == t:
            return flow, paths
        for child in c[n]:
            cf = c[n][child] - f[n][child]
            if child not in paths and cf > 0:
                paths[child] = n
                q.append((child, min(flow, cf)))

def edmonds_karp(c, s, t):
    """ Compute the flow of the graph with Edmonds-Karp. """
    f = {u: {v: 0 for v in c} for u in c}
    p = bfs(c, f, s, t)
    flow = 0
    while p is not None:
        df, path = p
        u, v = t, path[t]
        flow += df
        while v is not None:
            f[v][u] += df
            f[u][v] = -f[v][u]
            u, v = v, path[v]
        p = bfs(c, f, s, t)
    return flow, f
```

Problems:

Max Flow
- \url{http://poj.org/problem?id=1273} (USACO Drainage Ditches)
- \url{https://codeforces.com/problemset/problem/843/E}
- \url{https://www.spoj.com/problems/POTHOLE/}
- \url{https://www.spoj.com/problems/FASTFLOW/}
- \url{https://open.kattis.com/problems/maxflow}

Dynamic
- \url{https://codeforces.com/problemset/problem/903/G}

Min-cost
- \url{https://open.kattis.com/problems/mincostmaxflow}

Extensions
- Self edge: flow is 0 ($ f(u, u) = -f(u, u)$), therefore capacity is 0
- Multiple edges between $ (u, v) $: one edge with the sum of the capacities
- Undirected graph: add $ (u, v) $ and $ (v, u) $
- Unweighted graph: weight of 1
- If vertex $ v $ has capacity $ c $: make new
  vertices $ v_\text{in} $, $ v_\text{out} $.
  All edges going into $ v $ go into $ v_\text{in}
  $, going out of $ v $ goes from $ v_\text{out} $.
  Add edge $ (v_\text{in}, v_\text{out}) $ with capacity $ c $.
- Multiple sources/sinks: make supersource connected to each source with
  infinite capacity and each sink connected to supersink with infinite capacity

\url{https://en.wikipedia.org/wiki/Push–relabel_maximum_flow_algorithm}

#### Matching

Problems:

\url{https://www.spoj.com/problems/MATCHING/}

Notes:
- Reduce into max flow by assigning each edge a capacity of 1, add source
  connected to each left vertex and sink connected to each right vertex.

## Sorting

[Merge Sort](https://en.wikipedia.org/wiki/Merge_sort) -
[Verification: MBIT Hen Hackers](https://pdfhost.io/v/E8BXkbbdN_mBIT_Advancedpdf.pdf) -
Complexity: $ \BigO(n \log n) $
```python
def merge_two(l1: list, l2: list) -> list:
    """ Takes in two sorted lists and returns a sorted list. """
    rtn = []
    p1 = p2 = 0
    while p1 < len(l1) and p2 < len(l2):
        if l1[p1] < l2[p2]:
            rtn.append(l1[p1])
            p1 += 1
        else:
            rtn.append(l2[p2])
            p2 += 1
    return rtn + (l1[p1:] if p1 != len(l1) else l2[p2:])

def merge_sort(l: list) -> list:
    """ Sorts a list. """
    m = len(l)>>1
    return l if m == 0 else merge_two(merge_sort(l[:m]), merge_sort(l[m:]))
```

[Quick Sort](https://en.wikipedia.org/wiki/Quicksort) -
[Verification]() -
Complexity: expected $ \BigO(n \log n) $
```python
def median(n1: int, n2: int, n3: int) -> int:
    """ Finds the median of three numbers. """
    return sorted([n1, n2, n3])[1]

def quick_sort(l: list) -> list:
    """ Sorts a list. """
    if len(l) <= 1:
        return l
    m = median(l[0], l[len(l)>>1], l[-1])
    l1, l2 = [], []
    for n in l:
        (l1 if n < m else \
        (l2 if n > m else (l1 if len(l1) < len(l2) else l2))).append(n)
    return quick_sort(l1) + quick_sort(l2)
```

[Order statistics](https://activities.tjhsst.edu/computervision/lectures/kmeans_Handout.pdf#page=38) -
[Verification]() -
Complexity: $ \BigO(n) $
```python
def split(l: list, x: float) -> tuple:
    """ Splits the list by a particular value x. """
    left, mid, right = [], [], []
    for v in l:
        (left if v < x else (right if v > x else mid)).append(v)
    return left, mid, right

def median(l: list) -> float:
    """ Returns the upper median of l, via a sort. """
    return sorted(l)[len(l)//2]

def select(l: list, i: int) -> float:
    """ Returns sorted(l)[i] in O(n) with median of medians as a pivot. """
    if len(l) == 1: # base case
        return l[0]
    medians = [median(l[5*i: 5*(i + 1)]) for i in range(-(-len(l)//5))]
    left, mid, right = split(l, select(medians, len(medians)//2))
    k, m = len(left), len(mid)
    if k <= i <= k + m - 1: # pivot is the answer
        return mid[0]
    # recur on sublist and get rid of pivot
    return select(left, i) if i < k else select(right, i - k - m)
```

## Data Structures

### Monotonic Query

[Monotonic Queue]() -
[Verification](https://codeforces.com/group/M4wsRWBHyZ/contest/220486/problem/A) -
Complexity: $ \BigO(1) $
```python
class MinStack:
    """ O(1) append O(1) pop O(1) min query. """

    def __init__(self, f=min):
        self.stk, self.f = [], f

    def __len__(self) -> int:
        return len(self.stk)

    def __str__(self) -> str:
        return f"stack({self.stk})"

    def append(self, val):
        self.stk.append(
            (val, self.f(val, self.stk[-1][1]) if len(self.stk) > 0 else val)
        )

    def pop(self):
        return self.stk.pop()

    def min(self):
        if len(self.stk) > 0:
            return self.stk[-1][1]
        else:
            raise ValueError("min() arg is an empty sequence")

class MinQueue:
    """
    O(1) append O(1) pop O(1) min

    Implemented via two min-stacks.
    """

    def __init__(self, f=min):
        self.instk, self.outstk = MinStack(f), MinStack(f)
        self.f = f

    def __len__(self) -> int:
        return len(self.instk) + len(self.outstk)

    def __iter__(self) -> "MinQueue":
        self.i, self.l = 0, 0
        return self

    def __next__(self):
        if self.l == 2:
            raise StopIteration
        l = [self.outstk.stk, self.instk.stk][self.l]
        if self.i == len(l):
            self.l += 1
            self.i = 0
            return self.__next__()
        v = l[len(l) - 1 - self.i] if self.l == 0 else l[self.i]
        self.i += 1
        return v[0]

    def __str__(self) -> str:
        return f"queue({[x for x in self]})"

    def append(self, val) -> None:
        self.instk.append(val)

    def popleft(self):
        if len(self.outstk) == 0:
            while len(self.instk) != 0:
                self.outstk.append(self.instk.pop())
        return self.outstk.pop()

    def appendleft(self, val) -> None:
        self.outstk.append(val)

    def min(self):
        if len(self.instk) != 0 and len(self.outstk) != 0:
            return self.f(self.instk.min(), self.outstk.min())
        elif len(self.instk) == 0:
            return self.outstk.min()
        elif len(self.outstk) == 0:
            return self.instk.min()
        else:
            raise ValueError("min() arg is an empty sequence")
```

### Range Minimum Query

[Stanford lecture
slides](http://web.stanford.edu/class/cs166/lectures/00/Slides00.pdf)

[Fischer-Heun](http://web.stanford.edu/class/cs166/lectures/01/Slides01.pdf) -
[Verification: SPOJ RMQSQ](https://www.spoj.com/problems/RMQSQ/) -
Complexity: <O(n), O(1)>
```python
import math

def cartesian_number(l: list) -> int:
    """ Returns the Cartesian number for a given list. """
    stk = []
    num = 0
    for i in range(len(l)):
        while len(stk) > 0 and stk[-1] > l[i]:
            stk.pop()
            num <<= 1
        stk.append(l[i])
        num <<= 1
        num |= 1
    return num << (2*len(l) - msb(num) - 1)

def msb(n: int) -> int:
    """ Returns the index of the most significant bit of n. """
    return n.bit_length() - 1

def sparse_table(l: list) -> list:
    """ Computes a sparse table (think 2^n jump pointers) """
    n, m = len(l), math.ceil(math.log2(len(l)))
    dp = [[] for i in range(n)]

    for i in range(n):
        dp[i].append(i)

    k = 1
    for j in range(m):
        for i in range(n):
            if i + k >= n or j >= len(dp[i + k]):
                break
            # min with lambdas is REALLY slow
            # dp[i].append(min(dp[i][j], dp[i + k][j], key=lambda x: l[x]))
            dp[i].append(
                dp[i][j] if l[dp[i][j]] <= l[dp[i + k][j]] else dp[i + k][j]
            )
        k <<= 1

    return dp

def sparse_rmq(l: list, table: list, i: int, j: int) -> int:
    """ Returns the index of the minimum element between i and j. """
    k = msb(j - i + 1)
    # return min(table[i][k], table[j - (1 << k) + 1][k], key=lambda x: l[x])
    return table[i][k] if l[table[i][k]] < l[table[j - (1 << k) + 1][k]] else \
           table[j - (1 << k) + 1][k]

def full_table(l: list) -> list:
    """ Computes all possible ranges. """
    n = len(l)
    dp = [[] for i in range(n)]

    for i in range(n):
        dp[i].append(i)

    for j in range(n):
        for i in range(n - 1):
            if i + j >= n or j >= len(dp[i + 1]):
                break
            # dp[i].append(min(dp[i][j], dp[i + 1][j], key=lambda x: l[x]))
            dp[i].append(
                dp[i][j] if l[dp[i][j]] <= l[dp[i + 1][j]] else dp[i + 1][j]
            )

    return dp

def full_rmq(table: list, i: int, j: int) -> int:
    """ O(1) RMQ. """
    return table[i][j - i]

def fischer_heun(l: list) -> tuple:
    """ Constructs the structure in O(n). """
    b = max(int(math.log2(len(l))) >> 1, 1) # k = 1, not k = 1/2 (>> 2 for 1/2)
    blocks = [l[i: i + b] for i in range(0, len(l), b)]
    a = [min(block) for block in blocks]
    indexes = [min(range(i, min(i + b, len(l))), key=lambda x: l[x])
               for i in range(0, len(l), b)]
    table = sparse_table(a)
    ids = [cartesian_number(block) for block in blocks]

    tables = {}
    for i, block in enumerate(blocks):
        if ids[i] not in tables:
            tables[ids[i]] = full_table(block)

    return b, a, indexes, ids, table, tables

def rmq(o: list, b: int, a: list, indexes: list, ids: list,
        table: list, tables: list, i: int, j: int) -> int:
    """ O(1) RMQ. """
    if i > j:
        i, j = j, i
    l, r = i//b, j//b         # block indexes
    li, ri = i - l*b, j - r*b # index in the block
    # in same block
    if l == r:
        return b*l + full_rmq(tables[ids[l]], li, ri)
    i1 = b*l + full_rmq(tables[ids[l]], li, b - 1)
    i2 = indexes[sparse_rmq(a, table, l + 1, r - 1)] if r - 1 >= l + 1 else -1
    i3 = b*r + full_rmq(tables[ids[r]], 0, ri)

    v = i1 if o[i1] < o[i3] else i3
    return v if i2 == -1 or o[v] < o[i2] else i2
```

### Trees

#### Binary Indexed Trees (BITS)

[Binary Indexed Trees](https://activities.tjhsst.edu/sct/lectures/1920/2019_11_01_Binary_Index_Trees.pdf) -
[Verification]() -
Complexity: $ \BigO(n \log n) $ construction, $ \BigO(\log n) $ query
```python
class BIT:

    def __init__(self, n) -> None:
        if isinstance(n, list):
            self.l = [0]*(len(n) + 1)
            for i in range(len(n)):
                self.update(i, n[i])
        else:
            self.l = [0]*(n + 1)

    def __str__(self) -> str:
        return str(self.l)

    def query(self, i: int) -> float:
        """ sum of elements up to (and including) i """
        i += 1
        ans = 0
        while i > 0:
            ans += self.l[i]
            i -= (i & -i)
        return ans

    def range(self, i: int, j: int) -> float:
        """ sum of elements between i and j, inclusive on both ends """
        return self.query(j) - (self.query(i - 1) if i > 0 else 0)

    def update(self, i: int, v: float) -> None:
        i += 1
        while i < len(self.l):
            self.l[i] += v
            i += (i & -i)

class RBIT:

    """ BITs with range update. """

    def __init__(self, n) -> None:
        p = len(n) if isinstance(n, list) else n
        self.t1, self.t2 = BIT(p), BIT(p)

        if isinstance(n, list):
            for i in range(p):
                self.update(i, i, n[i])

    def __str__(self) -> str:
        return f"{self.t1} {self.t2}"

    def query(self, i: int) -> float:
        return i*self.t1.query(i) + self.t2.query(i)

    def range(self, i: int, j: int) -> float:
        return self.query(j) - (self.query(i - 1) if i > 0 else 0)

    def update(self, i: int, j: int, v: float) -> None:
        self.t1.update(i, v)
        self.t1.update(j + 1, -v)
        self.t2.update(i, -(i - 1)*v)
        self.t2.update(j + 1, j*v)
```


#### Segment Tree

[Segment Tree](https://activities.tjhsst.edu/sct/lectures/1920/2019_11_15_Segment_Trees.pdf) -
[Verification]() -
Complexity: $ \BigO(n) $ construction, $ \BigO(\log n) $ query
```python
class SegTree:

    def __init__(self, n) -> None:
        if isinstance(n, list):
            self.l = [0]*(len(n)<<2)
            self.build(n, 1, 0, len(n) - 1)
            self.n = len(n)
        else:
            self.l = [0]*(n<<2)
            self.n = n
        self.lazy = [0]*(self.n<<2)

    def __str__(self) -> str:
        return str(self.l)

    def build(self, a: list, p: int, l: int, r: int) -> None:
        # leaf
        if l == r:
            self.l[p] = a[l]
        else:
            pl, pr = p<<1, p<<1|1
            m = (l + r)>>1
            self.build(a, pl, l, m)
            self.build(a, pr, m + 1, r)
            self.l[p] = self.l[pl] + self.l[pr]

    def push(self, p: int, l: int, r: int):
        """ Propagate lazy values """
        pl, pr = p<<1, p<<1|1
        # update the actual value proportional
        # to the number of elements in the range
        self.l[p] += (r - l + 1)*self.lazy[p]
        # not leaf
        if l != r:
            self.lazy[pl] += self.lazy[p]
            self.lazy[pr] += self.lazy[p]
        self.lazy[p] = 0

    def query(self, i: int, j: int) -> float:
        return self.subquery(i, j, 1, 0, self.n - 1)

    def subquery(self, i: int, j: int, p: int, l: int, r: int) -> float:
        # segment outside query
        if i > r or j < l:
            return 0
        self.push(p, l, r)
        # segment inside query
        if i <= l and r <= j:
            return self.l[p]
        # partial
        pl, pr = p<<1, p<<1|1
        m = (l + r)>>1
        vl = self.subquery(i, j, pl, l, m)
        vr = self.subquery(i, j, pr, m + 1, r)
        return (vl + vr)

    def update(self, i: int, j: int, v: float) -> None:
        self.subupdate(i, j, 1, 0, self.n - 1, v)

    def subupdate(self, i: int, j: int, p: int,
                  l: int, r: int, v: float) -> None:
        if i > r or j < l:
            return
        self.push(p, l, r)
        if i <= l and r <= j:
            self.lazy[p] += v
            self.push(p, l, r)
        else:
            pl, pr = p<<1, p<<1|1
            m = (l + r)>>1
            self.subupdate(i, j, pl, l, m, v)
            self.subupdate(i, j, pr, m + 1, r, v)
            self.l[p] = self.l[pl] + self.l[pr]
```

#### AVL Tree

[AVL Tree](https://en.wikipedia.org/wiki/AVL_tree) -
[Verification]() -
Complexity: $ \BigO(n \log n) $ construction, $ \BigO(\log n) $ query
```python
class Node:

    def __init__(self, key, value=None, parent=None, left=None, right=None):
        self.key = key
        self.value = value
        self.parent = parent
        self.child = [left, right]
        self.balance = 0
        self.height = 1

    def __str__(self) -> str:
        return f"{self.key}:{self.value}:{self.balance}" \
            if self.value is not None else str(self.key)

    def extrema(self, i: int):
        n = self
        while n.child[i] is not None:
            n = n.child[i]
        return n

    def min(self): return self.extrema(0)

    def max(self): return self.extrema(1)

class BST:

    def __init__(self):
        self.root = None

    def __str__(self, n=None, s="", d=0) -> str:
        if d == 0: n = self.root
        if n is None: return s
        s = self.__str__(n.child[0], s, d + 1)
        s += " "*4*d + str(n) + "\n"
        s = self.__str__(n.child[1], s, d + 1)
        return s

    def __len__(self) -> int:
        # future: OS tree (302 in introduction to algorithms)
        return 0 if self.root is None else 1

    def min(self): return self.root.min()

    def max(self): return self.root.max()

    def add(self, key, value=None):
        if self.root is None:
            self.root = Node(key, value)
            return
        self.subadd(key, value, self.root)

        n = self.find(key, value)
        self.trace_heights(n)
        self.trace(n)

    def find(self, key, value=None):
        return self.subfind(key, value, self.root)

    def contains(self, key, value=None) -> bool:
        return self.find(key, value) is not None

    def delete(self, key, value=None):
        n = self.find(key, value)
        # leaf node
        if n.child[0] is None and n.child[1] is None:
            # root node
            if n.parent is None:
                self.root = None
                return
            self.set_children(n.parent, n.key, None)
            to_trace = n.parent
        elif n.child[0] is None:
            if n.parent is None:
                self.root = n.child[1]
                self.root.parent = None
                return
            self.set_children(n.parent, n.key, n.child[1])
            to_trace = n.child[1]
        elif n.child[1] is None:
            if n.parent is None:
                self.root = n.child[0]
                self.root.parent = None
                return
            self.set_children(n.parent, n.key, n.child[0])
            to_trace = n.child[0]
        # two children
        else:
            temp = n.child[0].max()
            n.key, n.value = temp.key, temp.value
            self.set_children(temp.parent, temp.key, temp.child[0])
            if temp.child[0] is not None:
                self.trace_heights(temp.child[0])
            self.trace_heights(temp)
            return

        self.trace_heights(to_trace)
        self.trace(to_trace)

    def update(self, key, value, new):
        self.delete(key, value)
        self.add(new, value)

    def pop(self):
        n = self.min()
        self.delete(n.key, n.value)
        return n.key, n.value

    def set_children(self, n, key, value):
        n.child[key > n.key] = value
        if value is not None:
            value.parent = n

    def subadd(self, key, value, n):
        if n.child[key > n.key] is None:
            n.child[key > n.key] = Node(key, value, n)
            return
        self.subadd(key, value, n.child[key > n.key])

    def subfind(self, key, value, n):
        if key == n.key and value == n.value:
            return n

        if n.child[key > n.key] is None:
            return None

        return self.subfind(key, value, n.child[key > n.key])

    def trace_heights(self, n):
        while n is not None:
            self.update_height(n)
            n = n.parent

    def update_height(self, n):
        left  = n.child[0].height if n.child[0] is not None else 0
        right = n.child[1].height if n.child[1] is not None else 0
        n.height = max(left, right) + 1
        n.balance = right - left

    def trace(self, n):
        while n.parent is not None:
            p = n.parent
            # right child
            if n.key > p.key:
                if p.balance == 2:
                    (self.rotate_right_left if n.balance < 0 else \
                     self.rotate_left)(p, n)
            else:
                if p.balance == -2:
                    (self.rotate_left_right if n.balance > 0 else \
                     self.rotate_right)(p, n)
            n = p

    def rotate(self, p, n, d):
        c = n.child[d]
        p.child[d ^ 1] = c

        if c is not None:
            c.parent = p
        n.child[d] = p

        n.parent = p.parent
        if p.parent is not None:
            p.parent.child[p.key > p.parent.key] = n
        else:
            self.root = n
        p.parent = n

        # order matters: update from the bottom up
        self.update_height(p)
        self.update_height(n)

        return n

    def rotate_left(self, p, n): return self.rotate(p, n, 0)

    def rotate_right(self, p, n): return self.rotate(p, n, 1)

    def rotate_left_right(self, p, n):
        return self.rotate_right(p, self.rotate_left(n, n.child[1]))

    def rotate_right_left(self, p, n):
        return self.rotate_left(p, self.rotate_right(n, n.child[0]))
```

#### Cartesian Tree

[Cartesian Tree](http://web.stanford.edu/class/cs166/lectures/01/Slides01.pdf) -
[Verification: SPOJ RMQSQ](https://www.spoj.com/problems/RMQSQ/) -
Complexity: $ \BigO(n) $
```python
class CartesianTree:

    def __init__(self, key, parent=None, left=None, right=None):
        self.key = key
        self.parent = parent
        self.child = [left, right]

    def __str__(self, n=None, s="", d=0) -> str:
        if d == 0: n = self
        if n is None: return s
        s = self.__str__(n.child[0], s, d + 1)
        s += " "*4*d + str(n.key) + "\n"
        s = self.__str__(n.child[1], s, d + 1)
        return s

def cartesian_tree(l: list) -> CartesianTree:
    """ Constructs a Cartesian tree for a given list in O(n). """
    stk = []
    for i in range(len(l)):
        c = None
        while len(stk) > 0 and stk[-1].key > l[i]:
            c = stk.pop()
        stk.append(CartesianTree(l[i], stk[-1] if len(stk) > 0 else None, c))
        # add right child
        if len(stk) > 1:
            stk[-2].child[1] = stk[-1]
    return stk[0]
```

#### kd-Tree

[CMU
lecture slides](https://www.cs.cmu.edu/~ckingsf/bioinfo-lectures/kdtrees.pdf)

[kd-Tree](https://activities.tjhsst.edu/computervision/lectures/kmeans_Handout.pdf) -
[Verification]() -
Complexity:
- $ \BigO(n \log n) $ construction
- $ \BigO(2^d + \log n) $ nearest neighbor query
```python
### median selection

def select_split(l: list, x: float) -> tuple:
    """ Splits the list by a particular value x. """
    left, mid, right = [], [], []
    for v in l:
        (left if v < x else (right if v > x else mid)).append(v)
    return left, mid, right

def median(l: list) -> float:
    """ Returns the upper median of l, via a sort. """
    return sorted(l)[len(l)//2]

def select(l: list, i: int) -> float:
    """ Returns sorted(l)[i] in O(n) with median of medians as a pivot. """
    if len(l) == 1: # base case
        return l[0]
    medians = [median(l[5*i: 5*(i + 1)]) for i in range(-(-len(l)//5))]
    left, mid, right = select_split(l, select(medians, len(medians)//2))
    k, m = len(left), len(mid)
    if k <= i <= k + m - 1: # pivot is the answer
        return mid[0]
    # recur on sublist and get rid of pivot
    return select(left, i) if i < k else select(right, i - k - m)

def split_median(points: list, cd: int) -> tuple:
    """ Picks the point which is the median along the dimension cd. """
    m = select([point[cd] for point in points], len(points)//2)
    for i in range(len(points)):
        if points[i][cd] == m: # pick any point with value m
            break
    return points[i], points[:i] + points[i + 1:]

### helper methods

def dist(p1: tuple, p2: tuple) -> float:
    """ Squared distance between two points."""
    return sum((p1[i] - p2[i])*(p1[i] - p2[i]) for i in range(len(p1)))

def closest(points: list, q: tuple) -> tuple:
    """ Returns the point closest to q in points (nearest neighbor query). """
    return min(points, key=lambda p: dist(p, q))

def distbb(p: tuple, bb: list) -> float:
    """ Squared distance between a point and a bounding box. """
    # three cases, use x if x is in the box, otherwise one of the bounds
    bbp = tuple(box[0] if x < box[0] else (box[1] if x > box[1] else x)
                for x, box in zip(p, bb))
    return dist(p, bbp)

def trimbb(bb: list, cd: int, p: int, d: int) -> list:
    """ Trims the bounding box by the plane x_cd = p[cd]. """
    if len(bb) == 0: return bb
    bb = list(list(box) for box in bb) # copy
    bb[cd][1 - d] = p[cd]              # update, assuming p[cd] is valid
    return bb

### kd-tree

def split(points: list, cd: int, p: int) -> tuple:
    """ Splits the list of points by the plane x_cd = p[cd]. """
    left, right = [], []
    for point in points:
        # add point with the same value as p at cd to the right side
        (left if point[cd] < p[cd] else right).append(point)
    return left, right

class kdNode:

    """ kd-tree node. """

    def __init__(self, point: tuple=None, cd: int=0) -> None:
        self.child, self.point, self.cd, self.bb = [None, None], point, cd, []
        self.D, self.tight_bb = len(point) if point is not None else 0, False

    def __str__(self, n: "kdNode"=None, d: int=0) -> str:
        """ Fancy string representation. """
        if d == 0: n = self # called with None by defualt, set to the root
        if n is None: return [] # leaf node
        s = [f"{' '*4*d}{n.point}"]
        s += self.__str__(n.child[0], d + 1)
        s += self.__str__(n.child[1], d + 1)
        return "\n".join(s) if d == 0 else s

    def dir(self, p: tuple) -> int:
        """ Gets the proper left/right child depending on the point. """
        return p[self.cd] >= self.point[self.cd]

    def tighten(self, t: "KdNode"=None) -> None:
        """ Tighten bounding boxes in O(nd). """
        if t is None: t = self # called with None, set to the root
        l, r, t.tight_bb = t.child[0], t.child[1], True
        # recur on children
        if l is not None: self.tighten(l)
        if r is not None: self.tighten(r)
        if l is None and r is None: # leaf node, box is just the singular point
            t.bb = [(t.point[d], t.point[d]) for d in range(t.D)]
        elif l is None or r is None: # one child, inherit box of child
            t.bb = l.bb if l is not None else r.bb
            t.bb = [(min(box[0], v), max(box[1], v))  # add node's point
                    for box, v in zip(t.bb, t.point)]
        else:                        # two children, combine boxes
            t.bb = [(min(bbl[0], bbr[0], v), max(bbl[1], bbr[1], v))
                    for bbl, bbr, v in zip(l.bb, r.bb, t.point)]

    def __add(self, t: "kdNode", p: tuple, parent: "kdNode"=None) -> None:
        """ Insert the given point into the tree. """
        if t is None:      # found leaf to insert new node in
            t = kdNode(p, (parent.cd + 1) % parent.D)
        elif t.point == p: # ignore duplicates
            return t
        else:              # update pointers
            t.child[t.dir(p)] = self.__add(t.child[t.dir(p)], p, t)
            t.tight_bb = False # no longer use tight bounding boxes
            # is it worth O(d log n) instead of O(log n) for tighter boxes?
            # if so, manually update t.bb over each of the d dimensions
        return t

    def add(self, p: tuple) -> None:
        """ Wrapper over the recursive helper function __add. """
        if self.point is None: # empty tree, simply change our own point
            self.__init__(p)
        self.__add(self, p)

    def __closest(self, t: "kdNode", p: tuple, curr_bb: list) -> tuple:
        """ Returns the closest point to p in the tree (nearest neighbor). """
        # all points in this bounding box farther than existing point
        bb = t.bb if t is not None and t.tight_bb else curr_bb
        if t is None or distbb(p, bb) > self.best_dist:
            return
        # update best point
        d = dist(p, t.point)
        if d < self.best_dist:
            self.best, self.best_dist = t.point, d
        # visit subtrees in order of distance from p
        i, j = t.dir(p), 1 - t.dir(p)
        self.__closest(t.child[i], p, trimbb(curr_bb, t.cd, t.point, i))
        self.__closest(t.child[j], p, trimbb(curr_bb, t.cd, t.point, j))

    def closest(self, p: tuple) -> tuple:
        """ Wrapper over the recursive helper function __closest. """
        self.best, self.best_dist = None, float("inf")
        bb = [[-float("inf"), float("inf")] for d in range(len(p))]
        self.__closest(self, p, [] if self.tight_bb else bb)
        return self.best

    def nnsearch(self, p: tuple, r: float) -> tuple:
        """ Finds the points that are within a radius of r from p. """
        l = []
        h = [(0, self)]
        while len(h) > 0:
            d, n = heapq.heappop(h)
            if d > r*r: # stop processing if out of circle
                return l
            if is_leaf(n):
                l.append(n.point)
            else:
                for child in n.child + [kdNode(n.point)]:
                    if child is not None:
                        d = dist(p, child.point) if is_leaf(child) else \
                            distbb(p, child.bb)
                        heapq.heappush(h, (d, child))
        return l

class kdTree(kdNode):

    """ Thin wrapper over a kd-node to build a tree from a list of points.  """

    def __init__(self, points: list=[]) -> None:
        super().__init__()
        if len(points) > 0:
            # no need for duplicate points
            self.__build_tree(self, list(set(points)))
            self.tighten()

    def __build_tree(self, t: kdNode, points: list, cd: int=0) -> kdNode:
        """ Constructs a kd-tree in O(n log n). """
        N, D, t.cd = len(points), len(points[0]), cd
        t.point, points = split_median(points, cd) # median
        t.D, next_cd = D, (cd + 1) % D
        t.child = [self.__build_tree(kdNode(), l, next_cd) if len(l) > 0
                   else None for l in split(points, cd, t.point)]
        return t
```

### Prefix Sums

[1D Prefix Sums]() -
[Verification](https://codeforces.com/group/M4wsRWBHyZ/contest/259141/problem/C) -
Complexity: $ \BigO(n) $
```python
def query(i, j):
    return prefix[j + 1] - prefix[i]

prefix = [0]*(N + 1)
for i in range(N):
    prefix[i + 1] = prefix[i] + a[i]
```

[2D Prefix Sums]() -
[Verification](https://codeforces.com/group/M4wsRWBHyZ/contest/232015/problem/D) -
Complexity: $ \BigO(nm) $
```python
def query(prefix, x1, y1, x2, y2):
    return prefix[x2 + 1][y2 + 1] - prefix[x2 + 1][y1] - prefix[x1][y2 + 1] + \
           prefix[x1][y1]

prefix = [[0]*(M + 1) for i in range(N + 1)]

for i in range(1, len(prefix)):
    for j in range(1, len(prefix[0])):
        prefix[i][j] = prefix[i - 1][j] + m[i - 1][j - 1]

for i in range(len(prefix)):
    for j in range(1, len(prefix[0])):
        prefix[i][j] = prefix[i][j - 1]
```

## Computational Geometry

Helper methods

```python
def dist(p1: tuple, p2: tuple) -> float:
    """ Squared l2 distance between the points p1 and p2. """
    (x1, y1), (x2, y2) = p1, p2
    return ((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2))

def ccw(p1: tuple, p2: tuple, p3: tuple) -> bool:
    """ Whether (p1, p2, p3) forms a counterclockwise turn.
    > 0 counterclockwise, = 0 collinear, and < 0 clockwise. """
    (x1, y1), (x2, y2), (x3, y3) = p1, p2, p3
    # this is reasonably numerically stable, no need to check > EPS
    return (x2 - x1)*(y3 - y1) - (y2 - y1)*(x3 - x1)
```

### Closest Pair
[Closest pair](https://en.wikipedia.org/wiki/Closest_pair_of_points_problem) -
[Verification: CV]() -
Complexity: $ \BigO(n \log n) $
```python
def slow_closest_pair(points: list) -> tuple:
    best = float("inf")
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if dist(points[i], points[j]) < best:
                best, p1, p2 = dist(points[i], points[j]), points[i], points[j]
    return p1, p2

def divide(points: list, x: float) -> tuple:
    return [p for p in points if p[0] <= x], [p for p in points if p[0] > x]

def recur_closest_pair(points: list, X: list, Y: list) -> tuple:
    if len(points) <= 3:
        return slow_closest_pair(points)

    mid = len(X)//2
    x = (X[mid][0] + X[mid - 1][0])/2 if len(X) % 2 == 0 else X[mid][0]
    pointsl, pointsr = divide(points, x)
    Xl, Xr = divide(X, x)
    Yl, Yr = divide(Y, x)

    l1, l2 = recur_closest_pair(pointsl, Xl, Yl)
    r1, r2 = recur_closest_pair(pointsr, Xr, Yr)

    dl, dr = dist(l1, l2), dist(*r1, *r2)
    d = min(dl, dr)

    dp = float("inf")
    Yp = [p for p in Y if x - d <= p[0] <= x + d]
    for i in range(len(Yp)):
        for j in range(i + 1, min(i + 8, len(Yp))):
            if dist(Yp[i], Yp[j]) < dp:
                dp, m1, m2 = dist(Yp[i], Yp[j]), Yp[i],  Yp[j]

    if dp < d:
        return m1, m2
    return (l1, l2) if dl < dr else (r1, r2)

def closest_pair(points: list) -> tuple:
    return recur_closest_pair(points,
                              sorted(points, key=lambda p: p[0]),
                              sorted(points, key=lambda p: p[1]))
```

### Convex Hull

[Graham scan](https://en.wikipedia.org/wiki/Graham_scan),
specifically [Andrew's Monotone Chain](https://en.wikibooks.org/wiki/Algorithm_Implementation/Geometry/Convex_hull/Monotone_chain) -
[Verification: Kattis convexhull](https://open.kattis.com/problems/convexhull) -
Complexity: $ \BigO(n \log n) $
```python
def convex_hull(points: list) -> list:
    """ Finds the convex hull of the point list. """
    points = sorted(set(points))
    if len(points) <= 1:
        return points

    final = []
    for half in range(2):
        hull = []
        for p in points:
            while len(hull) > 1 and ccw(hull[-2], hull[-1], p) <= 0:
                hull.pop()
            hull.append(p)
        final += hull[:-1]
        points = points[::-1]

    return final
```

## Ad-hoc

### Grid BFS Problems

```python
def rc(i): return i//N, i % N

def is_valid(i, j): return 0 <= i < M and 0 <= j < N

def get_children(i, j):
    return [child for child in [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]
            if is_valid(*child)]
```


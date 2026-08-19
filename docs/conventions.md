In the following we details the conventions and notations used in this package, particulary as found in the Theory page. 

## Notation

We will denote vectors/lists in boldface and use $\v{a}[n]$ to denote the $n$th element of $\v{a}$. Input indices use an input length $N$, while output indices may use an independently chosen length $M$. Where helpful, we will use the notation $(f(n))_n$ to represent a vector whose length is implicit and easily deduced from context, and the notation $f(\v{x})\equiv(f(\v{x}[n]))_n.$

We use the floor function, which takes a real number $x$ and returns $\lfloor x \rfloor$, the greatest integer that is less than or equal to $x$.

## Fourier transform conventions

We choose the ordinary-frequency convention for the continuous Fourier transform (CFT) and its inverse

$$
\begin{align}\label{eqn:FT}
    F(k) &= \hat{\mathcal{F}}_x[f(x)](k) = \left. \int_{-\infty}^\infty f(x)\exp(-i 2\pi \xi x) \d x \right|_{\xi = k}, \\
    f(x) &= \hat{\mathcal{F}}^{-1}_k[F(k)](x) = \left. \int_{-\infty}^\infty F(k)\exp(i 2\pi k \xi) \d k \right|_{\xi = x}.
\end{align}
$$

The fast Fourier transform (FFT) is a fast, mathematically equivalent way of performing the discrete Fourier transform (DFT). Please be aware that we casually use these terms interchangeably. We use the usual definition of the FFT

$$
\begin{align}
    \v{A}[m] &= \Big(\text{fft}(\v{a})\Big)[m] =  \sum_{n=0}^{N-1} \v{a}[n] \exp\left( -i 2\pi \frac{mn}{N} \right), \\
    \v{a}[m] &= \Big(\text{ifft}(\v{A})\Big)[m] = \frac{1}{N} \sum_{n=0}^{N-1} \v{A}[n] \exp\left( +i2\pi \frac{mn}{N} \right).
\end{align}
$$

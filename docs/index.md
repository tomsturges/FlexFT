This package evaluates numerical approximations to the continuous Fourier transform (CFT) on uniformly sampled data while allowing the input and output grids to be chosen independently. This is a significant difference from the Fast Fourier Transform (FFT), where the output grid is fixed by the input grid.

## Usage

Usage is straightforward. The user provides:

1. The desired spacing in real space $\delta_x$ and reciprocal space $\delta_k$, as well as the grid centers $x_0$ and $k_0$. These correspond to the uniform grids

    $$
    \begin{align}
    \v{x}[n] &= (n - c_N)\delta_x + x_0, & \v{k}[m] &= (m - c_M)\delta_k + k_0,
    \end{align}
    $$

    where $c_N = \text{floor}(N/2)$ and $c_M = \text{floor}(M/2)$ for
    independently chosen input and output lengths.

2. The samples of the function $f(x)$ at the sample points $\v{f}=f(\v{x})$.

The function `flexft` then returns a vector $\tilde{\v{F}}$ which is an approximation to CFT at the chosen sample points 

$$
\begin{align}
\text{flexft}(\delta_x, \delta_k, x_0, k_0, \v{f}) &\approx F(\v{k}).\\
F(\v{k}) \equiv \hat{\mathcal{F}}_x[f(x)](\v{k}) &= \left. \int_{-\infty}^\infty f(x)\exp(-i 2\pi \xi x) \d x \right|_{\xi = \v{k}}.
\end{align}
$$


## Implementation overview

For $N$ input and $M$ output samples, `method="direct"` provides an $O(NM)$
matrix evaluation and the default `method="bluestein"` provides an
$O((N+M)\log(N+M))$ convolution. The latter is based on the article authored by
D. H. Bailey and P. N. Swarztrauber [@BaileySwarztrauber1994]. Reusable plans
precompute the direct matrix or the FFT of the convolution kernel.

## References

\bibliography

This package evaluates numerical approximations to the continuous Fourier transform (CFT) on uniformally sampled data while allowing the input and output grids to be chosen independently. This is a significant difference from the Fast Fourier Transform (FFT), where the output grid is fixed by the input grid.

## Usage

Usage is straightforward. The user provides:

1. The desired spacing in real space $\delta_x$ and reciprocal space $\delta_k$, as well as the grid centers $x_0$ and $k_0$. These correspond to the uniform grids

    $$
    \begin{align}
    \v{x}[n] &= (n - c)\delta_x + x_0, & \v{k}[n] &= (n - c)\delta_k + k_0,
    \end{align}
    $$

    where the central index is $c = \text{floor}(N/2)$.

2. The samples of the function $f(x)$ at the sample points $\v{f}=f(\v{x})$.

The function `flexft` then returns a vector $\tilde{\v{F}}$ which is an approximation to CFT at the chosen sample points 

$$
\begin{align}
\text{flexft}(\delta_x, \delta_k, x_0, k_0, \v{f}) &\approx F(\v{k}).\\
F(\v{k}) \equiv \hat{\mathcal{F}}_x[f(x)](\v{k}) &= \left. \int_{-\infty}^\infty f(x)\exp(-i 2\pi \xi x) \d x \right|_{\xi = \v{k}}.
\end{align}
$$


## Implementation overview

The function `flexft` uses Bluestein's algorithm to rewrite the discretised approximation to the CFT as a linear convolution, which can then be evaluated as a circular convolution using FFTs. In the current implementation, this requires two forward FFTs and one inverse FFT, each acting on vectors of length $2N$, where $N$ is the number of input samples. This implementation is based on the article authored by D. H. Bailey and P. N. Swarztrauber [@BaileySwarztrauber1994]. Consequently, `flexft` has the same computational complexity as an FFT, although with a larger constant prefactor. When the FFT of the convolution kernel is precomputed, each subsequent evaluation requires only one forward FFT and one inverse FFT of length $2N$, giving an asymptotic arithmetic cost of approximately four times that of an $N$-point FFT.

## References

\bibliography


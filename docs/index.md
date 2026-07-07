This package evaluates numerical approximations to the continuous Fourier transform (CFT) on uniformally sampled data while allowing the input and output grids to be chosen independently. This is a significant difference from the Fast Fourier Transform (FFT), where the output grid is fixed by the input grid.

## Usage

Usage is straightforward. The user provides:

1. A set of uniformally spaced sample points $\v{x}$.
2. A set of samples of the function at the sample points $\v{f}=f(\v{x})$.
3. A set of desired sample points $\v{k}$ in reciprocal space where the user wants to evaulate an approximation to the CFT of $f(x)$. These must also be uniformally spaced and the same number of samples as $\v{x}$.

We briefly note that we choose the ordinary-frequency Fourier-transform convention

$$
F(k) = \hat{\mathcal{F}}_x[f(x)](k) = \left. \int_{-\infty}^\infty f(x)\exp(-i 2\pi \xi x) \d x \right|_{\xi = k}.
$$

The function `flexft` returns a vector $\tilde{\v{F}}$ which is an approximation to CFT at the chosen sample points 

$$
\text{flexft}(\v{x}, \v{k}, \v{f}) \equiv \tilde{\v{F}} \approx \v{F} \equiv F(\v{k}).
$$

## Implementation overview

The function `flexft` uses Bluestein's algorithm to rewrite the discretised approximation to the CFT as a linear convolution, which can then be evaluated as a circular convolution using FFTs. In the current implementation, this requires two forward FFTs and one inverse FFT, each acting on vectors of length $2N$, where $N$ is the number of input samples. This implementation is based on the article authored by D. H. Bailey and P. N. Swarztrauber [@BaileySwarztrauber1994]. Consequently, `flexft` has the same computational complexity as an FFT, although with a larger constant prefactor. When the FFT of the convolution kernel is precomputed, each subsequent evaluation requires only one forward FFT and one inverse FFT of length $2N$, giving an asymptotic arithmetic cost of approximately four times that of an $N$-point FFT.

## References

\bibliography


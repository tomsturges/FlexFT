See the page [Conventions](conventions.md) for details on the notations used, the Fourier conventions, and other definitions used here. See also the page [Approximation with DFT](theory_DFT.md) which covers how to approximate the continuous Fourier transform with the discrete Fourier transform. The derivations there are also very relevant to this section, and could help to make things clearer.

## The objective

Given a function $f(x)$ that is negligible outside the interval $(-L_x/2, L_x/2)$, we want to compute a numerical approximation to its exact continuous Fourier transform (CFT). Importantly, we want to be able to choose the output grid $\v{k}$ in reciprocal space independently from the input grid $\v{x}$ in direct space. As we saw in the previous [theory page](theory_DFT.md) this is not possible with the DFT, which intrinsically links the input and output grids. In addition we would like the algorithm to be reasonably fast, with the same computational complexity as the FFT.

## Approximating the CFT with the fractional DFT

Our derivation begins exactly the same as [equation 1 from the DFT theory page](theory_DFT.md#eqn-truncation-and-riemann). We define an input grid $\v{x}[n] = (n - c_N)\delta_x$ with $N$ samples and an output grid $\v{k}[m] = (m - c_M)\delta_k$ with $M$ samples, where $c_N = \lfloor N / 2 \rfloor$ and $c_M = \lfloor M / 2 \rfloor$. Truncating the integral within the CFT to $(-L_x/2, L_x/2)$ and replacing the integral with a Riemann sum allows us to approximate the CFT as

$$
\begin{equation}
\label{eqn:start}
\v{F}[m] \approx \tilde{\v{F}}[m] = \delta_x \sum_{n=0}^{N-1} \v{f}[n] \exp(-i2\pi \v{k}[m] \v{x}[n]),
\end{equation}
$$

where $\v{F} = F(\v{k})$ is the vector of samples of the *exact* CFT, and $\tilde{\v{F}}$ is the approximation. Now the difference to the DFT approach enters. To use a single DFT to approximate the CFT one *must* fix the grid spacing in reciprocal space to be the reciprocal of the input sample length, i.e. $\delta_k = 1 / L_x$. However, we want the freedom to choose the output sampling freely. Therefore we leave $\delta_k$ as a free parameter. Substituting $\v{k}$ and $\v{x}$ into equation \eqref{eqn:start} gives

$$
\begin{equation}
\label{eqn:pre_FRDFT}
\tilde{\v{F}}[m] = \delta_x \exp\left(i 2 \pi \delta(c_N m - c_M c_N) \right) \sum_n \left[\v{f}[n] \exp( i 2 \pi \delta c_M n) \right] \exp(-i 2 \pi \delta mn),
\end{equation}
$$

where we define $\delta = \delta_k \delta_x$ for convenience. To proceed further, we can note that the sum in equation \eqref{eqn:pre_FRDFT} looks somewhat like a DFT of the term in square brackets. If we were to replace $\delta$ with $1/N$ then this would be exactly a DFT. For this reason, we can think of this sum as some kind of *fractional* DFT. This terminology was introduced by D. H. Bailey and P. N. Swarztrauber [@BaileySwarztrauber1991; @BaileySwarztrauber1994] and is discussed in detail in the corresponding [theory page](theory_FRDFT.md). The fractional-DFT is defined as

$$
\begin{equation}
\label{eqn:define-frdft}
    \v{G}_\alpha[m] = \text{frdft}_\alpha(\v{g})[m] = \sum_{n=0}^{N-1} \v{g}[n] \exp(- i 2\pi \alpha m n), \qquad 0\leq m<M.
\end{equation}
$$

The fractional-DFT can be calculated efficiently as

$$
\begin{equation}
    \v{G}_\alpha[m] = \theta_{m,\alpha}^* \text{ifft} \big( \text{fft}(\v{Y}_\alpha) \cdot \text{fft}(\v{Z}_\alpha) \big)[m] \quad \text{for} \quad 0 \leq m < M,
\end{equation}
$$

where $\theta_{n,\alpha}=\exp(i \pi \alpha n^2)$ is a phase factor, $\v{Y}_\alpha$ is a zero-padded version of $\v{g}$ multiplied by the conjugated phase factor, and $\v{Z}_\alpha$ is an extended and wrapped-around version of the phase factor. Please see the [corresponding page](theory_FRDFT.md) for the full definitions. The convolution arrays have length $N+M$.

Using the definition of the fractional-DFT we can rewrite equation \eqref{eqn:pre_FRDFT} as 

!!! success "Final result"

    $$
    \begin{equation}
    \label{eqn:final}
    \tilde{\v{F}} = \delta_x \exp\big( i 2 \pi \delta (c_N \v{m} - c_M c_N) \big) \text{frdft}_\delta \Big( \v{f} \cdot \exp( i 2 \pi \delta c_M \v{n}) \Big),
    \end{equation}
    $$

where $\v{n}$ is the vectorised version of the integer indices and element-wise multiplication is implied.

## Computational cost

FlexFT provides two exact evaluations of the same flexible-grid finite sum.
`method="direct"` uses a matrix evaluation costing $O(NM)$, while the default
`method="bluestein"` uses a convolution of length $N+M$ and costs
$O((N+M)\log(N+M))$. A reusable plan precomputes either the direct matrix or
the FFT of the convolution kernel. The user chooses the method explicitly.

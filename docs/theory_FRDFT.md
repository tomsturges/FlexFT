

The fractional-DFT was introduced by D. H. Bailey and P. N. Swarztrauber [@BaileySwarztrauber1991; @BaileySwarztrauber1994]. Our motivation for discussing the fractional-DFT becomes natural when following the derivation of the discretised approximation to the continuous Fourier transform (CFT) on the corresponding [theory page](theory_FlexFT.md). Therefore, you may prefer to begin reading from that page. For an even more pedagogical introduction you could read the page on deriving an expression for the CFT in terms of a [normal DFT](theory_DFT.md).

## The fractional-DFT

We define the fractional-DFT as

$$
\begin{equation}
\label{eqn:define-frdft}
\v{G}_\alpha[m] = \text{frdft}_\alpha(\v{g})[m] = \sum_{n=0}^{N-1} \v{g}[n] \exp(- i 2\pi \alpha m n), \qquad 0 \leq m < M.
\end{equation}
$$

Here $N$ is the number of input samples and $M$ is the independently chosen
number of output samples. The implementation uses $M=N$ when no output length
is specified.

## Direct evaluation

The definition can be evaluated directly by precomputing the dense matrix

$$
\v{A}_\alpha[m,n] = \exp(-i2\pi\alpha mn)
$$

and calculating $\v{G}_\alpha=\v{A}_\alpha\v{g}$. Both the runtime and the
matrix storage scale as $O(NM)$. This is advantageous when $M$ is small, while
the FFT-based method below is preferable for larger output grids.

## Evaluation with Bluestein's algorithm

We can rewrite this expression in terms of FFTs, starting by using the Bluestein trick [@Bluestein1970] which is related to the chirp-$z$ transform. Using the identity $2mn = m^2 + n^2 - (m - n)^2$ we can factor the exponent into 

$$
\v{G}_\alpha[m] = \sum_n \v{g}[n]\exp\left(- i \pi \alpha\left(m^2 + n^2 - (m - n)^2 \right) \right),
$$

which we can rewrite as

$$
\begin{equation}
\label{eqn:thetas}
\v{G}_\alpha[m] = \theta_m^* \sum_n \v{g}[n] \theta_n^* \theta_{m-n},
\end{equation}
$$

where we defined $\theta_n = \exp(i \pi \alpha n^2)$. Please note that we have left the dependence of $\theta_n$ on $\alpha$ implicit to declutter the following derivation, but we must not forget it. 

Equation \eqref{eqn:thetas} contains a linear convolution between the finite array $\v{y} = \left( \v{g}[n] \theta_n^* \right)_{n=0}^{N-1}$ and the infinite sequence $\theta_{m-n} \forall\, n\in\mathbb{Z}$. We can rewrite this linear convolution as a circular convolution by creating a new finite array $\v{Z}$ that contains all the elements of $\theta_{m-n}$ needed for all values of $m$, and arranging them in such a way that they can be accessed by periodic array indexing. First let us look at an example in the table below for $N=M=4$. The row for $m=0$ clearly lays out the definition of the new array $\v{Z}$. It is written in such a way that as we increase $m$ (look at the row below for $m=2$) we slide the array over to the right, wrapping around the values that would have gone to out-of-bounds indices. We also see how we need to pad the array $\v{y}$ with zeros.

$$
\begin{array}{|c|c|c|c|c|c|c|c|c|}
\hline
n & 0 & 1 & 2 & 3 & 4 & 5 & 6 & 7 \\
\hline
\v{Y}[n] & \v{y}[0] & \v{y}[1] & \v{y}[2] & \v{y}[3] & 0 & 0 & 0 & 0 \\
\hline
\v{Z}[m-n]\big|_{m=0} & \theta_0 & \theta_{-1} & \theta_{-2} & \theta_{-3} & \theta_{4} & \theta_{3} & \theta_{2} & \theta_{1}  \\
\hline
\v{Z}[m-n]\big|_{m=2} & \theta_{2} & \theta_{1} & \theta_0 & \theta_{-1} & \theta_{-2} & \theta_{-3} & \theta_{-4} & \theta_{3}  \\
\hline
\end{array}
$$

Those with a keen eye may notice that one value will never actually multiply anything non-zero, so we have made the arrays one element longer than the minimum needed: a length of $N+M$ rather than $N+M-1$. This preserves the original length $2N$ construction when $M=N$; in particular, a power-of-two input then still produces a power-of-two convolution length. To make this discussion more formal, let us now explicitly define the arrays as

$$
\begin{align}
    \v{Y}[n] &= \begin{cases}
        \v{y}[n] = \v{g}[n]\theta_n^*, & 0\leq n < N \\
        0, & N \leq n < N+M
    \end{cases}
    \\
    \v{Z}[n] &= \begin{cases}
        \theta_n, & 0 \leq n < M \\
        \theta_{n-(N+M)}, & M \leq n < N+M
    \end{cases}
\end{align}
$$

which allows us to write the fractional-DFT as

$$
\begin{equation}
    \v{G}[m] = \theta_m^* \sum_{n=0}^{N+M-1} \v{Y}[n] \v{Z}[m-n] \quad \text{for} \quad 0 \leq m < M,
\end{equation}
$$

where the array indexing is modulo the array length $N+M$. This is a circular convolution between $\v{Y}$ and $\v{Z}$ which, as per the [circular convolution theorem](https://en.wikipedia.org/wiki/Discrete_Fourier_transform#Circular_convolution_theorem_and_cross-correlation_theorem), can be performed with an $(N+M)$-point FFT as

!!! success "Final result"

    $$
    \begin{equation}
        \v{G}_\alpha[m] = \theta_m^* \text{ifft} \big( \text{fft}(\v{Y}_\alpha) \cdot \text{fft}(\v{Z}_\alpha) \big)[m] \quad \text{for} \quad 0 \leq m < M,
    \end{equation}
    $$

where element-wise multiplication is implied. Notice that we have reintroduced the explicit dependence on the fractionality parameter $\alpha$. We must remember that the parameter $\theta_n$ within $\v{Y}_\alpha$ and $\v{Z}_\alpha$ also depends on $\alpha$. We emphasise that this expression is exactly the same as the original stated fractional-DFT for all $M$ requested output values. The remaining values of the circular convolution are discarded.



The fractional-DFT was introduced by D. H. Bailey and P. N. Swarztrauber [@BaileySwarztrauber1991; @BaileySwarztrauber1994]. Our motivation for discussing the fractional-DFT becomes natural when following the derivation of the discretised approximation to the continuous Fourier transform (CFT) on the corresponding [theory page](theory_FlexFT.md). Therefore, you may prefer to begin reading from that page. For an even more pedagogical introduction you could read the page on deriving an expression for the CFT in terms of a [normal DFT](theory_DFT.md).

## The fractional-DFT

We define the fractional-DFT as

$$
\begin{equation}
\label{eqn:define-frdft}
\v{G}_\alpha[m] = \text{frdft}_\alpha(\v{g})[m] = \sum_{n=0}^{N-1} \v{g}[n] \exp(- i 2\pi \alpha m n).
\end{equation}
$$

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

Equation \eqref{eqn:thetas} contains a linear convolution between the finite array $\v{y} = \left( \v{g}[n] \theta_n^* \right)_{n=0}^{N-1}$ and the infinite sequence $\theta_{m-n} \forall\, n\in\mathbb{Z}$. We can rewrite this linear convolution as a circular convolution by creating a new finite array $\v{Z}$ that contains all the elements of $\theta_{m-n}$ needed for all values of $m$, and arranging them in such a way that they can be accessed by periodic array indexing. First let us look at an example in table below for $N=4$. The row for $m=0$ clearly lays out the definition of the new array $\v{Z}$. It is written in such a way that as we increase $m$ (look at the row below for $m=2$) we slide the array over to the right, wrapping around the values that would have gone to out-of-bounds indices. We also see how we need to pad the array $\v{y}$ with zeros.

$$
\begin{array}{|c|c|c|c|c|c|c|c|c|}
\hline
n & 0 & 1 & 2 & 3 & 4 & 5 & 6 & 7 \\
\hline
\v{Y}[n] & \v{y}[0] & \v{y}[1] & \v{y}[2] & \v{y}[3] & 0 & 0 & 0 & 0 \\
\hline
\v{Z}[m-n]\big|_{m=0} & z_0 & z_{-1} & z_{-2} & z_{-3} & z_{4} & z_{3} & z_{2} & z_{1}  \\
\hline
\v{Z}[m-n]\big|_{m=2} & z_{2} & z_{1} & z_0 & z_{-1} & z_{-2} & z_{-3} & z_{-4} & z_{3}  \\
\hline
\end{array}
$$

Those with a keen eye may notice that the value $z_4$ will never actually multiply anything non-zero, so it seems that we have made the arrays one element longer than needed: a length of $2N$ rather than the possible $2N-1$. This is intentional because, if the original length $N$ is a power of 2 (the fastest length for an FFT), then the length $2N$ is also a power of 2. To make this discussion more formal, let us now define the arrays

$$
\begin{align}
    \v{Y}[n] &= \begin{cases}
        \v{y}[n] = \v{g}[n]\theta_n^*, & 0\leq n < N \\
        0, & N \leq n < 2N
    \end{cases}
    \\
    \v{Z}[n] &= \begin{cases}
        \theta_n, & 0 \leq n < N \\
        \theta_{n-2N}, & N \leq n < 2N
    \end{cases}
\end{align}
$$

which allows us to write the fractional-DFT as

$$
\begin{equation}
    \v{G}[m] = \theta_m^* \sum_{n=0}^{2N-1} \v{Y}[n] \v{Z}[m-n] \quad \text{for} \quad 0 \leq m < N,
\end{equation}
$$

where the array indexing is modulo the array length $2N$. This is a circular convolution between $\v{Y}$ and $\v{Z}$ which, as per the [circular convolution theorem](https://en.wikipedia.org/wiki/Discrete_Fourier_transform#Circular_convolution_theorem_and_cross-correlation_theorem), can be performed with a $2N$-point FFT as

!!! success "Final result"

    $$
    \begin{equation}
        \v{G}_\alpha[m] = \theta_m^* \text{ifft} \big( \text{fft}(\v{Y}_\alpha) \cdot \text{fft}(\v{Z}_\alpha) \big)[m] \quad \text{for} \quad 0 \leq m < N, 
    \end{equation}
    $$

where element-wise multiplication is implied. Notice that we have reintroduced the explicit dependence on the fractionality parameter $\alpha$. We must remember that the parameter $\theta_n$ within $\v{Y}_\alpha$ and $\v{Z}_\alpha$ also depends on $\alpha$. We emphasise that this expression is exactly the same as the original stated fractional-DFT for the first $N$ values. The rest are discarded.

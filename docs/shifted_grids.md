The derivations so far have assumed that zero lies at the central index of both the direct- and reciprocal-space grids. We can generalise the result to shifted grids using the [shift and modulation properties](https://en.wikipedia.org/wiki/Fourier_transform#Properties) of the continuous Fourier transform (CFT). Given $F(k) = \mathcal{F}_x[f(x)](k)$ these are

$$
\begin{align}
f(x - x_0) &\stackrel{\mathcal{F}}{\Longleftrightarrow} \exp(-i 2\pi k x_0) F(k),\\
\exp(i 2 \pi k_0 x) f(x) &\stackrel{\mathcal{F}}{\Longleftrightarrow} F(k-k_0).
\end{align}
$$

We will also use the following change-of-coordinate identity

$$
\mathcal{F}_{x'}[f(x')](k) = \exp(-i 2 \pi k x_0) \mathcal{F}_x [f(x + x_0)](k)\quad\text{where }x' = x + x_0.
$$

Let us start by considering a shifted input coordinate $x' = x + x_0$. We then have

$$
\begin{align}
\mathcal{F}_{x'}[f(x')](k) &=  \int_{-\infty}^\infty f(x') \text{e}^{-i 2 \pi k x'} \d{x'}\\
&= \text{e}^{-i 2 \pi k x_0} \int_{-\infty}^\infty f(x + x_0) \text{e}^{-i 2 \pi k x} \d{x} \\
&= \text{e}^{-i 2 \pi k x_0} \mathcal{F}_x[f(x + x_0)](k) 
\end{align}
$$

Next let us consider an additional shift of the output coordinate $k' = k + k_0$,

$$
\begin{align}
\mathcal{F}_{x'}[f(x')](k') &= \mathcal{F}_{x'}[f(x')](k + k_0),\\
&= \text{e}^{-i 2 \pi (k + k_0) x_0} \mathcal{F}_x[f(x + x_0)](k + k_0), \\
&= \text{e}^{-i 2 \pi (k + k_0) x_0} \mathcal{F}_x[\text{e}^{-i 2 \pi k_0 x} f(x + x_0)](k).
\end{align}
$$

To represent both the DFT-approach and the fractional-DFT approach (and perhaps other approaches), let us introduce the function

$$
\text{daft}_{\delta_x, \delta_k}(\v{f}),
$$

to represent some abstract discrete approximation to the continuous Fourier transform. This function assumes that the samples $\v{f} = f(\v{x})$ were taken on an input grid $\v{x}[n] = (n - c)\delta_x$ where $c = \lfloor N / 2 \rfloor$ (such that it is centered around the origin). It returns an approximation to $\mathcal{F}_x[f(x)](k)$ at the sample points $\v{k}[n] = (n - c)\delta_k$ (which are also centered around the origin). As such, the approximation to the CFT for shifted grids can be written as

$$
\begin{equation}
\tilde{\v{F}} = \exp\big(-i 2 \pi x_0(\v{k} + k_0)\big) \cdot \text{daft}_{\delta_x, \delta_k}\Big(\exp\big(-i 2 \pi k_0 \v{x}\big) \cdot f(\v{x} + x_0)\Big),
\end{equation}
$$

where element-wise multiplication is implied. In terms of the original coordinate system this can be written as

!!! success "Final result"

    $$
    \tilde{\v{F}} = \exp\big(-i 2 \pi x_0\v{k}'\big) \cdot \text{daft}_{\delta_x, \delta_k}\Big(\exp\big(-i 2 \pi k_0 (\v{x}' - x_0 )\big) \cdot f(\v{x}')\Big).
    $$
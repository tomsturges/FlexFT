See the page [Conventions](conventions.md) for details on the notations used, the Fourier conventions, and other definitions used here.

## The objective

Given a function $f(x)$ that is negligible outside the interval $(-L_x/2, L_x/2)$, we want to compute a numerical approximation to its exact continuous Fourier transform (CFT). 

## The DFT is not an approximation to the CFT and does not allow independent sampling grids

The [discrete Fourier transform](https://en.wikipedia.org/wiki/Discrete_Fourier_transform) (DFT) and (mathematically) equivalently the [fast Fourier transform](https://en.wikipedia.org/wiki/Fast_Fourier_transform) (FFT) are **not** approximations to the CFT. This is an important, albeit possibly pedantic point. Indeed, a suitably scaled and shifted DFT **can** be used to evaluate a finite-sum approximation to the CFT. However, the input and output sampling grids are intrinsically coupled and cannot be chosen independently. Overcoming this limitation is the motivation for the alternative approach used in this package.

## Approximating the CFT with the DFT

In the following we derive an approximation to the CFT that uses a single suitably scaled and shifted DFT. In doing so, we will explicitly see how the spacing of the output samples is fixed by the spacing of the input samples.

A CFT involves an infinite continuous integral. We assume the function $f(x)$ is negligible outside the interval $(-L_x/2, L_x/2)$. Therefore the first approximation amounts to the truncation

$$
\int_{-\infty}^\infty \rightarrow \int_{-L_x/2}^{L_x/2}.
$$

Our second approximation is to use a Riemann sum (rectangle method) to replace the integral

$$
\int_{-L_x/2}^{L_x/2} f(x) \rightarrow \delta_x \sum_{n=0}^{N-1} f(\v{x}[n]),
$$

where the centred grid $\v{x}$ is defined by its components $\v{x}[n] = (n - c)\delta_x$, with the central index $c = \lfloor N/2 \rfloor$ and the spacing $\delta_x = L_x / N$. To proceed with approximating the CFT we also need to define the output grid $\v{k}$ defined by its components $\v{k}[n] = (n - c) \delta_k$. Putting everything together we can approximate the CFT as

<a id="eqn-truncation-and-riemann"></a>

$$
\begin{align}
\v{F}[m] \equiv F(k_m) &= \hat{\mathcal{F}}_x[f(x)](k_m) = \left. \int_{-\infty}^\infty f(x)\exp(-i 2\pi \xi x) \d x \right|_{\xi = k_m},\notag\\
\label{eqn:truncation_and_riemann}
    &\approx \delta_x \sum_{n=0}^{N-1} \v{f}[n] \exp(-i2\pi \v{k}[m] \v{x}[n]) \equiv \tilde{\v{F}}[m],
\end{align}
$$

where $\v{f} = f(\v{x})$ (or equivalently $\v{f}[n]=f(\v{x}[n])$). Note that we call the vector of samples of the *exact* CFT as $\v{F}$, and the vector of samples of the approximation as $\tilde{\v{F}}$. The next step is where the intrinsic link between the input and output grids becomes apparent in the DFT approach. By fixing the output spacing $\delta_k$ to be the reciprocal of the input sample length $\delta_k \equiv 1 / L_x$ (or equivalently $\delta_k = 1 / (N \delta_x)$) then we have

$$
\begin{align}
\v{k}[m] \v{x}[n] &= (m - c)(n - c)\delta_k \delta_x, \notag\\
&= (1 / N)(m - c)(n - c).\notag
\end{align}
$$

To make the notation less cluttered let us introduce $\phi(x) = \exp(i 2 \pi x / N)$. Substituting this into equation \eqref{eqn:truncation_and_riemann} gives us

$$
\begin{align}
\tilde{\v{F}}[m] &= \delta_x \sum_{n=0}^{N-1} \v{f}[n] \phi\big(-(m-c)(n-c)\big),\notag\\
&= \delta_x \phi(-c^2)\phi(cm) \sum_n \v{f}[n] \phi(cn)\phi(-mn).\notag
\end{align}
$$

Using the definition of the DFT, $\text{fft}(\v{A})[m] = \sum_n \v{a}[n] \phi(-mn)$, we have

$$
\begin{equation}
\label{eqn:after_DFT_def}
\tilde{\v{F}}[m] = \delta_x \phi(-c^2)\phi(cm) \text{fft}(\v{g})[m],
\end{equation}
$$

where $\v{g}[n] = \v{f}[n]\phi(cn)$. Next, we use the cyclic shift property of the DFT

$$
\text{fft}\Big(S_n(\v{a})\Big)[m] = \phi(-n m) \text{fft}(\v{a})[m],
$$

where $S_n(\v{x})[m] = \v{x}[m-n]$ is the cyclic shift operator where all indices are interpreted modulo $N$. Indeed in this documentation, all indices in square brackets should be interpreted as modulo the vector length $N$. Applying the shift property to equation \eqref{eqn:after_DFT_def} gives us

$$
\begin{equation}
\label{eqn:after_shift}
\tilde{\v{F}}[m] = \delta_x \phi(-c^2) \text{fft} \Big( S_{-c}(\v{g}) \Big)[m].
\end{equation}
$$

Applying the shift property to $\v{g}$ gives $S_{-c}(\v{g})[n] = \phi(c^2)\phi(nc)S_{-c}(\v{f})[n]$, which substituted into equation \eqref{eqn:after_shift} gives

$$
\begin{equation}
\label{eqn:shifted_f}
\tilde{\v{F}}[m] = \delta_x \text{fft} \Bigg( \Big( \phi(n c) S_{-c}(\v{f})[n] \Big)_n \Bigg)[m],
\end{equation}
$$

where $(x(n))_n$ indicates a vector with components index by $n$. Applying the alternative (but equivalent) cyclic shift property

$$
\text{fft}\Big( \big( \phi(m n)\v{a}[n] \big)_n \Big) = S_m\big( \text{fft}(\v{a}) \big)
$$

to equation \eqref{eqn:shifted_f} gives

$$
\begin{equation}
\label{eqn:result_S_form}
\tilde{\v{F}} = \delta_x S_c(\text{fft}(S_{-c}(\v{f}))).
\end{equation}
$$

In many programming languages an operator (and its inverse) is defined to shift the zero-frequency component of a DFT to the centre of the vector. They are defined as

$$
\begin{align}
\text{fftshift}(\v{a}) = S_c(\v{a}),\notag\\
\text{ifftshift}(\v{a}) = S_{-c}(\v{a}),\notag
\end{align}
$$

where (just like before) $c=\lfloor N/2 \rfloor$ is the central index of the vector $\v{a}$ with length $N$. Using these common definitions we can rewrite equation \eqref{eqn:result_S_form} as

!!! success "Final result"

    \begin{equation}
    \tilde{\v{F}}
    =
    \delta_x
    \text{fftshift}(\text{fft}(\text{ifftshift}(\v{f})))
    \end{equation}

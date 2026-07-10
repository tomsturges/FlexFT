# FlexFT

[![Documentation](https://img.shields.io/badge/docs-online-blue)](https://tomsturges.github.io/FlexFT/)

FlexFT evaluates finite-sum approximations to continuous Fourier transforms
(CFTs) on uniform input and output grids whose spacings can be chosen
independently. It uses the Bailey-Swarztrauber fractional DFT/Bluestein
convolution, implemented with JAX FFTs.

For samples `f(x)` on

```text
x[n] = x0 + (n - floor(N / 2)) * dx
```

the forward transform `flexft(f, dx=dx, dk=dk, x0=x0, k0=k0)` returns
approximations on

```text
k[m] = k0 + (m - floor(N / 2)) * dk.
```

## Computational cost

`flexft` uses Bluestein's algorithm to rewrite the discretised CFT
approximation as a linear convolution, then evaluates that convolution with
FFTs. A one-shot transform currently needs two forward FFTs and one inverse FFT,
each of length `2N`, where `N` is the number of samples.

The asymptotic complexity is therefore the same as an FFT, but with a larger
constant prefactor. When the convolution kernel is precomputed by reusing a
`FlexFT` plan, each subsequent transform needs only one forward FFT and one
inverse FFT of length `2N`.

## Installation

```bash
pip install flexft
```

The quick-start plot below also uses Matplotlib.

## Quick start

```python
import jax.numpy as jnp
import matplotlib.pyplot as plt
from flexft import flexft

N = 512
dx = 0.04
dk = 0.01

x = (jnp.arange(N) - N // 2) * dx
k = (jnp.arange(N) - N // 2) * dk

# With the convention F(k) = integral f(x) exp(-i 2 pi k x) dx,
# exp(-pi x^2) is its own continuous Fourier transform.
f = jnp.exp(-jnp.pi * x**2)
F_flexft = flexft(f, dx=dx, dk=dk)
F_exact = jnp.exp(-jnp.pi * k**2)

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(k, F_exact, label="exact", linewidth=2)
ax.plot(k, F_flexft.real, "--", label="flexft", linewidth=2)
ax.set(xlabel="k", ylabel="F(k)")
ax.legend()
fig.tight_layout()
fig.savefig("docs/assets/readme-quick-start.png", dpi=200)
```

![FlexFT approximation compared with the exact Gaussian transform](https://raw.githubusercontent.com/tomsturges/FlexFT/v0.1.1/docs/assets/readme-quick-start.png)

Omit `dk` in the forward transform to use the FFT-compatible spacing
`1 / (N * dx)`. Conversely, omit `dx` in the inverse transform to use
`1 / (N * dk)`.

For repeated transforms with unchanged grids, construct and reuse `FlexFT` or
`IFlexFT`; this reuses the precomputed convolution kernel.

## Documentation

Read the [full documentation](https://tomsturges.github.io/FlexFT/) for the
derivation, shifted grids, 2D transforms, API reference, and more examples.

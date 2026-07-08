# FlexFT

[![Documentation](https://img.shields.io/badge/docs-online-blue)](https://tomsturges.github.io/FlexFT/)

FlexFT evaluates finite-sum approximations to continuous Fourier transforms on
uniform input and output grids whose spacings can be chosen independently. It
uses the Bailey--Swarztrauber fractional DFT/Bluestein convolution, implemented
with JAX FFTs.

For samples on

```text
x[n] = x0 + (n - floor(N / 2)) * dx
```

the forward transform returns approximations on

```text
k[m] = k0 + (m - floor(N / 2)) * dk.
```

## Installation

```bash
pip install flexft
```

## Quick start

```python
import jax.numpy as jnp
from flexft import flexft, iflexft

N = 256
dx = 0.05
dk = 0.02
x = (jnp.arange(N) - N // 2) * dx
f = jnp.exp(-(x**2))

F = flexft(f, dx=dx, dk=dk)
f_inverse_approximation = iflexft(F, dk=dk, dx=dx)
```

Omit `dk` in the forward transform to use the FFT-compatible spacing
`1 / (N * dx)`. Conversely, omit `dx` in the inverse transform to use
`1 / (N * dk)`.

For repeated transforms with unchanged grids, construct and reuse `FlexFT` or
`IFlexFT`; this reuses the precomputed convolution kernel.

## Documentation

Read the [full documentation](https://tomsturges.github.io/FlexFT/) for the
derivation, shifted grids, 2D transforms, API reference, and more examples.

# FlexFT

[![Documentation](https://img.shields.io/badge/docs-online-blue)](https://tomsturges.github.io/FlexFT/)

FlexFT evaluates finite-sum approximations to continuous Fourier transforms
(CFTs) on uniform input and output grids whose spacings can be chosen
independently. It uses the Bailey-Swarztrauber fractional DFT/Bluestein
convolution or direct summation, selected explicitly by the user.

For samples `f(x)` on

```text
x[n] = x0 + (n - floor(N / 2)) * dx
```

the forward transform `flexft(f, dx=dx, dk=dk, M=M, x0=x0, k0=k0)` returns
`M` approximations on

```text
k[m] = k0 + (m - floor(M / 2)) * dk.
```

## Computational cost

For `N` input samples and `M` output samples, `method="direct"` uses direct
summation with cost `O(NM)`, while the default `method="bluestein"` uses a
convolution of length `N + M` with cost `O((N + M) log(N + M))`. Reusing a plan
also reuses its direct matrix or the FFT of its Bluestein convolution kernel.

`recommend_method` can advise between these two numerically equivalent
flexible-grid methods. Its default estimate is immediate; its opt-in benchmark
mode measures synchronized JAX execution on the current platform:

```python
from flexft import FlexFT, recommend_method

dx = 0.05
dk = 0.002
recommendation = recommend_method(
    N=4096,
    M=4,
    mode="benchmark",
    batch_size=128,
    expected_calls=100,
)
transform = FlexFT(
    N=4096,
    M=4,
    dx=dx,
    dk=dk,
    method=recommendation.method,
)
```

The helper never recommends `method="fft"`, because FFT evaluation fixes the
output grid and is therefore a numerical choice rather than only a performance
choice.

## Installation

```bash
pip install flexft
```

The quick-start plot below also uses Matplotlib.

## Quick start

```python
import jax.numpy as jnp
import matplotlib.pyplot as plt
from flexft import FlexFT, flexft

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

The default `method="bluestein"` and `method="direct"` require an explicit
`dk`. For an ordinary FFT, use `method="fft"` and pass `dk=None`; the compatible
spacing `1 / (N * dx)` is then fixed by the input grid. The inverse API follows
the same rule with an explicit `dx=None`.

For repeated transforms with unchanged grids, construct and reuse `FlexFT` or
`IFlexFT`; this reuses the precomputed direct matrix or convolution kernel.

## Documentation

Read the [full documentation](https://tomsturges.github.io/FlexFT/) for the
derivation, shifted grids, 2D transforms, API reference, and more examples.

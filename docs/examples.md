# Examples

All one-dimensional functions accept or return arrays of shape `(N,)`. The grid
parameters always have the same meaning:

- `dx`: direct-space spacing
- `dk`: reciprocal-space spacing
- `x0`: direct-space grid centre
- `k0`: reciprocal-space grid centre

## One-shot forward transform

```python
import jax.numpy as jnp
from flexft import flexft

N = 256
dx = 0.05
dk = 0.02
x = (jnp.arange(N) - N // 2) * dx
f = jnp.exp(-(x**2))

F = flexft(f, dx=dx, dk=dk)
k = (jnp.arange(N) - N // 2) * dk
```

The forward transform requires an explicit `dk`. To use the FFT-compatible
grid and centered ordinary DFT path explicitly, construct an FFT plan:

```python
from flexft import FlexFT

fft_transform = FlexFT.fft(N=N, dx=dx)
F_fft = fft_transform(f)
```

## Reusing a transform plan

Constructing `FlexFT` precomputes grid phases and, for a non-FFT-compatible
spacing, the FFT of the convolution kernel. Reuse the object when transforming
many arrays on the same grids.

```python
from flexft import FlexFT

transform = FlexFT(N=N, dx=dx, dk=dk)
F1 = transform(f)
F2 = transform(2 * f)
```

## Inverse transform

The inverse transform requires both `dk` and `dx`.

```python
from flexft import iflexft

f_inverse_approximation = iflexft(F, dk=dk, dx=dx)
```

Use the class factory for the FFT-compatible spacing and ordinary inverse DFT:

```python
from flexft import IFlexFT

inverse_fft_transform = IFlexFT.fft(N=N, dk=dk)
f_fft_inverse = inverse_fft_transform(F)
```

For independently chosen spacings, forward and inverse calls are quadrature
approximations to their respective continuous transforms. They are not generally
exact matrix inverses. The FFT-compatible spacing makes the paired discrete
operations exact inverses up to floating-point roundoff.

## Shifted grids

```python
x0 = 1.5
k0 = -0.25
x = x0 + (jnp.arange(N) - N // 2) * dx
f = jnp.exp(-((x - x0) ** 2))

F = flexft(f, dx=dx, dk=dk, x0=x0, k0=k0)
k = k0 + (jnp.arange(N) - N // 2) * dk
```

## Two-dimensional transforms

```python
from flexft import flexft2d, iflexft2d

shape = (128, 96)
dx2 = (0.05, 0.08)
dk2 = (0.02, 0.03)

x1 = (jnp.arange(shape[0]) - shape[0] // 2) * dx2[0]
x2 = (jnp.arange(shape[1]) - shape[1] // 2) * dx2[1]
f2 = jnp.exp(-(x1[:, None] ** 2 + x2[None, :] ** 2))

F2 = flexft2d(f2, dx=dx2, dk=dk2)
f2_inverse_approximation = iflexft2d(F2, dk=dk2, dx=dx2)
```

Each 2D grid argument may be a pair ordered by array axis, as above, or a scalar
that is applied equally to both axes. For example, a square isotropic transform
can be written as:

```python
F2 = flexft2d(f2_square, dx=0.05, dk=0.02)
```

As in one dimension, the one-shot transform requires `dk`. Use the class
factory explicitly for an ordinary 2D FFT on the compatible grid:

```python
from flexft import FlexFT2D

fft_transform_2d = FlexFT2D.fft(N=shape, dx=dx2)
F2_fft = fft_transform_2d(f2)
```

The inverse 2D one-shot function also requires both spacings. Its ordinary-FFT
path is available through the corresponding class factory:

```python
from flexft import IFlexFT2D

inverse_fft_transform_2d = IFlexFT2D.fft(N=shape, dk=dk2)
f2_fft_inverse = inverse_fft_transform_2d(F2)
```

## Precision

FlexFT constructs its plan phases using reduced float64 arithmetic before
converting them to the precision selected by JAX. To run transforms in 64-bit
precision, enable it before constructing a plan:

```python
import jax

jax.config.update("jax_enable_x64", True)
```

# Examples

One-dimensional transforms accept arrays of shape `(N,)` and return arrays of
shape `(M,)`; `M` defaults to `N`. The grid parameters always have the same
meaning:

- `dx`: direct-space spacing
- `dk`: reciprocal-space spacing
- `x0`: direct-space grid centre
- `k0`: reciprocal-space grid centre

The default `method="bluestein"` evaluates the flexible-grid sum through FFT
convolution. `method="direct"` evaluates the same sum as a dense matrix-vector
product. `method="fft"` selects the ordinary FFT, fixes `M=N`, and determines
the output spacing from the input grid.

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

Request fewer output samples without changing `dk` by specifying `M`. The
output remains centred on `k0`:

```python
M = 9
F_region = flexft(f, dx=dx, dk=dk, M=M, k0=0.4)
k_region = 0.4 + (jnp.arange(M) - M // 2) * dk
```

The flexible methods require an explicit `dk`. To use the FFT-compatible grid
and centered ordinary DFT path, select it explicitly and omit `dk`:

```python
from flexft import FlexFT

fft_transform = FlexFT(N=N, dx=dx, method="fft")
F_fft = fft_transform(f)
```

## Reusing a transform plan

Constructing `FlexFT` precomputes grid phases and either a direct matrix or the
FFT of a convolution kernel. Reuse the object when transforming many arrays on
the same grids.

```python
from flexft import FlexFT

transform = FlexFT(N=N, M=32, dx=dx, dk=dk, method="direct")
F1 = transform(f)
F2 = transform(2 * f)
```

## Inverse transform

The flexible inverse methods require both `dk` and `dx`.

```python
from flexft import iflexft

f_inverse_approximation = iflexft(F, dk=dk, dx=dx, M=N)
```

Select the ordinary inverse DFT explicitly and omit `dx`:

```python
from flexft import IFlexFT

inverse_fft_transform = IFlexFT(N=N, dk=dk, method="fft")
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

An axis-specific output shape enables hybrid plans. Here the first axis is
reduced directly before the remaining transform is evaluated:

```python
from flexft import FlexFT2D

slice_transform = FlexFT2D(
    N=shape,
    M=(1, shape[1]),
    dx=dx2,
    dk=(dk2[0], None),
    method=("direct", "fft"),
)
F2_slice = slice_transform(f2)
print(slice_transform.axis_order)  # (0, 1)
```

Each 2D grid argument may be a pair ordered by array axis, as above, or a scalar
that is applied equally to both axes. For example, a square isotropic transform
can be written as:

```python
F2 = flexft2d(f2_square, dx=0.05, dk=0.02)
```

As in one dimension, flexible methods require `dk`. Select an ordinary 2D FFT
explicitly and omit `dk`:

```python
from flexft import FlexFT2D

fft_transform_2d = FlexFT2D(N=shape, dx=dx2, method="fft")
F2_fft = fft_transform_2d(f2)
```

The inverse 2D flexible methods require both spacings. For the ordinary-FFT
path, select it explicitly and omit `dx`:

```python
from flexft import IFlexFT2D

inverse_fft_transform_2d = IFlexFT2D(N=shape, dk=dk2, method="fft")
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

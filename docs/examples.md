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
product. Both methods evaluate the exact grid requested by the caller. The
ordinary FFT is a separate construction path because it fixes `M=N` and
derives the output spacing from the input grid.

## Recommending a flexible-grid method

Use `recommend_method` when the application should choose between direct and
Bluestein evaluation without hiding that decision inside the transform:

```python
from flexft import FlexFT, recommend_method

dx = 0.05
dk = 0.002
recommendation = recommend_method(N=4096, M=4)
print(recommendation.method)
print(recommendation.reason)

transform = FlexFT(
    N=4096,
    M=4,
    dx=dx,
    dk=dk,
    method=recommendation.method,
)
```

The default `mode="estimate"` uses approximate work and memory estimates and
does not execute a transform. For a recommendation calibrated to the active
JAX platform, explicitly request a benchmark:

```python
recommendation = recommend_method(
    N=4096,
    M=4,
    mode="benchmark",
    batch_size=128,
    expected_calls=100,
    jit=True,
)

for method, result in recommendation.results.items():
    print(method, result.execution_time, result.score)
```

Benchmark mode warms up and synchronizes each candidate before comparing
median execution times. `batch_size` should match the number of vectors the
application normally evaluates together. If `expected_calls` is supplied, the
score includes plan setup and JIT compilation as well as repeated execution;
otherwise it compares steady-state execution only. Results are cached within
the process for identical configurations, and `cache=False` forces a fresh
measurement.

Only `"direct"` and `"bluestein"` are candidates. Ordinary FFT evaluation is
constructed explicitly with `FlexFT.fft`, because it changes the allowable
output grid rather than only its implementation.

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

The flexible constructor requires an explicit `dk`. To derive the compatible
grid and use the centered ordinary FFT path, use the class factory:

```python
from flexft import FlexFT

fft_transform = FlexFT.fft(N=N, dx=dx)
F_fft = fft_transform(f)
```

`x0` and `k0` may also be supplied to the factory; only the output length and
spacing are constrained by FFT compatibility.

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

Construct an ordinary inverse FFT and its compatible output grid with the
corresponding factory:

```python
from flexft import IFlexFT

inverse_fft_transform = IFlexFT.fft(N=N, dk=fft_transform.dk)
f_fft_inverse = inverse_fft_transform(F_fft)
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

An axis-specific output shape and method pair can reduce one axis directly
before evaluating the remaining flexible transform:

```python
from flexft import FlexFT2D

slice_transform = FlexFT2D(
    N=shape,
    M=(1, shape[1]),
    dx=dx2,
    dk=dk2,
    method=("direct", "bluestein"),
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

As in one dimension, the flexible constructor requires `dk`. Construct an
ordinary 2D FFT and derive both output spacings with the class factory:

```python
from flexft import FlexFT2D

fft_transform_2d = FlexFT2D.fft(N=shape, dx=dx2)
F2_fft = fft_transform_2d(f2)
```

The inverse 2D flexible methods require both spacings. The ordinary inverse FFT
has its own factory:

```python
from flexft import IFlexFT2D

inverse_fft_transform_2d = IFlexFT2D.fft(N=shape, dk=fft_transform_2d.dk)
f2_fft_inverse = inverse_fft_transform_2d(F2_fft)
```

## Precision

FlexFT constructs its plan phases using reduced float64 arithmetic before
converting them to the precision selected by JAX. To run transforms in 64-bit
precision, enable it before constructing a plan:

```python
import jax

jax.config.update("jax_enable_x64", True)
```

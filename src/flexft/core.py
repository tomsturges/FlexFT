"""Core one- and two-dimensional FlexFT operators."""

from __future__ import annotations

import math
from numbers import Integral
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.numpy.fft import fft, fftshift, ifft, ifftshift


def _validate_positive_int(value: Any, *, name: str = "N") -> int:
    """Return a positive integer."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer, got {value!r}.")

    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive, got {result}.")
    return result


def _validate_finite_scalar(value: Any, *, name: str) -> float:
    """Return a finite scalar as a Python float."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a finite real scalar, got {value!r}.")

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a finite real scalar, got {value!r}.") from exc

    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    return result


def _validate_spacing(value: Any, *, name: str) -> float:
    """Return a finite, strictly positive grid spacing."""
    result = _validate_finite_scalar(value, name=name)
    if result <= 0:
        raise ValueError(f"{name} must be positive, got {result}.")
    return result


def _complex_dtype():
    """Use the precision selected by the user's JAX configuration."""
    return jnp.complex128 if jax.config.x64_enabled else jnp.complex64


def _unit_phase(cycles: Any):
    """Evaluate exp(i 2 pi cycles) accurately before conversion to JAX.

    Phase arrays are plan constants, so constructing them in NumPy float64 avoids
    int32 overflow and the large-argument loss of precision that otherwise occurs
    under JAX's default 32-bit configuration. Reducing modulo one also keeps the
    exponential's argument small.
    """
    cycles64 = np.asarray(cycles, dtype=np.float64)
    if not np.all(np.isfinite(cycles64)):
        raise ValueError("Phase values must be finite.")
    reduced = np.remainder(cycles64, 1.0)
    values = np.exp(2j * np.pi * reduced)
    return jnp.asarray(values, dtype=_complex_dtype())


def _as_vector(value: Any, *, length: int, name: str):
    array = jnp.asarray(value)
    if array.ndim != 1 or array.shape[0] != length:
        raise ValueError(
            f"{name} must have shape ({length},), got {tuple(array.shape)}."
        )
    return array


def _as_matrix(value: Any, *, shape: tuple[int, int], name: str):
    array = jnp.asarray(value)
    if array.ndim != 2 or tuple(array.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(array.shape)}.")
    return array


def _as_pair(value: Any, *, name: str):
    """Normalize a scalar or two-item iterable to a pair."""
    if value is None:
        raise TypeError(f"{name} must be a scalar or a pair.")

    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a scalar or a pair, got {value!r}.")

    try:
        result = tuple(value)
    except TypeError:
        return (value, value)

    if len(result) != 2:
        raise ValueError(f"{name} must be a scalar or contain exactly two values.")
    return result


_FLEXFT_METHODS = ("bluestein", "direct")


def _validate_method(value: Any, *, name: str = "method") -> str:
    """Return a supported FlexFT evaluation method."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be one of {_FLEXFT_METHODS}, got {value!r}.")
    if value == "fft":
        raise ValueError(
            f"{name}='fft' is not a flexible-grid method; use the .fft() "
            "class factory instead."
        )
    if value not in _FLEXFT_METHODS:
        raise ValueError(f"{name} must be one of {_FLEXFT_METHODS}, got {value!r}.")
    return value


def _as_method_pair(value: Any) -> tuple[Any, Any]:
    """Normalize one method name or an axis-specific pair."""
    if isinstance(value, str):
        return (value, value)
    return _as_pair(value, name="method")


class CenteredDFT:
    r"""Centered ordinary DFT operator.

    Computes

    $$
    \operatorname{fftshift}
    \left(
        \operatorname{fft}
        \left(
            \operatorname{ifftshift}(\mathbf g)
        \right)
    \right).
    $$
    """

    def __call__(self, g):
        g = jnp.asarray(g)
        if g.ndim != 1:
            raise ValueError(f"g must be one-dimensional, got shape {tuple(g.shape)}.")
        if g.shape[0] == 0:
            raise ValueError("g must contain at least one sample.")
        return fftshift(fft(ifftshift(g)))


class FRDFT:
    r"""Fractional discrete Fourier transform operator.

    This class represents

    $$
    \operatorname{frdft}_{\alpha}(\mathbf g)[m]
    =
    \sum_{n=0}^{N-1}
    \mathbf g[n]\exp(-i2\pi\alpha mn).
    $$

    The output index runs from $0$ to $M-1$, where $M=N$ by default. The
    ordinary DFT is recovered when $M=N$ and $\alpha=1/N$. The transform is
    evaluated using the Bailey--Swarztrauber/Bluestein chirp-convolution
    algorithm. Constructing an ``FRDFT`` precomputes the chirps and convolution
    kernel so the operator can be reused with different vectors of the same
    input length.

    Parameters
    ----------
    N
        Input length. Must be a positive integer.
    alpha
        Finite fractionality parameter $\alpha$.
    M
        Output length. Must be a positive integer. Defaults to ``N``.

    See Also
    --------
    frdft
        One-shot convenience function.
    """

    def __init__(self, N, alpha, M=None):
        self.N = _validate_positive_int(N)
        self.M = self.N if M is None else _validate_positive_int(M, name="M")
        self.L = self.N + self.M
        self.alpha = _validate_finite_scalar(alpha, name="alpha")
        # Float64 avoids int32 overflow in n**2 once n exceeds 46,340.
        n = np.arange(self.N, dtype=np.float64)
        m = np.arange(self.M, dtype=np.float64)
        # exp(i*pi*alpha*n**2) = exp(i*2*pi*(alpha/2)*n**2).
        theta_input = _unit_phase(0.5 * self.alpha * n**2)
        theta_output = _unit_phase(0.5 * self.alpha * m**2)
        theta_negative = _unit_phase(0.5 * self.alpha * (n - self.N) ** 2)

        self.ThetaStarInput = jnp.conjugate(theta_input)
        self.ThetaStarOutput = jnp.conjugate(theta_output)
        # Retain the original name as an alias for the input chirp.
        self.ThetaStar = self.ThetaStarInput
        self.Zfft = fft(jnp.concatenate((theta_output, theta_negative)))

    def __call__(self, g):
        """Transform a vector of length ``N`` into one of length ``M``."""
        g = _as_vector(g, length=self.N, name="g")
        Y = jnp.pad(g * self.ThetaStarInput, (0, self.M))
        conv = ifft(fft(Y) * self.Zfft)[: self.M]
        return self.ThetaStarOutput * conv


class CenteredFRDFT:
    r"""Centered fractional discrete Fourier transform operator.

    This wraps [`FRDFT`][flexft.core.FRDFT] with the phases needed to evaluate

    $$
    \sum_{n=0}^{N-1}
    \mathbf g[n]
    \exp\left[-i2\pi\alpha(m-c_M)(n-c_N)\right],
    $$

    for $0\leq m<M$, where $c_N=\lfloor N/2\rfloor$ and
    $c_M=\lfloor M/2\rfloor$. The output length defaults to the input length.
    """

    def __init__(self, N, alpha, M=None):
        self.N = _validate_positive_int(N)
        self.M = self.N if M is None else _validate_positive_int(M, name="M")
        self.alpha = _validate_finite_scalar(alpha, name="alpha")

        n = np.arange(self.N, dtype=np.float64)
        m = np.arange(self.M, dtype=np.float64)
        c_N = self.N // 2
        c_M = self.M // 2
        self.pre = _unit_phase(self.alpha * c_M * n)
        self.post = _unit_phase(self.alpha * (c_N * m - c_M * c_N))
        self.frdft = FRDFT(self.N, self.alpha, M=self.M)

    def __call__(self, g):
        """Transform a vector of length ``N`` into one of length ``M``."""
        g = _as_vector(g, length=self.N, name="g")
        return self.post * self.frdft(self.pre * g)


class CenteredDirectFRDFT:
    r"""Directly evaluated centered fractional DFT operator.

    This evaluates

    $$
    \sum_{n=0}^{N-1}
    \mathbf g[n]
    \exp\left[-i2\pi\alpha(m-c_M)(n-c_N)\right],
    $$

    for $0\leq m<M$, where $c_N=\lfloor N/2\rfloor$ and
    $c_M=\lfloor M/2\rfloor$. The centered dense kernel is precomputed so no
    additional centering phases are applied at runtime.

    Parameters
    ----------
    N
        Input length. Must be a positive integer.
    alpha
        Finite fractionality parameter $\alpha$.
    M
        Output length. Must be a positive integer. Defaults to ``N``.
    """

    def __init__(self, N, alpha, M=None):
        self.N = _validate_positive_int(N)
        self.M = self.N if M is None else _validate_positive_int(M, name="M")
        self.alpha = _validate_finite_scalar(alpha, name="alpha")

        n = np.arange(self.N, dtype=np.float64) - self.N // 2
        m = np.arange(self.M, dtype=np.float64) - self.M // 2
        self.kernel = _unit_phase(-self.alpha * np.outer(m, n))

    def __call__(self, g):
        """Transform a vector of length ``N`` into one of length ``M``."""
        g = _as_vector(g, length=self.N, name="g")
        return self.kernel @ g


def frdft(g, alpha, M=None):
    """Apply an ``N``-to-``M`` fractional DFT without constructing a plan."""
    g = jnp.asarray(g)
    if g.ndim != 1:
        raise ValueError(f"g must be one-dimensional, got shape {tuple(g.shape)}.")
    return FRDFT(g.shape[0], alpha, M=M)(g)


class FlexFT:
    r"""Approximate the continuous Fourier transform on uniform grids.

    The transform convention is

    $$
    F(k) = \int f(x)\exp(-i2\pi kx)\,\mathrm dx.
    $$

    The grids are

    $$
    \mathbf{x}[n] = x_0 + (n-c_N)\delta_x,
    \qquad
    \mathbf{k}[m] = k_0 + (m-c_M)\delta_k,
    $$

    where $c_N=\lfloor N/2\rfloor$ and $c_M=\lfloor M/2\rfloor$.

    ``method="bluestein"`` and ``method="direct"`` evaluate the same requested
    finite sum using FFT convolution and direct matrix multiplication,
    respectively. Use [`FlexFT.fft`][flexft.core.FlexFT.fft] when the output
    spacing should instead be derived from an FFT-compatible input grid.

    Parameters
    ----------
    N
        Positive input length.
    M
        Positive output length. Defaults to ``N``.
    dx
        Positive direct-space spacing $\delta_x$.
    dk
        Positive reciprocal-space spacing $\delta_k$.
    method
        Flexible-grid evaluation method: ``"bluestein"`` or ``"direct"``.
        Defaults to ``"bluestein"``.
    x0
        Direct-space grid centre. Defaults to zero.
    k0
        Reciprocal-space grid centre. Defaults to zero.

    """

    def __init__(
        self,
        *,
        N,
        dx,
        dk,
        M=None,
        method="bluestein",
        x0=0.0,
        k0=0.0,
    ):
        method = _validate_method(method)
        self._initialize(
            N=N,
            M=M,
            dx=dx,
            dk=dk,
            method=method,
            x0=x0,
            k0=k0,
        )

    @classmethod
    def fft(cls, *, N, dx, x0=0.0, k0=0.0):
        r"""Construct an ordinary centered FFT on its compatible output grid.

        The output length equals ``N`` and its spacing is
        $\delta_k=1/(N\delta_x)$. Grid centres ``x0`` and ``k0`` remain freely
        selectable and are implemented through the appropriate phase factors.
        """
        transform = cls.__new__(cls)
        transform._initialize(
            N=N,
            dx=dx,
            method="fft",
            x0=x0,
            k0=k0,
        )
        return transform

    def _initialize(
        self,
        *,
        N,
        dx,
        method,
        x0,
        k0,
        M=None,
        dk=None,
    ):
        """Initialize a validated public construction path."""
        self.N = _validate_positive_int(N)
        self.dx = _validate_spacing(dx, name="dx")
        self.method = method
        self.x0 = _validate_finite_scalar(x0, name="x0")
        self.k0 = _validate_finite_scalar(k0, name="k0")

        if method == "fft":
            self.M = self.N
            self.dk = 1.0 / (self.N * self.dx)
            self.core = CenteredDFT()
        else:
            self.M = self.N if M is None else _validate_positive_int(M, name="M")
            self.dk = _validate_spacing(dk, name="dk")
            if method == "direct":
                self.core = CenteredDirectFRDFT(self.N, self.dx * self.dk, M=self.M)
            else:
                self.core = CenteredFRDFT(self.N, self.dx * self.dk, M=self.M)

        n = np.arange(self.N, dtype=np.float64)
        m = np.arange(self.M, dtype=np.float64)
        x = (n - self.N // 2) * self.dx
        k = (m - self.M // 2) * self.dk
        self.pre = 1.0 if self.k0 == 0 else _unit_phase(-self.k0 * x)
        self.post = 1.0 if self.x0 == 0 else _unit_phase(-self.x0 * (k + self.k0))

    def __call__(self, f):
        """Transform samples ``f`` with shape ``(N,)``."""
        f = _as_vector(f, length=self.N, name="f")
        return self.dx * self.post * self.core(self.pre * f)


def flexft(f, *, dx, dk, M=None, method="bluestein", x0=0.0, k0=0.0):
    """Apply a forward FlexFT without explicitly constructing a reusable plan."""
    f = jnp.asarray(f)
    if f.ndim != 1:
        raise ValueError(f"f must be one-dimensional, got shape {tuple(f.shape)}.")
    return FlexFT(N=f.shape[0], M=M, dx=dx, dk=dk, method=method, x0=x0, k0=k0)(f)


class IFlexFT:
    r"""Approximate the inverse continuous Fourier transform on uniform grids.

    This approximates

    $$
    f(x) = \int F(k)\exp(i2\pi kx)\,\mathrm dk.
    $$

    ``dk`` always denotes reciprocal-space spacing and ``dx`` always denotes
    direct-space spacing.

    Parameters
    ----------
    N
        Positive input length.
    M
        Positive output length. Defaults to ``N``.
    dk
        Positive reciprocal-space spacing $\delta_k$.
    dx
        Positive direct-space spacing $\delta_x$.
    method
        Flexible-grid evaluation method: ``"bluestein"`` or ``"direct"``.
        Defaults to ``"bluestein"``. Use
        [`IFlexFT.fft`][flexft.core.IFlexFT.fft] for an FFT-compatible grid.
    x0
        Direct-space grid centre. Defaults to zero.
    k0
        Reciprocal-space grid centre. Defaults to zero.

    """

    def __init__(
        self,
        *,
        N,
        dk,
        dx,
        M=None,
        method="bluestein",
        x0=0.0,
        k0=0.0,
    ):
        method = _validate_method(method)
        forward_like = FlexFT(
            N=N,
            M=M,
            dx=dk,
            dk=dx,
            method=method,
            x0=k0,
            k0=x0,
        )
        self._initialize_from_forward(forward_like)

    @classmethod
    def fft(cls, *, N, dk, x0=0.0, k0=0.0):
        r"""Construct an ordinary centered inverse FFT on its compatible grid.

        The output length equals ``N`` and its spacing is
        $\delta_x=1/(N\delta_k)$. Grid centres ``x0`` and ``k0`` remain freely
        selectable.
        """
        forward_like = FlexFT.fft(
            N=N,
            dx=dk,
            x0=k0,
            k0=x0,
        )
        transform = cls.__new__(cls)
        transform._initialize_from_forward(forward_like)
        return transform

    def _initialize_from_forward(self, forward_like):
        """Initialize from the conjugated forward-transform representation."""
        self.forward_like = forward_like
        self.N = forward_like.N
        self.M = forward_like.M
        self.dk = forward_like.dx
        self.dx = forward_like.dk
        self.x0 = forward_like.k0
        self.k0 = forward_like.x0
        self.method = forward_like.method

    def __call__(self, F):
        """Inverse-transform samples ``F`` with shape ``(N,)``."""
        F = _as_vector(F, length=self.N, name="F")
        return jnp.conj(self.forward_like(jnp.conj(F)))


def iflexft(F, *, dk, dx, M=None, method="bluestein", x0=0.0, k0=0.0):
    """Apply an inverse FlexFT without constructing a reusable plan."""
    F = jnp.asarray(F)
    if F.ndim != 1:
        raise ValueError(f"F must be one-dimensional, got shape {tuple(F.shape)}.")
    return IFlexFT(N=F.shape[0], M=M, dk=dk, dx=dx, method=method, x0=x0, k0=k0)(F)


class FlexFT2D:
    """Approximate the 2D CFT on uniform tensor-product grids.

    ``dx`` contains the direct-space spacings and ``dk`` contains the
    reciprocal-space spacings. Each grid argument may be a scalar, which is
    applied to both axes, or an axis-specific pair. ``N`` and ``M`` are the
    input and output shapes. ``method`` may be one method for both axes or an
    axis-specific pair containing ``"bluestein"`` or ``"direct"``. Use
    [`FlexFT2D.fft`][flexft.core.FlexFT2D.fft] when both output axes should be
    derived from an FFT-compatible input grid. The axis with the greater output
    compression is evaluated first.
    """

    def __init__(
        self,
        *,
        N,
        dx,
        dk,
        M=None,
        method="bluestein",
        x0=0.0,
        k0=0.0,
    ):
        N1, N2 = _as_pair(N, name="N")
        M1, M2 = (N1, N2) if M is None else _as_pair(M, name="M")
        dx1, dx2 = _as_pair(dx, name="dx")
        dk1, dk2 = _as_pair(dk, name="dk")
        method1, method2 = _as_method_pair(method)
        x01, x02 = _as_pair(x0, name="x0")
        k01, k02 = _as_pair(k0, name="k0")

        N_pair = (
            _validate_positive_int(N1, name="N[0]"),
            _validate_positive_int(N2, name="N[1]"),
        )
        M_pair = (
            _validate_positive_int(M1, name="M[0]"),
            _validate_positive_int(M2, name="M[1]"),
        )
        methods = (
            _validate_method(method1, name="method[0]"),
            _validate_method(method2, name="method[1]"),
        )
        op1 = FlexFT(
            N=N_pair[0],
            M=M_pair[0],
            dx=dx1,
            dk=dk1,
            method=methods[0],
            x0=x01,
            k0=k01,
        )
        op2 = FlexFT(
            N=N_pair[1],
            M=M_pair[1],
            dx=dx2,
            dk=dk2,
            method=methods[1],
            x0=x02,
            k0=k02,
        )
        self._initialize_axis_plans(op1, op2)

    @classmethod
    def fft(cls, *, N, dx, x0=0.0, k0=0.0):
        """Construct a separable 2D FFT on its compatible output grid."""
        N1, N2 = _as_pair(N, name="N")
        dx1, dx2 = _as_pair(dx, name="dx")
        x01, x02 = _as_pair(x0, name="x0")
        k01, k02 = _as_pair(k0, name="k0")
        op1 = FlexFT.fft(N=N1, dx=dx1, x0=x01, k0=k01)
        op2 = FlexFT.fft(N=N2, dx=dx2, x0=x02, k0=k02)
        transform = cls.__new__(cls)
        transform._initialize_axis_plans(op1, op2)
        return transform

    def _initialize_axis_plans(self, op1, op2):
        """Initialize the separable operator from two internal axis plans."""
        self.op1 = op1
        self.op2 = op2
        self.N = (op1.N, op2.N)
        self.M = (op1.M, op2.M)

        self.dx = (self.op1.dx, self.op2.dx)
        self.dk = (self.op1.dk, self.op2.dk)
        self.x0 = (self.op1.x0, self.op2.x0)
        self.k0 = (self.op1.k0, self.op2.k0)
        self.method = (self.op1.method, self.op2.method)

        self._op1_vm = jax.vmap(self.op1, in_axes=1, out_axes=1)
        self._op2_vm = jax.vmap(self.op2, in_axes=0, out_axes=0)

        if self.M[0] * self.N[1] <= self.M[1] * self.N[0]:
            self.axis_order = (0, 1)
        else:
            self.axis_order = (1, 0)

    def __call__(self, f):
        f = _as_matrix(f, shape=self.N, name="f")
        if self.axis_order == (0, 1):
            return self._op2_vm(self._op1_vm(f))
        return self._op1_vm(self._op2_vm(f))


def flexft2d(f, *, dx, dk, M=None, method="bluestein", x0=0.0, k0=0.0):
    """Apply a 2D forward FlexFT without constructing a reusable plan."""
    f = jnp.asarray(f)
    if f.ndim != 2:
        raise ValueError(f"f must be two-dimensional, got shape {tuple(f.shape)}.")
    return FlexFT2D(N=f.shape, M=M, dx=dx, dk=dk, method=method, x0=x0, k0=k0)(f)


class IFlexFT2D:
    """Approximate the inverse 2D CFT on uniform tensor-product grids.

    ``dk`` contains the reciprocal-space spacings and ``dx`` contains the
    direct-space spacings. Each grid argument may be a scalar, which is applied
    to both axes, or an axis-specific pair. ``N`` and ``M`` are the input and
    output shapes. Use [`IFlexFT2D.fft`][flexft.core.IFlexFT2D.fft] when both
    output axes should be derived from an FFT-compatible input grid.
    """

    def __init__(
        self,
        *,
        N,
        dk,
        dx,
        M=None,
        method="bluestein",
        x0=0.0,
        k0=0.0,
    ):
        N_pair = _as_pair(N, name="N")
        M_pair = N_pair if M is None else _as_pair(M, name="M")
        dk_pair = _as_pair(dk, name="dk")
        dx_pair = _as_pair(dx, name="dx")
        method_pair = _as_method_pair(method)
        x0_pair = _as_pair(x0, name="x0")
        k0_pair = _as_pair(k0, name="k0")

        forward_like = FlexFT2D(
            N=N_pair,
            M=M_pair,
            dx=dk_pair,
            dk=dx_pair,
            method=method_pair,
            x0=k0_pair,
            k0=x0_pair,
        )
        self._initialize_from_forward(forward_like)

    @classmethod
    def fft(cls, *, N, dk, x0=0.0, k0=0.0):
        """Construct a separable 2D inverse FFT on its compatible grid."""
        forward_like = FlexFT2D.fft(
            N=N,
            dx=dk,
            x0=k0,
            k0=x0,
        )
        transform = cls.__new__(cls)
        transform._initialize_from_forward(forward_like)
        return transform

    def _initialize_from_forward(self, forward_like):
        """Initialize from the conjugated forward-transform representation."""
        self.forward_like = forward_like
        self.N = forward_like.N
        self.M = forward_like.M
        self.dk = forward_like.dx
        self.dx = forward_like.dk
        self.x0 = forward_like.k0
        self.k0 = forward_like.x0
        self.method = forward_like.method
        self.axis_order = forward_like.axis_order

    def __call__(self, F):
        F = _as_matrix(F, shape=self.N, name="F")
        return jnp.conj(self.forward_like(jnp.conj(F)))


def iflexft2d(F, *, dk, dx, M=None, method="bluestein", x0=0.0, k0=0.0):
    """Apply a 2D inverse FlexFT without constructing a reusable plan."""
    F = jnp.asarray(F)
    if F.ndim != 2:
        raise ValueError(f"F must be two-dimensional, got shape {tuple(F.shape)}.")
    return IFlexFT2D(N=F.shape, M=M, dk=dk, dx=dx, method=method, x0=x0, k0=k0)(F)

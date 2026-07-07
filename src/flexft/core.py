import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
from jaxtyping import Float, Int, Array, Complex, Num
from jax.numpy.fft import fft, ifft
Float2D = Float[Array, "x1 x2"]
Complex1D = Complex[Array, "x1"]
Complex2D = Complex[Array, "x1 x2"]
Real1D = Float[Array, "x1"] | Int[Array, "x1"]
pi = jnp.pi
exp = jnp.exp
from typing import NamedTuple
from jax.numpy.fft import fft, fftshift, ifftshift

class CenteredDFT:
    r"""
    Centered ordinary DFT operator.

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
        return fftshift(fft(ifftshift(g)))

class FRDFT:
    r"""
    Fractional discrete Fourier transform operator. This class represents the fractional DFT

    $$
    \operatorname{frdft}_{\alpha}(\mathbf g)[m]
    =
    \sum_{n=0}^{N-1}
    \mathbf g[n]\exp(-i2\pi\alpha mn),
    $$

    where $N$ is the length of the vector $\v{g}$. The ordinary DFT is recovered when

    $$
    \alpha = \frac{1}{N}.
    $$

    The transform is evaluated using the Bailey--Swarztrauber /
    Bluestein chirp-convolution algorithm. This rewrites the fractional
    DFT as a linear convolution, which is then evaluated using FFTs. The
    resulting algorithm has the same asymptotic complexity as an FFT,
    although with a larger constant prefactor.

    Creating an ``FRDFT`` object precomputes the chirp factors and the FFT
    of the convolution kernel. The resulting object can then be reused to
    transform many different input vectors with the same ``N`` and
    ``alpha``.

    Parameters
    ----------
    N
        Length of the input and output vectors.
    alpha
        Fractionality parameter $\alpha$.

    See Also
    --------
    frdft
        One-shot convenience function.
    """
    def __init__(self, N, alpha):
        self.N = int(N)
        self.alpha = alpha
        n = jnp.arange(self.N)
        theta = lambda n: jnp.exp(1j * jnp.pi * alpha * n**2)
        self.ThetaStar = jnp.conjugate(theta(n))
        Z = jnp.concatenate((theta(n), theta(n - self.N)))
        self.Zfft = fft(Z)

    def __call__(self, g):
        r"""
        Apply the fractional DFT to an input vector.

        Parameters
        ----------
        g
            Input vector of length ``N``.

        Returns
        -------
        Array
            The fractional DFT of ``g``.
        """
        if g.shape[0] != self.N:
            raise ValueError(f"Expected input of length {self.N}, got {g.shape[0]}.")
        Y = jnp.pad(g * self.ThetaStar, (0, self.N))
        conv = ifft(fft(Y) * self.Zfft)[:self.N]
        return self.ThetaStar * conv

class CenteredFRDFT:
    r"""
    Centered fractional discrete Fourier transform operator.

    This wraps [`FRDFT`][flexft.core.FRDFT] with the phase factors needed to
    evaluate

    $$
    \sum_{n=0}^{N-1}
    \mathbf g[n]
    \exp\left[-i2\pi\alpha(m-c)(n-c)\right],
    $$

    where \(c=\lfloor N/2\rfloor\).
    """
    
    def __init__(self, N, alpha):
        self.N = N
        n = jnp.arange(N)
        c = N // 2
        pre_phase = 1j * 2 * jnp.pi * alpha * c * n
        self.pre = jnp.exp(pre_phase)
        post_phase = 1j * 2 * jnp.pi * alpha * (c * n - c**2)
        self.post = jnp.exp(post_phase)
        self.frdft = FRDFT(N, alpha)

    def __call__(self, g):
        r"""
        Apply the *centered* fractional DFT to an input vector.

        Parameters
        ----------
        g
            Input vector of length ``N``.

        Returns
        -------
        Array
            The *centered* fractional DFT of ``g``.
        """
        if g.shape[0] != self.N:
            raise ValueError(f"Expected input of length {self.N}, got {g.shape[0]}.")
        return self.post * self.frdft(self.pre * g)

def frdft(g, alpha):
    r"""
    One-shot wrapper around FRDFT.

     Parameters
        ----------
        g
            Input vector of length ``N``.
        alpha
            Fractionality parameter $\alpha$.

        Returns
        -------
        Array
            The fractional DFT of ``g``.
    """
    return FRDFT(g.shape[-1], alpha)(g)

class FlexFT:
    r"""
    Approximates the continuous Fourier transform
    
    $$
    F(k) = \int f(x)\exp(-i2\pi k x)\,\mathrm dx.
    $$
    
    on the uniform grids

    $$
    \v{x}[n] = x_0 + (n-c) \delta_x,
    \qquad
    \v{k}[m] = k_0 + (m-c) \delta_k,
    $$

    where \(c=\lfloor N/2\rfloor\).

    If ``dk`` is not provided, the FFT-compatible spacing

    $$
    \delta_k = \frac{1}{N\delta_x}
    $$

    is used and the core transform is a centered ordinary DFT. If ``dk``
    is provided, the core transform is a centered fractional DFT.

    Parameters
    ----------
    N
        Length of the input and output vectors.
    dx
        Direct-space grid spacing $\delta_x$. 
    dk
        Reciprocal-space grid spacing $\delta_k$. If not provided,
        the FFT-compatible spacing ``1 / (N * dx)`` is used.
    x0
        Centre of the direct-space grid. If not provided, zero is used.
    k0
        Centre of the reciprocal-space grid. If not provided, zero is used.
    """

    def __init__(self, *, N, dx, dk=None, x0=None, k0=None):
        self.dx = dx
        self.N = int(N)
        
        if dk is None:
            self.dk = 1 / (self.N * self.dx)
            self.core = CenteredDFT
        else:
            self.dk = dk
            self.core = CenteredFRDFT(self.N, self.dx * self.dk)

        c = N // 2
        n = jnp.arange(self.N)
        x = (n - c) * self.dx
        k = (n - c) * self.dk

        if k0 is None:
            self.k0 = 0
            self.pre = 1
        else:
            self.k0 = k0
            pre_phase = -1j * 2 * jnp.pi * self.k0 * x
            self.pre = jnp.exp(pre_phase)
        if x0 is None:
            self.x0 = 0
            self.post = 1
        else:
            self.x0=x0
            post_phase = -1j * 2 * jnp.pi * (k + self.k0) * x0
            self.post = jnp.exp(post_phase)

    def __call__(self, f):
        if f.shape[0] != self.N:
            raise ValueError(f"Expected input of length {self.N}, got {f.shape[0]}.")

        return self.dx * self.post * self.core(self.pre * f)

def flexft(f, *, dx, dk=None, x0=None, k0=None):
    r"""
    One-shot wrapper around [`FlexFT`][flexft.core.FlexFT].
    """
    return IFlexFT(
        N=f.shape[0],
        dx=dx,
        dk=dk,
        x0=x0,
        k0=k0,
    )(f)

class IFlexFT:
    r"""
    Approximate the inverse continuous Fourier transform on uniform grids.

    This approximates

    $$
    f(x) = \int F(k)\exp(i2\pi kx)\,\mathrm dk.
    $$

    The original direct- and reciprocal-space grids are assumed to be

    $$
    \v{x}[n] = x_0 + (n-c)\delta_x,
    \qquad
    \v{k}[m] = k_0 + (m-c)\delta_k.
    $$

    Parameters
    ----------
    N
        Length of the input and output vectors.
    dx
        Direct-space grid spacing $\delta_x$. If not provided,
        the FFT-compatible spacing ``1 / (N * dk)`` is used.
    dk
        Reciprocal-space grid spacing $\delta_k$. 
    x0
        Centre of the direct-space grid. If not provided, zero is used.
    k0
        Centre of the reciprocal-space grid. If not provided, zero is used.
    """

    def __init__(self, *, N, dk, dx=None, x0=None, k0=None):
        self.N = int(N)
        self.dk = dk
        self.x0 = x0
        self.k0 = k0

        self.forward_like = FlexFT(
            N=self.N,
            dx=dk,
            dk=dx,
            x0=k0,
            k0=x0,
        )

        self.dx = self.forward_like.dk

    def __call__(self, F):
        if F.shape[0] != self.N:
            raise ValueError(f"Expected input of length {self.N}, got {F.shape[0]}.")

        return jnp.conj(self.forward_like(jnp.conj(F)))

def iflexft(F, *, dk, dx=None, x0=None, k0=None):
    r"""
    One-shot wrapper around [`IFlexFT`][flexft.core.IFlexFT].
    """
    return IFlexFT(
        N=F.shape[0],
        dk=dk,
        dx=dx,
        x0=x0,
        k0=k0,
    )(F)

def _pair(value, *, name):
    if value is None:
        return (None, None)
    if len(value) != 2:
        raise ValueError(f"{name} must be a pair.")
    return value


class FlexFT2D:
    r"""
    Approximate the 2D continuous Fourier transform on uniform tensor-product grids.

    The transform is applied separably: first along axis 0, then along axis 1.
    """

    def __init__(self, *, N, dx, dk=None, x0=None, k0=None):
        N1, N2 = N
        dx1, dx2 = dx
        dk1, dk2 = _pair(dk, name="dk")
        x01, x02 = _pair(x0, name="x0")
        k01, k02 = _pair(k0, name="k0")

        self.N = (int(N1), int(N2))

        self.op1 = FlexFT(N=N1, dx=dx1, dk=dk1, x0=x01, k0=k01)
        self.op2 = FlexFT(N=N2, dx=dx2, dk=dk2, x0=x02, k0=k02)

        self._op1_vm = jax.vmap(self.op1, in_axes=1, out_axes=1)
        self._op2_vm = jax.vmap(self.op2, in_axes=0, out_axes=0)

    def __call__(self, f):
        if f.shape[:2] != self.N:
            raise ValueError(f"Expected input shape {self.N}, got {f.shape[:2]}.")

        return self._op2_vm(self._op1_vm(f))

def flexft2d(f, *, dx, dk=None, x0=None, k0=None):
    r"""
    One-shot wrapper around [`FlexFT2D`][flexft.core.FlexFT2D].
    """
    return FlexFT2D(N=f.shape[:2], dx=dx, dk=dk, x0=x0, k0=k0)(f)

class IFlexFT2D:
    r"""
    Approximate the inverse 2D continuous Fourier transform on uniform grids.
    """

    def __init__(self, *, N, dx, dk=None, x0=None, k0=None):
        N1, N2 = N
        dx1, dx2 = dx

        if dk is None:
            dk1 = 1 / (int(N1) * dx1)
            dk2 = 1 / (int(N2) * dx2)
        else:
            dk1, dk2 = dk

        x01, x02 = _pair(x0, name="x0")
        k01, k02 = _pair(k0, name="k0")

        self.N = (int(N1), int(N2))

        self.forward_like = FlexFT2D(
            N=self.N,
            dx=(dk1, dk2),
            dk=(dx1, dx2),
            x0=(k01, k02),
            k0=(x01, x02),
        )

    def __call__(self, F):
        if F.shape[:2] != self.N:
            raise ValueError(f"Expected input shape {self.N}, got {F.shape[:2]}.")

        return jnp.conj(self.forward_like(jnp.conj(F)))

def iflexft2d(F, *, dx, dk=None, x0=None, k0=None):
    r"""
    One-shot wrapper around [`IFlexFT2D`][flexft.core.IFlexFT2D].
    """
    return IFlexFT2D(N=F.shape[:2], dx=dx, dk=dk, x0=x0, k0=k0)(F)
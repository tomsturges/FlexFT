"""Show why an ordinary FFT cannot use an independently chosen output grid.

The Fourier-transform convention is

    F(k) = integral f(x) exp(-i 2 pi k x) dx.

For the Gaussian used here,

    f(x) = exp(-pi (x / width)^2),
    F(k) = width exp(-pi (width k)^2).

Run this file to write two PNG figures to ``figures/``.
"""

from pathlib import Path

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# Sampling parameters ---------------------------------------------------------
N = 128
DX = 0.08
WIDTH = 1.0

# Deliberately incompatible with the ordinary FFT requirement
# DK_FFT = 1 / (N * DX).
DK_REQUESTED = 0.04

OUTPUT_DIR = Path(__file__).with_name("figures")


def gaussian(x: np.ndarray) -> np.ndarray:
    """Return f(x) = exp(-pi (x / WIDTH)^2)."""
    return np.exp(-np.pi * (x / WIDTH) ** 2)


def gaussian_transform(k: np.ndarray) -> np.ndarray:
    """Return the analytical continuous Fourier transform of ``gaussian``."""
    return WIDTH * np.exp(-np.pi * (WIDTH * k) ** 2)


def direct_sum(
    samples: np.ndarray, x: np.ndarray, k: np.ndarray
) -> np.ndarray:
    """Evaluate the Riemann-sum CFT approximation on any requested k-grid."""
    kernel = np.exp(-2j * np.pi * np.outer(k, x))
    return DX * kernel @ samples


def centered_fft(samples: np.ndarray) -> np.ndarray:
    """Evaluate the scaled, centered ordinary FFT."""
    return DX * np.fft.fftshift(np.fft.fft(np.fft.ifftshift(samples)))


def style_axes(ax: plt.Axes) -> None:
    ax.set_xlabel(r"spatial frequency $k$")
    ax.set_ylabel(r"$|F(k)|$")
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-0.025, 1.075)
    ax.legend()


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUTPUT_DIR / f"{stem}.png",
        dpi=200,
        facecolor="white",
        transparent=False,
        bbox_inches="tight",
    )
    plt.close(fig)


def new_figure() -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
    ax.set_facecolor("white")
    return fig, ax


def main() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "legend.fontsize": 11,
        }
    )

    x = (np.arange(N) - N // 2) * DX
    f = gaussian(x)

    k_requested = (np.arange(N) - N // 2) * DK_REQUESTED
    direct_requested = direct_sum(f, x, k_requested)

    fft_values = centered_fft(f)
    dk_fft = 1.0 / (N * DX)

    k_curve = np.linspace(-1.25, 1.25, 1000)
    exact_curve = gaussian_transform(k_curve)
    exact_requested = gaussian_transform(k_requested)

    # 1. A brute-force sum can evaluate the independently requested grid.
    fig, ax = new_figure()
    ax.plot(k_curve, exact_curve, label="Analytical")
    ax.plot(
        k_requested,
        np.abs(direct_requested),
        linestyle="none",
        marker="o",
        markersize=4,
        label="Direct DFT",
    )
    ax.set_title("Direct DFT on requested grid")
    style_axes(ax)
    save_figure(fig, "01_direct_sum_matches_analytic")

    # 2. The common mistake: put the FFT values on the requested coordinates.
    #    The FFT did not compute values at these k positions, so this comparison
    #    is intentionally wrong.
    fig, ax = new_figure()
    ax.plot(k_curve, exact_curve, label="Analytical")
    ax.plot(
        k_requested,
        np.abs(fft_values),
        linestyle="none",
        marker="o",
        markersize=4,
        label="FFT",
    )
    ax.set_title("FFT values assigned to requested grid")
    style_axes(ax)
    save_figure(fig, "02_naive_fft_on_wrong_grid")

    direct_error = np.max(np.abs(direct_requested - exact_requested))
    naive_error = np.max(np.abs(fft_values - exact_requested))
    print(f"N = {N}, dx = {DX}, L = {N * DX:.2f}")
    print(f"requested dk = {DK_REQUESTED:.5f}")
    print(f"FFT-compatible dk = {dk_fft:.5f}")
    print(f"direct sum max abs error = {direct_error:.3e}")
    print(f"naive mislabeled FFT max abs error = {naive_error:.3e}")
    print(f"wrote figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

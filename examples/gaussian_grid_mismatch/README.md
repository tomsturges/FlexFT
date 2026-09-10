# Gaussian grid mismatch

This example uses the transform convention

\[
F(k) = \int_{-\infty}^{\infty} f(x)\exp(-i2\pi kx)\,\mathrm{d}x
\]

and the analytical pair

\[
f(x) = \exp\!\left[-\pi(x/w)^2\right], \qquad
F(k) = w\exp\!\left[-\pi(wk)^2\right].
\]

The requested reciprocal-space spacing is deliberately different from the
ordinary FFT spacing:

\[
\Delta k_{\mathrm{requested}} = 0.04, \qquad
\Delta k_{\mathrm{FFT}} = \frac{1}{N\Delta x} = 0.09765625.
\]

Run the example from the repository root:

```console
uv run --extra dev python examples/gaussian_grid_mismatch/gaussian_grid_mismatch.py
```

The script writes two PNG figures on white backgrounds to `figures/`:

1. The direct Riemann sum agrees with the analytical transform on the freely
   chosen output grid.
2. The ordinary FFT appears incorrect when its values are naively assigned to
   that requested grid.

The second figure intentionally demonstrates a coordinate error. The FFT
values themselves are valid, but their physical coordinates are spaced by
`1 / (N * dx)`, not by the requested `dk`.

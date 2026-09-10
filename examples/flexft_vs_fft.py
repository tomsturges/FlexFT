# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # FlexFT, FFT, and analytical Fourier transforms
#
# This interactive example compares the FFT with FlexFT on independently
# chosen output grids. It supports forward and inverse transforms for several
# functions with known analytical transforms.

# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
from IPython.display import display

from flexft import FlexFT, IFlexFT


jax.config.update("jax_enable_x64", True)


# %% [markdown]
# ## Functions with analytical transform pairs
#
# The transform convention is
#
# \[
# F(k) = \int_{-\infty}^{\infty} f(x)\exp(-i2\pi kx)\,\mathrm{d}x.
# \]

# %%
def f_gaussian(x):
    return jnp.exp(-(x**2))


def F_gaussian(k):
    return np.sqrt(np.pi) * np.exp(-((np.pi * k) ** 2))


def Finv_gaussian(x):
    return np.sqrt(np.pi) * np.exp(-((np.pi * x) ** 2))


def f_step(x):
    return jnp.where(jnp.abs(x) <= 0.5, 1.0, 0.0)


def F_step(k):
    return np.sinc(k)


def Finv_step(x):
    return np.sinc(x)


def f_triangle(x):
    return jnp.maximum(1 - jnp.abs(x), 0.0)


def F_triangle(k):
    return np.sinc(k) ** 2


def Finv_triangle(x):
    return np.sinc(x) ** 2


def f_sech(x):
    return 1 / jnp.cosh(x)


def F_sech(k):
    return np.pi / np.cosh(np.pi**2 * k)


def Finv_sech(x):
    # sech is real and even, so its inverse transform equals its forward one.
    return np.pi / np.cosh(np.pi**2 * x)


FUNCTION_TRIPLES = {
    "Gaussian": (f_gaussian, F_gaussian, Finv_gaussian),
    "Step": (f_step, F_step, Finv_step),
    "Triangle": (f_triangle, F_triangle, Finv_triangle),
    "Sech": (f_sech, F_sech, Finv_sech),
}


# %% [markdown]
# ## Comparison figure

# %%
def comparison_figure(
    function_name,
    direction,
    N,
    input_width,
    output_half_width,
    plot_half_width,
):
    sampled_function, exact_forward, exact_inverse = FUNCTION_TRIPLES[
        function_name
    ]
    forward = direction == "Forward"

    input_spacing = input_width / N
    input_grid = (np.arange(N) - N // 2) * input_spacing
    input_samples = sampled_function(input_grid)

    # FlexFT permits this spacing to be chosen independently of input_spacing.
    output_spacing = 2 * output_half_width / N

    if forward:
        flexible_plan = FlexFT(
            N=N,
            dx=input_spacing,
            dk=output_spacing,
            method="bluestein",
        )
        fft_plan = FlexFT.fft(N=N, dx=input_spacing)

        flexible_result = flexible_plan(input_samples)
        fft_result = fft_plan(input_samples)
        flexible_grid = flexible_plan.k
        fft_grid = fft_plan.k
        exact_transform = exact_forward

        input_xlabel = r"$x$"
        input_ylabel = r"$|f(x)|$"
        output_xlabel = r"$k$"
        output_ylabel = r"$|F(k)|$"
        spacing_symbol = r"\Delta k"
        fft_output_spacing = fft_plan.dk
        fft_label = "FFT"
        flexible_label = "FlexFT"
    else:
        flexible_plan = IFlexFT(
            N=N,
            dk=input_spacing,
            dx=output_spacing,
            method="bluestein",
        )
        fft_plan = IFlexFT.fft(N=N, dk=input_spacing)

        flexible_result = flexible_plan(input_samples)
        fft_result = fft_plan(input_samples)
        flexible_grid = flexible_plan.x
        fft_grid = fft_plan.x
        exact_transform = exact_inverse

        input_xlabel = r"$k$"
        input_ylabel = r"$|F(k)|$"
        output_xlabel = r"$x$"
        output_ylabel = r"$|f(x)|$"
        spacing_symbol = r"\Delta x"
        fft_output_spacing = fft_plan.dx
        fft_label = "IFFT"
        flexible_label = "IFlexFT"

    exact_grid = np.linspace(-plot_half_width, plot_half_width, 1000)
    exact_values = np.abs(exact_transform(exact_grid))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].plot(input_grid, np.abs(input_samples))
    axes[0].set_xlabel(input_xlabel)
    axes[0].set_ylabel(input_ylabel)
    axes[0].set_title(f"{function_name}: {direction}")

    axes[1].plot(exact_grid, exact_values, color="black", label="Analytical")
    axes[1].plot(
        np.asarray(fft_grid),
        np.abs(np.asarray(fft_result)),
        "rx",
        markersize=4,
        label=fft_label,
    )
    axes[1].set_title(
        rf"{fft_label}: ${spacing_symbol}={fft_output_spacing:.3g}$"
    )

    axes[2].plot(exact_grid, exact_values, color="black", label="Analytical")
    axes[2].plot(
        np.asarray(flexible_grid),
        np.abs(np.asarray(flexible_result)),
        "bx",
        markersize=4,
        label=flexible_label,
    )
    axes[2].set_title(
        rf"{flexible_label}: ${spacing_symbol}={output_spacing:.3g}$"
    )

    for axis in axes[1:]:
        axis.set_xlabel(output_xlabel)
        axis.set_ylabel(output_ylabel)
        axis.set_xlim(-plot_half_width, plot_half_width)
        axis.legend()

    fig.tight_layout()
    return fig


# %% [markdown]
# ## Interactive controls

# %%
widget_style = {"description_width": "initial"}

w_function = widgets.ToggleButtons(
    options=list(FUNCTION_TRIPLES),
    description="Function:",
    style=widget_style,
)
w_direction = widgets.ToggleButtons(
    options=["Forward", "Inverse"],
    description="Direction:",
    style=widget_style,
)
w_N = widgets.IntSlider(
    value=100,
    min=16,
    max=512,
    step=16,
    description="Samples N:",
    continuous_update=False,
    style=widget_style,
)
w_input_width = widgets.FloatSlider(
    value=5.0,
    min=0.5,
    max=20.0,
    step=0.5,
    description="Input width:",
    continuous_update=False,
    style=widget_style,
)
w_output_half_width = widgets.FloatSlider(
    value=0.2,
    min=0.01,
    max=20.0,
    step=0.01,
    description="FlexFT output half-width:",
    continuous_update=False,
    style=widget_style,
)
w_plot_half_width = widgets.FloatSlider(
    value=5.0,
    min=0.5,
    max=25.0,
    step=0.5,
    description="Plot half-width:",
    continuous_update=False,
    style=widget_style,
)

output = widgets.Output()


def update_plot(
    function_name,
    direction,
    N,
    input_width,
    output_half_width,
    plot_half_width,
):
    figure = comparison_figure(
        function_name,
        direction,
        N,
        input_width,
        output_half_width,
        plot_half_width,
    )
    with output:
        output.clear_output(wait=True)
        display(figure)
        plt.close(figure)


controls = widgets.VBox(
    [
        widgets.HBox([w_function, w_direction]),
        widgets.HBox([w_N, w_input_width]),
        widgets.HBox([w_output_half_width, w_plot_half_width]),
        output,
    ]
)
display(controls)

widgets.interactive_output(
    update_plot,
    {
        "function_name": w_function,
        "direction": w_direction,
        "N": w_N,
        "input_width": w_input_width,
        "output_half_width": w_output_half_width,
        "plot_half_width": w_plot_half_width,
    },
)

# %%

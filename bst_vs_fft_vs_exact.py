# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
import sys
print(sys.executable)

# %% [markdown]
# ### Imports

# %%
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
from jaxtyping import Float, Int, Array, Complex, Num
from jax.numpy.fft import fft, ifft

Float2D = Float[Array, "x1 x2"]
Complex1D = Complex[Array, "x1"]
Array1D = Num[Array, "x1"]

# %% [markdown]
# ### Interactive comparison of 1D continuous Fourier transform approximations
#
# The following cell creats an interactive widget that can be used to compare the FlexFT vs FFT vs the exact anlytical continuous Fourier transform

# %%
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from flexft.flexft import generate_bst, ibst

jax.config.update("jax_enable_x64", True)

# --- function triples: (f, F{f}, F^{-1}{f}) ---
def f_gaussian(x):    return jnp.exp(-(x**2))
def F_gaussian(nu):   return np.sqrt(np.pi) * np.exp(-((np.pi * nu) ** 2))
def Finv_gaussian(x): return np.sqrt(np.pi) * np.exp(-(np.pi * x)**2)

def f_step(x):        return jnp.where(jnp.abs(x) <= 0.5, 1.0, 0.0)
def F_step(nu):       return np.sinc(nu)
def Finv_step(x):     return np.sinc(x)

def f_triangle(x):    return jnp.maximum(1 - jnp.abs(x), 0.0)
def F_triangle(nu):   return np.sinc(nu)**2
def Finv_triangle(x): return np.sinc(x)**2

def f_sech(x):        return 1 / jnp.cosh(x)
def F_sech(nu):       return np.pi / np.cosh(np.pi**2 * nu)
def Finv_sech(x):     return (1 / np.pi) / np.cosh(x / np.pi**2)

FUNCTION_TRIPLES = {
    "Gaussian": (f_gaussian, F_gaussian, Finv_gaussian),
    "Step":     (f_step,     F_step,     Finv_step),
    "Triangle": (f_triangle, F_triangle, Finv_triangle),
    "Sech":     (f_sech,     F_sech,     Finv_sech),
}

# --- widgets ---
w_func      = widgets.ToggleButtons(options=list(FUNCTION_TRIPLES), description="Function:")
w_direction = widgets.ToggleButtons(options=["Forward", "Inverse"], description="Direction:")
w_N         = widgets.IntSlider(value=100, min=16, max=512, step=16,  description="N")
w_Linput    = widgets.FloatSlider(value=5.0,  min=0.5,  max=20.0, step=0.5,  description="L_input")
w_Lbst     = widgets.FloatSlider(value=0.2,  min=0.01, max=20.0, step=0.01, description="L_bst")
w_Lex       = widgets.FloatSlider(value=10.0, min=1.0,  max=50.0, step=1.0,  description="L_exact")

out = widgets.Output()

def plot_bst(func_name, direction, N, L_input, L_bst, L_exact):
    f, F_fwd, F_inv = FUNCTION_TRIPLES[func_name]
    forward = (direction == "Forward")

    # input is always f(x) sampled on a spatial grid
    x         = np.linspace(-L_input / 2, L_input / 2, N, endpoint=False)
    dx        = x[1] - x[0]
    dnu       = 1 / (N * dx)
    f_sampled = f(x)

    if forward:
        nu_bst    = np.linspace(-L_bst, L_bst, N, endpoint=False)
        G_bst     = generate_bst(x, nu_bst)(f_sampled)

        G_fft     = jnp.fft.fftshift(jnp.fft.fft(f_sampled))
        nu_fft    = jnp.fft.fftshift(jnp.fft.fftfreq(N, dx))

        nu_exact  = np.linspace(-L_exact, L_exact, 1000)

        ex_x,  ex_y  = nu_exact, np.abs(F_fwd(nu_exact))
        fft_x, fft_y = nu_fft,   dx * np.abs(G_fft)
        bst_x, bst_y = nu_bst,   np.abs(G_bst)

        xlabel_out, ylabel_out = "ν",  "|F(ν)|"
        fft_label, bst_label   = "FFT", "FlexFT"

    else:
        x_bst     = np.linspace(-L_bst, L_bst, N, endpoint=False)
        g_bst     = ibst(x, x_bst, f_sampled)
        dx_bst    = x_bst[1] - x_bst[0]

        g_ifft    = jnp.fft.fftshift(jnp.fft.ifft(jnp.fft.ifftshift(f_sampled))) * N * dx
        x_ifft    = jnp.fft.fftshift(jnp.fft.fftfreq(N, dx))

        x_exact   = np.linspace(-L_exact, L_exact, 400)

        ex_x,  ex_y  = x_exact, np.abs(F_inv(x_exact))
        fft_x, fft_y = x_ifft,  np.abs(g_ifft)
        bst_x, bst_y = x_bst,   np.abs(g_bst)

        xlabel_out, ylabel_out = "x",  "|F⁻¹(x)|"
        fft_label, bst_label   = "IFFT", "IBST"


    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    
    
    ax[0].plot(x, np.abs(f_sampled))
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("f(x)")
    ax[0].set_title(f"{func_name} — {direction}")

    ax[1].plot(ex_x, ex_y, "black", label="Exact")
    ax[1].plot(fft_x, fft_y, "rx", markersize=4, label=fft_label)
    ax[1].legend(loc="lower center", bbox_to_anchor=(0.5, 1.02))
    ax[1].set_xlabel(xlabel_out)
    ax[1].set_ylabel(ylabel_out)
    ax[1].set_xlim(-L_exact / 2, L_exact / 2)

    ax[2].plot(ex_x, ex_y, "black", label="Exact")
    ax[2].plot(bst_x, bst_y, "bx", markersize=4, label=bst_label)
    ax[2].legend(loc="lower center", bbox_to_anchor=(0.5, 1.02))
    ax[2].set_xlabel(xlabel_out)
    ax[2].set_ylabel(ylabel_out)
    ax[2].set_xlim(-L_exact / 2, L_exact / 2)

    plt.tight_layout()
    with out:
        out.clear_output(wait=True)
        plt.show()

ui = widgets.VBox([
    widgets.HBox([w_func, w_direction]),
    widgets.HBox([w_N, w_Linput]),
    widgets.HBox([w_Lbst, w_Lex]),
    out,
])
display(ui)

widgets.interactive_output(plot_bst, {
    "func_name":  w_func,
    "direction":  w_direction,
    "N":          w_N,
    "L_input":    w_Linput,
    "L_bst":      w_Lbst,
    "L_exact":    w_Lex,
})

# %%

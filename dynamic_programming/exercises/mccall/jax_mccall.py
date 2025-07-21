import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from typing import NamedTuple
import quantecon as qe


class McCallModel(NamedTuple):
                 # unemployment compensation
                 # job separation rate
                 # discount factor
                 # grid of points for fitted VFI
                 # draws of wages for Monte Carlo


def create_mccall_model(c=1.0,
                        α=0.1,
                        β=0.96,
                        μ=2.5,
                        σ=0.5,
                        grid_min=1e-10,
                        grid_max=5,
                        grid_size=100,
                        mc_size=1_000,
                        seed=1234):
    " Create an instance of the McCall model. "
    # Complete


def bellman(model, v, d):
    " Update d and v "
    # Complete


def solve_model(model, tol=1e-5, max_iter=2000):
    " Iterates to convergence on the Bellman equations "
    # Complete


def compute_reservation_wage(model):
    """
    Computes the reservation wage of an instance of the McCall model
    by finding the smallest w such that v(w) >= h.

    If no such w exists, then w_bar is set to jnp.inf.
    """
    # Complete


mu_vals = jnp.linspace(0.0, 2.0, 15)
w_bar_vals = []
fig, ax = plt.subplots()

print("Computing reservation wages")
qe.tic()
for m in mu_vals:
    model = create_mccall_model(μ=m)
    w_bar = compute_reservation_wage(model)
    w_bar_vals.append(w_bar)
time = qe.toc()
print(f"Compute reservation wages in {time:.2f} seconds")

ax.set(xlabel='mean', ylabel='reservation wage')
ax.plot(mu_vals, w_bar_vals, label=r'$\bar w$ as a function of $\mu$')
ax.legend()

plt.show()


import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from typing import NamedTuple
import quantecon as qe


class McCallModel(NamedTuple):
    c: float              # unemployment compensation
    α: float              # job separation rate
    β: float              # discount factor
    w_grid: jnp.ndarray   # grid of points for fitted VFI
    w_draws: jnp.ndarray  # draws of wages for Monte Carlo


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
    key = jax.random.PRNGKey(seed)
    z = jax.random.normal(key, mc_size)
    w_draws = jnp.exp(μ + σ * z)
    w_grid = jnp.linspace(grid_min, grid_max, grid_size)
    return McCallModel(c=c, α=α, β=β, w_grid=w_grid, w_draws=w_draws)


@jax.jit
def bellman(model, v, d):
    # Unpack and simplify names
    c, α, β, w_grid, w_draws = model
    w, u = w_grid, jnp.log
    # Interpolate array representing value function
    vf = lambda x: jnp.interp(x, w, v)
    # Update d using Monte Carlo to evaluate integral
    d_new = jnp.mean(jnp.maximum(vf(w_draws), u(c) + β * d))
    # Update v
    v_new = u(w) + β * ((1 - α) * v + α * d)
    return v_new, d_new


@jax.jit
def solve_model(model, tol=1e-5, max_iter=2000):
    " Iterates to convergence on the Bellman equations "
    c, α, β, w_grid, w_draws = model
    v = jnp.ones_like(w_grid)       # Initial guess of v
    d = 1                           # Initial guess of d

    def update(state):
        v, d, i, error = state
        v_new, d_new = bellman(model, v, d)
        error_1 = jnp.max(jnp.abs(v_new - v))
        error_2 = jnp.abs(d_new - d)
        error = jnp.maximum(error_1, error_2)
        return v_new, d_new, i + 1, error

    def test(state):
        v, d, i, error = state
        return (error > tol) & (i < max_iter)

    initial_state = v, d, 0, tol + 1
    out_state = jax.lax.while_loop(test, update, initial_state)
    v, d, i, error = out_state
    return v, d


@jax.jit
def compute_reservation_wage(model):
    """
    Computes the reservation wage of an instance of the McCall model
    by finding the smallest w such that v(w) >= h.

    If no such w exists, then w_bar is set to jnp.inf.
    """
    c, α, β, w_grid, w_draws = model
    u = jnp.log
    v, d = solve_model(model)
    h = u(c) + β * d
    w_bar = jnp.inf
    i = jnp.searchsorted(v, h)
    w_bar = w_grid[i]
    return w_bar


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


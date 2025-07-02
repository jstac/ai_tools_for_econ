"""
We consider a version of the Long-Plosser multisector optimal growth model with
Bellman equation

    v(y) = max_{l, X, c} { u(c, z) + β E v(f(l, X, λ)) }

subject to

    c_j + Σ_i X_{ij} = y_j,
    z + Σ_i l_i = H
    [f(l, X, λ)]_i = λ_i l_i^{b_i} Π_j X_{ij}^{a_{ij}}

Here

* y is an N-vector of outputs 
* λ is an N-vector of IID shocks 
* c is an N-vector of consumption quantities
* z is leisure
* X is an NxN matrix of inputs, with X_{ij} the quantity of commodity j used to produce commodity i
* u(c, z) = θ_0 ln z + Σ_i θ_i ln c_i

"""

import numpy as np
import quantecon as qe
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from functools import partial
from typing import NamedTuple


class Model(NamedTuple):
    """
    Class for the Long and Plosser (1983) model
    """
    θ: jax.Array        # utility parameters
    b: jax.Array        # production parameters 
    A: jax.Array        # production elasticities
    grid_size: int
    H: float = 1        # labor supply normalized to 1
    β: float = 0.97


def u(model, c, z):
    "Utility"
    H, β, θ, b, A, N = model
    return θ[0] * jnp.log(z) + jnp.dot(θ[1:], jnp.log(c))

def f(model, L, X, λ):
    """Production function f(L, X, λ) -> (N, ) array of output
    L: (N, ) array of labor inputs
    X: (N, N) array of commodity inputs
    λ: (N, ) array of shocks"""
    H, β, θ, b, A, N = model
    return λ * L**b * jnp.prod(X**A, 1)

@partial(jax.jit, static_argnames=('N'))
def v_star(model, Y):
    """
    Analytical value function. Equation (9) in the paper, where J(λ) = 0 for
    lognormal iid shocks by (11) and K is derived by plugging (9) into the
    Bellman equation.
    """
    H, β, θ, b, A, N = model
    gamma = θ[1:] @ jnp.linalg.inv(jnp.eye(N) - β * A)  # equation (10b)
    sum_ga = gamma[:, None] * A  # matrix that represents gamma_i * a_ij
    sum_gb = θ[0] + β * jnp.sum(gamma * b)
    K = (θ[0] * jnp.log(H * θ[0] / sum_gb) # leisure
         + jnp.sum(θ[1:] * jnp.log(θ[1:]/gamma))  # consumption
         + β * jnp.sum(gamma * b * jnp.log(β * H * gamma * b / sum_gb))
         + β * jnp.sum(sum_ga * jnp.log(β * sum_ga/gamma[None, :]))) / (1 - β)
    return jnp.sum(gamma * jnp.log(Y)) + K


class Config(NamedTuple):
    """
    Class for Bellman operator configurations
    """
    pass



θ = jnp.array([1., 1.])
b = jnp.array([0.3])
A = jnp.array([[0.3]])

θ = np.array([1, 1, 1])
b = jnp.array([[0.2, 0.6]])
A = jnp.array([[0.1, 0.7], 
               [0.3, 0.1]])


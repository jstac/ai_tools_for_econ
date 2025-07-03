"""
We consider a version of the Long-Plosser multisector optimal growth model with
Bellman equation

    v(y) = max_{l, X, c} { u(c, z) + β E v(f(l, X, λ)) }

subject to

    c_j + Σ_i X_{ij} = y_j,
    Σ_i l_i = 1.0
    [f(l, X, λ)]_i = λ_i l_i^{b_i} Π_j X_{ij}^{a_{ij}}

Here

* y is an N-vector of outputs 
* λ is an N-vector of IID shocks 
* c is an N-vector of consumption quantities
* X is an NxN matrix of inputs, with X_{ij} the quantity of commodity j used to produce commodity i
* u(c) = Σ_i θ_i ln c_i

"""

import numpy as np
import quantecon as qe
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from functools import partial
from typing import NamedTuple


## == Model == ##

class Model(NamedTuple):
    """
    Class for the Long and Plosser (1983) model
    """
    θ: jax.Array        # utility parameters
    b: jax.Array        # production parameters 
    A: jax.Array        # production elasticities
    β: float = 0.97     # discount rate


def u(model, c, z):
    "Utility"
    θ, b, A, β = model
    return jnp.dot(θ, jnp.log(c))


def f(model, l, X, λ):
    "Production function"
    θ, b, A, β = model
    return λ * l**b * jnp.prod(X**A, axis=1)


def v_star(model, y):
    """
    Analytical value function. Equation (9) in the paper, where J(λ) = 0 for
    lognormal iid shocks by (11) and K is derived by plugging (9) into the
    Bellman equation.
    """
    θ, b, A, β = model
    N = len(A)
    gamma = θ @ jnp.linalg.inv(jnp.eye(N) - β * A)  # equation (10b)
    sum_ga = gamma[:, None] * A  # matrix that represents gamma_i * a_ij
    sum_gb = β * jnp.sum(gamma * b)
    K = (β * jnp.sum(gamma * b * jnp.log(β * 1.0 * gamma * b / sum_gb))
         + β * jnp.sum(sum_ga * jnp.log(β * sum_ga/gamma[None, :]))) / (1 - β)
    return jnp.sum(gamma * jnp.log(y)) + K


## == Test == ##

θ = np.array([1.0, 1.0])
b = jnp.array([[0.2, 0.6]])
A = jnp.array([[0.1, 0.7], 
               [0.3, 0.1]])

def plot_along_ray():
    model = Model(θ=θ, b=b, A=A)
    grid_size = 200
    N = len(A)
    scaling_values = np.linspace(0.001, 5, grid_size)
    # Create a matrix where columns are constant and equal to each scaling value
    ys = np.ones((N, grid_size)) * scaling_values
    # Vectorize v_star to work on each column of ys  (in_axes=1 means columns)
    v_star_vectorized = jax.vmap(lambda y: v_star(model, y), in_axes=1)
    # Apply it to all columns of ys
    values = v_star_vectorized(ys)

    fig, ax = plt.subplots()
    ax.plot(scaling_values, values)
    plt.show()


## == Network == ##

class LayerParams(NamedTuple):
    """
    Stores parameters for one layer of the neural network.

    """
    W: jnp.ndarray     # weights
    b: jnp.ndarray     # biases


class Config:
    """
    Configuration and parameters for training the neural network.

    """
    seed = 42
    epochs = 200
    n_paths = 1_000
    path_length = 1_000
    layer_sizes = 1, 8, 8, 1
    init_lr = 0.0015
    min_lr = 0.0001
    warmup_steps = 100
    decay_steps = 300


def initialize_layer(in_dim, out_dim, key):
    """
    Initialize weights and biases for a single layer of a feedforward network.
    Use LeCun initialization.

    """
    s = jnp.sqrt(1.0 / in_dim)
    W = jax.random.normal(key, (in_dim, out_dim)) * s
    b = jnp.ones((out_dim,))
    return LayerParams(W, b)


def initialize_network(key, layer_sizes):
    """
    Build a network by initializing all of the parameters.
    A network is a list of LayerParams instances, each of which
    contains a weight-bias pair (W, b).

    """
    params = []
    # For all layers but the output 
    for i in range(len(layer_sizes) - 1):
        key, subkey = jax.random.split(key)
        layer = initialize_layer(
            layer_sizes[i],      # in dimension for layer
            layer_sizes[i + 1],  # out dimension for layer
            subkey 
        )
        params.append(layer)

    return params


def forward(params, y):
    """
    Evaluate neural network policy: maps vector y to vector of allocations x and
    consumption quantity c for a given sector. For example, if the sector is j,
    then it maps to 

        (x_{1j}, x_{2j}, ..., x_{Nj}, c_j)

    The output is expressed as fractions, so that the output vector sums to one.
    The fractions are converted to quantities by multiplying by y_j.
    """
    # Iterate using the primary activation function
    σ = jax.nn.selu     
    for W, b in params[:-1]:
        y = σ(y @ W + b)
    # Switch to softmax at the last step to get weights
    W, b = params[-1]
    p = jax.nn.softmax(y @ W + b, axis=-1)
    return p


def policy(all_params, y):
    """
    Using the ANNs, map state vector y to X, c.

    """
    for j, params in enumerate(all_params):
        p[j] = forward(params, y)


@partial(jax.jit, static_argnames=('path_length', 'n_paths'))
def simulate_paths(params, model, key, path_length, n_paths):
    """
    Simulate n_paths paths using policy rollout and return 
    their present values.

    """
    
    θ, b, A, β = model
    policy = jax.vmap(lambda y: forward(params, y))
    initial_y = jnp.ones((n_paths,))  # All paths start at y = 1.0

    def update(t, state):
        # Set up
        y_vec, values, discount, key = state
        key, subkey = random.split(key)
        z = random.normal(subkey, (n_paths,))
        ξ = s * jnp.exp(z)

        # Compute consumption given y and update income
        consumption_rate = policy(y_vec)
        c = consumption_rate * y_vec
        y_vec = f(y_vec - c, A, α) * ξ
        # Update lifetime value
        values = values + discount * u(c) 
        discount = discount * β
        new_state = y_vec, values, discount, key
        return new_state
    
    values, discount = jnp.zeros((n_paths,)), 1.0
    state = initial_y, values, discount, key
    _, final_values, discount, key = jax.lax.fori_loop(
        0, path_length, update, state
    )
    return final_values

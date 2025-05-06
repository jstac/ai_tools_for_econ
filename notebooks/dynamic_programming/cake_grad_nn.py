"""

Solve infinite horizon cake eating problem using policy gradient ascent with
JAX. Policy is represented as a simple neural network.

Utility is u(c) = c^(1-γ) / (1-γ) and the discount factor is β.

Wealth evolves according to 

    w' = R(w-c) 

where R > 0 is the gross interest rate.  

To ensure stability we assume that β R^(1-γ) < 1.

Note that the optimal policy is c = κ w, where

    κ := 1 - [β R^(1-γ)]^(1/γ)

The initial size of the cake is 1.0.

"""

import jax
import jax.numpy as jnp
from jax import grad, jit, random
import optax
import matplotlib.pyplot as plt
from typing import List, Tuple, NamedTuple

class Model(NamedTuple):
    """
    Stores parameters for the model.

    """
    γ: int = 2.0
    β: int = 0.95
    R: int = 1.01


class LayerParams(NamedTuple):
    """
    Stores parameters for one layer of the neural network.

    """
    W: jnp.ndarray     # weights
    b: jnp.ndarray     # biases


class LearningConfig:
    """
    Configuration and parameters for training the neural network.

    """
    seed = 42
    learning_rate = 0.01
    n_iter = 10_000
    n_paths = 5_000
    n_steps = 50
    layer_sizes = 1, 16, 16, 1

def initialize_layer(in_dim, out_dim, key):
    """
    Initialize weights and biases for a single layer of a feedforward network.
    Use LeCun initialization.

    """
    s = jnp.sqrt(1.0 / in_dim)
    W = jax.random.normal(key, (in_dim, out_dim)) * s
    b = jnp.zeros((out_dim,))
    return W, b
    

def initialize_network(key, layer_sizes):
    """
    Build a network by initializing all of the parameters.
    A network is a list of tuples (W, b).

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


# Neural network forward pass
def policy_network(params, w):
    """
    Neural network policy: maps cake size to consumption rate.

    """
    σ = jax.nn.selu         # Activation function
    x = jnp.array([w])      # Make it a 1D array
    
    # Forward pass through network
    for W, b in params[:-1]:
        x = σ(x @ W + b)
    # Output layer with sigmoid activation for consumption rate
    W, b = params[-1]
    x = x @ W + b
    consumption = jax.nn.sigmoid(x[0])
    
    # Clip to [0.001, 0.999] range and return
    consumption = 0.001 + 0.998 * consumption
    return consumption


def u(c, γ):
    """ Utility function. """
    c = jnp.maximum(c, 1e-10)
    return c**(1 - γ) / (1 - γ)


def simulate_path(params, key):
    """
    Simulate one path and return its present value.

    """
    def update(t, state):
        w, value, discount = state
        continue_sim = w > 1e-10
        
        # Get consumption from policy network
        c = policy_network(params, w)
        
        # Ensure feasible consumption
        c = jnp.minimum(c, w - 1e-10)
        
        # Update state
        w = jnp.maximum(R * (w - c), 0.0)
        value = value + discount * u(c, γ) * continue_sim
        discount = discount * β
        new_state = w, value, discount
        return new_state
    
    initial_w, initial_value, initial_discount = 1.0, 0.0, 1.0
    initial_state = initial_w, initial_value, initial_discount
    final_w, final_value, discount = jax.lax.fori_loop(0, n_steps, update, initial_state)
    return final_value

# Vectorized simulation for multiple paths
simulate_paths = jax.vmap(simulate_path, in_axes=(None, 0))

# Estimate policy value by averaging multiple paths
@jit
def estimate_negative_value(params, key):
    """Estimate value function using Monte Carlo sampling"""
    keys = random.split(key, n_paths)
    values = simulate_paths(params, keys)
    return - jnp.mean(values)


# == Train and solve == # 

# Unpack names

seed, γ, β, R = Config.seed, Config.γ, Config.β, Config.R
learning_rate, n_iter = Config.learning_rate, Config.n_iter
n_paths, n_steps = Config.n_paths, Config.n_steps
layer_sizes = Config.layer_sizes

# Test stability
assert β * R**(1 - γ) < 1, "Parameters fail stability test."

# Optimizer
optimizer = optax.adam(learning_rate)

# Initialize neural network parameters
key = random.PRNGKey(seed)
params = initialize_network(key, layer_sizes)
opt_state = optimizer.init(params)

# Store history
value_history = []

# Training loop
for i in range(n_iter):
    key = random.PRNGKey(i)
    
    # Compute value and gradients at existing parameterization
    neg_value, grads = jax.value_and_grad(estimate_negative_value)(params, key)
    value = - neg_value
    
    # Update parameters using optimizer
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    
    # Store history
    value_history.append(value)
    
    if i % 100 == 0:
        print(f"Iteration {i}: Value = {value:.4f}")


# Plot learning progress

fig, ax = plt.subplots()
ax.plot(value_history, 'b-', linewidth=2)
ax.set_xlabel('iteration')
ax.set_ylabel('policy value')
ax.set_title('learning progress')
plt.show()

# Visualize the learned policy
w_grid = jnp.linspace(0.01, 1.0, 1000)
policy_vmap = jax.vmap(lambda w: policy_network(params, w))
consumption = policy_vmap(w_grid)
consumption_rate = consumption / w_grid

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot consumption policy
ax1.plot(w_grid, consumption, 'b-', linewidth=2, label='Neural Network Policy')
ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='45° line')
ax1.set_xlabel('Cake Size')
ax1.set_ylabel('Consumption')
ax1.set_title('Neural Network Consumption Policy')
ax1.legend()

# Plot consumption rate
ax2.plot(w_grid, consumption_rate, 'g-', linewidth=2, label='Neural Network Policy')
ax2.set_xlabel('Cake Size')
ax2.set_ylabel('Consumption Rate (c/w)')
ax2.set_title('Consumption Rate Policy')
ax2.legend()

plt.tight_layout()
plt.show()



def simulate_consumption_path(params, T=50):
    """Simulate consumption path using neural network policy"""
    w_path = [1.0]   # 1.0 is the initial size
    c_path = []
    
    w = 1.0
    for t in range(T):
        # Forward pass through neural network
        x = jnp.array([w])
        
        for i, (W, b) in enumerate(params[:-1]):
            x = jnp.matmul(x, W) + b
            x = jnp.tanh(x)
        
        W, b = params[-1]
        x = jnp.matmul(x, W) + b
        c = float(jax.nn.sigmoid(x[0]))
        c = 0.001 + 0.998 * c
        
        c_path.append(float(c))
        w = max(0, w - c)
        w_path.append(float(w))
        
        if w <= 1e-10:
            break
    
    return w_path, c_path

# Simulate and plot path
w_path, c_path = simulate_consumption_path(params)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(w_path, 'b-', marker='o')
ax1.set_xlabel('Time')
ax1.set_ylabel('Cake Size')
ax1.set_title('Cake Size Over Time')

ax2.plot(c_path, 'r-', marker='o')
ax2.set_xlabel('Time')
ax2.set_ylabel('Consumption')
ax2.set_title('Optimal Consumption Over Time')

plt.tight_layout()
plt.show()

print(f"\nFinal value: {value_history[-1]:.4f}")

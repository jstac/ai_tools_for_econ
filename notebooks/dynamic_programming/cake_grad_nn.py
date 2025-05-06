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
    γ = 2.0
    β = 0.95
    R = 1.01


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
    n_iter = 2_000
    n_paths = 5_000
    path_length = 120
    layer_sizes = 1, 16, 16, 1
    init_lr = 0.0025
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


def forward(params, w):
    """
    Evaluate neural network policy: maps cake size to consumption rate c/w
    by running a forward pass through the network.

    """
    σ = jax.nn.selu         # Activation function
    x = jnp.array([w])      # Make state a 1D array
    # Forward pass through network
    for W, b in params[:-1]:
        x = σ(x @ W + b)
    # Output layer with sigmoid activation for consumption rate
    W, b = params[-1]
    x = jax.nn.sigmoid(x @ W + b)
    consumption_rate = x[0]
    return consumption_rate


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
        c = forward(params, w) * w
        
        # Ensure feasible consumption
        c = jnp.minimum(c, w - 1e-10)
        
        # Update state
        w = jnp.maximum(R * (w - c), 0.0)
        value = value + discount * u(c, γ) 
        discount = discount * β
        new_state = w, value, discount
        return new_state
    
    initial_w, initial_value, initial_discount = 1.0, 0.0, 1.0
    initial_state = initial_w, initial_value, initial_discount
    final_w, final_value, discount = jax.lax.fori_loop(
            0, path_length, update, initial_state
        )
    return final_value

# Vectorized simulation for multiple paths
simulate_paths = jax.vmap(simulate_path, in_axes=(None, 0))

# Estimate policy value by averaging multiple paths
@jit
def loss_function(params, key):
    """
    The loss function is minus the estimated lifetime value of a policy
    identified by params, computed using Monte Carlo sampling via rollouts.

    """
    keys = random.split(key, n_paths)
    # Compute lifetime values across many paths
    values = simulate_paths(params, keys)
    # Loss is negative of mean
    return - jnp.mean(values)


def create_lr_schedule():
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=Config.init_lr,
        transition_steps=Config.warmup_steps
    )
    
    decay_fn = optax.exponential_decay(
        init_value=Config.init_lr,
        transition_steps=Config.decay_steps,
        decay_rate=0.5,
        end_value=Config.min_lr
    )
    
    return optax.join_schedules(
        schedules=[warmup_fn, decay_fn],
        boundaries=[Config.warmup_steps]
    )


# == Train and solve == # 

# Unpack names

γ, β, R = Model.γ, Model.β, Model.R
seed, n_iter = Config.seed, Config.n_iter
n_paths, path_length = Config.n_paths, Config.path_length
layer_sizes = Config.layer_sizes

# Test stability
assert β * R**(1 - γ) < 1, "Parameters fail stability test."

# Compute optimal consumption rate and lifetime value
κ = 1 - (β * R**(1 - γ))**(1/γ)
v_max = κ**(-γ) * u(1.0, γ)
print(f"Maximum possible lifetime value = {v_max}.\n")

# Optimizer
lr_schedule = create_lr_schedule()
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),  # Gradient clipping for stability
    optax.adam(learning_rate=lr_schedule)
)

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
    loss, grads = jax.value_and_grad(loss_function)(params, key)
    lifetime_value = - loss
    
    # Update parameters using optimizer
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    
    # Store history
    value_history.append(lifetime_value)
    
    if i % 100 == 0:
        print(f"Iteration {i}: Value = {lifetime_value:.4f}")


# Plot learning progress

fig, ax = plt.subplots()
ax.plot(value_history, 'b-', linewidth=2)
ax.set_xlabel('iteration')
ax.set_ylabel('policy value')
ax.set_title('learning progress')
plt.show()

# Visualize the learned policy
w_grid = jnp.linspace(0.01, 1.0, 1000)
policy_vmap = jax.vmap(lambda w: forward(params, w))
consumption_rate = policy_vmap(w_grid)

# Plot consumption rate
fig, ax = plt.subplots()
ax.plot(w_grid, consumption_rate, lw=2, label='policy-based')
ax.plot(w_grid, κ * jnp.ones(len(w_grid)), lw=2, label='optimal')
ax.set_xlabel('Cake size')
ax.set_ylabel('Consumption rate (c/w)')
ax.set_title('Consumption rate')
ax.legend()
plt.show()



def simulate_consumption_path(params, T=50):
    """
    Compute consumption path using neural network policy identified by params.

    """
    w_path = [1.0]   # 1.0 is the initial size
    c_path = []
    
    w = 1.0
    for t in range(T):
        c = forward(params, w) * w
        c_path.append(float(c))
        w = R * (w - c)
        w_path.append(float(w))
        
        if w <= 1e-10:
            break
    
    return w_path, c_path

def compute_optimal_path(T=50):
    """
    Compute optimal consumption path.

    """
    w_path = [1.0]   # 1.0 is the initial size
    c_path = []
    
    w = 1.0
    for t in range(T):
        c = κ * w
        c_path.append(c)
        w = R * (w - c)
        w_path.append(w)
        
        if w <= 1e-10:
            break
    
    return w_path, c_path


# Simulate and plot path
w_sim, c_sim = simulate_consumption_path(params)
w_opt, c_opt = compute_optimal_path()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(w_sim, label='policy-based')
ax1.plot(w_opt, label='optimal')
ax1.set_xlabel('Time')
ax1.set_ylabel('Cake size')
ax1.set_title('Cake size over time')
ax1.legend()

ax2.plot(c_sim, label='policy-based')
ax2.plot(c_opt, label='optimal')
ax2.set_xlabel('Time')
ax2.set_ylabel('Consumption')
ax2.set_title('Consumption over time')
ax2.legend()

plt.tight_layout()
plt.show()

print(f"\nFinal value: {value_history[-1]:.4f}")

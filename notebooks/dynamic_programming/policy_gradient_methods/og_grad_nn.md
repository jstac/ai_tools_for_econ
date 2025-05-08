---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Policy Gradient-Based Stochastic Optimal Growth

## Introduction

In this notebook we solve infinite horizon optimal growth problem using policy gradient ascent with JAX.

Each policy is represented as a fully connected neural network.

The growth problem is a classic Brock--Mirman type problem with log utility and Cobb-Douglas production.

We compare the computational result with the known analytical solution.

Utility is $u(c) = \ln c$ and the discount factor is $\beta$.

Income evolves according to 

$$
    y' = f(y-c) ξ
$$

where $f(k) = A k^α$ is producton.

Note that the optimal policy is $c = \kappa w$, where

$$
    \kappa := 1 - \alpha \beta 
$$

We use the following imports.

```{code-cell} ipython3
import jax
import jax.numpy as jnp
from jax import grad, jit, random
import optax
import matplotlib.pyplot as plt
from typing import List, Tuple, NamedTuple
```

## Set up

+++

We define a `Model` class to store model parameters.

```{code-cell} ipython3
class Model(NamedTuple):
    """
    Stores parameters for the model.

    """
    β = 0.96  # discount factor
    A = 4.0   # multiplicative productivity parameter
    α = 0.7   # power productivity parameter
    s = 0.10  # volatility term for shock
```

We also define a `LayerParams` class to store the weights `W` and biases `b` associated with a single network layer.

```{code-cell} ipython3
class LayerParams(NamedTuple):
    """
    Stores parameters for one layer of the neural network.

    """
    W: jnp.ndarray     # weights
    b: jnp.ndarray     # biases
```

We use a `Config` class to hold parameters for training the network.

```{code-cell} ipython3
class Config:
    """
    Configuration and parameters for training the neural network.

    """
    seed = 42
    epochs = 100
    n_paths = 1_000
    path_length = 1_000
    layer_sizes = 1, 8, 8, 1
    init_lr = 0.0015
    min_lr = 0.0001
    warmup_steps = 100
    decay_steps = 300
```

The following function is used to initialize a single layer.

```{code-cell} ipython3
def initialize_layer(in_dim, out_dim, key):
    """
    Initialize weights and biases for a single layer of a feedforward network.
    Use LeCun initialization.

    """
    s = jnp.sqrt(1.0 / in_dim)
    W = jax.random.normal(key, (in_dim, out_dim)) * s
    b = jnp.ones((out_dim,))
    return W, b
```

The next function initializes the parameters of the full network.

```{code-cell} ipython3
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
```

We create a function to run a forward pass through the network, mapping parameters and an input `y` to a corresponding consumption rate.

```{code-cell} ipython3
def forward(params, y):
    """
    Evaluate neural network policy: maps income consumption rate c/y
    by running a forward pass through the network.

    """
    σ = jax.nn.selu         # Activation function
    x = jnp.array([y])      # Make state a 1D array
    # Forward pass through network
    for W, b in params[:-1]:
        x = σ(x @ W + b)
    # Output layer with sigmoid activation 
    W, b = params[-1]
    x = jax.nn.sigmoid(x @ W + b)
    consumption_rate = x[0]
    return consumption_rate
```

Here's the utility function.

```{code-cell} ipython3
def u(c):
    c = jnp.maximum(c, 1e-10)
    return jnp.log(c)
```

Here's the production function.

```{code-cell} ipython3
def f(k, A, α):
    return A * k**α
```

The next code runs a collection of `n_paths` policy rollouts, each of which
starts from $y=1.0$.

```{code-cell} ipython3
def simulate_paths(params, model, key, n_paths):
    """
    Simulate n_paths paths and return their present values.

    """
    
    α, A, β, s = model.α, model.A, model.β, model.s
    policy = jax.vmap(lambda y: forward(params, y))

    def update(t, state):
        # Set up
        y_vec, values, discount, key = state
        key, subkey = random.split(key)
        z = random.normal(subkey, (n_paths,))
        ξ = s * jnp.exp(z)

        # Compute consumption given y
        consumption_rate = policy(y_vec)
        c = consumption_rate * y_vec
        c = jnp.minimum(c, y_vec - 1e-10)  # ensure c < y
        
        # Update income
        y_vec = f(y_vec - c, A, α) * ξ

        # Update state
        values += discount * u(c) 
        discount = discount * β
        new_state = y_vec, values, discount, key
        return new_state
    
    y_vec = jnp.ones((n_paths,))  # All paths start at y = 1.0
    values, discount =  jnp.zeros((n_paths,)), 1.0
    state = y_vec, values, discount, key
    _, final_values, discount, key = jax.lax.fori_loop(
            0, path_length, update, state
    )
    return final_values
```

We estimate policy value by averaging values along multiple policy rollouts.

This value is then negated to create a loss function.

```{code-cell} ipython3
@jit
def loss_function(params, model, key):
    """
    The loss function is minus the estimated lifetime value of a policy
    identified by params, computed using Monte Carlo sampling via rollouts.

    """
    # Compute lifetime values across paths
    values = simulate_paths(params, model, key, n_paths)
    # Loss is negative of mean
    return - jnp.mean(values)
```

The next function creates a learning rate scheduler.

```{code-cell} ipython3
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
```

## Training 

Let's now train the model.

First we unpack names.

```{code-cell} ipython3
model = Model()
β, α, A, s = model.β, model.α, model.A, model.s
seed, epochs = Config.seed, Config.epochs
n_paths, path_length = Config.n_paths, Config.path_length
layer_sizes = Config.layer_sizes
```

Next we generate a learning rate scheduler select the Adam optimizer.

```{code-cell} ipython3
lr_schedule = create_lr_schedule()
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),  # Gradient clipping for stability
    optax.adam(learning_rate=lr_schedule)
)
```

Finally, we initialize the network parameters and the optimizer.

```{code-cell} ipython3
# Initialize neural network parameters
key = random.PRNGKey(seed)
params = initialize_network(key, layer_sizes)
opt_state = optimizer.init(params)
```

We are now ready to run the training loop.

```{code-cell} ipython3
# Training loop
value_history = []
for i in range(epochs):
    key = random.PRNGKey(i)
    
    # Compute value and gradients at existing parameterization
    loss, grads = jax.value_and_grad(loss_function)(params, model, key)
    lifetime_value = - loss
    
    # Update parameters using optimizer
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    
    # Store history
    value_history.append(lifetime_value)
    
    if i % 100 == 0:
        print(f"Iteration {i}: Value = {lifetime_value:.4f}")
```

Let's plot the value history over the training epochs.

```{code-cell} ipython3
# Plot learning progress
fig, ax = plt.subplots()
ax.plot(value_history, 'b-', linewidth=2)
ax.set_xlabel('iteration')
ax.set_ylabel('policy value')
ax.set_title('learning progress')
plt.show()
```

Now we visualize the learned policy and compare it to the optimal policy.

```{code-cell} ipython3
y_grid = jnp.linspace(0.01, 1.0, 1000)
policy_vmap = jax.vmap(lambda y: forward(params, y))
consumption_rate = policy_vmap(y_grid)
κ = 1 - (α * β) # Compute optimal consumption rate 
fig, ax = plt.subplots()
ax.plot(y_grid, consumption_rate, linestyle='--', lw=2, label='policy-based')
ax.plot(y_grid, κ * jnp.ones(len(y_grid)), lw=2, label='optimal')
ax.set_xlabel('Income')
ax.set_ylabel('Consumption rate (c/y)')
ax.set_title('Consumption rate')
ax.set_ylim((0, 1))
ax.legend()
plt.show()
```

We simulate a consumption path using the learned policy, as well as
one using the optimal policy.

```{code-cell} ipython3
def simulate_consumption_path(params, T=120):
    """
    Compute consumption path using neural network policy identified by params,
    as well as corresponding optimal path.

    """

    # 1.0 is the initial state
    y_sim = [1.0]   
    c_sim = []
    y_opt = [1.0]  
    c_opt = []

    y = 1.0
    for t in range(T):
        key = random.PRNGKey(t)
        z = random.normal(key, (1,))
        ξ = s * float(jnp.exp(z[0]))

        # Update policy path
        c = forward(params, y) * y
        c_sim.append(float(c))
        y = f(y - c, A, α) * ξ
        y_sim.append(float(y))
        
        if y <= 1e-10:
            break

    y = 1.0
    for t in range(T):
        key = random.PRNGKey(t)
        z = random.normal(key, (1,))
        ξ = s * float(jnp.exp(z[0]))

        # Update optimal path
        c = κ * y
        c_opt.append(float(c))
        y = f(y - c, A, α) * ξ
        y_opt.append(float(y))
        
        if y <= 1e-10:
            break
    
    return y_sim, c_sim, y_opt, c_opt
```

```{code-cell} ipython3
# Simulate and plot path
y_sim, c_sim, y_opt, c_opt = simulate_consumption_path(params)
```

```{code-cell} ipython3
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(y_sim, lw=4, linestyle='--', label='policy-based')
ax1.plot(y_opt, lw=2, label='optimal')
ax1.set_xlabel('Time')
ax1.set_ylabel('Income')
ax1.set_title('Income over time')
ax1.legend()

ax2.plot(c_sim, lw=4, linestyle='--', label='policy-based')
ax2.plot(c_opt, lw=2, label='optimal')
ax2.set_xlabel('Time')
ax2.set_ylabel('Consumption')
ax2.set_title('Consumption over time')
ax2.legend()

plt.tight_layout()
plt.show()
```

```{code-cell} ipython3
print(f"\nFinal value: {value_history[-1]:.4f}")
```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

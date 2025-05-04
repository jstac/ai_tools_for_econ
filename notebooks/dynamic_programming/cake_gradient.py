import jax
import jax.numpy as jnp
from jax import grad, jit, random
import optax
import matplotlib.pyplot as plt

def solve_cake_eating_policy_gradient(gamma=2.0, beta=0.95, initial_cake=1.0, 
                                     learning_rate=0.01, n_iter=1000, n_paths=100, 
                                     n_steps=50, policy_type='linear'):
    """
    Solve infinite horizon cake eating problem using policy gradient ascent with JAX.
    """
    
    # Utility function
    def utility(c):
        # Ensure c is positive for numerical stability
        c = jnp.maximum(c, 1e-10)
        # Use jnp.where for conditional computation
        return jnp.where(
            gamma == 1,
            jnp.log(c),
            (c ** (1 - gamma)) / (1 - gamma)
        )
    
    # Policy function - parameterized as consumption rate c/w
    def policy(params, w):
        # Use jnp.where instead of if statement
        c_rate = jnp.where(
            policy_type == 'linear',
            jnp.clip(params[0] + params[1] * w, 0.001, 0.999),
            0.001 + 0.998 * jax.nn.sigmoid(params[0] + params[1] * jnp.log(jnp.maximum(w, 1e-10)))
        )
        return c_rate * w
    
    # Simulate one path and calculate value
    def simulate_path(params, key, initial_w):
        """Simulate one path and return its present value"""
        def body_fn(carry, t):
            w, value = carry
            # Check if we should continue
            continue_sim = w > 1e-10
            
            # Get consumption from policy
            c = policy(params, w)
            
            # Ensure feasible consumption
            c = jnp.minimum(c, w - 1e-10)
            
            # Update value
            discount = beta ** t
            value = value + discount * utility(c) * continue_sim
            
            # Update state
            new_w = jnp.maximum(w - c, 0.0)
            
            return (new_w, value), None
        
        # Initialize state
        initial_state = (initial_w, 0.0)
        
        # Run the loop
        (final_w, final_value), _ = jax.lax.scan(body_fn, initial_state, jnp.arange(n_steps))
        
        return final_value
    
    # Vectorized simulation for multiple paths
    simulate_paths = jax.vmap(simulate_path, in_axes=(None, 0, None))
    
    # Estimate policy value by averaging multiple paths
    @jit
    def estimate_value(params, key):
        """Estimate value function using Monte Carlo sampling"""
        keys = random.split(key, n_paths)
        values = simulate_paths(params, keys, initial_cake)
        return jnp.mean(values)
    
    # Optimizer
    optimizer = optax.adam(learning_rate)
    
    # Initialize parameters
    if policy_type == 'linear':
        # For log utility, optimal is c/w = 1-beta, so start there
        initial_guess = jnp.array([(1-beta), 0.0])
    else:
        # Sigmoid policy parameters
        initial_guess = jnp.array([0.0, 0.0])
    
    params = initial_guess
    opt_state = optimizer.init(params)
    
    # Store history
    value_history = []
    param_history = []
    
    # Training loop
    for i in range(n_iter):
        key = random.PRNGKey(i)
        
        # Compute value and gradients
        value, grads = jax.value_and_grad(estimate_value)(params, key)
        
        # Update parameters using optimizer
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        
        # Store history
        value_history.append(value)
        param_history.append(params.copy())
        
        if i % 100 == 0:
            print(f"Iteration {i}: Value = {value:.4f}, Params = {params}")
    
    return params, value_history, param_history

# Solve the problem
print("Solving cake eating problem with policy gradient...")
params, value_history, param_history = solve_cake_eating_policy_gradient(
    gamma=2.0, beta=0.95, initial_cake=1.0, learning_rate=0.01, n_iter=1000
)

# Visualize the learned policy
def plot_policy(params, policy_type='linear'):
    """Plot the learned policy function"""
    w_grid = jnp.linspace(0.01, 1.0, 1000)
    
    def policy(params, w):
        c_rate = jnp.where(
            policy_type == 'linear',
            jnp.clip(params[0] + params[1] * w, 0.001, 0.999),
            0.001 + 0.998 * jax.nn.sigmoid(params[0] + params[1] * jnp.log(jnp.maximum(w, 1e-10)))
        )
        return c_rate * w
    
    # Vectorize policy function
    policy_vmap = jax.vmap(lambda w: policy(params, w))
    consumption = policy_vmap(w_grid)
    consumption_rate = consumption / w_grid
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot consumption policy
    ax1.plot(w_grid, consumption, 'b-', linewidth=2)
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)  # 45 degree line
    ax1.set_xlabel('Cake Size')
    ax1.set_ylabel('Optimal Consumption')
    ax1.set_title('Consumption Policy Function')
    ax1.grid(True)
    
    # Plot consumption rate
    ax2.plot(w_grid, consumption_rate, 'g-', linewidth=2)
    ax2.axhline(y=0.05, color='r', linestyle='--', label='1-β = 0.05')
    ax2.set_xlabel('Cake Size')
    ax2.set_ylabel('Consumption Rate (c/w)')
    ax2.set_title('Consumption Rate Policy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# Plot learning progress
plt.figure(figsize=(10, 6))
plt.plot(value_history, 'b-', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Policy Value')
plt.title('Learning Progress')
plt.grid(True)
plt.show()

# Plot parameter evolution
param_array = jnp.array(param_history)
plt.figure(figsize=(10, 6))
plt.plot(param_array[:, 0], 'b-', label='θ₀', linewidth=2)
plt.plot(param_array[:, 1], 'r-', label='θ₁', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Parameter Value')
plt.title('Parameter Evolution')
plt.legend()
plt.grid(True)
plt.show()

# Visualize the learned policy
plot_policy(params)

# Final parameter values
print(f"\nFinal parameters: {params}")
print(f"Theoretical optimal for log utility: c/w = {1-0.95}")

# Simulate a consumption path with learned policy
def simulate_consumption_path(params, initial_cake=1.0, T=50):
    """Simulate consumption path using learned policy - non-jitted version"""
    w_path = [initial_cake]
    c_path = []
    
    w = initial_cake
    for t in range(T):
        # Get consumption rate
        consumption_rate = float(params[0] + params[1] * w)
        consumption_rate = max(0.001, min(0.999, consumption_rate))
        c = consumption_rate * w
        
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
ax1.grid(True)

ax2.plot(c_path, 'r-', marker='o')
ax2.set_xlabel('Time')
ax2.set_ylabel('Consumption')
ax2.set_title('Optimal Consumption Over Time')
ax2.grid(True)

plt.tight_layout()
plt.show()

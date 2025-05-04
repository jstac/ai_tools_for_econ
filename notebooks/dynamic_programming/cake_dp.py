import numpy as np
import matplotlib.pyplot as plt

def solve_infinite_cake_eating(gamma, beta, initial_cake=1.0, grid_size=1000, 
                              max_iter=1000, tol=1e-6):
    """
    Solve the infinite horizon cake eating problem with CRRA utility.
    
    Parameters:
    gamma: CRRA parameter (gamma > 0, gamma != 1)
    beta: Discount factor (0 < beta < 1)
    initial_cake: Initial cake size
    grid_size: Number of grid points for state space
    max_iter: Maximum number of iterations
    tol: Convergence tolerance
    
    Returns:
    w_grid: State space grid (cake sizes)
    policy: Optimal consumption policy function
    value: Value function
    """
    
    # Create grid for cake sizes (avoid exactly zero)
    w_grid = np.linspace(1e-10, initial_cake, grid_size)
    
    # Initialize value function and policy
    value = np.zeros(grid_size)
    value_new = np.zeros(grid_size)
    policy = np.zeros(grid_size)
    
    def utility(c):
        """CRRA utility function"""
        # Ensure c is positive to avoid NaN
        c = np.maximum(c, 1e-10)
        if gamma == 1:
            return np.log(c)
        else:
            return (c ** (1 - gamma)) / (1 - gamma)
    
    # Initialize value function with simple guess
    for i, w in enumerate(w_grid):
        value[i] = utility(w)
    
    # Value function iteration
    for iteration in range(max_iter):
        for i, w in enumerate(w_grid):
            # Grid search over possible consumption values
            c_grid = np.linspace(1e-10, w - 1e-10, 100)
            remaining_cake = w - c_grid
            
            # Ensure non-negative remaining cake
            remaining_cake = np.maximum(remaining_cake, 0)
            
            # Interpolate value function at remaining cake amounts
            v_next = np.interp(remaining_cake, w_grid, value)
            
            # Value for each consumption choice
            v_choices = utility(c_grid) + beta * v_next
            
            # Find optimal consumption
            opt_idx = np.argmax(v_choices)
            policy[i] = c_grid[opt_idx]
            value_new[i] = v_choices[opt_idx]
        
        # Check convergence (handle potential NaN)
        if np.any(np.isnan(value_new)):
            print("Warning: NaN detected in value function")
            break
            
        max_diff = np.max(np.abs(value_new - value))
        if max_diff < tol:
            print(f"Converged after {iteration + 1} iterations")
            break
        
        # Update value function
        value = value_new.copy()
    
    if iteration == max_iter - 1:
        print(f"Warning: Maximum iterations reached. Max difference: {max_diff}")
    
    return w_grid, policy, value

def simulate_infinite_consumption(policy, w_grid, initial_cake, T=50):
    """Simulate consumption path using the policy function for T periods"""
    cake_path = np.zeros(T + 1)
    consumption_path = np.zeros(T)
    
    cake_path[0] = initial_cake
    
    for t in range(T):
        # Find optimal consumption at current cake size
        consumption_path[t] = np.interp(cake_path[t], w_grid, policy)
        cake_path[t + 1] = max(0, cake_path[t] - consumption_path[t])
        
        # Stop if cake is essentially gone
        if cake_path[t + 1] < 1e-10:
            cake_path = cake_path[:t+2]
            consumption_path = consumption_path[:t+1]
            break
    
    return cake_path, consumption_path

# Example usage
gamma = 2.0  # CRRA parameter
beta = 0.95  # Discount factor
initial_cake = 1.0

# Solve the infinite horizon problem
w_grid, policy, value = solve_infinite_cake_eating(gamma, beta, initial_cake, grid_size=500)

# Simulate consumption path
cake_path, consumption_path = simulate_infinite_consumption(policy, w_grid, initial_cake, T=50)

# Plotting
fig, axes  = plt.subplots(2, 2, figsize=(15, 12))

(ax1, ax2, ax3, ax4) = axes.flatten()

# Plot value function
ax1.plot(w_grid, value, 'g-', linewidth=2)
ax1.set_xlabel('Cake Size')
ax1.set_ylabel('Value')
ax1.set_title('Value Function')
ax1.grid(True)

# Plot policy function
ax2.plot(w_grid, policy, 'b-', linewidth=2, label='Numerical Solution')
ax2.plot([0, initial_cake], [0, initial_cake], 'k--', alpha=0.5, label='45° line')
ax2.set_xlabel('Cake Size')
ax2.set_ylabel('Optimal Consumption')
ax2.set_title(f'Policy Function (γ={gamma}, β={beta})')
ax2.legend()
ax2.grid(True)

# Plot cake size over time
ax3.plot(cake_path, 'b-', marker='o')
ax3.set_xlabel('Time')
ax3.set_ylabel('Cake Size')
ax3.set_title('Cake Size Over Time')
ax3.grid(True)

# Plot consumption over time
ax4.plot(consumption_path, 'r-', marker='o')
ax4.set_xlabel('Time')
ax4.set_ylabel('Consumption')
ax4.set_title('Optimal Consumption Over Time')
ax4.grid(True)

plt.tight_layout()
plt.show()

# Analyze the policy function
print("\nPolicy Function Analysis:")
print(f"Consumption at w=1.0: {np.interp(1.0, w_grid, policy):.4f}")
print(f"Consumption at w=0.5: {np.interp(0.5, w_grid, policy):.4f}")
print(f"Consumption at w=0.1: {np.interp(0.1, w_grid, policy):.4f}")

# Calculate consumption rate (c/w) - avoid division by zero
consumption_rate = np.zeros_like(policy)
mask = w_grid > 1e-10
consumption_rate[mask] = policy[mask] / w_grid[mask]

plt.figure(figsize=(10, 6))
valid_mask = ~np.isnan(consumption_rate)
plt.plot(w_grid[valid_mask], consumption_rate[valid_mask], 'g-', linewidth=2)
plt.xlabel('Cake Size')
plt.ylabel('Consumption Rate (c/w)')
plt.title(f'Consumption Rate Policy (γ={gamma}, β={beta})')
plt.grid(True)

# Add analytical consumption rate for log utility
if abs(gamma - 1) < 1e-6:
    plt.axhline(y=(1-beta), color='r', linestyle='--', label=f'Analytical: 1-β = {1-beta}')
    plt.legend()

plt.show()

print(f"\nFor log utility (γ=1), the optimal consumption rate is constant: c/w = {1-beta}")

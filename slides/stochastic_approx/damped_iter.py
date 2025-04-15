import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm, eig

# Define the matrix A
A = np.array([[0.5, 0.6], [0.4, -0.7]])

# Display the spectral radius
eigenvalues = eig(A)[0]  # Extract eigenvalues
spectral_radius = max(abs(eigenvalues))
print(f'Spectral radius of matrix A: {spectral_radius:.6f}')

# Set a random initial vector 
np.random.seed(42)  
x0 = np.random.rand(2)
print(f'Initial vector: [{x0[0]:.4f}, {x0[1]:.4f}]\n')

num_iterations = 20

# ---- Standard iteration ---- #

# Store all vectors for plotting 
X = np.zeros((num_iterations+1, 2))  # Each row is a vector [x1, x2]
X[0, :] = x0
norms = np.zeros(num_iterations+1)
norms[0] = norm(x0)

# Perform iterations: x_{k+1} = A * x_k
for k in range(num_iterations):
    X[k+1] = A @ X[k]  # Matrix multiplication with flat array
    norms[k+1] = norm(X[k+1])
    

# ---- Damped iteration ---- #

omega = 0.7  # Damping parameter between 0 and 1
X_damped = np.zeros((num_iterations+1, 2))  # Each row is a vector [x1, x2]
X_damped[0] = x0
norms_damped = np.zeros(num_iterations+1)
norms_damped[0] = norm(x0)

# Perform damped iterations: x_{k+1} = ω*A*x_k + (1-ω)*x_k
for k in range(num_iterations):
    X_damped[k+1] = omega * (A @ X_damped[k]) + (1-omega) * X_damped[k]
    norms_damped[k+1] = norm(X_damped[k+1])
    
# --- Figures --- #

# Subplot 1: Vector trajectory (Standard)
fig, ax = plt.subplots()
ax.plot(X[:, 0], X[:, 1], 'b.-', markersize=8)
ax.plot(X[0, 0], X[0, 1], 'go', markersize=8, linewidth=2)  # Initial point
ax.plot(0, 0, 'mo', markersize=8, linewidth=2)  # Origin
ax.set_xlabel('$\\theta_1$')
ax.set_ylabel('$\\theta_2$')
ax.set_title('Standard iteration: trajectory')
ax.legend(['trajectory', 'initial point', 'fixed point (origin)'], frameon=False)
plt.savefig('damped_1.pdf')
plt.show()

# Subplot 3: Vector trajectory (Damped)
fig, ax = plt.subplots()
ax.plot(X_damped[:, 0], X_damped[:, 1], 'b.-', markersize=8)
ax.plot(X_damped[0, 0], X_damped[0, 1], 'go', markersize=8, linewidth=2)  # Initial point
ax.plot(0, 0, 'mo', markersize=8, linewidth=2)  # Origin
ax.set_xlabel('$\\theta_1$')
ax.set_ylabel('$\\theta_2$')
ax.set_title(f'Damped iteration: trajectory')
ax.legend(['trajectory', 'initial point', 'fixed point (origin)'], frameon=False)
plt.savefig('damped_2.pdf')
plt.show()

# Subplot 4: Norm vs iteration number (Damped)
fig, ax = plt.subplots()
ax.semilogy(range(num_iterations+1), norms_damped, markersize=8)
ax.set_xlabel('iteration')
ax.set_ylabel('norm (log scale)')
ax.set_title('Error (norm of current iterate)')
ax.semilogy(range(num_iterations+1), norms, alpha=0.6)
ax.legend(['damped iteration', 'standard iteration'], frameon=False)
plt.savefig('damped_3.pdf')
plt.show()

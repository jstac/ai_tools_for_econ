NOTE:

Remove brute force.  Review HPI and make sure it's understandable.

Potentially use this as an exercise.

# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Inventory Management Model: Vectorization Practice
#
# This notebook demonstrates a stochastic dynamic inventory management model and compares different computational approaches for calculating transition probabilities.
#
# ## Problem Overview
#
# We have an inventory system with:
#
# - $K$: Maximum inventory capacity
# - $\beta$: Discount factor
# - $c$: Marginal cost (unit cost per order)
# - $\kappa$: Fixed cost of ordering
# - $p$: Parameter for demand shock distribution
#
# Inventory evolves according to 
#
# $$
#     X_{t+1} = \max(X_t - D_{t+1}, 0) + A_t
# $$
#
# where
#
# - $X_t$ is current inventory (number of units),
# - $D_{t+1}$ is an IID demand shock, and
# - $A_t$ is the current order (number of units).
#
# It will be convenient to work with the transition probability kernel
#
# $$P(x, a, y) := \mathbb P\{X_{t+1}=y \,|\, X_t = x, A_t = a \}$$
#
# With $\phi$ as the probability density function for demand,
# the transition probability kernel obeys
#
# \begin{align}
# P(x, a, y) &= \sum_{d \geq 0} \mathbb{1}\{\max(x - d, 0) + a = y\} \phi(d) \\
# &= \sum_{d < x} \mathbb{1}\{x - d + a = y\} \phi(d) + \sum_{d \geq x} \mathbb{1}\{a = y\} \phi(d) \\
# &= \sum_{d < x} \mathbb{1}\{d = x + a - y\} \phi(d) + \mathbb{1}\{y = a\} F(x) \\
# &= \mathbb{1}\{0 \leq x + a - y < x\} \phi(x + a - y) + \mathbb{1}\{y = a\} F(x)
# \end{align}
#
# Where $F(x) = P\{D \geq x\}$ is the survival function.
#

import jax
import jax.numpy as jnp
import numpy as np
from typing import NamedTuple
from functools import partial
import matplotlib.pyplot as plt


class Model(NamedTuple):
    K: int = 5       # max inventory
    β: float = 0.97  # discount factor
    c: float = 0.2   # marginal cost
    κ: float = 0.8   # fixed cost
    p: float = 0.6   # demand shock parameter


def ϕ(p, d):
    """PDF for demand shock: ϕ(d) = (1-p)^d * p"""
    return (1 - p)**d * p


def F(p, x):
    """Survival function: F(x) = P{D ≥ x} = (1-p)^x"""
    return (1 - p)**x


@partial(jax.jit, static_argnums=(0,))
def P_function(model, x, a, y):
    """
    Scalar function to compute P(x, a, y) for a single (x, a, y) triple.
    See the mathematical derivation above for the complete formula.

    """
    K, β, c, κ, p = model
    d = x + a - y
    # Test 0 <= x + a - y < x 
    mask = jnp.logical_and(0 <= d, d < x)
    # Compute 1{0 ≤ x + a - y < x} φ(x + a - y) + 1{y = a} F(x)
    return mask * ϕ(p, d) + (y == a) * F(p, x)


@partial(jax.jit, static_argnums=(0,))
def P_array(model): 
    """
    Vmap-based computation of the transition probability kernel P(x, a, y).
    Uses JAX's vmap to vectorize the scalar function over all (x, a, y) combinations.
    """
    K, β, c, κ, p = model
    S = K + 1

    # Create all combinations of (x, a, y) indices
    x_vals = jnp.arange(S)
    a_vals = jnp.arange(S)  
    y_vals = jnp.arange(S)
    
    # Use vmap to compute P(x,a,y) for all combinations
    P_vmap_y   = jax.vmap(P_function, (None, None, None, 0))
    P_vmap_ay  = jax.vmap(P_vmap_y,   (None, None, 0, None))
    P_vmap_xay = jax.vmap(P_vmap_ay,  (None, 0, None, None))
    
    return P_vmap_xay(model, x_vals, a_vals, y_vals)


def reward_function(model, x, a):
    """
    The flow (expected) reward function

        r(x, a) = Σ_{d >= 0} min(x, d) ϕ(d) - ca - κ (a > 0)

    We use 

        Σ_{d >= 0} min(x, d) = Σ_{d < x} d ϕ(d) + Σ_{d >= x} x ϕ(d)
                             = Σ_{d < x} d ϕ(d) + x F(x)
                             = (1-p)/p * (1 - (1 + x*p)*(1-p)**x) + x*(1-p)**x

    Derivation of Σ_{d < x} d ϕ(d):
    
        Σ_{d=0}^{x-1} d * (1-p)^d * p = p * Σ_{d=0}^{x-1} d * (1-p)^d

    For the finite geometric series Σ_{d=0}^{x-1} d * r^d with r = (1-p), we use:
    
        Σ_{d=0}^{x-1} d * r^d = r/(1-r)² - r^x * [x/(1-r) - (x-1)r/(1-r)²]
        
    Substituting r = (1-p):
    
        Σ_{d=0}^{x-1} d * (1-p)^d = (1-p)/p² - (1-p)^x * [x/p - (x-1)(1-p)/p²]
        
    Multiplying by p and simplifying:
    
        p * Σ_{d=0}^{x-1} d * (1-p)^d = (1-p)/p * (1 - (1 + x*p)*(1-p)^x)
        
    This represents the expected sales revenue when holding x units of inventory.
    """
    K, β, c, κ, p = model
    expected_sales = (1-p)/p * (1 - (1 + x*p)*(1-p)**x) + x*(1-p)**x
    return expected_sales - c * a - jnp.greater(a, 0) * κ 


@partial(jax.jit, static_argnums=(0,))
def reward_array(model): 
    """
    Vmap-based computation of the reward array r(x, a).

    """
    K, β, c, κ, p = model
    S = K + 1

    # Create all combinations of (x, a) indices
    x_vals = jnp.arange(S)
    a_vals = jnp.arange(S)  
    
    # Use vmap to compute r(x,a) for all combinations
    r_vmap_a   = jax.vmap(reward_function, (None, None, 0))
    r_vmap_xa  = jax.vmap(r_vmap_a,        (None, 0,    None))
    
    return r_vmap_xa(model, x_vals, a_vals)


@partial(jax.jit, static_argnums=(0,))
def T(model, P, r, v):
    K, β, c, κ, p = model
    B = r + β * jnp.sum(P * v, axis=2)
    return jnp.max(B, axis=1)


def vfi(model, max_iter=10_000, tol=1e-6):
    K, β, c, κ, p = model
    P = P_array(model)
    r = reward_array(model)
    error = tol + 1
    i = 0
    v = jnp.zeros(K+1)

    while i < max_iter and error > tol:
        new_v = T(model, P, r, v)
        error = jnp.max(jnp.abs(new_v - v))
        v = new_v

    return v


@partial(jax.jit, static_argnums=(0,))
def policy_evaluation(model, P, r, policy):
    """
    Policy evaluation: solve (I - β P_σ) v = r_σ for value function v
    where P_σ and r_σ are transition matrix and rewards under policy σ
    
    Solves the linear system directly using matrix inversion
    """
    K, β, c, κ, p = model
    S = K + 1
    
    # Extract transition probabilities and rewards for the given policy
    P_policy = P[jnp.arange(S), policy, :]  # Shape: (S, S)
    r_policy = r[jnp.arange(S), policy]     # Shape: (S,)
    
    # Solve (I - β P_σ) v = r_σ directly
    A = jnp.eye(S) - β * P_policy
    v = jnp.linalg.solve(A, r_policy)
    
    return v


@partial(jax.jit, static_argnums=(0,))
def policy_improvement(model, P, r, v):
    """
    Policy improvement: compute greedy policy with respect to value function v
    Returns new policy σ'(x) = argmax_a [r(x,a) + β Σ_y P(x,a,y) v(y)]
    """
    K, β, c, κ, p = model
    
    # Compute Q(x,a) = r(x,a) + β Σ_y P(x,a,y) v(y)
    Q = r + β * jnp.sum(P * v, axis=2)
    
    # Return greedy policy
    return jnp.argmax(Q, axis=1)


def howard_policy_iteration(model, max_iter=1000, tol=1e-6):
    """
    Howard's policy iteration algorithm.
    Alternates between policy evaluation and policy improvement until convergence.
    """
    K, β, c, κ, p = model
    S = K + 1
    P = P_array(model)
    r = reward_array(model)
    
    # Initialize with zero policy (order nothing)
    policy = jnp.zeros(S, dtype=int)
    
    for i in range(max_iter):
        # Policy evaluation
        v = policy_evaluation(model, P, r, policy)
        
        # Policy improvement
        new_policy = policy_improvement(model, P, r, v)
        
        # Check for convergence
        if jnp.array_equal(policy, new_policy):
            return v, new_policy
        
        policy = new_policy
    
    return v, policy


def get_optimal_policy(model, v):
    """
    Extract the optimal policy from a value function using policy improvement.
    """
    P = P_array(model)
    r = reward_array(model)
    return policy_improvement(model, P, r, v)


def generate_all_policies_array(model):
    """
    Generate all possible policies as a JAX array for vectorized computation.
    """
    K, β, c, κ, p = model
    S = K + 1
    
    # Generate all combinations using JAX operations
    # Each policy is S integers, each from 0 to S-1
    total_policies = S ** S
    
    policies = jnp.zeros((total_policies, S), dtype=int)
    
    # Generate policies by treating them as base-S numbers
    for i in range(total_policies):
        policy = []
        num = i
        for _ in range(S):
            policy.append(num % S)
            num //= S
        policies = policies.at[i].set(jnp.array(policy))
    
    return policies


@partial(jax.jit, static_argnums=(0,))
def evaluate_all_policies(model, P, r, policies):
    """
    Vectorized evaluation of all policies using vmap.
    """
    def evaluate_single_policy(policy):
        return policy_evaluation(model, P, r, policy)[0]  # Return only v(0)
    
    # Vectorize over all policies
    return jax.vmap(evaluate_single_policy)(policies)


def brute_force_optimal_policy(model):
    """
    Find optimal policy by brute force using vectorized JAX operations.
    """
    P = P_array(model)
    r = reward_array(model)
    
    print(f"Generating all {(model.K + 1)**(model.K + 1)} possible policies...")
    all_policies = generate_all_policies_array(model)
    
    print("Evaluating all policies with vectorized computation...")
    all_v0_values = evaluate_all_policies(model, P, r, all_policies)
    
    # Find the policy with maximum v(0)
    best_idx = jnp.argmax(all_v0_values)
    best_policy = all_policies[best_idx]
    best_value_at_0 = all_v0_values[best_idx]
    
    print(f"Completed brute force search. Best v(0): {best_value_at_0:.6f}")
    return best_policy, best_value_at_0


model = Model()

P = P_array(model)
r = reward_array(model)

# Solve using value function iteration
print("Solving with Value Function Iteration...")
v_vfi = vfi(model)
policy_vfi = get_optimal_policy(model, v_vfi)

# Solve using Howard's policy iteration
print("Solving with Howard's Policy Iteration...")
v_hpi, policy_hpi = howard_policy_iteration(model)

# Solve using brute force
print("\nSolving with Brute Force...")
policy_bf, v0_bf = brute_force_optimal_policy(model)

# Compare results
print("\nComparison of methods:")
print(f"VFI vs HPI - Value functions equal: {jnp.allclose(v_vfi, v_hpi, atol=1e-6)}")
print(f"VFI vs HPI - Policies equal: {jnp.array_equal(policy_vfi, policy_hpi)}")
print(f"VFI vs HPI - Max value function difference: {jnp.max(jnp.abs(v_vfi - v_hpi))}")

print(f"\nBrute Force vs VFI - Policies equal: {jnp.array_equal(policy_bf, policy_vfi)}")
print(f"Brute Force v(0): {v0_bf:.6f}")
print(f"VFI v(0): {v_vfi[0]:.6f}")

print("\nOptimal Policy (inventory level -> order amount):")
for x in range(model.K + 1):
    print(f"  x={x}: order {policy_vfi[x]} units")

v = v_vfi

# Plot value function
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(range(model.K + 1), v, 'o-', linewidth=2, markersize=6)
ax.set_xlabel('Inventory Level')
ax.set_ylabel('Value Function')
ax.set_title('Value Function vs Inventory Level')
plt.show()

# Plot policies
fig, ax = plt.subplots(figsize=(8, 6))
x_vals = range(model.K + 1)
ax.plot(x_vals, policy_vfi, 'o-', linewidth=2, markersize=6, label='VFI Policy')
ax.plot(x_vals, policy_hpi, '--s', linewidth=2, markersize=4, label='HPI Policy')
ax.plot(x_vals, policy_bf, ':^', linewidth=2, markersize=4, label='Brute Force Policy')
ax.set_xlabel('Inventory Level')
ax.set_ylabel('Order Amount')
ax.set_title('Optimal Policy Comparison - All Methods')
ax.legend()
ax.set_xticks(x_vals)
ax.set_yticks(range(model.K + 1))
plt.show()



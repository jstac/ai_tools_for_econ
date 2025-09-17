---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Inventory Management Model: Vectorization Practice

*Prepared for the Computational Economics Workshop at Hitotsubashi*

Author: [John Stachurski](https://johnstachurski.net)

This notebook demonstrates a stochastic dynamic inventory management model and compares different computational approaches for calculating transition probabilities.

## Problem Overview

We have an inventory system with:

- $K$: Maximum inventory capacity
- $p$: Parameter for demand shock distribution

Inventory evolves according to 

$$
    X_{t+1} = \max(X_t - D_{t+1}, 0) + A_t
$$

where

- $X_t$ is current inventory (number of units),
- $D_{t+1}$ is an IID demand shock, and
- $A_t$ is the current order (number of units).

We are interested in computing the transition probability kernel

$$P(x, a, y) := \mathbb P\{X_{t+1}=y \,|\, X_t = x, A_t = a \}$$

More explicitly,

$$P(x, a, y) = \sum_{d \geq 0} \mathbb{1}\{\max(x - d, 0) + a = y\} \phi(d)$$

Here

- $d$ is the demand shock
- $\phi$ is the probability density function for demand


## Mathematical Derivation

The transition probability kernel obeys

\begin{align}
P(x, a, y) &= \sum_{d \geq 0} \mathbb{1}\{\max(x - d, 0) + a = y\} \phi(d) \\
&= \sum_{d < x} \mathbb{1}\{x - d + a = y\} \phi(d) + \sum_{d \geq x} \mathbb{1}\{a = y\} \phi(d) \\
&= \sum_{d < x} \mathbb{1}\{d = x + a - y\} \phi(d) + \mathbb{1}\{y = a\} F(x) \\
&= \mathbb{1}\{0 \leq x + a - y < x\} \phi(x + a - y) + \mathbb{1}\{y = a\} F(x)
\end{align}

Where $F(x) = P\{D \geq x\}$ is the survival function.

+++

## Implementation Approaches

We'll compare three different computational approaches:
1. **Loop-based** (Numba JIT compiled)
2. **Vectorized** (JAX vectorized operations)
3. **Vmap** (JAX's functional transformation)

```{code-cell} ipython3
import numba
import jax
import jax.numpy as jnp
import numpy as np
import time
from typing import NamedTuple
import matplotlib.pyplot as plt

np.set_printoptions(precision=6, suppress=True)
```

```{code-cell} ipython3
class Model(NamedTuple):
    K: int = 50     # max inventory
    p: float = 0.6  # demand shock parameter


def ϕ(p, d):
    """PDF for demand shock: ϕ(d) = (1-p)^d * p"""
    return (1 - p)**d * p


def F(p, x):
    """Survival function: F(x) = P{D ≥ x} = (1-p)^x"""
    return (1 - p)**x
```

Let's create a model instance.

```{code-cell} ipython3
model = Model()
print(f"Created {model}")
print(f"Demand PDF: ϕ(d) = (1-p)^d * p = (1-{model.p})^d * {model.p}")
print(f"Survival function: F(x) = (1-p)^x = (1-{model.p})^x")
```

## Method 1: Loop-based Implementation (Numba)

This approach uses traditional nested loops with Numba JIT compilation for speed.

```{code-cell} ipython3
@numba.jit
def generate_kernel_loops(model):
    """
    Loop-based computation of the transition probability kernel P(x, a, y).
    See the mathematical derivation above for the complete formula.
    """
    K, p = model
    S = K + 1
    P = np.zeros((S, S, S))

    def ϕ(d):
        return (1 - p)**d * p

    def F(x):
        return (1 - p)**x

    for x in range(S):
        for a in range(S):
            for y in range(S):
                # implement 1{0 ≤ x + a - y < x} φ(x + a - y) + 1{y = a} F(x)
                phi_value = (0 <= x + a - y < x) * ϕ(x + a - y) 
                f_value = (y == a) * F(x)
                P[x, a, y] = phi_value + f_value
    return P
```

## Method 2: Vectorized Implementation (JAX)

This approach uses JAX's vectorized operations to compute all probabilities simultaneously.

```{code-cell} ipython3
def generate_kernel_vectorized(model):
    """
    Fully vectorized JAX-based computation of the transition probability kernel P(x, a, y).
    See the mathematical derivation above for the complete formula.
    """
    K, p = model
    S = K + 1
    
    # Create meshgrids for vectorized computation
    x_grid, a_grid, y_grid = jnp.meshgrid(
        jnp.arange(S), jnp.arange(S), jnp.arange(S), indexing='ij'
    )
    
    # Initialize probability tensor
    P = jnp.zeros((S, S, S))
    
    # Vectorized computation of the first term: 1{0 ≤ x + a - y < x} φ(x + a - y)
    d_candidate = x_grid + a_grid - y_grid
    valid_d = jnp.logical_and(d_candidate >= 0, d_candidate < x_grid) 
    phi_values = valid_d * ϕ(p, d_candidate)
    
    # Second term: I{y = a} F(x)
    y_eq_a = jnp.equal(y_grid, a_grid)
    f_values = y_eq_a * F(p, x_grid) 
  
    # Combine both terms
    P = phi_values + f_values
    
    return P
```

## Method 3: Vmap Implementation (JAX)

This approach uses JAX's `vmap` (vectorized map) to transform a scalar function into a vectorized one.

```{code-cell} ipython3
def generate_kernel_vmap(model): 
    """
    Vmap-based computation of the transition probability kernel P(x, a, y).
    Uses JAX's vmap to vectorize the scalar function over all (x, a, y) combinations.
    """
    K, p = model
    S = K + 1

    def P(x, a, y):
        """
        Scalar function to compute P(x, a, y) for a single (x, a, y) triple.
        See the mathematical derivation above for the complete formula.
        """
        d = x + a - y
        # Test 0 <= x + a - y < x (first term)
        valid_d = jnp.logical_and(0 <= d, d < x)
        # Test y = a (second term)
        y_eq_a = jnp.equal(y, a)
        # Combine: 1{0 ≤ x + a - y < x} φ(x + a - y) + 1{y = a} F(x)
        return valid_d * ϕ(p, d) + y_eq_a * F(p, x)

    # Create all combinations of (x, a, y) indices
    x_vals = jnp.arange(S)
    a_vals = jnp.arange(S)  
    y_vals = jnp.arange(S)
    
    # Use vmap to compute P(x,a,y) for all combinations
    vmap_y = jax.vmap(P,      (None, None, 0))
    vmap_a = jax.vmap(vmap_y, (None, 0, None))
    vmap_x = jax.vmap(vmap_a, (0, None, None))
    
    return vmap_x(x_vals, a_vals, y_vals)
```

## Correctness Verification

Let's verify that all three methods produce identical results.

```{code-cell} ipython3
# Compute results using all three methods
loops_P = generate_kernel_loops(model)
vectorized_P = generate_kernel_vectorized(model)
vmap_P = generate_kernel_vmap(model)

print("=== Correctness Check ===")
print(f"Vectorized P equals loops P: {np.allclose(loops_P, vectorized_P)}")
print(f"Vmap P equals loops P: {np.allclose(loops_P, vmap_P)}")
print()

# Show some sample probabilities
print("Sample transition probabilities P(x=2, a=1, y):")
for y in range(min(6, model.K + 1)):
    prob = loops_P[2, 1, y]
    print(f"P(2, 1, {y}) = {prob:.6f}")
print()

# Verify that probabilities sum to 1 for each (x, a) pair
prob_sums = np.sum(loops_P, axis=2)
print(f"All probability sums equal 1: {np.allclose(prob_sums, 1.0)}")
print(f"Max deviation from 1: {np.max(np.abs(prob_sums - 1.0)):.2e}")
```

## Performance Benchmarking

Now let's compare the performance of all three approaches.

```{code-cell} ipython3
# Create JIT-compiled versions
gen_kernel_vectorized_jit = jax.jit(generate_kernel_vectorized, static_argnums=(0,))
gen_kernel_vmapped_jit = jax.jit(generate_kernel_vmap, static_argnums=(0,))

# Warm up JIT compilation
print("Warming up JIT compilation...")
_ = generate_kernel_loops(model)
_ = gen_kernel_vectorized_jit(model)
_ = gen_kernel_vmapped_jit(model)

print("Warmup complete.")
print()
```

```{code-cell} ipython3
# Run performance benchmarks
print("Running performance benchmarks (100 iterations each)...")
print()

times_loops_jit = []
times_vec_jit = []
times_vmap_jit = []

n_iterations = 1000

for i in range(n_iterations):
    if (i + 1) % 100 == 0:
        print(f"Progress: {i + 1}/{n_iterations}")
    
    # Benchmark loops
    start = time.perf_counter()
    _ = generate_kernel_loops(model)
    end = time.perf_counter()
    times_loops_jit.append(end - start)

    # Benchmark vectorized
    start = time.perf_counter()
    _ = gen_kernel_vectorized_jit(model).block_until_ready()
    end = time.perf_counter()
    times_vec_jit.append(end - start)
    
    # Benchmark vmap
    start = time.perf_counter()
    _ = gen_kernel_vmapped_jit(model).block_until_ready()
    end = time.perf_counter()
    times_vmap_jit.append(end - start)

print("\n=== Performance Results ===")
print(f"Loops (Numba):  {np.mean(times_loops_jit)*1000000:.1f} ± {np.std(times_loops_jit)*1000000:.1f} μs")
print(f"Vectorized:     {np.mean(times_vec_jit)*1000000:.1f} ± {np.std(times_vec_jit)*1000000:.1f} μs")
print(f"Vmap:           {np.mean(times_vmap_jit)*1000000:.1f} ± {np.std(times_vmap_jit)*1000000:.1f} μs")
print()

# Calculate speedups
loops_mean = np.mean(times_loops_jit)
vec_mean = np.mean(times_vec_jit) 
vmap_mean = np.mean(times_vmap_jit)

print("=== Relative Performance ===")
print(f"Vectorized vs Loops: {loops_mean / vec_mean:.1f}x {'faster' if vec_mean < loops_mean else 'slower'}")
print(f"Vmap vs Loops:       {loops_mean / vmap_mean:.1f}x {'faster' if vmap_mean < loops_mean else 'slower'}")
print(f"Vmap vs Vectorized:  {vec_mean / vmap_mean:.1f}x {'faster' if vmap_mean < vec_mean else 'slower'}")
```

## Summary and Conclusions

This notebook demonstrated three different approaches to computing transition probabilities in a stochastic inventory model:

1. **Loop-based with Numba**: Traditional nested loops with JIT compilation
2. **Vectorized JAX**: Fully vectorized operations using meshgrids
3. **Vmap JAX**: Functional transformation of scalar operations

### Key Findings:

- All three methods produce identical results, confirming correctness
- The JAX-based approaches are faster.

```{code-cell} ipython3

```

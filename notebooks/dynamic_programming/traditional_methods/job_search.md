---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.7
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Job Search

Note: This lecture was built using a machine with JAX installed and access to a
GPU.

To run this lecture on [Google Colab](https://colab.research.google.com/), click
on the “play” icon top right, select Colab, and set the runtime environment to
include a GPU.


## Introduction

In this lecture we study a basic infinite-horizon job search problem with Markov
wage draws

(For background on infinite horizon job search see, e.g., [DP1](https://dp.quantecon.org/).)

The exercise at the end asks you to add risk-sensitive preferences and see how
the main results change.

In addition to what’s in Anaconda, this lecture will need the following libraries:

```{code-cell} ipython3
:hide-output: false

!pip install quantecon
```

We use the following imports.

```{code-cell} ipython3
:hide-output: false

import matplotlib.pyplot as plt
import quantecon as qe
import jax
import jax.numpy as jnp
from typing import NamedTuple

jax.config.update("jax_enable_x64", True)
```

## Model

We study an elementary model where

- jobs are permanent  
- unemployed workers receive current compensation $ c $  
- the horizon is infinite  
- an unemployment agent discounts the future via discount factor $ \beta \in (0,1) $

+++

### Set up

At the start of each period, an unemployed worker receives wage offer $ W_t $.

To build a wage offer process we consider the dynamics

$$
    W_{t+1} = \rho W_t + \nu Z_{t+1}
$$

where $ (Z_t)_{t \geq 0} $ is IID and standard normal.

We then discretize this wage process using Tauchen’s method to produce a stochastic matrix $ P $.

Successive wage offers are drawn from $ P $.

+++

### Rewards

Since jobs are permanent, the return to accepting wage offer $ w $ today is

$$
w + \beta w + \beta^2 w + 
    \cdots = \frac{w}{1-\beta}
$$

The Bellman equation is

$$
v(w) = \max
    \left\{
            \frac{w}{1-\beta}, c + \beta \sum_{w'} v(w') P(w, w')
    \right\}
$$

We solve this model using value function iteration.

+++

## Code

Let’s set up a type to store information needed to solve the model.

```{code-cell} ipython3
class Model(NamedTuple): 
    n: int
    w_vals: jnp.ndarray
    P: jnp.ndarray
    β: float
    c: float
```

The function below fixes default values and generates an instance.

```{code-cell} ipython3
def create_js_model(
        n=500,       # wage grid size
        ρ=0.9,       # wage persistence
        ν=0.2,       # wage volatility
        β=0.99,      # discount factor
        c=1.0,       # unemployment compensation
    ):
    "Creates an instance of the job search model with Markov wages."
    mc = qe.tauchen(n, ρ, ν)
    w_vals, P = jnp.exp(mc.state_values), jnp.array(mc.P)
    return Model(n=n, w_vals=w_vals, P=P, β=β, c=c)
```

Let’s test it:

```{code-cell} ipython3
model = create_js_model(β=0.98)
```

```{code-cell} ipython3
:hide-output: false

model.β, model.c
```

```{code-cell} ipython3
:hide-output: false

model.w_vals.mean()
```

Next we implement the Bellman operator, which has the form

$$
    (Tv)(w) 
    = \max
    \left\{
            \frac{w}{1-\beta}, c + \beta \sum_{w'} v(w') P(w, w')
    \right\}
$$

```{code-cell} ipython3
:hide-output: false

@jax.jit
def T(v: jnp.ndarray, model: Model) -> jnp.ndarray:
    "Bellman operator"
    n, w_vals, P, β, c = model
    h = c + β * P @ v
    e = w_vals / (1 - β)

    return jnp.maximum(e, h)
```

The next function computes the optimal policy under the assumption that $ v $ is
the value function.

The policy takes the form

$$
\sigma(w) = \mathbf 1 
        \left\{
            \frac{w}{1-\beta} \geq c + \beta \sum_{w'} v(w') P(w, w')
        \right\}
$$

Here $ \mathbf 1 $ is an indicator function.

- $ \sigma(w) = 1 $ means stop  
- $ \sigma(w) = 0 $ means continue.

```{code-cell} ipython3
:hide-output: false

@jax.jit
def get_greedy(v: jnp.ndarray, model: Model) -> jnp.ndarray:
    "Get a v-greedy policy."
    n, w_vals, P, β, c = model
    e = w_vals / (1 - β)
    h = c + β * P @ v
    σ = jnp.where(e >= h, 1, 0)
    return σ
```

Here’s a routine for value function iteration.

```{code-cell} ipython3
:hide-output: false

def vfi(model: Model, 
        max_iter: int = 10_000, 
        tol: float = 1e-4
    ) -> tuple[jnp.ndarray]:
    "Solve the infinite-horizon Markov job search model by VFI."
    print("Starting VFI iteration.")
    v = jnp.zeros_like(model.w_vals)    # Initial guess
    i = 0
    error = tol + 1

    while error > tol and i < max_iter:
        new_v = T(v, model)
        error = jnp.max(jnp.abs(new_v - v))
        i += 1
        v = new_v

    v_star = v
    σ_star = get_greedy(v_star, model)
    return v_star, σ_star
```

## Computing the solution

Let’s set up and solve the model.

```{code-cell} ipython3
:hide-output: false

model = create_js_model()
n, w_vals, P, β, c = model

v_star, σ_star = vfi(model)
```

Here’s the optimal policy:

```{code-cell} ipython3
:hide-output: false

fig, ax = plt.subplots()
ax.plot(σ_star)
ax.set_xlabel("wage values")
ax.set_ylabel("optimal choice (stop=1)")
plt.show()
```

We compute the reservation wage as the first $ w $ such that $ \sigma(w)=1 $.

```{code-cell} ipython3
:hide-output: false

stop_indices = jnp.where(σ_star == 1)
stop_indices
```

```{code-cell} ipython3
:hide-output: false

res_wage_index = min(stop_indices[0])
```

```{code-cell} ipython3
:hide-output: false

res_wage = w_vals[res_wage_index]
```

Here’s a joint plot of the value function and the reservation wage.

```{code-cell} ipython3
:hide-output: false

fig, ax = plt.subplots()
ax.plot(w_vals, v_star, alpha=0.8, label="value function")
ax.vlines((res_wage,), 150, 400, 'k', ls='--', label="reservation wage")
ax.legend(frameon=False, fontsize=12, loc="lower right")
ax.set_xlabel("$w$", fontsize=12)
plt.show()
```

## Exercise

+++

## Exercise 10.1

In the setting above, the agent is risk-neutral vis-a-vis future utility risk.

Now solve the same problem but this time assuming that the agent has risk-sensitive
preferences, which are a type of nonlinear recursive preferences.

The Bellman equation becomes

$$
v(w) = \max
    \left\{
            \frac{w}{1-\beta}, 
            c + \frac{\beta}{\theta}
            \ln \left[ 
                      \sum_{w'} \exp(\theta v(w')) P(w, w')
                \right]
    \right\}
$$

When $ \theta < 0 $ the agent is risk averse.

Solve the model when $ \theta = -0.1 $ and compare your result to the risk neutral
case.

Try to interpret your result.

You can start with the following code:

```{code-cell} ipython3
class RiskModel(NamedTuple): 
    n: int
    w_vals: jnp.ndarray
    P: jnp.ndarray
    β: float
    c: float
    θ: float

def create_risk_sensitive_js_model(
        n=500,       # wage grid size
        ρ=0.9,       # wage persistence
        ν=0.2,       # wage volatility
        β=0.99,      # discount factor
        c=1.0,       # unemployment compensation
        θ=-0.1       # risk parameter
    ):
    "Creates an instance of the job search model with Markov wages."
    mc = qe.tauchen(n, ρ, ν)
    w_vals, P = jnp.exp(mc.state_values), mc.P
    P = jnp.array(P)
    return RiskModel(n, w_vals, P, β, c, θ)
```

Now you need to modify `T` and `get_greedy` and then run value function iteration again.

+++

## Solution to[ Exercise 10.1](https://jax.quantecon.org/#job_search_1)

```{code-cell} ipython3
@jax.jit
def T_rs(v, model):
    """
    The Bellman operator Tv = max{e, c + β R v} with 

        e(w) = w / (1-β) and

        (Rv)(w) = (1/θ) ln{E_w[ exp(θ v(W'))]}

    """
    n, w_vals, P, β, c, θ = model
    h = c + (β / θ) * jnp.log(P @ (jnp.exp(θ * v)))
    e = w_vals / (1 - β)

    return jnp.maximum(e, h)


@jax.jit
def get_greedy_rs(v, model):
    " Get a v-greedy policy."
    n, w_vals, P, β, c, θ = model
    e = w_vals / (1 - β)
    h = c + (β / θ) * jnp.log(P @ (jnp.exp(θ * v)))
    σ = jnp.where(e >= h, 1, 0)
    return σ



def vfi(model, max_iter=10_000, tol=1e-4):
    "Solve the infinite-horizon Markov job search model by VFI."
    print("Starting VFI iteration.")
    v = jnp.zeros_like(model.w_vals)    # Initial guess
    i = 0
    error = tol + 1

    while error > tol and i < max_iter:
        new_v = T_rs(v, model)
        error = jnp.max(jnp.abs(new_v - v))
        i += 1
        v = new_v

    v_star = v
    σ_star = get_greedy_rs(v_star, model)
    return v_star, σ_star



model_rs = create_risk_sensitive_js_model()

n, w_vals, P, β, c, θ = model_rs

v_star_rs, σ_star_rs = vfi(model_rs)
```

Let’s plot the results together with the original risk neutral case and see what we get.

```{code-cell} ipython3
:hide-output: false

stop_indices = jnp.where(σ_star_rs == 1)
res_wage_index = min(stop_indices[0])
res_wage_rs = w_vals[res_wage_index]
```

```{code-cell} ipython3
:hide-output: false

fig, ax = plt.subplots()
ax.plot(w_vals, v_star,  alpha=0.8, label="risk neutral $v$")
ax.plot(w_vals, v_star_rs, alpha=0.8, label="risk sensitive $v$")
ax.vlines((res_wage,), 100, 400,  ls='--', color='darkblue', 
          alpha=0.5, label=r"risk neutral $\bar w$")
ax.vlines((res_wage_rs,), 100, 400, ls='--', color='orange', 
          alpha=0.5, label=r"risk sensitive $\bar w$")
ax.legend(frameon=False, fontsize=12, loc="lower right")
ax.set_xlabel("$w$", fontsize=12)
plt.show()
```

The figure shows that the reservation wage under risk sensitive preferences (RS $ \bar w $) shifts down.

This makes sense – the agent does not like risk and hence is more inclined to
accept the current offer, even when it’s lower.

```{code-cell} ipython3

```

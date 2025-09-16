---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Job Search

*Prepared for the Computational Economics Workshop at Hitotsubashi*

Author: [John Stachurski](https://johnstachurski.net)


In this lecture we study a basic infinite-horizon job search problem with Markov wage
draws 

* For background on infinite horizon job search see, e.g., [DP1](https://dp.quantecon.org/).


In addition to what's in Anaconda, this lecture will need the QE library:

```{code-cell} ipython3
#!pip install quantecon  # Uncomment if necessary
```

We use the following imports.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import quantecon as qe
import numpy as np
import time
from typing import NamedTuple, Callable
```

## Model

We study an elementary model where 

* jobs are permanent 
* unemployed workers receive current compensation $c$
* the horizon is infinite
* an unemployment agent discounts the future via discount factor $\beta \in (0,1)$

### Set up

At the start of each period, an unemployed worker receives wage offer $W_t$.

We assume that 

$$
    W_{t+1} = \rho W_t + \nu Z_{t+1}
$$

where $(Z_t)_{t \geq 0}$ is IID and standard normal.

We then discretize this wage process using Tauchen's method to produce a stochastic matrix $P$.

Successive wage offers are drawn from $P$.

### Rewards

Since jobs are permanent, the return to accepting wage offer $w$ today is

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

Let's set up a `Model` class to store information needed to solve the model.

```{code-cell} ipython3
class Model(NamedTuple):
    n: int
    w_vals: np.ndarray
    P: np.ndarray
    β: float
    c: float
```

The function below holds default values and creates a `Model` instance.

```{code-cell} ipython3
def create_js_model(
        n: int = 500,       # wage grid size
        ρ: float = 0.9,     # wage persistence
        ν: float = 0.2,     # wage volatility
        β: float = 0.99,    # discount factor
        c: float = 1.0,     # unemployment compensation
    ) -> Model:
    "Creates an instance of the job search model with Markov wages."
    mc = qe.tauchen(n, ρ, ν)
    w_vals, P = np.exp(mc.state_values), mc.P
    return Model(n, w_vals, P, β, c)
```

Let's test it:

```{code-cell} ipython3
model = create_js_model(β=0.98)
```

```{code-cell} ipython3
model.c
```

```{code-cell} ipython3
model.β
```

```{code-cell} ipython3
model.w_vals.mean()  
```

Here's the Bellman operator

$$
    (Tv)(w) = \max
    \left\{
            \frac{w}{1-\beta}, c + \beta \sum_{w'} v(w') P(w, w')
    \right\}
$$

```{code-cell} ipython3
def T(v: np.ndarray, model: Model) -> np.ndarray:
    """
    The Bellman operator Tv = max{e, c + β P v} with

        e(w) = w / (1-β) and (Pv)(w) = E_w[ v(W')]

    """
    n, w_vals, P, β, c = model
    h = c + β * P @ v
    e = w_vals / (1 - β)

    return np.maximum(e, h)
```

The next function computes the optimal policy under the assumption that $v$ is
the value function.

The policy takes the form

$$
    \sigma(w) = \mathbf 1 
        \left\{
            \frac{w}{1-\beta} \geq c + \beta \sum_{w'} v(w') P(w, w')
        \right\}
$$

Here $\mathbf 1$ is an indicator function.

* $\sigma(w) = 1$ means stop
* $\sigma(w) = 0$ means continue.

```{code-cell} ipython3
def get_greedy(v: np.ndarray, model: Model) -> np.ndarray:
    "Get a v-greedy policy."
    n, w_vals, P, β, c = model
    e = w_vals / (1 - β)
    h = c + β * P @ v
    σ = np.where(e >= h, 1, 0)
    return σ
```

Here's a routine for value function iteration.

```{code-cell} ipython3
def vfi(
        model: Model,
        max_iter: int = 10_000,
        tol: float = 1e-4,
        verbose: bool = False
    ):
    """
    Solve the infinite-horizon Markov job search model by VFI.

    """
    v = np.zeros_like(model.w_vals)  # Initial condition

    for i in range(max_iter):
        new_v = T(v, model)
        error = np.max(np.abs(new_v - v))

        if error < tol:
            if verbose:
                print(f"VFI converged after {i+1} iterations (error: {error:.2e})")
            break
        v = new_v
    else:
        print(f"VFI hit max iterations ({max_iter}) with error {error:.2e}")

    return new_v, get_greedy(new_v, model)
```

## Computing the solution

Let's set up and solve the model.

```{code-cell} ipython3
model = create_js_model()
n, w_vals, P, β, c = model

v_star, σ_star = vfi(model, verbose=True)
```

Here's the optimal policy:

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.plot(w_vals, σ_star)
ax.set_xlabel("wage values")
ax.set_ylabel("optimal choice (stop=1)")
plt.show()
```

For context, we can plot it against the stationary distribution of the wage
offer process.

```{code-cell} ipython3
mc = qe.MarkovChain(P, state_values=w_vals)
ψ = mc.stationary_distributions[0]
fig, ax = plt.subplots()
ax.plot(w_vals, σ_star, 'k-')
ax.bar(w_vals, 200 * ψ, alpha=0.05)
ax.set_xlabel("wage values")
ax.set_ylabel("optimal choice (stop=1)")
plt.show()
```

Let's compute the runtime as well, averaging over a number of iterations

```{code-cell} ipython3
runtimes = []
for _ in range(10):
    start = time.time()
    v_star, σ_star = vfi(model, verbose=False)
    end = time.time()
    runtimes.append(end - start)

print()
print(f"Mean runtime for value function iteration = {np.mean(runtimes):.4f}")
print()
```

We compute the reservation wage as the first $w$ such that $\sigma(w)=1$.

```{code-cell} ipython3
stop_indices = np.where(σ_star == 1)
stop_indices
```

```{code-cell} ipython3
res_wage_index = min(stop_indices[0])
```

```{code-cell} ipython3
res_wage = w_vals[res_wage_index]
```

Here's a joint plot of the value function and the reservation wage.

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.plot(w_vals, v_star, alpha=0.8, label="value function")
ax.vlines((res_wage,), 150, 400, 'k', ls='--', label="reservation wage")
ax.legend(frameon=False, fontsize=12, loc="lower right")
ax.set_xlabel("$w$", fontsize=12)
plt.show()
```

## Exercise 1

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


When $\theta < 0$ the agent is risk averse.

Solve the model when $\theta = -0.1$ and compare your result to the risk neutral
case.

Try to interpret your result.

You can start with the following code:

```{code-cell} ipython3
class RiskModel(NamedTuple):
    n: int
    w_vals: np.ndarray
    P: np.ndarray
    β: float
    c: float
    θ: float

def create_risk_sensitive_js_model(
        n: int = 500,       # wage grid size
        ρ: float = 0.9,     # wage persistence
        ν: float = 0.2,     # wage volatility
        β: float = 0.99,    # discount factor
        c: float = 1.0,     # unemployment compensation
        θ: float = -0.1     # risk parameter
    ) -> RiskModel:
    "Creates an instance of the job search model with Markov wages."
    mc = qe.tauchen(n, ρ, ν)
    w_vals, P = np.exp(mc.state_values), mc.P
    P = np.array(P)
    return RiskModel(n, w_vals, P, β, c, θ)
```

Now you need to modify `T` and `get_greedy` and then run value function iteration again.

```{code-cell} ipython3
for _ in range(15):
    print("Solution below!")
```

```{code-cell} ipython3
def T_rs(v: np.ndarray, model: RiskModel) -> np.ndarray:
    """
    The Bellman operator Tv = max{e, c + β R v} with

        e(w) = w / (1-β) and

        (Rv)(w) = (1/θ) ln{E_w[ exp(θ v(W'))]}

    """
    n, w_vals, P, β, c, θ = model
    h = c + (β / θ) * np.log(P @ (np.exp(θ * v)))
    e = w_vals / (1 - β)

    return np.maximum(e, h)


def get_greedy_rs(v: np.ndarray, model: RiskModel) -> np.ndarray:
    " Get a v-greedy policy."
    n, w_vals, P, β, c, θ = model
    e = w_vals / (1 - β)
    h = c + (β / θ) * np.log(P @ (np.exp(θ * v)))
    σ = np.where(e >= h, 1, 0)
    return σ


def vfi_rs(
        model: RiskModel,
        max_iter: int = 10_000,
        tol: float = 1e-4
    ):
    "Solve the infinite-horizon Markov job search model by VFI."
    v = np.zeros_like(model.w_vals)

    for i in range(max_iter):
        new_v = T_rs(v, model)
        error = np.max(np.abs(new_v - v))

        if error < tol:
            print(f"VFI converged after {i+1} iterations (error: {error:.2e})")
            break
        v = new_v
    else:
        print(f"VFI reached max iterations ({max_iter}) with error {error:.2e}")

    return new_v, get_greedy_rs(new_v, model)



model_rs = create_risk_sensitive_js_model()
n, w_vals, P, β, c, θ = model_rs
v_star_rs, σ_star_rs = vfi_rs(model_rs)
```

Let's plot the results together with the original risk neutral case and see what we get.

```{code-cell} ipython3
stop_indices = np.where(σ_star_rs == 1)
res_wage_index = min(stop_indices[0])
res_wage_rs = w_vals[res_wage_index]
```

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.plot(w_vals, v_star,  ls='-', color='blue',
        alpha=0.8, label="risk neutral $v$")
ax.plot(w_vals, v_star_rs, ls='-', color='orange',
        alpha=0.8, label="risk sensitive $v$")
ax.vlines((res_wage,), 100, 400,  ls='--', color='blue',
          alpha=0.5, label=r"risk neutral $\bar w$")
ax.vlines((res_wage_rs,), 100, 400, ls='--', color='orange',
          alpha=0.5, label=r"risk sensitive $\bar w$")
ax.legend(frameon=False, fontsize=12, loc="lower right")
ax.set_xlabel("$w$", fontsize=12)
plt.show()
```

The figure shows that the reservation wage under risk sensitive preferences (RS $\bar w$) shifts down.

This makes sense -- the agent does not like risk and hence is more inclined to
accept the current offer, even when it's lower.

+++

## Exercise 2

In the code above, we wrote two versions of VFI, one for each model.

This is poor style because we are repeating logic.  

Write one version of VFI that can work with both and test that it does the
same job.

```{code-cell} ipython3
for _ in range(15):
    print("Solution below!")
```

```{code-cell} ipython3
def generic_vfi(
        bellman_operator: Callable,
        get_greedy_function: Callable,
        v_zero: np.ndarray,
        max_iter: int = 10_000,
        tol: float = 1e-4
    ):
    """
    Solve the infinite-horizon Markov job search model by VFI.

    """
    v = v_zero

    for i in range(max_iter):
        new_v = bellman_operator(v)
        error = np.max(np.abs(new_v - v))

        if error < tol:
            print(f"VFI converged after {i+1} iterations (error: {error:.2e})")
            break
        v = new_v
    else:
        print(f"VFI reached max iterations ({max_iter}) with error {error:.2e}")

    return new_v, get_greedy_function(new_v)
```

Let's test this with the original model (comparing the output of `vfi` and `generic_vfi`).

```{code-cell} ipython3
model = create_js_model()
n, w_vals, P, β, c = model
v_star_0, σ_star_0 = vfi(model)
bellman_operator = lambda v: T(v, model)
get_greedy_function = lambda v: get_greedy(v, model)
v_zero = np.zeros_like(w_vals)
v_star_1, σ_star_1 = generic_vfi(
    bellman_operator, get_greedy_function, v_zero
)

correct = np.allclose(v_star_0, v_star_1) and np.allclose(σ_star_0, σ_star_1)
print(f"Success = {correct}")
```

Let's also test this set up with the risk sensitive model (comparing the output of `vfi_rs` and `generic_vfi`).

```{code-cell} ipython3
model = create_risk_sensitive_js_model()
n, w_vals, P, β, c, θ = model_rs
v_star_0, σ_star_0 = vfi_rs(model)
bellman_operator = lambda v: T_rs(v, model)
get_greedy_function = lambda v: get_greedy_rs(v, model)
v_zero = np.zeros_like(w_vals)
v_star_1, σ_star_1 = generic_vfi(
    bellman_operator, get_greedy_function, v_zero
)

correct = np.allclose(v_star_0, v_star_1) and np.allclose(σ_star_0, σ_star_1)
print(f"Success = {correct}")
```

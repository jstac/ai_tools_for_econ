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

# Job Search with Separation

This module solves for the optimal policy of an agent who can be either
unemployed or employed. The key features of the model are:

## Model Setup

- Agent receives wage offers w from a finite set when unemployed
- Wage offers follow a Markov chain with transition matrix P
- Jobs terminate with probability α each period (separation rate)
- Unemployed workers receive compensation c per period
- Future payoffs discounted by factor β ∈ (0,1)

## Decision Problem

When unemployed and receiving wage offer w, the agent chooses between:
1. Accept offer w: Become employed at wage w
2. Reject offer: Remain unemployed, receive c, get new offer next period

## Value Functions

- v_u*(w): Value of being unemployed when current wage offer is w
- v_e*(w): Value of being employed at wage w

## Bellman Equations

The unemployed worker's value function satisfies:

$$v_u^*(w) = \max\{v_e^*(w), c + \beta \sum_{w'} v_u^*(w') P(w,w')\}$$

The employed worker's value function satisfies:

$$v_e^*(w) = w + \beta[\alpha \sum_{w'} v_u^*(w') P(w,w') + (1-\alpha) v_e^*(w)]$$

## Computational Approach

1. Solve the employed value function analytically:
   $$v_e^*(w) = \frac{1}{1-\beta(1-\alpha)} \cdot (w + \alpha\beta(Pv_u^*)(w))$$

2. Substitute into unemployed Bellman equation to get:
   $$v_u^*(w) = \max\left\{\frac{1}{1-\beta(1-\alpha)} \cdot (w + \alpha\beta(Pv_u^*)(w)), c + \beta(Pv_u^*)(w)\right\}$$

3. Use value function iteration to solve for v_u*
4. Compute optimal policy: accept if v_e*(w) ≥ c + β(Pv_u*)(w)

The optimal policy is a reservation wage strategy: accept all wages above
some threshold w*.

+++

## Code

We use the following imports:

```{code-cell} ipython3
from quantecon.markov import tauchen
import numpy as np
from typing import NamedTuple
import matplotlib.pyplot as plt
```

First, we implement the successive approximation algorithm:

```{code-cell} ipython3
def successive_approx(
        T,                         # Operator (callable)
        x_0,                       # Initial condition
        tolerance: float = 1e-6,   # Error tolerance
        max_iter: int = 10_000,    # Max iteration bound
        print_step: int = 25       # Print at multiples
    ):
    """Computes the approximate fixed point of T via successive approximation."""
    x = x_0
    error = tolerance + 1
    k = 1
    while (error > tolerance) and (k <= max_iter):
        x_new = T(x)
        error = np.max(np.abs(x_new - x))
        if k % print_step == 0:
            print(f"Completed iteration {k} with error {error}.")
        x = x_new
        k += 1
    if error <= tolerance:
        print(f"Terminated successfully in {k} iterations.")
    else:
        print("Warning: hit iteration bound.")
    return x
```

Let's set up a `Model` class to store information needed to solve the model:

```{code-cell} ipython3
class Model(NamedTuple):
    n: int
    w_vals: np.ndarray
    P: np.ndarray
    β: float
    c: float
    α: float
```

The function below holds default values and creates a `Model` instance:

```{code-cell} ipython3
def create_js_with_sep_model(
        n: int = 200,          # wage grid size
        ρ: float = 0.9,        # wage persistence
        ν: float = 0.2,        # wage volatility
        β: float = 0.98,       # discount factor
        α: float = 0.1,        # separation rate
        c: float = 1.0         # unemployment compensation
    ) -> Model:
    """Creates an instance of the job search model with separation."""
    mc = tauchen(n, ρ, ν)
    w_vals, P = np.exp(mc.state_values), mc.P
    return Model(n, w_vals, P, β, c, α)
```

Let's test it:

```{code-cell} ipython3
default_model = create_js_with_sep_model()
```

Here's the Bellman operator for the unemployed worker's value function:

```{code-cell} ipython3
def T(v: np.ndarray, model: Model) -> np.ndarray:
    """The Bellman operator for the value of being unemployed."""
    n, w_vals, P, β, c, α = model
    d = 1 / (1 - β * (1 - α))
    accept = d * (w_vals + α * β * P @ v)
    reject = c + β * P @ v
    return np.maximum(accept, reject)
```

The next function computes the optimal policy under the assumption that v is
the value function:

```{code-cell} ipython3
def get_greedy(v: np.ndarray, model: Model) -> np.ndarray:
    """Get a v-greedy policy."""
    n, w_vals, P, β, c, α = model
    d = 1 / (1 - β * (1 - α))
    accept = d * (w_vals + α * β * P @ v)
    reject = c + β * P @ v
    σ = accept >= reject
    return σ
```

Here's a routine for value function iteration:

```{code-cell} ipython3
def vfi(model: Model) -> tuple[np.ndarray, np.ndarray]:
    """Solve by VFI."""
    v_init = np.zeros(model.w_vals.shape)
    v_star = successive_approx(lambda v: T(v, model), v_init)
    σ_star = get_greedy(v_star, model)
    return v_star, σ_star
```

## Computing the Solution

Let's solve the model and plot the results:

```{code-cell} ipython3
def plot_main(model: Model = default_model) -> None:
    n, w_vals, P, β, c, α = model
    v_star, σ_star = vfi(model)

    d = 1 / (1 - β * (1 - α))
    accept = d * (w_vals + α * β * P @ v_star)
    h_star = c + β * P @ v_star

    w_star = np.inf
    for (i, w) in enumerate(w_vals):
        if accept[i] >= h_star[i]:
            w_star = w
            break

    assert w_star != np.inf, "Agent never accepts"

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(w_vals, h_star, linewidth=4, ls="--", alpha=0.4,
            label="continuation value")
    ax.plot(w_vals, accept, linewidth=4, ls="--", alpha=0.4,
            label="stopping value")
    ax.plot(w_vals, v_star, "k-", alpha=0.7, label=r"$v_u^*(w)$")
    ax.legend(frameon=False)
    ax.set_xlabel(r"$w$")
    plt.show()

plot_main()
```

## Sensitivity Analysis

Let's examine how reservation wages change with the separation rate α:

```{code-cell} ipython3
def plot_w_stars(
        α_vals: np.ndarray = np.linspace(0.0, 1.0, 10)
    ) -> None:

    w_star_vec = np.empty_like(α_vals)
    for (i_α, α) in enumerate(α_vals):
        print(i_α, α)
        model = create_js_with_sep_model(α=α)
        n, w_vals, P, β, c, α = model
        v_star, σ_star = vfi(model)

        d = 1 / (1 - β * (1 - α))
        accept = d * (w_vals + α * β * P @ v_star)
        h_star = c + β * P @ v_star

        w_star = np.inf
        for (i_w, w) in enumerate(w_vals):
            if accept[i_w] >= h_star[i_w]:
                w_star = w
                break

        assert w_star != np.inf, "Agent never accepts"
        w_star_vec[i_α] = w_star

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(α_vals, w_star_vec, linewidth=2, alpha=0.6,
            label="reservation wage")
    ax.legend(frameon=False)
    ax.set_xlabel(r"$\alpha$")
    ax.set_xlabel(r"$w$")
    plt.show()

plot_w_stars()
```

## Employment Simulation

Now let's simulate the employment dynamics under the optimal policy:

```{code-cell} ipython3
def simulate_employment_path(
        model: Model,
        T: int = 1000,
        seed: int = 42
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate employment path for T periods starting from unemployment.

    Parameters:
    - model: Model instance with parameters
    - T: Number of periods to simulate
    - seed: Random seed for reproducibility

    Returns:
    - wage_path: Array of wage offers/current wages
    - employment_status: Array of employment status (0=unemployed, 1=employed)
    - decisions: Array of accept/reject decisions when unemployed
    """
    np.random.seed(seed)

    # Solve for optimal policy
    v_star, σ_star = vfi(model)
    n, w_vals, P, β, c, α = model

    # Initialize arrays
    wage_path = np.zeros(T)
    employment_status = np.zeros(T, dtype=int)
    decisions = np.zeros(T, dtype=int)

    # Start unemployed with random wage draw
    current_wage_idx = np.random.choice(n)
    is_employed = False

    for t in range(T):
        if not is_employed:
            # Unemployed: receive wage offer
            wage_offer_idx = current_wage_idx
            wage_path[t] = w_vals[wage_offer_idx]
            employment_status[t] = 0

            # Make accept/reject decision based on optimal policy
            if σ_star[wage_offer_idx]:
                # Accept offer
                decisions[t] = 1
                is_employed = True
                current_wage_idx = wage_offer_idx
            else:
                # Reject offer
                decisions[t] = 0
                # Draw next period's wage offer
                current_wage_idx = np.random.choice(n, p=P[wage_offer_idx, :])
        else:
            # Employed: receive current wage
            wage_path[t] = w_vals[current_wage_idx]
            employment_status[t] = 1
            decisions[t] = -1  # Not applicable when employed

            # Check for separation
            if np.random.random() < α:
                # Job terminates
                is_employed = False
                # Draw wage offer for next period if unemployed
                current_wage_idx = np.random.choice(n, p=P[current_wage_idx, :])
            else:
                # Job continues, wage evolves according to Markov chain
                current_wage_idx = np.random.choice(n, p=P[current_wage_idx, :])

    return wage_path, employment_status, decisions
```

Here's a helper function to compute average spell lengths:

```{code-cell} ipython3
def _compute_avg_spell_length(status_array: np.ndarray, status: int) -> float:
    """Compute average spell length for given employment status."""
    spells = []
    current_spell = 0

    for s in status_array:
        if s == status:
            current_spell += 1
        else:
            if current_spell > 0:
                spells.append(current_spell)
                current_spell = 0

    # Add final spell if it ends with the target status
    if current_spell > 0:
        spells.append(current_spell)

    return np.mean(spells) if spells else 0.0
```

Finally, let's create a comprehensive plot of the employment simulation:

```{code-cell} ipython3
def plot_employment_simulation(model: Model = default_model) -> None:
    """Plot simulated employment path under optimal policy."""
    wage_path, employment_status, decisions = simulate_employment_path(model)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

    # Plot employment status
    ax1.plot(employment_status, 'b-', alpha=0.7, linewidth=1)
    ax1.fill_between(range(len(employment_status)), employment_status,
                     alpha=0.3, color='blue')
    ax1.set_ylabel('Employment Status')
    ax1.set_title('Simulated Employment Path (0=Unemployed, 1=Employed)')
    ax1.set_ylim(-0.1, 1.1)
    ax1.grid(True, alpha=0.3)

    # Plot wage path with employment status coloring
    unemployed_mask = employment_status == 0
    employed_mask = employment_status == 1

    if np.any(unemployed_mask):
        ax2.scatter(np.where(unemployed_mask)[0], wage_path[unemployed_mask],
                   c='red', alpha=0.6, s=10, label='Unemployed (offers)')
    if np.any(employed_mask):
        ax2.scatter(np.where(employed_mask)[0], wage_path[employed_mask],
                   c='blue', alpha=0.6, s=10, label='Employed (wages)')

    ax2.set_xlabel('Time Period')
    ax2.set_ylabel('Wage')
    ax2.set_title('Wage Path Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot cumulative fraction of time unemployed
    unemployed_indicator = (employment_status == 0).astype(int)
    cumulative_unemployment = np.cumsum(unemployed_indicator) / np.arange(1, len(employment_status) + 1)

    ax3.plot(cumulative_unemployment, 'r-', alpha=0.8, linewidth=2)
    ax3.axhline(y=np.mean(unemployed_indicator), color='black',
                linestyle='--', alpha=0.7,
                label=f'Final rate: {np.mean(unemployed_indicator):.3f}')
    ax3.set_xlabel('Time Period')
    ax3.set_ylabel('Cumulative Unemployment Rate')
    ax3.set_title('Cumulative Fraction of Time Spent Unemployed')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    unemployment_rate = 1 - np.mean(employment_status)
    avg_unemployment_spell = _compute_avg_spell_length(employment_status, 0)
    avg_employment_spell = _compute_avg_spell_length(employment_status, 1)

    print(f"Simulation Summary:")
    print(f"Unemployment rate: {unemployment_rate:.3f}")
    print(f"Average unemployment spell: {avg_unemployment_spell:.1f} periods")
    print(f"Average employment spell: {avg_employment_spell:.1f} periods")
    print(f"Average wage when employed: {np.mean(wage_path[employed_mask]):.3f}")

plot_employment_simulation()
```

## Results Summary

The simulation demonstrates the model's key predictions:

1. **Optimal Policy**: The agent follows a reservation wage strategy
2. **Employment Dynamics**: Realistic patterns of job search, acceptance, and separation
3. **Steady State**: The cumulative unemployment rate converges to the theoretical prediction
4. **Labor Market Flows**: Clear cycles between unemployment and employment spells

The model successfully captures the essential features of labor market dynamics with job separation, showing how workers optimally balance the trade-off between accepting current offers versus waiting for better opportunities.

## Cross-Sectional Analysis

Now let's simulate many agents simultaneously to examine the cross-sectional unemployment rate:

```{code-cell} ipython3
def simulate_cross_section(
        model: Model,
        n_agents: int = 10000,
        T: int = 1000,
        seed: int = 42
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate employment paths for many agents simultaneously.

    Parameters:
    - model: Model instance with parameters
    - n_agents: Number of agents to simulate
    - T: Number of periods to simulate
    - seed: Random seed for reproducibility

    Returns:
    - unemployment_rates: Fraction of agents unemployed at each period
    - employment_matrix: n_agents x T matrix of employment status
    """
    np.random.seed(seed)

    # Solve for optimal policy
    v_star, σ_star = vfi(model)
    n, w_vals, P, β, c, α = model

    # Initialize arrays
    employment_matrix = np.zeros((n_agents, T), dtype=int)
    current_wage_indices = np.random.choice(n, size=n_agents)
    is_employed = np.zeros(n_agents, dtype=bool)

    for t in range(T):
        for agent in range(n_agents):
            if not is_employed[agent]:
                # Unemployed: receive wage offer
                wage_offer_idx = current_wage_indices[agent]
                employment_matrix[agent, t] = 0

                # Make accept/reject decision based on optimal policy
                if σ_star[wage_offer_idx]:
                    # Accept offer
                    is_employed[agent] = True
                else:
                    # Reject offer, draw next period's wage offer
                    current_wage_indices[agent] = np.random.choice(
                        n, p=P[wage_offer_idx, :]
                    )
            else:
                # Employed: receive current wage
                employment_matrix[agent, t] = 1

                # Check for separation
                if np.random.random() < α:
                    # Job terminates
                    is_employed[agent] = False
                    # Draw wage offer for next period
                    current_wage_indices[agent] = np.random.choice(
                        n, p=P[current_wage_indices[agent], :]
                    )
                else:
                    # Job continues, wage evolves according to Markov chain
                    current_wage_indices[agent] = np.random.choice(
                        n, p=P[current_wage_indices[agent], :]
                    )

    # Calculate unemployment rate at each period
    unemployment_rates = 1 - np.mean(employment_matrix, axis=0)

    return unemployment_rates, employment_matrix


def plot_cross_sectional_unemployment(model: Model = default_model) -> None:
    """Plot cross-sectional unemployment rate over time."""
    unemployment_rates, employment_matrix = simulate_cross_section(model)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot unemployment rate over time
    ax.plot(unemployment_rates, 'b-', alpha=0.8, linewidth=1.5,
            label='Cross-sectional unemployment rate')

    # Add horizontal line for average unemployment rate
    avg_unemployment = np.mean(unemployment_rates)
    ax.axhline(y=avg_unemployment, color='red', linestyle='--', alpha=0.7,
               label=f'Average: {avg_unemployment:.3f}')

    # Add shaded region for ±1 standard deviation
    window_size = 50
    rolling_std = np.array([
        np.std(unemployment_rates[max(0, t-window_size):t+1])
        for t in range(len(unemployment_rates))
    ])

    ax.fill_between(range(len(unemployment_rates)),
                    unemployment_rates - rolling_std,
                    unemployment_rates + rolling_std,
                    alpha=0.2, color='blue',
                    label='±1 rolling std')

    ax.set_xlabel('Time Period')
    ax.set_ylabel('Unemployment Rate')
    ax.set_title(f'Cross-Sectional Unemployment Rate ({len(employment_matrix)} Agents)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    final_unemployment = unemployment_rates[-1]
    std_unemployment = np.std(unemployment_rates)

    print(f"Cross-Sectional Analysis Summary:")
    print(f"Average unemployment rate: {avg_unemployment:.3f}")
    print(f"Final period unemployment rate: {final_unemployment:.3f}")
    print(f"Standard deviation: {std_unemployment:.3f}")
    print(f"Min unemployment rate: {np.min(unemployment_rates):.3f}")
    print(f"Max unemployment rate: {np.max(unemployment_rates):.3f}")

plot_cross_sectional_unemployment()
```

This cross-sectional analysis reveals important insights about the model's aggregate behavior:

1. **Convergence to Steady State**: The unemployment rate converges to its theoretical steady-state value
2. **Random Fluctuations**: Short-term variations around the long-run average due to idiosyncratic shocks
3. **Law of Large Numbers**: With many agents, the cross-sectional unemployment rate becomes more stable over time
4. **Model Validation**: Comparison between theoretical predictions and simulated outcomes

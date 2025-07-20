# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python
#     language: python3
#     name: python3
# ---

# %% [markdown]
# # A Bayesian Formulation of Friedman and Wald’s Problem
#
# %% [markdown]
# ## Contents
#
# %% [markdown]
# ## Overview
#
# This lecture revisits the  statistical decision problem presented to Milton
# Friedman and W. Allen Wallis during World War II when they were analysts at
# the U.S. Government’s  Statistical Research Group at Columbia University.
#
#
# In this lecture, we reformulate Friedman and Wald’s  problem  by adopting
# an explicitly ‘subjective’ perspective taken by a Bayesian decision maker
# who regards parameters not as fixed numbers but as (hidden) random variables
# that are jointly distributed with the random variables that can be observed by
# sampling from that joint distribution.
#
#
# - an initial prior subjective probability $\pi_{-1} \in (0,1)$  that nature
# uses $f_1$ to  generate  $\{z_k\} $ as a sequence of i.i.d. draws rather than $ f_0 $.  
# - faith in Bayes’ law as a way to revise his subjective beliefs as observations on $ \{z_k\} $ sequence arrive each period.  
# - a loss function that tells how the decision maker values different types of errors.  
#
# We use the following imports

# %%
import numpy as np
import matplotlib.pyplot as plt
from typing import NamedTuple
import jax
import jax.numpy as jnp
from jax.scipy.special import gamma


# %% [markdown]
# ## A Dynamic Programming Approach
#
# A decision-maker can observe a sequence of draws of a random variable $ z $.
#
# He (or she) wants to know which of two probability distributions $ f_0 $ or $ f_1 $ governs $ z $.
#
# Conditional on knowing that successive observations are drawn from distribution $ f_0 $, the sequence of
# random variables is independently and identically distributed (IID).
#
# Conditional on knowing that successive observations are drawn from distribution $ f_1 $, the sequence of
# random variables is also independently and identically distributed (IID).
#
# But the observer does not know which of the two distributions generated the sequence.
#
# The observer has something to learn, namely, whether the observations are drawn from  $ f_0 $ or from $ f_1 $.
#
# The decision maker   wants  to decide
# which of the  two distributions is generating outcomes.
#
# We adopt a Bayesian formulation.
#
# The decision maker begins  with a prior probability
#
# $$
# \pi_{-1} =
# \mathbb P \{ f = f_1 \mid \textrm{ no observations} \} \in (0, 1)
# $$
#
#
# After observing $ k+1 $ observations $ z_k, z_{k-1}, \ldots, z_0 $, he updates his personal probability that the observations are described by distribution $ f_1 $  to
#
# $$
# \pi_k = \mathbb P \{ f = f_1 \mid z_k, z_{k-1}, \ldots, z_0 \}
# $$
#
# which is calculated recursively by applying Bayes’ law:
#
# $$
# \pi_{k+1} = \frac{ \pi_k f_1(z_{k+1})}{ (1-\pi_k) f_0(z_{k+1}) + \pi_k f_1 (z_{k+1}) },
# \quad k = -1, 0, 1, \ldots
# $$
#
# After observing $ z_k, z_{k-1}, \ldots, z_0 $, the decision-maker believes
# that $ z_{k+1} $ has probability distribution
#
# $$
# f_{{\pi}_k} (v) = (1-\pi_k) f_0(v) + \pi_k f_1 (v) ,
# $$
#
# which  is a mixture of distributions $ f_0 $ and $ f_1 $, with the weight
# on $ f_1 $ being the posterior probability that $ f = f_1 $ <sup><a href=#f1 id=f1-link>[1]</a></sup>.
#

# %% [markdown]
# ### Losses and Costs
#
# After observing $ z_k, z_{k-1}, \ldots, z_0 $, the decision-maker
# chooses among three distinct actions:
#
# - He decides that $ f = f_0 $ and draws no more $ z $’s  
# - He decides that $ f = f_1 $ and draws no more $ z $’s  
# - He postpones deciding now and instead chooses to draw a
#   $ z_{k+1} $  
#
#
# Associated with these three actions, the decision-maker can suffer three
# kinds of losses:
#
# - A loss $ L_0 $ if he decides $ f = f_0 $ when actually
#   $ f=f_1 $  
# - A loss $ L_1 $ if he decides $ f = f_1 $ when actually
#   $ f=f_0 $  
# - A cost $ c $ if he postpones deciding and chooses instead to draw
#   another $ z $  

# %% [markdown]
# ### Intuition
#
# Before proceeding,  let’s try to guess what an optimal decision rule might look like.
#
# Suppose at some given point in time that $ \pi $ is close to 1.
#
# Then our prior beliefs and the evidence so far point strongly to $ f = f_1 $.
#
# If, on the other hand, $ \pi $ is close to 0, then $ f = f_0 $ is strongly favored.
#
# Finally, if $ \pi $ is in the middle of the interval $ [0, 1] $, then we are confronted with more uncertainty.
#
# This reasoning suggests a sequential  decision rule that we illustrate  in the following figure:
#
# ![https://python.quantecon.org/_static/lecture_specific/wald_friedman_2/wald_dec_rule.png](https://python.quantecon.org/_static/lecture_specific/wald_friedman_2/wald_dec_rule.png)
#
#   
# As we’ll see, this is indeed the correct form of the decision rule.
#
# Our problem is to determine threshold values $ A, B $ that somehow depend on the parameters described  above.
#
# You might like to pause at this point and try to predict the impact of a
# parameter such as $ c $ or $ L_0 $ on $ A $ or $ B $.

# %% [markdown]
# ### A Bellman Equation
#
# Let $ J(\pi) $ be the total loss for a decision-maker with current belief $ \pi $ who chooses optimally.
#
# With some thought, you will agree that $ J $ should satisfy the Bellman equation
#
#
# <a id='equation-new1'></a>
# $$
# J(\pi) =
#     \min
#     \left\{
#         \underbrace{\pi L_0}_{ \text{accept } f_0 } \; , \; \underbrace{(1-\pi) L_1}_{ \text{accept } f_1 } \; , \;
#         \underbrace{c + \mathbb E [ J (\pi') ]}_{ \text{draw again} }
#     \right\} \tag{24.1}
# $$
#
# where $ \pi' $ is the random variable defined by Bayes’ Law
#
# $$
# \pi' = \kappa(z', \pi) = \frac{ \pi f_1(z')}{ (1-\pi) f_0(z') + \pi f_1 (z') }
# $$
#
# when $ \pi $ is fixed and $ z' $ is drawn from the current best guess, which is the distribution $ f $ defined by
#
# $$
# f_{\pi}(v) = (1-\pi) f_0(v) + \pi f_1 (v)
# $$
#
# In the Bellman equation, minimization is over three actions:
#
# 1. Accept the hypothesis that $ f = f_0 $  
# 1. Accept the hypothesis that $ f = f_1 $  
# 1. Postpone deciding and draw again  
#
#
# We can represent the  Bellman equation as
#
#
# <a id='equation-optdec'></a>
# $$
# J(\pi) =
# \min \left\{ \pi L_0, \; (1-\pi) L_1, \; h(\pi) \right\} \tag{24.2}
# $$
#
# where $ \pi \in [0,1] $ and
#
# - $ \pi L_0 $ is the expected loss associated with accepting
#   $ f_0 $ 
# - $ (1-\pi) L_1 $ is the expected loss associated with accepting
#   $ f_1 $
# - $ h(\pi) :=  c + \mathbb E [J(\pi')] $; this is the continuation value; i.e.,
#   the expected cost associated with drawing one more $ z $.  
#
#
# The optimal decision rule is characterized by two numbers $ A, B \in (0,1) \times (0,1) $ that satisfy
#
# $$
# \pi L_0 < \min \{ (1-\pi) L_1, c + \mathbb E [J(\pi')] \}  \textrm { if } \pi \leq B
# $$
#
# and
#
# $$
# (1- \pi) L_1 < \min \{ \pi L_0,  c + \mathbb E [J(\pi')] \} \textrm { if } \pi \geq A
# $$
#
# The optimal decision rule is then
#
# $$
# \begin{aligned}
# \textrm { accept } f=f_1 \textrm{ if } \pi \geq A \\
# \textrm { accept } f=f_0 \textrm{ if } \pi \leq B \\
# \textrm { draw another }  z \textrm{ if }  B < \pi < A
# \end{aligned}
# $$
#
# Our aim is to compute the cost function $ J $ as well as  the associated cutoffs $ A $
# and $ B $.
#
# To make our computations manageable, using [(24.2)](#equation-optdec), we can write the continuation cost $ h(\pi) $ as
#
#
# <a id='equation-optdec2'></a>
# $$
# \begin{aligned}
# h(\pi) &= c + \mathbb E [J(\pi')] \\
# &= c + \mathbb E_{\pi'} \min \{ \pi' L_0, (1 - \pi') L_1, h(\pi') \} \\
# &= c + \int \min \{ \kappa(z', \pi) L_0, (1 - \kappa(z', \pi) ) L_1, h(\kappa(z', \pi) ) \} f_\pi (z') dz'
# \end{aligned} \tag{24.3}
# $$
#
# The equality
#
#
# <a id='equation-funceq'></a>
# $$
# h(\pi) =
# c + \int \min \{ \kappa(z', \pi) L_0, (1 - \kappa(z', \pi) ) L_1, h(\kappa(z', \pi) ) \} f_\pi (z') dz' \tag{24.4}
# $$
#
# is an equation  in an unknown function  $ h $.
#
# >**Note**
# >
# >Such an equation is called a **functional equation**.
#
# Using the functional equation, [(24.4)](#equation-funceq), for the continuation cost, we can back out
# optimal choices using the right side of [(24.2)](#equation-optdec).
#
# This functional equation can be solved by taking an initial guess and iterating
# to find a fixed point.
#
# Thus, we iterate with an operator $ Q $, where
#
# $$
# Q h(\pi) =
# c + \int \min \{ \kappa(z', \pi) L_0, (1 - \kappa(z', \pi) ) L_1, h(\kappa(z', \pi) ) \} f_\pi (z') dz'
# $$

# %% [markdown]
# ## Implementation

# %% 
@jax.jit
def p(x, a, b):
    " Beta(a, b) density. "
    r = gamma(a + b) / (gamma(a) * gamma(b))
    return r * x**(a-1) * (1 - x)**(b-1)

class Model(NamedTuple):
    a0: float            # Parameters of beta distributions
    b0: float
    a1: float
    b1: float
    c: float             # Cost of another draw
    grid_size: int
    L0: float            # Cost of selecting f0 when f1 is true
    L1: float            # Cost of selecting f1 when f0 is true
    π_grid: jnp.ndarray
    mc_size: int
    u: jnp.ndarray


# %% hide-output=false
def create_model_instance(
        c=1.25, a0=1, b0=1, a1=3, b1=1.2, L0=25, L1=25, 
        grid_size=200, mc_size=1000, seed=1234
    ):

    key = jax.random.PRNGKey(seed)
    π_grid = np.linspace(0, 1, grid_size)
    mc_size = mc_size
    z0 = jax.random.uniform(key, (mc_size,))
    return Model(
        a0=a0, b0=b0, a1=a1, b1=b1, c=c, grid_size=grid_size,
        L0=L0, L1=L1, π_grid=π_grid, mc_size=mc_size, u=u
    )


def f0(model, x):
    return p(x, model.a0, model.b0)

def f1(model, x):
    return p(x, model.a1, model.b1)

def f0_rvs(model):
    return np.random.beta(model.a0, model.b0)

def f1_rvs(model):
    return np.random.beta(model.a1, model.b1)

def κ(model, z, π):
    """
    Updates π using Bayes' rule and the current observation z

    """
    f0, f1 = model.f0, model.f1
    π_f0, π_f1 = (1 - π) * f0(z), π * f1(z)
    π_new = π_f1 / (π_f0 + π_f1)
    return π_new


# %% [markdown]
#
# - We iterate at a finite grid of possible values of $ \pi $.  
# - When we evaluate $ \mathbb E[J(\pi')] $ between grid points, we use linear interpolation.  

# %% 
@jax.jit
def Q(h, model):

    c, π_grid = model.c, model.π_grid
    L0, L1 = model.L0, model.L1
    z0, z1 = model.z0, model.z1
    mc_size = model.mc_size
    h_new = np.empty_like(π_grid)
    h_func = lambda p: np.interp(p, π_grid, h)

    for i in prange(len(π_grid)):
        π = π_grid[i]

        # Find the expected value of J by integrating over z
        integral_f0, integral_f1 = 0, 0
        for m in range(mc_size):
            π_0 = κ(model, z0[m], π)  # Draw z from f0 and update π
            integral_f0 += min(π_0 * L0, (1 - π_0) * L1, h_func(π_0))

            π_1 = κ(model, z1[m], π)  # Draw z from f1 and update π
            integral_f1 += min(π_1 * L0, (1 - π_1) * L1, h_func(π_1))

        integral = ((1 - π) * integral_f0 + π * integral_f1) / mc_size

        h_new[i] = c + integral

    return h_new


# %% [markdown]
# To solve the key functional equation, we will iterate using `Q` to find the fixed point

# %% 
def solve_model(model, tol=1e-4, max_iter=1000):
    " Compute the continuation cost function. "

    h = np.zeros(len(model.π_grid))
    i = 0
    error = tol + 1

    while i < max_iter and error > tol:
        h_new = Q(h, model)
        error = np.max(np.abs(h - h_new))
        i += 1
        h = h_new

    if error > tol:
        print("Failed to converge!")

    return h_new


# %% [markdown]
# ## Analysis
#
# Let’s inspect outcomes.
#
# We will be using the default parameterization with distributions like so

# %% hide-output=false
model = Model()

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(model.f0(model.π_grid), label="$f_0$")
ax.plot(model.f1(model.π_grid), label="$f_1$")
ax.set(ylabel="probability of $z_k$", xlabel="$z_k$", title="Distributions")
ax.legend()

plt.show()

# %% [markdown]
# ### Cost Function
#
# To solve the model, we will call our `solve_model` function

# %% hide-output=false
h_star = solve_model(model)    # Solve the model


# %% [markdown]
# We will also set up a function to compute the cutoffs $ A $ and $ B $
# and plot these on our cost function plot

# %% hide-output=false
@jit
def find_cutoff_rule(model, h):

    """
    This function takes a continuation cost function and returns the
    corresponding cutoffs of where you transition between continuing and
    choosing a specific model
    """

    π_grid = model.π_grid
    L0, L1 = model.L0, model.L1

    # Evaluate cost at all points on grid for choosing a model
    cost_f0 = π_grid * L0
    cost_f1 = (1 - π_grid) * L1
    
    # Find B: largest π where cost_f0 <= min(cost_f1, h)
    optimal_cost = np.minimum(np.minimum(cost_f0, cost_f1), h)
    choose_f0 = (cost_f0 <= cost_f1) & (cost_f0 <= h)
    
    if np.any(choose_f0):
        B = π_grid[choose_f0][-1]  # Last point where we choose f0
    else:
        assert False, "No point where we choose f0"
    
    # Find A: smallest π where cost_f1 <= min(cost_f0, h)  
    choose_f1 = (cost_f1 <= cost_f0) & (cost_f1 <= h)
    
    if np.any(choose_f1):
        A = π_grid[choose_f1][0]  # First point where we choose f1
    else:
        assert False, "No point where we choose f1"

    return (B, A)

B, A = find_cutoff_rule(model, h_star)
cost_L0 = model.π_grid * model.L0
cost_L1 = (1 - model.π_grid) * model.L1

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(model.π_grid, h_star, label='sample again')
ax.plot(model.π_grid, cost_L1, label='choose f1')
ax.plot(model.π_grid, cost_L0, label='choose f0')
ax.plot(model.π_grid,
        np.amin(np.column_stack([h_star, cost_L0, cost_L1]),axis=1),
        lw=15, alpha=0.1, color='b', label=r'$J(\pi)$')

ax.annotate(r"$B$", xy=(B + 0.01, 0.5), fontsize=14)
ax.annotate(r"$A$", xy=(A + 0.01, 0.5), fontsize=14)

plt.vlines(B, 0, (1 - B) * model.L1, linestyle="--")
plt.vlines(A, 0, A * model.L0, linestyle="--")

ax.set(xlim=(0, 1), ylim=(0, 0.5 * max(model.L0, model.L1)), ylabel="cost",
       xlabel=r"$\pi$", title=r"Cost function $J(\pi)$")

plt.legend(borderpad=1.1)
plt.show()


# %% [markdown]
# The cost function $ J $ equals $ \pi L_0 $ for $ \pi \leq B $, and $ (1-\pi) L_1 $ for $ \pi
# \geq A $.
#
# The slopes of the two linear pieces of the cost   function $ J(\pi) $ are determined by $ L_0 $
# and $ -L_1 $.
#
# The cost function $ J $ is smooth in the interior region, where the posterior
# probability assigned to $ f_1 $ is in the indecisive region $ \pi \in (B, A) $.
#
# The decision-maker continues to sample until the probability that he attaches to
# model $ f_1 $ falls below $ B $ or above $ A $.

# %% [markdown]
# ### Simulations
#
# The next figure shows the outcomes of 500 simulations of the decision process.
#
# On the left is a histogram of **stopping times**, i.e.,  the number of draws of $ z_k $ required to make a decision.
#
# The average number of draws is around 6.6.
#
# On the right is the fraction of correct decisions at the stopping time.
#
# In this case, the decision-maker is correct 80% of the time

# %% hide-output=false
def simulate(model, true_dist, h_star, π_0=0.5):

    """
    This function takes an initial condition and simulates until it
    stops (when a decision is made)
    """

    f0, f1 = model.f0, model.f1
    f0_rvs, f1_rvs = model.f0_rvs, model.f1_rvs
    π_grid = model.π_grid
    κ = model.κ

    if true_dist == "f0":
        f, f_rvs = model.f0, model.f0_rvs
    elif true_dist == "f1":
        f, f_rvs = model.f1, model.f1_rvs

    # Find cutoffs
    B, A = find_cutoff_rule(model, h_star)

    # Initialize a couple of useful variables
    decision_made = False
    π = π_0
    t = 0

    while decision_made is False:
        z = f_rvs()
        t = t + 1
        π = κ(z, π)
        if π < B:
            decision_made = True
            decision = 0
        elif π > A:
            decision_made = True
            decision = 1

    if true_dist == "f0":
        if decision == 0:
            correct = True
        else:
            correct = False

    elif true_dist == "f1":
        if decision == 1:
            correct = True
        else:
            correct = False

    return correct, π, t

def stopping_dist(model, h_star, ndraws=250, true_dist="f0"):

    """
    Simulates repeatedly to get distributions of time needed to make a
    decision and how often they are correct
    """

    tdist = np.empty(ndraws, int)
    cdist = np.empty(ndraws, bool)

    for i in range(ndraws):
        correct, π, t = simulate(model, true_dist, h_star)
        tdist[i] = t
        cdist[i] = correct

    return cdist, tdist

def simulation_plot(model):
    h_star = solve_model(model)
    ndraws = 500
    cdist, tdist = stopping_dist(model, h_star, ndraws)

    fig, ax = plt.subplots(1, 2, figsize=(16, 5))

    ax[0].hist(tdist, bins=np.max(tdist))
    ax[0].set_title(f"Stopping times over {ndraws} replications")
    ax[0].set(xlabel="time", ylabel="number of stops")
    ax[0].annotate(f"mean = {np.mean(tdist)}", xy=(max(tdist) / 2,
                   max(np.histogram(tdist, bins=max(tdist))[0]) / 2))

    ax[1].hist(cdist.astype(int), bins=2)
    ax[1].set_title(f"Correct decisions over {ndraws} replications")
    ax[1].annotate(f"% correct = {np.mean(cdist)}",
                   xy=(0.05, ndraws / 2))

    plt.show()

simulation_plot(model)

# %% [markdown]
# ### Comparative Statics
#
# Now let’s consider the following exercise.
#
# We double the cost of drawing an additional observation.
#
# Before you look, think about what will happen:
#
# - Will the decision-maker be correct more or less often?  
# - Will he make decisions sooner or later?  

# %% hide-output=false
model = WaldFriedman(c=2.5)
simulation_plot(model)

# %% [markdown]
# Increased cost per draw has induced the decision-maker to take fewer draws before deciding.
#
# Because he decides with fewer draws, the percentage of time he is correct drops.
#
# This leads to him having a higher expected loss when he puts equal weight on both models.
#
# To facilitate comparative statics, we invite you to adjust the parameters of the model
# and investigate
#
# - effects on the smoothness of the value function in the indecisive middle range
#   as we increase the number of grid points in the piecewise linear  approximation.  
# - effects of different settings for the cost parameters $ L_0, L_1, c $, the
#   parameters of two beta distributions $ f_0 $ and $ f_1 $, and the number
#   of points and linear functions $ m $ to use in the piece-wise continuous approximation to the value function.  
# - various simulations from $ f_0 $ and associated distributions of waiting times to making a decision.  
# - associated histograms of correct and incorrect decisions.  
#
#
# <p><a id=f1 href=#f1-link><strong>[1]</strong></a> The decision maker acts as if he believes that the sequence of random variables
# $ [z_{0}, z_{1}, \ldots] $ is *exchangeable*.  See [Exchangeability and Bayesian Updating](https://python.quantecon.org/exchangeable.html) and
# [[Kreps, 1988](https://python.quantecon.org/zreferences.html#id111)] chapter 11, for  discussions of exchangeability.

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

# Stochastic Approximation

If you haven't installed the quantecon Python library yet and you want to run this lecture, you need to execute the following line first:

```{code-cell} ipython3
!pip install quantecon
```

## Introduction

In this lecture we analyze a technique for optimizing functions and finding zeros (i.e., roots) of functions in settings where outcomes are only partially observable.

The technique is called [stochastic approximation](https://en.wikipedia.org/wiki/Stochastic_approximation) and dates back to early work by [Munro and Robbins](https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-22/issue-3/A-Stochastic-Approximation-Method/10.1214/aoms/1177729586.full).

Stochastic approximation can be thought of as a generalization of several other famous algorithms, including stochastic gradient descent.

It forms one of the key tools for training models built on neural networks, as well as in reinforcement learning.

Here we focus on relatively simple problems, in order to convey the key ideas.

We will use the following imports.

```{code-cell} ipython3
import numpy as np
import quantecon as qe
from numba import jit
import matplotlib.pyplot as plt
from scipy.optimize import bisect
```

## Background

+++

As mentioned above, stochastic approximation helps us solve optimization problems and nonlinear equations in settings of uncertainty.

Before we explain what it is, let's think about how it might be useful.

+++

### A firm problem

Suppose there is a firm that, each period, 

1. chooses inputs,
2. produces,
3. sells a certain number of its goods, and
4. receives a corresponding profit.

It wants to choose the input vector that maximizes expected profit.

+++

In Econ 101, we typically assume that the firm knows the exact relationship between inputs and outputs, as well as the full details of the demand function it faces.  

But is this true in reality?  

For example, the firm might be testing out a new product -- in which case it won't yet know exact production costs, or what the demand function will look like.

How can the firm maximize expected profits in this setting?

An econometrician might suggest that the firm 

1. makes some assumptions about functional forms, and
2. tries to estimate profit / demand / supply relationships.

But will these assumptions be valid?  

Do we have data to accurately estimate all these relationships?

Also, how can we get data on demand when the firm is selling a brand new product?

(Econometricians have methods for handling missing data but -- again -- will their assumptions be valid?)

+++

### A Monte Carlo solution

Here's one "model-free" way the firm could deal with its optimization problem while avoiding questionable assumptions:

1. Try many different combinations of inputs, recording resulting profits
2. Choose the combination that achieved highest profits.
  
But this approach could be very costly!

Not only would the firm have to try many input combinations, searching for the best one, they would also have to try each one of these combinations many times.

This is because trying a fixed input vector $x$ onces is not enough, since the result profit is random.

(Random disturbances push profits up and down in each period, even if inputs are fixed.)

Each input vector $x$ would have to be tried many times in order to estimate expected profits at $x$.

This means many trials $\times$ many trials, potentially losing money all the while.

+++

### Stochastic approximation

Stochastic approximation offers a way around this problem.

It suggests an algorithm for updating the input vector $x$ in each period, based on the observed outcome in that period, which converges in probability to a point that maximizes expected profits.

The algorithm is iterative and "on-line", meaning that the firm learns the best input vector while operating, rather than before starting  -- which is necessary in this case, since the firm needs to try different input vectors in order to learn about expected profits.

Before we discuss such this application further, we start with some very simple mathematical examples.

+++

## Deterministic problems

+++

As a starting point, let's suppose that we are interested in finding the zero (i.e., root) of the function 

$$ 
    f(x) = \ln x + \exp(x/4) - \tanh (x - 5) / 3 - 4
$$

```{code-cell} ipython3
def f(x):
    return np.log(x) + np.exp(x / 4) - np.tanh(x - 5) / 3 - 4
```

Here's what the function looks like:

```{code-cell} ipython3
x = np.linspace(0.1, 10, 200)
fig, ax = plt.subplots()
ax.plot(x, f(x), label='$f$')
ax.plot(x, np.zeros(len(x)), 'k--')
ax.legend()
plt.show()
```

Just from looking at the function, it seems like the zero is close to 3.5 ($x \approx 3.5$ implies $f(x) \approx 0$).

But we want a more accurate measurement.

+++

### A simple idea

+++

One we we could solve this problem numerically is by choosing some initial $x_0$ in the displayed region and then updating by

$$ x_{n+1} = x_n - \alpha_n f(x_n) $$

where $(\alpha_n)$ is a positive sequence converging to zero.

In other words, we take a

* small step up when $f(x_n) < 0$ and
* a small step down when $f(x_n) > 0$.

The size of the steps will decrease as we get closer to the zero of $f$ because 

1. $|f(x_n)|$ is (hopefully) getting close to zero, and
2. $\alpha_n$ is converging to zero.

We will call this routine **stochastic approximation**, even though no aspect of the routine is stochastic at this stage.

(Later we'll use an analogous method for stochastic problems.)

+++

Here's an implementation:

```{code-cell} ipython3
def s_approx(f, x_0=8, tol=1e-6, max_iter=10_000):
    x = x_0
    error, n = tol + 1, 0
    while n < max_iter and error > tol:
        α = (n + 1)**(-.5)
        x_new = x - α * f(x)
        error = abs(x_new - x)
        x = x_new
        n += n
    return x
```

The routine seems to work:

```{code-cell} ipython3
s_approx(f)
```

You should be able to confirm that in the code above that,

* convergence holds for for other initial conditions not too far from the root of $f$, and
* a constant learning rate $\alpha$ also works (can you see why?).

+++

### Special cases

+++

The stochastic approximation update rule

$$ x_{n+1} = x_n - \alpha_n f(x_n) $$

includes some important routines as special case.

One is [Newton iteration](https://en.wikipedia.org/wiki/Newton%27s_method), which updates according to

$$ x_{n+1} = x_n - (1/f'(x_n)) f(x_n) $$

We can see that Newton iteration is a version of stochastic approximation with a special choice of learning rate.

(In this case, we require that $f$ is increasing so that the learning rate $\alpha_n = 1/f'(x_n)$ is positive.)

+++

Another special case of stochastic approximation is **gradient descent**.

Indeed, suppose $F$ is a differentiable real function and we seek the minimum of this function (assuming it exists).

Gradient descent tells us to choose an initial guess $x_0$ and then update according to 

$$ x_{n+1} = x_n - \alpha_n f(x_n) 
    \quad \text{where} \quad
    f := F'
$$

The sequence $(\alpha_n)$ is a given learning rate.

We see immediately that this is a special case of stochastic approximation when the zero we seek is a zero of a derivative of a function one wishes to minimize.

+++

### Decreasing functions

+++

The stochastic approximation routine discussed above is aimed at handling increasing functions, since we take a small step down when $f(x_n)$ is positive and a small step up when $f(x_n)$ is negative.

When the function in question is decreasing we need to reverse the sign.

For example, suppose now that


$$ 
    f(x) = 10 - \ln x - \exp(x/4) + \tanh (x - 5) / 3 - 4
$$

```{code-cell} ipython3
def f(x):
    return 10 - np.log(x) - np.exp(x / 4) + np.tanh(x - 5) / 3 - 4
    
x = np.linspace(0.1, 10, 200)
fig, ax = plt.subplots()
ax.plot(x, f(x), label='$f$')
ax.plot(x, np.zeros(len(x)), 'k--')
ax.legend()
plt.show()
```

Now we set

$$ x_{n+1} = x_n + \alpha_n f(x_n) $$

where, as before, $(\alpha_n)$ is a positive sequence converging to zero.

Since $f(x_n)$ is positive when $x_n$ is low, we take a small step up and continue.

Similarly, when $x_n$ is above the root, we have $f(x_n) < 0$ and $x_{n+1}$ will be smaller.

This leads to convergence.

+++

**Exercise:** 

Modify the stochastic approximation code above to obtain an approximation of the zero of $f$.

+++

### An ODE perspective

+++

There is an ODE (ordinary differential equation) perspective on the stochastic approximation routine that provides another way to analyze the routine.

To see this, consider the scalar autonomous ODE

$$ \dot x = f(x) $$

To approximate the trajectory of associated with these dynamics we can use a standard numerical scheme, which starts with the standard first order approximation for a differentiable function:  for small $h$,

$$ x(t+h) \approx x(t) + h x'(t) $$

This suggests constructing a discrete trajectory via

$$ x_{n+1} = x_n + h f(x_n), $$

which is the same as our stochastic approximation rule for decreasing functions.

To understand why this matches stochastic approximation for decreasing rather than increasing functions, note that when $f$ is strictly decreasing and has a unique zero $x^*$ in the interval $(a, b)$, any solution to the ODE converges to $x^*$.

* For $x(t) < x^*$, we have $f(x(t))>0$ and hence $x(t)$ increases towards $x^*$.
* For $x(t) > x^*$, we have $f(x(t))<0$, and hence $x(t)$ decreases towards $x^*$.

Thus, the decreasing case corresponds to the setting where the zero $x^*$ is attracting for the ODE.

This means that, for this stable case, the stochastic approximation rule $x_{n+1} = x_n + \alpha_n f(x_n)$ converges to the stationary point of the ODE $\dot x = f(x)$.

+++

## Stochastic Problems

+++

As the name suggestions, stochastic approximation is really about solving problems that have random components.

For example, let's suppose that we want to find the zero of $f$ but we cannot directly observe $f(x)$ at arbitrary $x$.

Instead, after input $x_n$, we only observe a draw from $f(x_n) + D_n$, where $D_n$ is an independent draw from a fixed distribution $\phi$ with zero mean.

* we observe the sum $f(x_n) + D_n$ but cannot directly observe $f(x_n)$

How should we proceed?

+++

Drawing a direct analogy with the deterministic case, and assuming again that $f$ is an increasing function, we set

$$ 
    x_{n+1} = x_n - \alpha_n (f(x_n) + D_n) 
$$

Following [Monro and Robbins](https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-22/issue-3/A-Stochastic-Approximation-Method/10.1214/aoms/1177729586.full), we require that the learning rate obeys

$$ 
    \sum_n \alpha_n = \infty \quad \text{and} \quad \sum_n \alpha_n^2 < \infty 
$$

In particular, we will take $\alpha_n = (n+1)^{-0.6}$

We'll use the same function $f$ we used at the start, so that

```{code-cell} ipython3
@jit
def f(x):
    return np.log(x) + np.exp(x / 4) - np.tanh(x - 5) / 3 - 4
    
x = np.linspace(0.1, 10, 200)
fig, ax = plt.subplots()
ax.plot(x, f(x), label='$f$')
ax.plot(x, np.zeros(len(x)), 'k--')
ax.legend()
plt.show()
```

This means that the zero we are searching for is approximately $x=3.557$.

+++

The shocks will be normal with zero mean.

```{code-cell} ipython3
@jit
def generate_draw(x, σ=0.5):
    return f(x) + σ * np.random.randn()

@jit
def s_approx_beta(x_0=5.0, num_iter=1_000):
    x_sequence = np.empty(num_iter)
    x = x_0
    for n in range(num_iter):
        x_sequence[n] = x
        α = (n + 1)**(-0.6)
        x = x - α * generate_draw(x)
    return x_sequence
```

```{code-cell} ipython3
x = s_approx_beta()
```

```{code-cell} ipython3
x_star = 3.557
fig, ax = plt.subplots()
ax.plot(x, label='$(x_n)$')
ax.plot(range(len(x)), np.full(len(x), x_star), 'k--', label='$x^*$')
ax.legend()
plt.show()
```

We see that the sequence converges quite well to $x^* = 3.557$.

+++

Here's a function that simply returns the final estimate, rather than storing the whole sequence.

We'll run it at a higher number of iterations and see if the final estimate is close to 3.557.

```{code-cell} ipython3
@jit
def s_approx(x_0=5.0, num_iter=10_000_000):
    x = x_0
    for n in range(num_iter):
        α = (n + 1)**(-.6)
        x = x - α * generate_draw(x)
    return x
```

```{code-cell} ipython3
s_approx()
```

The result is pretty good.

For more on the convergence properties of of stochastic approximation, see, e.g., [this reference](https://arxiv.org/abs/2205.01303).

Note that the routine looks essentially the same in higher dimensions: when $f$ maps $\mathbb R^k$ to itself and each $D_k$ is a random vector in $\mathbb R^k$, we choose an initial $k$-vector $x_0$ and update via

$$ x_{n+1} = x_n - \alpha_n (f(x_n) + D_n)) $$

+++

## Application: profit maximization

+++

Now let's go back to the problem involving maximization of expected profit that we discussed at the start of the lecture.

We'll take on a toy version of this problem with just one input --- let's say it's labor.

Profits take the form

$$ \pi_n = \pi(x_n, \xi_n) $$

where $x_n$ is a value in $(0, \infty)$ denoting current labor input and $\xi_n$ is an independent draw from a fixed distribution $\phi$ on $\mathbb R$.

The firm manager hopes to maximize 

$$
    \bar \pi(x) := \mathbb E \, \pi(x, \xi) = \int \pi(x, z) \phi(dz)
$$

The manager doesn't know the function $\pi$ or the distribution $\phi$.

However, the manager can observe profits in each period.

To make our life a little easier, we'll also assume that the firm manager can observe local changes to profits.

(Perhaps, during one period, the manager can slightly vary work hours up and down to get an estimate of marginal returns.)

In particular, the manager can observe

$$
    \Delta_n := \pi^\prime_1(x_n, \xi_n) := \frac{\partial}{\partial x} \pi(x_n, \xi_n),
    \qquad n = 0, 1, \ldots
$$

Given a starting value of labor input, the firm now follows the update rule

$$
    x_{n+1} = x_n + \alpha_n \Delta_n
$$

where $\alpha_n$ is the learning rate.

+++

To experiment with these ideas, let's suppose that, unknown to the manager, 

* output obeys $q = q(x) = x^\alpha$ for some $\alpha \in (0,1)$,
* wages are constant at $w > 0$, and
* the inverse demand curve obeys $p = \xi \exp(- q)$, where $\ln \xi \sim N(\mu, \sigma)$.

As a result, profits are given by 

$$
    \pi = p q - w x = \xi \exp(-x^\alpha) x^\alpha - w x
$$

and expected profits are 

$$
    \bar \pi = \bar \xi \exp(-x^\alpha) x^\alpha - wx,
    \qquad \bar \xi := \mathbb E \xi = \exp\left(\mu + \frac{\sigma^2}{2} \right).
$$

+++

Here's the function for expected profits, plotted at a default set of parameters.

```{code-cell} ipython3
def bar_pi(x, params):
    α, w, μ, σ = params
    bar_xi = np.exp(μ + σ**2 / 2)
    return bar_xi * np.exp(-x**α) * x**α - w * x
```

```{code-cell} ipython3
default_params = 0.85, 1.0, 1.0, 0.05  # α, w, μ, σ

fig, ax = plt.subplots()
x = np.linspace(0.01, 1, 200)
y = bar_pi(x, default_params)
ax.plot(x, y)
plt.show()
```

Here's the expectation of the gradient of the profit function, derived from the calculations above.

```{code-cell} ipython3
def diff_bar_pi(x, params):
    α, w, μ, σ = params
    bar_xi = np.exp(μ + σ**2 / 2)
    return bar_xi * α * np.exp(-x**α) * (x**(α - 1) - x**(2*α - 1)) - w
```

```{code-cell} ipython3
fig, ax = plt.subplots()
x = np.linspace(0.01, 1, 200)
y = diff_bar_pi(x, default_params)
ax.plot(x, y)
ax.plot(x, 0 * x, 'k--')
plt.show()
```

If we have all this information in hand, we can easily find the root of the derivative and hence the maximizer.

```{code-cell} ipython3
x_star = bisect(lambda x: diff_bar_pi(x, default_params), 0.2, 0.6)
x_star
```

Now let's go back to the setting where we only get to see observations and need to apply stochastic approximation.

Here's a function to generate draws from the gradient

$$ 
 \frac{\partial}{\partial x} \pi(x_n, \xi_n)
$$

```{code-cell} ipython3
def generate_gradient_draw(x, params):
    α, w, μ, σ = params
    # Generate a draw of ξ_n
    xi = np.exp(μ + σ * np.random.randn())
    # Return the observation of the local gradient
    return xi * α * np.exp(-x**α) * (x**(α - 1) - x**(2*α - 1)) - w
```

Here's a routine for stochastic approximation, similar to the ones above.

This routine stores the sequence $(x_n)$, so we can plot it.

```{code-cell} ipython3
def stochastic_approx(params=default_params, x_0=0.1, num_iter=40):
    x_sequence = np.empty(num_iter+1)
    x = x_0
    x_sequence[0] = x
    for n in range(num_iter):
        α = 1 / (n + 1)
        x = x + α * generate_gradient_draw(x, params)
        x_sequence[n+1] = x
    return x_sequence
```

Let's generate and plot the sequence.

```{code-cell} ipython3
x = stochastic_approx()
```

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.plot(x, label='$(x_n)$')
ax.plot(range(len(x)), np.full(len(x), x_star), 'k--', label='$x^*$')
ax.legend()
plt.show()
```

In this case, the convergence is quite good.

+++

## Concluding comments

In this lecture we described the stochastic approximation algorithm and studied its behavior in some applications.

One important application of stochatic approximation in the field of artificial intelligence is reinforcement learning.

We study these connections in a separate lecture.

```{code-cell} ipython3

```

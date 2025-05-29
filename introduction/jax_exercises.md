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

# JAX Exercises

```{code-cell} ipython3
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
```

```{code-cell} ipython3
jax.devices()
```

## Linear Algebra

Consider the polynomial expression

$$
p(x) = a_0 + a_1 x + a_2 x^2 + \cdots a_N x^N 
$$

Write a function `p(x, coeff)` to evaluate`p(x)` using JAX arrays
and no for loop.

Your solution should be vectorized, so that `p` acts element by element on
arrays of `x` values.

Check your solution matches the output of `jnp.polyval` by plotting the output of
the two methods with the following data:

```{code-cell} ipython3
x = jnp.linspace(-4, 4, 200)               # Evaluation points
coef = jnp.array((2.0, -6.2, 10.6, 1.1))   # Polynomial coefficients
# Evaluate p at all points in x
# Do the same for jnp.polyval
# Plot the two functions using matplotlib, one with dashed lines.
# Check that the two functions agree.
# How would you check equality more precisely?
```

```{code-cell} ipython3
for _ in range(12):
    print("Solution below!")
```

This code does the job

```{code-cell} ipython3
def p(x, coef):
    powers = jnp.arange(len(coef))   # 0, 1, 2,...
    y = jnp.power(x, powers)         # 1, x, x**2,...
    return coef @ y
```

We replace `p` with a vectorized version:

```{code-cell} ipython3
p_vec = jax.vmap(p, (0, None))
```

We can do the plot like this:

```{code-cell} ipython3
x = jnp.linspace(-4, 4, 200)               # Evaluation points
coef = jnp.array((2.0, -6.2, 10.6, 1.1))   # Polynomial coefficients
y_1 = p_vec(x, coef)
y_2 = jnp.polyval(jnp.flip(coef), x)

fig, ax = plt.subplots()
ax.plot(x, y_1, '--', lw=2, label='$p$')
ax.plot(x, y_2, alpha=0.5, label='built-in JAX function')
ax.legend()
plt.show()
```

```{code-cell} ipython3
print("Checking equality more precisely:")
jnp.allclose(y_1, y_2)
```

## Random Numbers

The following code produces an estimate of $\pi$ that improves with $n$.

The code is accelerated on the CPU using Numba.

```{code-cell} ipython3
from random import uniform
import numpy as np
import numba

@numba.jit
def calculate_pi(n=10_000_000):
    count = 0
    for i in range(n):
        u, v = uniform(0, 1), uniform(0, 1)
        d = np.sqrt((u - 0.5)**2 + (v - 0.5)**2)
        if d < 0.5:
            count += 1

    area_estimate = count / n
    return area_estimate * 4  # dividing by radius**2
```

```{code-cell} ipython3
calculate_pi()
```

```{code-cell} ipython3
%timeit calculate_pi()
```

```{code-cell} ipython3
for _ in range(12):
    print("Solution below!")
```

```{code-cell} ipython3
@jax.jit
def jax_calculate_pi(keys, n=10_000_000):
    key1, key2 = keys
    u = jax.random.uniform(key1, (n,))
    v = jax.random.uniform(key2, (n,))
    d = jnp.sqrt((u - 0.5)**2 + (v - 0.5)**2)
    area_estimate = jnp.mean(d < 0.5)
    return area_estimate * 4  # dividing by radius**2
```

```{code-cell} ipython3
key = jax.random.PRNGKey(42)
key, subkey_1, subkey_2 = jax.random.split(key, 3)
```

```{code-cell} ipython3
jax_calculate_pi((subkey_1, subkey_2))
```

```{code-cell} ipython3
%timeit jax_calculate_pi((subkey_1, subkey_2))
```

## Linear Regression

Consider a linear regression problem of the form

$$
    y_i = x_i' \beta + u_i
    \quad i=1, \ldots, n
$$


The OLS parameter $\beta$ can be estimated using 

1. standard packages like `statsmodels` or
2. matrix algebra.

### Statsmodels

+++

Here's a regression using `statsmodels`:

```{code-cell} ipython3
import statsmodels.api as sm
import pandas as pd

# Load in data
df1 = pd.read_stata('https://github.com/QuantEcon/lecture-python/blob/master/source/_static/lecture_specific/ols/maketable1.dta?raw=true')
df1 = df1.dropna(subset=['logpgp95', 'avexpr'])

# Add a constant term
df1['const'] = 1

reg1 = sm.OLS(endog=df1['logpgp95'], exog=df1[['const', 'avexpr']], \
    missing='drop')
type(reg1)
results = reg1.fit()
print(results.summary())
```

We see that 

- the intercept $\hat{\beta}_0$ is $4.63$
- the slope $\hat{\beta}_1$ is $0.53$


### Matrix algebra

The linear equation we want to estimate is (written in matrix form)

$$
y = X\beta + u
$$

To solve for the unknown parameter $\beta$, we want to minimize
the sum of squared residuals

$$
\underset{\hat{\beta}}{\min} \hat{u}'\hat{u}
$$

Rearranging the first equation and substituting into the second
equation, we can write

$$
\underset{\hat{\beta}}{\min} \ (Y - X\hat{\beta})' (Y - X\hat{\beta})
$$

Solving this optimization problem gives the solution for the
$\hat{\beta}$ coefficients

$$
\hat{\beta} = (X'X)^{-1}X'y
$$

Using the above information, compute $\hat{\beta}$ for the same data set, using

```{code-cell} ipython3
# Define the X and y variables
y = jnp.asarray(df1['logpgp95'])
X = jnp.asarray(df1[['const', 'avexpr']])
```

Your results should be the same as those from the `statsmodels` output above.

```{code-cell} ipython3
for _ in range(12):
    print("Solution below!")
```

```{code-cell} ipython3
β_hat = jnp.linalg.solve(X.T @ X, X.T @ y)

# Print out the results from the 2 x 1 vector β_hat
print(f'β_0 = {β_hat[0]:.2}')
print(f'β_1 = {β_hat[1]:.2}')
```

```{code-cell} ipython3

```

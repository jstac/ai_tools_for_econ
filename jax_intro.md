---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.7
kernelspec:
  display_name: Python
  language: python3
  name: python3
---

# An Introduction to JAX

+++

# GPU

This lecture was built using a machine with JAX installed and access to a GPU.

To run this lecture on [Google Colab](https://colab.research.google.com/), click on the “play” icon top right, select Colab, and set the runtime environment to include a GPU.

To run this lecture on your own machine, you need to install [Google JAX](https://github.com/google/jax).

This lecture provides a short introduction to [Google JAX](https://github.com/google/jax).

Let’s see if we have an active GPU:

```{code-cell}
:hide-output: false

!nvidia-smi
```

## JAX as a NumPy Replacement

One way to use JAX is as a plug-in NumPy replacement. Let’s look at the
similarities and differences.

+++

### Similarities

The following import is standard, replacing `import numpy as np`:

```{code-cell}
:hide-output: false

import jax
import jax.numpy as jnp
```

Now we can use `jnp` in place of `np` for the usual array operations:

```{code-cell}
:hide-output: false

a = jnp.asarray((1.0, 3.2, -1.5))
```

```{code-cell}
:hide-output: false

print(a)
```

```{code-cell}
:hide-output: false

print(jnp.sum(a))
```

```{code-cell}
:hide-output: false

print(jnp.mean(a))
```

```{code-cell}
:hide-output: false

print(jnp.dot(a, a))
```

However, the array object `a` is not a NumPy array:

```{code-cell}
:hide-output: false

a
```

```{code-cell}
:hide-output: false

type(a)
```

Even scalar-valued maps on arrays return JAX arrays.

```{code-cell}
:hide-output: false

jnp.sum(a)
```

JAX arrays are also called “device arrays,” where term “device” refers to a
hardware accelerator (GPU or TPU).

(In the terminology of GPUs, the “host” is the machine that launches GPU operations, while the “device” is the GPU itself.)

Operations on higher dimensional arrays are also similar to NumPy:

```{code-cell}
:hide-output: false

A = jnp.ones((2, 2))
B = jnp.identity(2)
A @ B
```

```{code-cell}
:hide-output: false

from jax.numpy import linalg
```

```{code-cell}
:hide-output: false

linalg.inv(B)   # Inverse of identity is identity
```

```{code-cell}
:hide-output: false

linalg.eigh(B)  # Computes eigenvalues and eigenvectors
```

### Differences

One difference between NumPy and JAX is that JAX currently uses 32 bit floats by default.

This is standard for GPU computing and can lead to significant speed gains with small loss of precision.

However, for some calculations precision matters.  In these cases 64 bit floats can be enforced via the command

```{code-cell}
:hide-output: false

jax.config.update("jax_enable_x64", True)
```

Let’s check this works:

```{code-cell}
:hide-output: false

jnp.ones(3)
```

As a NumPy replacement, a more significant difference is that arrays are treated as **immutable**.

For example, with NumPy we can write

```{code-cell}
:hide-output: false

import numpy as np
a = np.linspace(0, 1, 3)
a
```

and then mutate the data in memory:

```{code-cell}
:hide-output: false

a[0] = 1
a
```

In JAX this fails:

```{code-cell}
:hide-output: false

a = jnp.linspace(0, 1, 3)
a
```

```{code-cell}
:hide-output: false

a[0] = 1
```

In line with immutability, JAX does not support inplace operations:

```{code-cell}
:hide-output: false

a = np.array((2, 1))
a.sort()
a
```

```{code-cell}
:hide-output: false

a = jnp.array((2, 1))
a_new = a.sort()
a, a_new
```

The designers of JAX chose to make arrays immutable because JAX uses a
functional programming style.  More on this below.

Note that, while mutation is discouraged, it is in fact possible with `at`, as in

```{code-cell}
:hide-output: false

a = jnp.linspace(0, 1, 3)
id(a)
```

```{code-cell}
:hide-output: false

a
```

```{code-cell}
:hide-output: false

a.at[0].set(1)
```

We can check that the array is mutated by verifying its identity is unchanged:

```{code-cell}
:hide-output: false

id(a)
```

## Random Numbers

Random numbers are also a bit different in JAX, relative to NumPy.  Typically, in JAX, the state of the random number generator needs to be controlled explicitly.

```{code-cell}
:hide-output: false

import jax.random as random
```

First we produce a key, which seeds the random number generator.

```{code-cell}
:hide-output: false

key = random.PRNGKey(1)
```

```{code-cell}
:hide-output: false

type(key)
```

```{code-cell}
:hide-output: false

print(key)
```

Now we can use the key to generate some random numbers:

```{code-cell}
:hide-output: false

x = random.normal(key, (3, 3))
x
```

If we use the same key again, we initialize at the same seed, so the random numbers are the same:

```{code-cell}
:hide-output: false

random.normal(key, (3, 3))
```

To produce a (quasi-) independent draw, best practice is to “split” the existing key:

```{code-cell}
:hide-output: false

key, subkey = random.split(key)
```

```{code-cell}
:hide-output: false

random.normal(key, (3, 3))
```

```{code-cell}
:hide-output: false

random.normal(subkey, (3, 3))
```

The function below produces `k` (quasi-) independent random `n x n` matrices using this procedure.

```{code-cell}
:hide-output: false

def gen_random_matrices(key, n, k):
    matrices = []
    for _ in range(k):
        key, subkey = random.split(key)
        matrices.append(random.uniform(subkey, (n, n)))
    return matrices
```

```{code-cell}
:hide-output: false

matrices = gen_random_matrices(key, 2, 2)
for A in matrices:
    print(A)
```

One point to remember is that JAX expects tuples to describe array shapes, even for flat arrays.  Hence, to get a one-dimensional array of normal random draws we use `(len, )` for the shape, as in

```{code-cell}
:hide-output: false

random.normal(key, (5, ))
```

## JIT compilation

The JAX just-in-time (JIT) compiler accelerates logic within functions by fusing linear
algebra operations into a single optimized kernel that the host can
launch on the GPU / TPU (or CPU if no accelerator is detected).

+++

### A first example

To see the JIT compiler in action, consider the following function.

```{code-cell}
:hide-output: false

def f(x):
    a = 3*x + jnp.sin(x) + jnp.cos(x**2) - jnp.cos(2*x) - x**2 * 0.4 * x**1.5
    return jnp.sum(a)
```

Let’s build an array to call the function on.

```{code-cell}
:hide-output: false

n = 50_000_000
x = jnp.ones(n)
```

How long does the function take to execute?

```{code-cell}
:hide-output: false

%time f(x).block_until_ready()
```

>**Note**
>
>Here, in order to measure actual speed, we use the `block_until_ready()` method
to hold the interpreter until the results of the computation are returned from
the device. This is necessary because JAX uses asynchronous dispatch, which
allows the Python interpreter to run ahead of GPU computations.

The code doesn’t run as fast as we might hope, given that it’s running on a GPU.

But if we run it a second time it becomes much faster:

```{code-cell}
:hide-output: false

%time f(x).block_until_ready()
```

This is because the built in functions like `jnp.cos` are JIT compiled and the
first run includes compile time.

Why would JAX want to JIT-compile built in functions like `jnp.cos` instead of
just providing pre-compiled versions, like NumPy?

The reason is that the JIT compiler can specialize on the *size* of the array
being used, which is helpful for parallelization.

For example, in running the code above, the JIT compiler produced a version of `jnp.cos` that is
specialized to floating point arrays of size `n = 50_000_000`.

We can check this by calling `f` with a new array of different size.

```{code-cell}
:hide-output: false

m = 50_000_001
y = jnp.ones(m)
```

```{code-cell}
:hide-output: false

%time f(y).block_until_ready()
```

Notice that the execution time increases, because now new versions of
the built-ins like `jnp.cos` are being compiled, specialized to the new array
size.

If we run again, the code is dispatched to the correct compiled version and we
get faster execution.

```{code-cell}
:hide-output: false

%time f(y).block_until_ready()
```

The compiled versions for the previous array size are still available in memory
too, and the following call is dispatched to the correct compiled code.

```{code-cell}
:hide-output: false

%time f(x).block_until_ready()
```

### Compiling the outer function

We can do even better if we manually JIT-compile the outer function.

```{code-cell}
:hide-output: false

f_jit = jax.jit(f)   # target for JIT compilation
```

Let’s run once to compile it:

```{code-cell}
:hide-output: false

f_jit(x)
```

And now let’s time it.

```{code-cell}
:hide-output: false

%time f_jit(x).block_until_ready()
```

Note the speed gain.

This is because the array operations are fused and no intermediate arrays are created.

Incidentally, a more common syntax when targetting a function for the JIT
compiler is

```{code-cell}
:hide-output: false

@jax.jit
def f(x):
    a = 3*x + jnp.sin(x) + jnp.cos(x**2) - jnp.cos(2*x) - x**2 * 0.4 * x**1.5
    return jnp.sum(a)
```

## Functional Programming

From JAX’s documentation:

*When walking about the countryside of Italy, the people will not hesitate to tell you that JAX has “una anima di pura programmazione funzionale”.*

In other words, JAX assumes a functional programming style.

The major implication is that JAX functions should be pure.

A pure function will always return the same result if invoked with the same inputs.

In particular, a pure function has

- no dependence on global variables and  
- no side effects  


JAX will not usually throw errors when compiling impure functions but execution becomes unpredictable.

Here’s an illustration of this fact, using global variables:

```{code-cell}
:hide-output: false

a = 1  # global

@jax.jit
def f(x):
    return a + x
```

```{code-cell}
:hide-output: false

x = jnp.ones(2)
```

```{code-cell}
:hide-output: false

f(x)
```

In the code above, the global value `a=1` is fused into the jitted function.

Even if we change `a`, the output of `f` will not be affected — as long as the same compiled version is called.

```{code-cell}
:hide-output: false

a = 42
```

```{code-cell}
:hide-output: false

f(x)
```

Changing the dimension of the input triggers a fresh compilation of the function, at which time the change in the value of `a` takes effect:

```{code-cell}
:hide-output: false

x = jnp.ones(3)
```

```{code-cell}
:hide-output: false

f(x)
```

Moral of the story: write pure functions when using JAX!

+++

## Gradients

JAX can use automatic differentiation to compute gradients.

This can be extremely useful for optimization and solving nonlinear systems.

We will see significant applications later in this lecture series.

For now, here’s a very simple illustration involving the function

```{code-cell}
:hide-output: false

def f(x):
    return (x**2) / 2
```

Let’s take the derivative:

```{code-cell}
:hide-output: false

f_prime = jax.grad(f)
```

```{code-cell}
:hide-output: false

f_prime(10.0)
```

Let’s plot the function and derivative, noting that $ f'(x) = x $.

```{code-cell}
:hide-output: false

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
x_grid = jnp.linspace(-4, 4, 200)
ax.plot(x_grid, f(x_grid), label="$f$")
ax.plot(x_grid, [f_prime(x) for x in x_grid], label="$f'$")
ax.legend(loc='upper center')
plt.show()
```

We defer further exploration of automatic differentiation with JAX until [Adventures with Autodiff](https://jax.quantecon.org/autodiff.html).

+++

## Writing vectorized code

Writing fast JAX code requires shifting repetitive tasks from loops to array processing operations, so that the JAX compiler can easily understand the whole operation and generate more efficient machine code.

This procedure is called **vectorization** or **array programming**, and will be familiar to anyone who has used NumPy or MATLAB.

In most ways, vectorization is the same in JAX as it is in NumPy.

But there are also some differences, which we highlight here.

As a running example, consider the function

$$
f(x,y) = \frac{\cos(x^2 + y^2)}{1 + x^2 + y^2}
$$

Suppose that we want to evaluate this function on a square grid of $ x $ and $ y $ points and then plot it.

To clarify, here is the slow `for` loop version.

```{code-cell}
:hide-output: false

@jax.jit
def f(x, y):
    return jnp.cos(x**2 + y**2) / (1 + x**2 + y**2)

n = 80
x = jnp.linspace(-2, 2, n)
y = x

z_loops = np.empty((n, n))
```

```{code-cell}
:hide-output: false

%%time
for i in range(n):
    for j in range(n):
        z_loops[i, j] = f(x[i], y[j])
```

Even for this very small grid, the run time is extremely slow.

(Notice that we used a NumPy array for `z_loops` because we wanted to write to it.)

OK, so how can we do the same operation in vectorized form?

If you are new to vectorization, you might guess that we can simply write

```{code-cell}
:hide-output: false

z_bad = f(x, y)
```

But this gives us the wrong result because JAX doesn’t understand the nested for loop.

```{code-cell}
:hide-output: false

z_bad.shape
```

Here is what we actually wanted:

```{code-cell}
:hide-output: false

z_loops.shape
```

To get the right shape and the correct nested for loop calculation, we can use a `meshgrid` operation designed for this purpose:

```{code-cell}
:hide-output: false

x_mesh, y_mesh = jnp.meshgrid(x, y)
```

Now we get what we want and the execution time is very fast.

```{code-cell}
:hide-output: false

%%time
z_mesh = f(x_mesh, y_mesh).block_until_ready()
```

Let’s run again to eliminate compile time.

```{code-cell}
:hide-output: false

%%time
z_mesh = f(x_mesh, y_mesh).block_until_ready()
```

Let’s confirm that we got the right answer.

```{code-cell}
:hide-output: false

jnp.allclose(z_mesh, z_loops)
```

Now we can set up a serious grid and run the same calculation (on the larger grid) in a short amount of time.

```{code-cell}
:hide-output: false

n = 6000
x = jnp.linspace(-2, 2, n)
y = x
x_mesh, y_mesh = jnp.meshgrid(x, y)
```

```{code-cell}
:hide-output: false

%%time
z_mesh = f(x_mesh, y_mesh).block_until_ready()
```

Let’s run again to get rid of compile time.

```{code-cell}
:hide-output: false

%%time
z_mesh = f(x_mesh, y_mesh).block_until_ready()
```

But there is one problem here: the mesh grids use a lot of memory.

```{code-cell}
:hide-output: false

x_mesh.nbytes + y_mesh.nbytes
```

By comparison, the flat array `x` is just

```{code-cell}
:hide-output: false

x.nbytes  # and y is just a pointer to x
```

This extra memory usage can be a big problem in actual research calculations.

So let’s try a different approach using [jax.vmap](https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html)

First we vectorize `f` in `y`.

```{code-cell}
:hide-output: false

f_vec_y = jax.vmap(f, in_axes=(None, 0))  
```

In the line above, `(None, 0)` indicates that we are vectorizing in the second argument, which is `y`.

Next, we vectorize in the first argument, which is `x`.

```{code-cell}
:hide-output: false

f_vec = jax.vmap(f_vec_y, in_axes=(0, None))
```

With this construction, we can now call the function $ f $ on flat (low memory) arrays.

```{code-cell}
:hide-output: false

%%time
z_vmap = f_vec(x, y).block_until_ready()
```

We run it again to eliminate compile time.

```{code-cell}
:hide-output: false

%%time
z_vmap = f_vec(x, y).block_until_ready()
```

The execution time is essentially the same as the mesh operation but we are using much less memory.

And we produce the correct answer:

```{code-cell}
:hide-output: false

jnp.allclose(z_vmap, z_mesh)
```

## Exercises

+++

## Exercise 2.1

In the Exercise section of [a lecture on Numba and parallelization](https://python-programming.quantecon.org/parallelization.html), we used Monte Carlo to price a European call option.

The code was accelerated by Numba-based multithreading.

Try writing a version of this operation for JAX, using all the same
parameters.

If you are running your code on a GPU, you should be able to achieve
significantly faster execution.

+++

## Solution to[ Exercise 2.1](https://jax.quantecon.org/#jax_intro_ex2)

Here is one solution:

```{code-cell}
:hide-output: false

M = 10_000_000

n, β, K = 20, 0.99, 100
μ, ρ, ν, S0, h0 = 0.0001, 0.1, 0.001, 10, 0

@jax.jit
def compute_call_price_jax(β=β,
                           μ=μ,
                           S0=S0,
                           h0=h0,
                           K=K,
                           n=n,
                           ρ=ρ,
                           ν=ν,
                           M=M,
                           key=jax.random.PRNGKey(1)):

    s = jnp.full(M, np.log(S0))
    h = jnp.full(M, h0)
    for t in range(n):
        key, subkey = jax.random.split(key)
        Z = jax.random.normal(subkey, (2, M))
        s = s + μ + jnp.exp(h) * Z[0, :]
        h = ρ * h + ν * Z[1, :]
    expectation = jnp.mean(jnp.maximum(jnp.exp(s) - K, 0))
        
    return β**n * expectation
```

Let’s run it once to compile it:

```{code-cell}
:hide-output: false

%%time 
compute_call_price_jax().block_until_ready()
```

And now let’s time it:

```{code-cell}
:hide-output: false

%%time 
compute_call_price_jax().block_until_ready()
```

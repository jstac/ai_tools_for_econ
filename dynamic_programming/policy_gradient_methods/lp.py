"""
We consider a version of the Long-Plosser multisector optimal growth model with
Bellman equation

    v(y) = max_{l, X, c} { u(c, z) + β E v(f(l, X, λ)) }

subject to

    c_j + Σ_i X_{ij} = y_j,
    z + Σ_i l_i = H
    [f(l, X, λ)]_i = λ_i l_i^{b_i} Π_j X_{ij}^{a_{ij}}

Here

* y is an N-vector of outputs 
* λ is an N-vector of IID shocks 
* c is an N-vector of consumption quantities
* z is leisure
* X is an NxN matrix of inputs, with X_{ij} the quantity of commodity j used to produce commodity i
* u(c, z) = θ_0 ln z + Σ_i θ_i ln c_i

"""

import numpy as np
from dataclasses import dataclass, astuple, field
import quantecon as qe
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from functools import partial


# ## Code for Solving the Bellman Equation

# +
@dataclass
class LP:
    """
    Class for the Long and Plosser (1983) model
    """
    θ: jax.Array  # parameters in u(C, Z)
    b: jax.Array  # parameters in production function
    A: jax.Array
    grid_size: int
    num_nodes: int = 1  # number of nodes for quadrature
    grid_min: float = 1e-5
    grid_max: float = 20
    H: float = 1  # labor supply normalized to 1
    β: float = 0.97


def u(self, C, Z):
    """Utility function U(C_t, Z_t)
    C: (N,) array
    Z: float
    """
    return θ[0] * jnp.log(Z) + jnp.dot(θ[1:], jnp.log(C))

def f(self, L, X, λ):
    """Production function f(L, X, λ) -> (N, ) array of output
    L: (N, ) array of labor inputs
    X: (N, N) array of commodity inputs
    λ: (N, ) array of shocks"""
    b, A = self.b, self.A
    return λ * L**b * jnp.prod(X**A, 1)

@partial(jax.jit, static_argnames=('N'))
def v_star(self, Y):
    """
    Analytical value function. Equation (9) in the paper, where J(λ) = 0 for
    lognormal iid shocks by (11) and K is derived by plugging (9) into the
    Bellman equation.
    """
    H, β, θ, b, A, N = self.H, self.β, self.θ, self.b, self.A, self.N
    gamma = θ[1:] @ jnp.linalg.inv(jnp.eye(N) - β * A)  # equation (10b)
    sum_ga = gamma[:, None] * A  # matrix that represents gamma_i * a_ij
    sum_gb = θ[0] + β * jnp.sum(gamma * b)
    K = (θ[0] * jnp.log(H * θ[0] / sum_gb) # leisure
         + jnp.sum(θ[1:] * jnp.log(θ[1:]/gamma))  # consumption
         + β * jnp.sum(gamma * b * jnp.log(β * H * gamma * b / sum_gb))
         + β * jnp.sum(sum_ga * jnp.log(β * sum_ga/gamma[None, :]))) / (1 - β)
    return jnp.sum(gamma * jnp.log(Y)) + K


# Utilities

def get_XCL(x, Y, N):
    """Return commodity input matrix X, consumption vector C,
    and labor input vector L."""
    X = x.reshape(N, N)
    # budget constraints: equations (3)--(4) in the paper
    C = Y - jnp.maximum(X, 0.).sum(0)  # negative inputs not allowed
    L = jnp.ones(N)
    return X, C, L


def compute_utility(C, H, u, tol):
    Z = H  # eliminate labor choice
    util = u(jnp.maximum(C, tol), Z)  # negative consumption not allowed
    return util


def sigmoid(x):
    return 1/(1 + jnp.exp(-x)) - 0.5


def compute_penalty(C, X, penalty_factor):
    # return jnp.sum(jnp.minimum(C, 0.)**2 + jnp.minimum(X, 0.)**2)*penalty_factor
    return jnp.sum(-jnp.minimum(C, 0.) - jnp.minimum(X, 0.))*penalty_factor
    # return jnp.sum(sigmoid(-jnp.minimum(C, 0.)) + sigmoid(-jnp.minimum(X, 0.))) * penalty_factor


# -

@dataclass
class BellmanConfig:
    """
    Class for Bellman operator configurations
    """
    max_iter: int = 200
    tol: float = 1e-10
    opt_tol: float = 1e-16
    penalty_factor: float = 1e4
    learning_rate: float = 0.03
    opt_name: str = 'adam'
    kwargs: dict = field(default_factory=dict)


def configure_optimizer(opt_name, learning_rate, kwargs):
    """Configure an optax optimizer"""
    if opt_name == "adam":
        return optax.adam(learning_rate=learning_rate, **kwargs)
    elif opt_name == "fromage":
        return optax.fromage(learning_rate=learning_rate, **kwargs)
    elif opt_name == "sgd":
        return optax.sgd(learning_rate=learning_rate, **kwargs)
    elif opt_name == "lamb":
        return optax.lamb(learning_rate=learning_rate, **kwargs)
    elif opt_name == "lars":
        return optax.lars(learning_rate=learning_rate, **kwargs)
    elif opt_name == "lion":
        return optax.lion(learning_rate=learning_rate, **kwargs)
    elif opt_name == "novograd":
        return optax.novograd(learning_rate=learning_rate, **kwargs)
    elif opt_name == "rmsprop":
        return optax.rmsprop(learning_rate=learning_rate, **kwargs)
    elif opt_name == "sm3":
        return optax.sm3(learning_rate=learning_rate, **kwargs)
    elif opt_name == "yogi":
        return optax.yogi(learning_rate=learning_rate, **kwargs)
    elif opt_name == "adabelief":
        return optax.adabelief(learning_rate=learning_rate, **kwargs)
    elif opt_name == "adadelta":
        return optax.adadelta(learning_rate=learning_rate, **kwargs)
    elif opt_name == "nadam":
        return optax.nadam(learning_rate=learning_rate, **kwargs)
    else:
        raise ValueError('Wrong optimizer.')


def T_factory(lp, config=BellmanConfig()):
    """Generate a jitted Bellman operator"""
    grids, N, shocks, weights = lp.grids, lp.N, lp.shocks, lp.weights
    β, H, u, f = lp.β, lp.H, lp.u, lp.f
    max_iter, tol, opt_tol, penalty_factor, learning_rate, opt_name, kwargs = astuple(config)

    @jax.jit
    def T_y(Y, v):
        """Evaluate Tv at a single state"""
        v_fun = RegularGridInterpolator(grids, v, bounds_error=False, fill_value=None)

        def cont_value_quad(L, X):
            """Compute continuation value βEv(S') using Gauss-Hermite quadrature"""
            Y_next = f(L, X, jnp.exp(shocks))
            # restrict the state to be inside the grid
            # without this, v_fun might extrapolate to negative values
            Y_next = jnp.clip(Y_next, lp.grids[:, 0], lp.grids[:, -1])
            return β * jnp.dot(v_fun(Y_next), weights)

        def state_action_value(x):
            """Compute - u(C, Z) - βEv(S')
            x is an array of shape (N^2,)
            """
            X, C, L = get_XCL(x, Y, N)
            util = compute_utility(C, H, u, tol)
            
            # penalty term for infeasible x
            penalty = compute_penalty(C, X, penalty_factor)
            res = -(util + cont_value_quad(L, jnp.clip(X, tol, Y)) - penalty)
            return res

        # solve the minimization problem using gradient descent from jaxopt
        x0 = jnp.tile(jnp.array([y/(N+1) for y in Y]), N)
        # x0 = jnp.zeros(N**2) + tol
        lb = jnp.zeros(N**2) + tol
        ub = jnp.tile(Y, N) - tol

        u_min = lp.u(lp.grids[:, 0], 1)
        u_max = lp.u(lp.grids[:, -1], 1)
        # lr = learning_rate * ((lp.u(Y, lp.H) - u_min + 0.1) / (u_max - u_min))**0.7
        # lr = learning_rate * jnp.sum(Y) / jnp.sum(lp.grids[:, -1])
        opt = configure_optimizer(opt_name, learning_rate, kwargs)
        params, state = run_opt(x0, state_action_value, opt, max_iter=max_iter, tol=opt_tol, 
                                lb=lb, ub=ub)

        return -state_action_value(params), params, state

    # vmap T_hat_y to accept a vector of states of shape (N, M)
    T = jax.vmap(T_y, in_axes=(1, None))
    return T


def T_hat_factory(lp, h, h_inv, config=BellmanConfig()):
    """Generate a jitted conjugate Bellman operator"""
    grids, N, shocks, weights = lp.grids, lp.N, lp.shocks, lp.weights
    β, H, u, f = lp.β, lp.H, lp.u, lp.f
    max_iter, tol, opt_tol, penalty_factor, learning_rate, opt_name, kwargs = astuple(config)

    @jax.jit
    def T_hat_y(Y, w):
        """Evaluate Tv at a single state"""
        w_fun = RegularGridInterpolator(grids, w, bounds_error=False, fill_value=None)

        def cont_value_quad(L, X):
            """Compute continuation value βEv(S') using Gauss-Hermite quadrature"""
            Y_next = f(L, X, jnp.exp(shocks))
            # restrict the state to be inside the grid
            # without this, w_fun might extrapolate to negative values
            Y_next = jnp.clip(Y_next, lp.grids[:, 0], lp.grids[:, -1])
            return β * jnp.dot(h_inv(w_fun(Y_next)), weights)

        def state_action_value(x):
            """Compute - u(C, Z) - βEv(S')
            x is an array of shape (N^2,)
            """
            X, C, L = get_XCL(x, Y, N)
            util = compute_utility(C, H, u, tol)
            
            # penalty term for infeasible x
            penalty = compute_penalty(C, X, penalty_factor)
            res = -h(util + cont_value_quad(L, jnp.clip(X, tol, Y)) - penalty)
            return res

        # solve the minimization problem using optax
        x0 = jnp.tile(jnp.array([y/(N+1) for y in Y]), N)
        # x0 = jnp.zeros(N**2) + tol
        lb = jnp.zeros(N**2) + tol
        ub = jnp.tile(Y, N) - tol

        u_min = lp.u(lp.grids[:, 0], 1)
        u_max = lp.u(lp.grids[:, -1], 1)
        # lr = learning_rate * ((lp.u(Y, lp.H) - u_min + 0.1) / (u_max - u_min))**0.7
        # lr = learning_rate * jnp.sum(Y) / jnp.sum(lp.grids[:, -1])
        opt = configure_optimizer(opt_name, learning_rate, kwargs)
        params, state = run_opt(x0, state_action_value, opt, max_iter=max_iter, tol=opt_tol,
                               lb=lb, ub=ub)
        
        return -state_action_value(params), params, state

    # vmap T_hat_y to accept a vector of states of shape (N, M)
    T_hat = jax.vmap(T_hat_y, in_axes=(1, None))
    return T_hat



def solve_model(lp,
                v0=None,
                tol=1e-6,
                method="value_iteration",
                max_iter=1000,
                config=BellmanConfig(opt_tol=1e-8),
                verbose=True,
                print_skip=25):
    """
    Solve model by iterating with the Bellman operator.
    """

    # Set up loop
    if v0 is None:
        v = sum([np.log(lp.mesh[n]) for n in range(lp.N)])  # Initial condition
    else:
        v = v0
    i = 0
    error = tol + 1
    T = T_factory(lp, config)

    if method == "value_iteration":
        while i < max_iter and error > tol:
            v_new, x, state = T(jnp.stack(lp.mesh).reshape(lp.N, -1), v)
            v_new = v_new.reshape((lp.grid_size,)*lp.N)
            error = jnp.max(jnp.abs(v - v_new))
            i += 1
            if verbose and i % print_skip == 0:
                print(f"Error at iteration {i} is {error}.")
            v = v_new
#     elif method == "policy_iteration":
    else:
        raise ValueError('Wrong method')
    
    if error > tol:
        print("Failed to converge!")
    elif verbose:
        print(f"\nConverged in {i} iterations.")

    return v_new


def solve_model_conjugate(lp,
                          h, h_inv,
                          w0=None,
                          tol=1e-6,
                          method="value_iteration",
                          max_iter=1000,
                          config=BellmanConfig(),
                          verbose=True,
                          print_skip=25):
    """
    Solve model by iterating with the conjugate Bellman operator.
    """

    # Set up loop
    if w0 is None:
        w = jnp.exp(sum([jnp.log(lp.mesh[n]) for n in range(lp.N)]))  # Initial condition
    else:
        w = w0
    i = 0
    error = tol + 1
    T_hat = T_hat_factory(lp, h, h_inv, config)

    if method == "value_iteration":
        while i < max_iter and error > tol:
            w_new, x, state = T_hat(jnp.stack(lp.mesh).reshape(lp.N, -1), w)
            w_new = w_new.reshape((lp.grid_size,)*lp.N)
            error = jnp.max(jnp.abs(h_inv(w) - h_inv(w_new)))
            i += 1
            if verbose and i % print_skip == 0:
                # print(otu.tree_get(state, 'count'))
                print(f"Error at iteration {i} is {error}.")
            w = w_new
#     elif method == "policy_iteration":
    else:
        raise ValueError('Wrong method')
    
    if error > tol:
        print("Failed to converge!")
    elif verbose:
        print(f"\nConverged in {i} iterations.")

    return w_new





# ## One-Sector Case

# +
# Here the transformation is chosen by trial and error to make the transformed value function appear linear
factor = 0.75
cons = 15

@jax.jit
def h(x):
    return jnp.exp(factor*x + cons)

@jax.jit
def h_inv(x):
    return (jnp.log(x) - cons)/factor


# -

n = 1000
gmin = 1e-7
gmax = 20.0
θ = jnp.array([1., 1.])
b = jnp.array([0.3])
A = jnp.array([[0.3]])

# %%time
lp = LP(grid_size=n, θ=θ, b=b, A=A, grid_min=gmin, grid_max=gmax)
v = solve_model(lp, tol=1e-20, max_iter=1000, config=BellmanConfig(opt_tol=1e-8, learning_rate=0.001, max_iter=1000, opt_name='lamb'))

# %%time
lp = LP(grid_size=n, θ=θ, b=b, A=A, grid_min=gmin, grid_max=gmax)
w = solve_model_conjugate(lp, h, h_inv, tol=1e-20, config=BellmanConfig(opt_tol=1e-8, learning_rate=0.001, max_iter=1000, opt_name='lamb'))

# One thing I noticed about solving these problems using `optax` is that VFI might not "converge" but the results are still very accurate. So here I just let them run 1000 iterations and compare results.
#
# My guess is that linear interpolation has caused some problems with the gradient descent algorithms.

# Next, we compute the analytical value function. Since we elliminate labor choice, we set $b \approx 0$ to get the correct result.

lp0 = LP(grid_size=n, θ=θ, b=jnp.array([1e-50]), A=A, grid_min=gmin, grid_max=gmax)
v0 = jax.vmap(lp0.v_star, in_axes=(1))(jnp.stack(lp0.mesh))

# As we can see, the value function computated from the original problem is much more inaccurate close to 0

# +
fig, ax = plt.subplots()

ax.plot(lp.grids[0], v, '--', lw=2, alpha=0.6,
        label='Approximate value function')
ax.plot(lp.grids[0], v0, lw=2, alpha=0.6,
        label='Analytical value function')
ax.legend()
plt.show()

# +
fig, ax = plt.subplots()

ax.plot(lp.grids[0], h_inv(w), '--', lw=2, alpha=0.6,
        label='Approximate value function (conjugate)')

ax.plot(lp.grids[0], v0, lw=2, alpha=0.6,
        label='Analytical value function')
ax.legend()
plt.show()
# -

# The transformed value function is more linear and much more accurate

# +
fig, ax = plt.subplots()

ax.plot(lp.grids[0], w, '--', lw=2, alpha=0.6,
        label='Approximate transformed value function')

ax.plot(lp.grids[0], h(v0), lw=2, alpha=0.6,
        label='Analytical transformed value function')
ax.legend()
plt.show()
# -

# Compare the $L_\infty$ distance to the analytical value function

jnp.max(jnp.abs(v - v0))

jnp.max(jnp.abs(h_inv(w) - v0))

# ## Two-Sector Case

# For the two-sector case, we get similar results.

# +
# By trial and error
factor = 0.4
cons = 60

@jax.jit
def h(x):
    return jnp.exp(factor*x + cons)

@jax.jit
def h_inv(x):
    return (jnp.log(x) - cons)/factor


# -

n = 500
gmin = 1e-6
gmax = 20.0
θ = np.array([1, 1, 1])
b = jnp.array([[0.2, 0.6]])
A = jnp.array([[0.1, 0.7], [0.3, 0.1]])
tol = gmin*0.01

# schedule = optax.warmup_cosine_decay_schedule(init_value=0.0, peak_value=1e-1, warmup_steps=200, 
                                              # decay_steps=500, end_value=0.0)

# %%time
lp = LP(θ=θ, b=b, A=A, grid_min=gmin, grid_size=n, grid_max=gmax)
v = solve_model(lp, max_iter=500, print_skip=1, 
                # v0=v,
                config=BellmanConfig(tol=tol, opt_tol=1e-8, max_iter=10000, learning_rate=1e-3, opt_name='adam'))

# %%time
lp = LP(θ=θ, b=b, A=A, grid_min=gmin, grid_size=n, grid_max=gmax)
w = solve_model_conjugate(lp, h, h_inv, max_iter=500, print_skip=1,
                          # w0=w,
                          config=BellmanConfig(tol=tol, max_iter=10000, learning_rate=1e-3, opt_name='adam'))

# analytical results
lp0 = LP(θ=θ, b=np.array([[1e-50, 1e-50]]), A=A, grid_min=gmin, grid_size=n, grid_max=gmax)
v0 = jax.vmap(lp0.v_star, in_axes=(1))(jnp.stack(lp0.mesh).reshape(lp0.N, -1))
v0 = v0.reshape((lp0.grid_size,)*lp0.N)

v - v0

h_inv(w) - v0

# The $L_\infty$ distance to the analytical value function

jnp.max(jnp.abs(v - v0))

jnp.max(jnp.abs(h_inv(w) - v0))

# Some 3d graphs that show the distance to the anlytical value function (blue surface is $v$ of the original problem).

fig, ax = plt.subplots(subplot_kw=dict(projection='3d'), figsize=(16, 8))
ax.plot_surface(*lp.mesh, v,antialiased=False, alpha=0.4, label='original')
ax.plot_surface(*lp.mesh, h_inv(w),antialiased=False, alpha=0.4, label='conjugate')
ax.plot_surface(*lp.mesh, v0,antialiased=False, alpha=0.4, label='analytical')
plt.legend()
plt.show()

fig, ax = plt.subplots(subplot_kw=dict(projection='3d'), figsize=(16, 8))
ax.plot_surface(*lp.mesh, h(v),antialiased=False, alpha=0.4, label='original')
ax.plot_surface(*lp.mesh, w,antialiased=False, alpha=0.4, label='conjugate')
ax.plot_surface(*lp.mesh, h(v0),antialiased=False, alpha=0.4, label='analytical')
plt.legend()
plt.show()

# Here I fix $Y_1$ at $1 \times 10^{-6}$, the smallest value of the state space, and plot $v$ as a function of $Y_2$. This clearly shows that the transformed solution is much more accurate. 

index = 0
# plt.plot(v[index, ], label='ori', alpha=0.6)
plt.plot(v0[:,index], label='analytical', alpha=0.6)
plt.plot(v[:,index], '--', label='original', alpha=0.6)
plt.plot(h_inv(w[:,index]), '-.', label='conjugate', alpha=0.6)
plt.legend()
plt.show()



# + [markdown] jp-MarkdownHeadingCollapsed=true
# # Test
# -

def fun_factory(lp, Y, v, penalty_factor=1e2, tol=1e-6):
    """Generate a jitted Bellman operator"""
    grids, N, shocks, weights = lp.grids, lp.N, lp.shocks, lp.weights
    β, H, u, f = lp.β, lp.H, lp.u, lp.f
    v_fun = RegularGridInterpolator(grids, v, bounds_error=False, fill_value=None)

    def cont_value_quad(L, X):
        """Compute continuation value βEv(S') using Gauss-Hermite quadrature"""
        Y_next = f(L, X, jnp.exp(shocks))
        # restrict the state to be inside the grid
        # without this, v_fun might extrapolate to negative values
        Y_next = jnp.clip(Y_next, lp.grids[:, 0], lp.grids[:, -1])
        return β * jnp.dot(v_fun(Y_next), weights)

    def state_action_value(x):
        """Compute - u(C, Z) - βEv(S')
        x is an array of shape (N^2,)
        """
        X, C, L = get_XCL(x, Y, N)
        util = compute_utility(C, H, u, tol)
        penalty = compute_penalty(C, X, penalty_factor)
        res = util + cont_value_quad(L, jnp.clip(X, tol, Y)) - penalty
        return -res

    return state_action_value


def fun_factory_conjugate(lp, Y, w, h, h_inv, penalty_factor=1e2, tol=1e-6):
    """Generate a jitted Bellman operator"""
    grids, N, shocks, weights = lp.grids, lp.N, lp.shocks, lp.weights
    β, H, u, f = lp.β, lp.H, lp.u, lp.f
    w_fun = RegularGridInterpolator(grids, w, bounds_error=False, fill_value=None)

    def cont_value_quad(L, X):
        """Compute continuation value βEv(S') using Gauss-Hermite quadrature"""
        Y_next = f(L, X, jnp.exp(shocks))
        # restrict the state to be inside the grid
        # without this, v_fun might extrapolate to negative values
        Y_next = jnp.clip(Y_next, lp.grids[:, 0], lp.grids[:, -1])
        return β * jnp.dot(h_inv(w_fun(Y_next)), weights)
     
    def state_action_value(x):
        """Compute - u(C, Z) - βEv(S')
        x is an array of shape (N^2,)
        """
        X, C, L = get_XCL(x, Y, N)
        util = compute_utility(C, H, u, tol)
        # penalty term for infeasible x
        penalty = compute_penalty(C, X, penalty_factor)
        res = -h(util + cont_value_quad(L, jnp.clip(X, tol, Y)) - penalty)
        return res

    return state_action_value


n = 10000
gm = 1e-6
lp = LP(grid_size=n, θ=jnp.array([1., 1.]), b=jnp.array([0.3]), A=jnp.array([[0.3]]), grid_min=gm)

lp0 = LP(grid_size=n, θ=jnp.array([1., 1.]), b=jnp.array([1e-20]), A=jnp.array([[0.3]]), grid_min=gm, num_nodes=1)
v0 = jax.vmap(lp0.v_star, in_axes=(1))(jnp.stack(lp0.mesh))

1

RegularGridInterpolator(grids, v, bounds_error=False, fill_value=None)

Y = jnp.array([10.0])
state_action_value = fun_factory(lp, Y, v0, penalty_factor=1e-2)

state_action_value_conjugate = fun_factory_conjugate(lp, Y, h(v0), h, h_inv, penalty_factor=1e-6)

sav_vmap = jax.vmap(state_action_value)
sav_vmap_conjugate = jax.vmap(state_action_value_conjugate)

x_arr = jnp.linspace(-0.5, 20., 1000)
sav_vmap_conjugate(x_arr).argmin()

x_arr[166]

plt.plot(sav_vmap(x_arr))

plt.plot(sav_vmap_conjugate(x_arr))

x0 = jnp.array([0.])
opt = configure_optimizer('adam', 300, kwargs={'b1': 0.1, 'b2': 0.9})
params = x0
fun = state_action_value
state = opt.init(x0)
for i in range(2000):
    # grad = jax.grad(fun)(params)
    value, grad = jax.value_and_grad(fun)(params)
    updates, state = opt.update(grad, state, params)
    params = optax.apply_updates(params, updates)
    params = optax.projections.projection_box(params, 0., 20.)
    grad = jax.grad(fun)(params)
    print(params, grad)
    err = otu.tree_l2_norm(grad)
    rel_err = abs(err/value)
    if rel_err < 1e-30:
        break


# try custom scheduler
scheduler = optax.schedules.cosine_onecycle_schedule(transition_steps=200, peak_value=2)
# opt = optax.chain(optax.scale_by_adam(b1=0.1, b2=0.3), optax.scale_by_schedule(scheduler), optax.scale(-1))
opt = configure_optimizer('adam', scheduler, kwargs={'b1': 0.1, 'b2':0.3})

x0 = jnp.array([1.])
# opt = configure_optimizer('adam', 30, kwargs={'b1': 0.1, 'b2':0.3})
params, state = run_opt(x0, state_action_value_conjugate, opt, 5000, 1e-6, lb=0., ub=20.)

params

state

n = 1000
gmin = 1e-6
gmax = 2.0
θ = np.array([1, 1, 1])
b = jnp.array([[0.2, 0.6]])
A = jnp.array([[0.1, 0.7], [0.3, 0.1]])

lp = LP(θ=θ, b=b, A=A, grid_min=gmin, grid_size=n, grid_max=gmax)

lp0 = LP(θ=θ, b=np.array([[1e-50, 1e-50]]), A=A, grid_min=gmin, grid_size=n, grid_max=gmax)
v0 = jax.vmap(lp0.v_star, in_axes=(1))(jnp.stack(lp0.mesh).reshape(lp0.N, -1))
v0 = v0.reshape((lp0.grid_size,)*lp0.N)

Y = jnp.array([lp.grids[0, 800], lp.grids[1, 1000]])
state_action_value = fun_factory(lp, Y, v0, penalty_factor=1e-1, tol=1e-6)

Y

sav_vmap = jax.vmap(state_action_value, in_axes=(0,))

x_arr = jnp.tile(jnp.array([0.1, 0.51399463, 0.30734807, 0.09843366]), (200, 1))
# x_arr = jnp.tile(jnp.array([0.1, 1e-6, 1e-6, 1e-6]), (200, 1))
x_arr = x_arr.at[:, 0].set(jnp.linspace(0., 1., 200))

sav_vmap(x_arr).argmin()

x_arr[41]

plt.plot(sav_vmap(x_arr))

x0 = jnp.tile(jnp.array([y/(lp.N+1) for y in Y]), lp.N)
# x0 = jnp.zeros(4)
schedule = optax.warmup_cosine_decay_schedule(init_value=0.0, peak_value=1, warmup_steps=100, 
                                              decay_steps=500, end_value=0.0)
opt = configure_optimizer('adam', schedule, 
                          kwargs={}
                         )
tol = 1e-6
params, state = run_opt(x0, state_action_value, opt, 2000, 1e-8, lb=tol, ub=jnp.tile(Y, lp.N) - tol)

params

state

params

state


# + [markdown] jp-MarkdownHeadingCollapsed=true
# # Obsolete
# -

def T_factory(lp, max_iter=200, tol=1e-20, opt_tol=1e-10, penalty_factor=1e4):
    """Generate a jitted Bellman operator"""
    grids, N, shocks, weights = lp.grids, lp.N, lp.shocks, lp.weights
    β, H, u, f = lp.β, lp.H, lp.u, lp.f

    @jax.jit
    def T_y(Y, v):
        """Evaluate Tv at a single state"""
        v_fun = RegularGridInterpolator(grids, v, bounds_error=False, fill_value=None)

        def cont_value_quad(L, X):
            """Compute continuation value βEv(S') using Gauss-Hermite quadrature"""
            Y_next = f(L, X, jnp.exp(shocks))
            # restrict the state to be inside the grid
            # without this, v_fun might extrapolate to negative values
            Y_next = jnp.clip(Y_next, lp.grids[:, 0], lp.grids[:, -1])
            return β * jnp.dot(v_fun(Y_next), weights)

        def state_action_value(x):
            """Compute - u(C, Z) - βEv(S')
            x is an array of shape (N^2,)
            """
            X, C, L = get_XCL(x, Y, N)
            util = compute_utility(C, H, u, tol)
            
            # penalty term for infeasible x
            penalty = compute_penalty(C, X, penalty_factor)
            res = -(util + cont_value_quad(L, jnp.maximum(X, tol)) - penalty)
            return res

        # solve the minimization problem using gradient descent from jaxopt
        x0 = jnp.tile(jnp.array([y/(N+1) for y in Y]), N)
        lb = jnp.zeros(N**2) + tol
        ub = jnp.tile(Y, N) - tol
        # here projection is for setting the constraints
        # solver = ProjectedGradient(fun=state_action_value, projection=projection.projection_box,
        #                            tol=1e-8, maxiter=max_iter)
        # # params is the minimizer
        # params, state = solver.run(x0, hyperparams_proj=(lb, ub))

        # use optax
        # opt = optax.lbfgs(learning_rate=0.003)
        # params, state = run_opt_extra_args(x0, state_action_value, opt, max_iter=max_iter, tol=1e-8)
        
        # opt = optax.adam(learning_rate=0.003)
        opt = optax.fromage(learning_rate=0.003)
        # opt = optax.lamb(learning_rate=0.003)
        params, state = run_opt(x0, state_action_value, opt, max_iter=max_iter, tol=opt_tol)

        return -state_action_value(params), params, state

    # vmap T_hat_y to accept a vector of states of shape (N, M)
    T = jax.vmap(T_y, in_axes=(1, None))
    return T


def T_hat_factory(lp, h, h_inv, max_iter=200, tol=1e-20, opt_tol=1e-10, penalty_factor=1e4):
    """Generate a jitted conjugate Bellman operator"""
    grids, N, shocks, weights = lp.grids, lp.N, lp.shocks, lp.weights
    β, H, u, f = lp.β, lp.H, lp.u, lp.f

    @jax.jit
    def T_hat_y(Y, w):
        """Evaluate Tv at a single state"""
        w_fun = RegularGridInterpolator(grids, w, bounds_error=False, fill_value=None)

        def cont_value_quad(L, X):
            """Compute continuation value βEv(S') using Gauss-Hermite quadrature"""
            Y_next = f(L, X, jnp.exp(shocks))
            # restrict the state to be inside the grid
            # without this, v_fun might extrapolate to negative values
            Y_next = jnp.clip(Y_next, lp.grids[:, 0], lp.grids[:, -1])
            return β * jnp.dot(h_inv(w_fun(Y_next)), weights)

        def state_action_value(x):
            """Compute - u(C, Z) - βEv(S')
            x is an array of shape (N^2,)
            """
            X, C, L = get_XCL(x, Y, N)
            util = compute_utility(C, H, u, tol)
            
            # penalty term for infeasible x
            penalty = compute_penalty(C, X, penalty_factor)
            res = -h(util + cont_value_quad(L, jnp.maximum(X, tol)) - penalty)
            return res

        # solve the minimization problem using gradient descent from jaxopt
        x0 = jnp.tile(jnp.array([y/(N+1) for y in Y]), N)
        lb = jnp.zeros(N**2) + tol
        ub = jnp.tile(Y, N) - tol
        # here projection is for setting the constraints
        # solver = ProjectedGradient(fun=state_action_value, projection=projection.projection_box,
                                   # tol=1e-8, maxiter=max_iter)
        # # params is the minimizer
        # params, state = solver.run(x0, hyperparams_proj=(lb, ub))

        # solver = LBFGSB(fun=state_action_value, tol=1e-8, maxiter=max_iter)
        # params, state = solver.run(x0, bounds=(lb, ub))

        # use optax
        # opt = optax.adam(learning_rate=0.003)
        # opt = optax.fromage(learning_rate=0.03)
        opt = optax.lamb(learning_rate=0.003)
        params, state = run_opt(x0, state_action_value, opt, max_iter=max_iter, tol=opt_tol)
        
        return -state_action_value(params), params, state

    # vmap T_hat_y to accept a vector of states of shape (N, M)
    T_hat = jax.vmap(T_hat_y, in_axes=(1, None))
    return T_hat


Y = jnp.array([1, 2])
x0 = jnp.tile(jnp.array([y/(lp.N+1) for y in Y]), lp.N)

x0


# +
def run_opt_extra_args(init_params, fun, opt, max_iter, tol):
  value_and_grad_fun = optax.value_and_grad_from_state(fun)

  def step(carry):
    params, state = carry
    value, grad = value_and_grad_fun(params, state=state)
    updates, state = opt.update(
        grad, state, params, value=value, grad=grad, value_fn=fun
    )
    params = optax.apply_updates(params, updates)
    return params, state

  def continuing_criterion(carry):
    _, state = carry
    iter_num = otu.tree_get(state, 'count')
    grad = otu.tree_get(state, 'grad')
    err = otu.tree_l2_norm(grad)
    return (iter_num == 0) | ((iter_num < max_iter) & (err >= tol))

  init_carry = (init_params, opt.init(init_params))
  final_params, final_state = jax.lax.while_loop(
      continuing_criterion, step, init_carry
  )
  return final_params, final_state

# test the optimizer

def test_fun(x):
    return -jnp.log(x).sum() + jnp.exp(x).sum()


opt = optax.lbfgs()
init_params = jnp.ones((4,))
print(
    f'Initial value: {test_fun(init_params):.2e} '
    f'Initial gradient norm: {otu.tree_l2_norm(jax.grad(test_fun)(init_params)):.2e}'
)
final_params, _ = run_opt_extra_args(init_params, test_fun, opt, max_iter=100, tol=1e-20)
print(
    f'Final value: {test_fun(final_params):.2e}, '
    f'Final gradient norm: {otu.tree_l2_norm(jax.grad(test_fun)(final_params)):.2e}'
)

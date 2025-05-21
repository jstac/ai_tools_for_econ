import numpy as np
import quantecon as qe
from numba import jit


import matplotlib
matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt


def test_stability(Q):
    """
    Stability test for a given matrix Q.
    """
    sr = np.max(np.abs(np.linalg.eigvals(Q)))
    error_msg = f"Spectral radius condition failed with radius = {sr}"
    assert sr < 1, error_msg

# == fix parameters == #

n = 10
rho = 0.9
sigma = 0.02
β=0.96
γ=2.0
mc = qe.tauchen(n, rho, sigma)
P = mc.P
states = mc.state_values

# == solve using linear algebra == #

# Compute the matrix K
g=np.exp
K = β * P * g(states)**(1 - γ)

# Make sure that a unique solution exists
test_stability(K)

# Compute v
I = np.identity(n)
v_star = np.linalg.solve(I - K, K @ np.ones(n))


# == solve using stochastic approximation == #

P_cdf = np.cumsum(P, axis=1)

@jit
def compute_fixed_point_sa(series_length=1_000_000):
    v = np.zeros(n)
    new_v = np.empty_like(v)
    for k in range(series_length):
        alpha = (k + 1)**(-0.55)
        for i in range(n):
            j = qe.random.draw(P_cdf[i, :])  # an index draw from P[i, :]
            Y = states[j]  # the update state
            update = β * g(Y)**(1 - γ) * (v[j] + 1)
            new_v[i] = v[i] + alpha * (update - v[i])
        error = np.max(np.abs(new_v - v))
        v[:] = new_v
    return v

v = compute_fixed_point_sa()

fig, ax = plt.subplots()
ax.plot(states, v_star, alpha=0.8, label='linear algebra')
ax.plot(states, v, lw=2, alpha=0.8, ls='dashed', 
        label='stochastic approx')
ax.legend()
plt.show()

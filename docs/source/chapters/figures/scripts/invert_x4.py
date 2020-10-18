import os
import matplotlib.pyplot as plt
import numpy as np

# Internal
from lwsspy import DOCFIGURES  # Location to store figure
from lwsspy import updaterc    # Makes figure pretty in my opinion
from lwsspy import Optimization
updaterc()


def x4(r):
    """x4 values"""
    x = r[0]
    f = x ** 4 - 4 * x ** 2 - 2*x + 7
    return f


def x4_prime(r):
    """x4 values"""
    x = r[0]
    f = 4 * x ** 3 - 8 * x - 2
    return np.array([f])


def cost_and_grad(r):
    """Just takes in both and does both"""
    #      Cost & Grad
    return x4(r), x4_prime(r)


def x4_preco(q):
    """x4 Preconditioner
    """
    x = 1
    h = np.zeros(1)
    h[0] = 12 * x ** 2 - 8
    # h(3) = -400*x;
    q = q/h
    return q


# Define initial model
model = np.array([-3])

# Prepare optim bfgs
print(50 * "*", " BFGS ", 54 * "*")
optim = Optimization("bfgs")
optim.compute_cost = x4
optim.compute_gradient = x4_prime
optim.apply_preconditioner = x4_preco
optim.is_preco = False
optim.niter_max = 50
optim.stopping_criterion = 1e-10
optim.n = len(model)
optim_bfgs = optim.solve(optim, model)

# BFGS preco
optim = Optimization("bfgs")
optim.compute_cost = x4
optim.compute_gradient = x4_prime
optim.apply_preconditioner = x4_preco
optim.is_preco = True
optim.niter_max = 50
optim.stopping_criterion = 1e-10
optim.n = len(model)
optim_pbfgs = optim.solve(optim, model)

print(50 * "*", " Steepest ", 50 * "*")
# Prepare optim steepest
optim = Optimization("steepest")
# optim.compute_cost = x4
# optim.compute_gradient = x4_prime
optim.compute_cost_and_gradient = cost_and_grad
optim.apply_preconditioner = x4_preco
optim.is_preco = False
optim.niter_max = 50
optim.stopping_criterion = 1e-10
optim.n = len(model)
optim_step = optim.solve(optim, model)

# Steepest preco
optim = Optimization("steepest")
optim.compute_cost = x4
optim.compute_gradient = x4_prime
optim.apply_preconditioner = x4_preco
optim.is_preco = True
optim.niter_max = 50
optim.stopping_criterion = 1e-10
optim.n = len(model)
optim.alpha = 1
optim_pstep = optim.solve(optim, model)

print(50 * "*", " NLCG ", 54 * "*")
# Prepare optim nlcg
optim = Optimization("nlcg")
optim.compute_cost = x4
optim.compute_gradient = x4_prime
optim.apply_preconditioner = x4_preco
optim.is_preco = False
optim.niter_max = 50
optim.stopping_criterion = 1e-10
optim.n = len(model)
optim_nlcg = optim.solve(optim, model)

# Steepest preco
optim = Optimization("nlcg")
optim.compute_cost = x4
optim.compute_gradient = x4_prime
optim.apply_preconditioner = x4_preco
optim.is_preco = True
optim.niter_max = 50
optim.stopping_criterion = 1e-10
optim.n = len(model)
optim_pnlcg = optim.solve(optim, model)

# Get costs
f1 = optim_bfgs.fcost_hist
i1 = np.arange(len(f1))
f2 = optim_pbfgs.fcost_hist
i2 = np.arange(len(f2))
f3 = optim_step.fcost_hist
i3 = np.arange(len(f3))
f4 = optim_pstep.fcost_hist
i4 = np.arange(len(f4))
f5 = optim_nlcg.fcost_hist
i5 = np.arange(len(f5))
f6 = optim_pnlcg.fcost_hist
i6 = np.arange(len(f6))


# Plot
plt.figure(figsize=(11, 5))

ax = plt.subplot(1, 2, 1)
ax.set_yscale("log")
plt.title("Misfit Convergence x4")
plt.plot(i1, f1, label="bfgs")
plt.plot(i2, f2, label="pbfgs")
plt.plot(i3, f3, label="steep")
plt.plot(i4, f4, label="psteep")
plt.plot(i5, f5, label="nlcg")
plt.plot(i6, f6, label="pnlcg")
plt.legend(loc=4)

ax2 = plt.subplot(1, 2, 2)
x = [np.linspace(-2.5, 2.75, 200)]
plt.plot(optim_bfgs.msave[0, :], x4([optim_bfgs.msave[0, :]]), label="bfgs")
plt.plot(optim_pbfgs.msave[0, :], x4([optim_pbfgs.msave[0, :]]), label="pbfgs")
plt.plot(optim_step.msave[0, :], x4([optim_step.msave[0, :]]), label="steep")
plt.plot(optim_pstep.msave[0, :], x4(
    [optim_pstep.msave[0, :]]), label="psteep")
plt.plot(optim_nlcg.msave[0, :], x4([optim_nlcg.msave[0, :]]), label="steep")
plt.plot(optim_pnlcg.msave[0, :], x4(
    [optim_pnlcg.msave[0, :]]), label="psteep")
plt.plot(x[0], x4(x), label=r"$f(x) = x^4 - 4x^2 -2x$")
# ax2.set_aspect('equal', 'box')
plt.legend(loc=1)
plt.title('Model Movement')
plt.savefig(os.path.join(DOCFIGURES, "optimization_x4.svg"), dpi=300)
plt.show()

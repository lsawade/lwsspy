import os
import matplotlib.pyplot as plt
import numpy as np

# Internal
from lwsspy import DOCFIGURES  # Location to store figure
from lwsspy import updaterc    # Makes figure pretty in my opinion
from lwsspy import Optimization
updaterc()


def rosenbrock(r):
    """Rosenbrock values"""
    x = r[0]
    y = r[1]
    f = (1-x) ** 2 + 100*(y-x*x) ** 2
    return f


def rosenbrock_prime(r):
    """Rosenbrock gradient"""
    x = r[0]
    y = r[1]
    g = np.zeros(2)
    g[0] = -2*(1-x) - 400*x*(y-x*x)
    g[1] = 200*(y-x*x)
    return g


def rosenbrock_pprime(r):
    """Rosenbrock gradient"""
    x = r[0]
    y = r[1]
    h = np.zeros(2, 2)
    h[0, 0] = 2 - 400*y + 1200*x*x
    h[0, 1] = - 400*x
    h[1, 0] = h[0, 1]
    h[1, 1] = 200
    return h


def rosenbrock_preco(q):
    """Rosenbrock Preconditioner
    """
    x = 1.5
    y = 1.5
    h = np.zeros(2)
    h[0] = 2 - 400*y + 1200*x*x
    h[1] = 200
    # h(3) = -400*x;
    q = q/h
    return q


def compute_cost_and_grad_and_hess(r):
    return rosenbrock(r), rosenbrock_prime(r), rosenbrock_pprime(r)


# Define initial model
model = np.array([1.5, 1.5])

# Prepare optim bfgs
print(50 * "*", " BFGS ", 54 * "*")
optim = Optimization("bfgs")
optim.compute_cost = rosenbrock
optim.compute_gradient = rosenbrock_prime
optim.apply_preconditioner = rosenbrock_preco
optim.is_preco = False
optim.niter_max = 50
optim.stopping_criterion = 1e-10
optim.n = len(model)
optim_bfgs = optim.solve(optim, model)

# BFGS preco
optim = Optimization("bfgs")
optim.compute_cost = rosenbrock
optim.compute_gradient = rosenbrock_prime
optim.apply_preconditioner = rosenbrock_preco
optim.is_preco = True
optim.niter_max = 50
optim.stopping_criterion = 1e-10
optim.n = len(model)
optim_pbfgs = optim.solve(optim, model)

print(50 * "*", " Steepest ", 50 * "*")
# Prepare optim steepest
optim = Optimization("steepest")
optim.compute_cost = rosenbrock
optim.compute_gradient = rosenbrock_prime
optim.apply_preconditioner = rosenbrock_preco
optim.is_preco = False
optim.niter_max = 50
optim.stopping_criterion = 1e-10
optim.n = len(model)
optim_step = optim.solve(optim, model)

# Steepest preco
optim = Optimization("steepest")
optim.compute_cost = rosenbrock
optim.compute_gradient = rosenbrock_prime
optim.apply_preconditioner = rosenbrock_preco
optim.is_preco = True
optim.niter_max = 50
optim.stopping_criterion = 1e-10
optim.n = len(model)
optim.alpha = 1
optim_pstep = optim.solve(optim, model)

print(50 * "*", " NLCG ", 54 * "*")
# Prepare optim nlcg
optim = Optimization("nlcg")
optim.compute_cost = rosenbrock
optim.compute_gradient = rosenbrock_prime
optim.apply_preconditioner = rosenbrock_preco
optim.is_preco = False
optim.niter_max = 50
optim.stopping_criterion = 1e-10
optim.n = len(model)
optim_nlcg = optim.solve(optim, model)

# Steepest preco
optim = Optimization("nlcg")
optim.compute_cost = rosenbrock
optim.compute_gradient = rosenbrock_prime
optim.apply_preconditioner = rosenbrock_preco
optim.is_preco = True
optim.niter_max = 50
optim.stopping_criterion = 1e-10
optim.n = len(model)
optim_pnlcg = optim.solve(optim, model)

# Steepest preco
print(50 * "*", " Gauss-Newton ", 50 * "*")
optim = Optimization("gn")
# optim.compute_cost = rosenbrock
# optim.compute_gradient = rosenbrock_prime
optim.compute_cost_and_grad_and_hess = compute_cost_and_grad_and_hess
optim.apply_preconditioner = rosenbrock_preco
optim.is_preco = False
optim.niter_max = 50
optim.stopping_criterion = 1e-10
optim.n = len(model)
optim_gn = optim.solve(optim, model)

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
f7 = optim_gn.fcost_hist
i7 = np.arange(len(f6))

# Plot
plt.figure(figsize=(11, 5))

ax = plt.subplot(1, 2, 1)
ax.set_yscale("log")
plt.title("Misfit Convergence Rosenbrock")
plt.plot(i1, f1, label="bfgs")
plt.plot(i2, f2, label="pbfgs")
plt.plot(i3, f3, label="steep")
plt.plot(i4, f4, label="psteep")
plt.plot(i5, f5, label="nlcg")
plt.plot(i6, f6, label="pnlcg")
plt.plot(i7, f7, label="gn")
plt.legend(loc=4)

ax2 = plt.subplot(1, 2, 2)
x, y = np.meshgrid(np.linspace(-2.5, 2.5, 300), np.linspace(-0.5, 2.5, 200))
plt.pcolormesh(x, y, np.log10(rosenbrock([x, y])), zorder=-11, cmap='gray')
plt.plot(optim_bfgs.msave[0, :optim_bfgs.current_iter],
         optim_bfgs.msave[1, :optim_bfgs.current_iter], label="bfgs")
plt.plot(optim_pbfgs.msave[0, :optim_pbfgs.current_iter],
         optim_pbfgs.msave[1, :optim_pbfgs.current_iter], label="pbfgs")
plt.plot(optim_step.msave[0, :optim_step.current_iter],
         optim_step.msave[1, :optim_step.current_iter], label="steep")
plt.plot(optim_pstep.msave[0, :optim_pstep.current_iter],
         optim_pstep.msave[1, :optim_pstep.current_iter], label="psteep")
plt.plot(optim_nlcg.msave[0, :optim_nlcg.current_iter],
         optim_nlcg.msave[1, :optim_nlcg.current_iter], label="steep")
plt.plot(optim_pnlcg.msave[0, :optim_pnlcg.current_iter],
         optim_pnlcg.msave[1, :optim_pnlcg.current_iter], label="psteep")
plt.plot(optim_gn.msave[0, :optim_pnlcg.current_iter],
         optim_gn.msave[1, :optim_pnlcg.current_iter], label="gn")
ax2.set_rasterization_zorder(-10)
ax2.set_aspect('equal', 'box')
plt.legend(loc=3)
plt.title('Model Movement')
plt.savefig(os.path.join(DOCFIGURES, "optimization.svg"), dpi=300)
plt.savefig(os.path.join(DOCFIGURES, "optimization.pdf"), dpi=300)
plt.show()
plt.show()

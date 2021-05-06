import os
import matplotlib.pyplot as plt
import numpy as np
import logging

# Internal
import lwsspy as lpy
from lwsspy import DOCFIGURES  # Location to store figure
from lwsspy import updaterc    # Makes figure pretty in my opinion
from lwsspy import Optimization
from lwsspy import plot_model_history
from lwsspy import plot_optimization

updaterc()

# create logger
logger = logging.getLogger(f"Optimizer")
logger.setLevel(logging.DEBUG)
logger.handlers = []
logger.propagate = False
formatter = lpy.CustomFormatter()
sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
sh.setFormatter(formatter)
logger.addHandler(sh)


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
    h = np.zeros((2, 2))
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


def rosenbrock_preco_steepest(r):
    """Rosenbrock Preconditioner
    """
    x = 1.5
    y = 1.5
    h = np.zeros(2)
    h[0] = 2 - 400*y + 1200*x*x
    h[1] = 200
    # h(3) = -400*x;
    q = 0
    h = rosenbrock_pprime(r)
    l = 0.0001 * np.max(np.abs(np.diag(h))) * np.eye(2)
    q = np.linalg.solve(h + l, rosenbrock_prime(r))
    return q


def compute_cost_and_grad_and_hess(r):
    return rosenbrock(r), rosenbrock_prime(r), rosenbrock_pprime(r)


# Define initial model
model = np.array([1.5, 1.5])
niter = 100
nls = 20

# Prepare optim bfgs
logger.info(50 * "*", " BFGS ", 54 * "*")
optim = Optimization("bfgs")
optim.logger = logger.info
optim.compute_cost = rosenbrock
optim.compute_gradient = rosenbrock_prime
optim.apply_preconditioner = rosenbrock_preco
optim.is_preco = False
optim.niter_max = niter
optim.nls_max = nls
optim.stopping_criterion = 1e-10
optim.n = len(model)
optim_bfgs = optim.solve(optim, model)

# BFGS preco
optim = Optimization("bfgs")
optim.logger = logger.info
optim.compute_cost = rosenbrock
optim.compute_gradient = rosenbrock_prime
optim.apply_preconditioner = rosenbrock_preco
optim.is_preco = True
optim.niter_max = niter
optim.nls_max = nls
optim.stopping_criterion = 1e-10
optim.n = len(model)
optim_pbfgs = optim.solve(optim, model)

logger.info(50 * "*", " Steepest ", 50 * "*")
# Prepare optim steepest
optim = Optimization("steepest")
optim.logger = logger.info
optim.compute_cost = rosenbrock
optim.compute_gradient = rosenbrock_prime
optim.apply_preconditioner = rosenbrock_preco
optim.is_preco = False
optim.niter_max = niter
optim.nls_max = nls
optim.stopping_criterion = 1e-10
optim.n = len(model)
optim_step = optim.solve(optim, model)

# Steepest preco
optim = Optimization("steepest")
optim.logger = logger.info
optim.compute_cost = rosenbrock
optim.compute_gradient = rosenbrock_prime
optim.apply_preconditioner = rosenbrock_preco_steepest
optim.is_preco = True
optim.niter_max = niter
optim.nls_max = nls
optim.stopping_criterion = 1e-10
optim.n = len(model)
optim.alpha = 1
optim_pstep = optim.solve(optim, model)

logger.info(50 * "*", " NLCG ", 54 * "*")
# Prepare optim nlcg
optim = Optimization("nlcg")
optim.logger = logger.info
optim.compute_cost = rosenbrock
optim.compute_gradient = rosenbrock_prime
optim.apply_preconditioner = rosenbrock_preco
optim.is_preco = False
optim.niter_max = niter
optim.nls_max = nls
optim.stopping_criterion = 1e-10
optim.n = len(model)
optim_nlcg = optim.solve(optim, model)

# Steepest preco
optim = Optimization("nlcg")
optim.logger = logger.info
optim.compute_cost = rosenbrock
optim.compute_gradient = rosenbrock_prime
optim.apply_preconditioner = rosenbrock_preco
optim.is_preco = True
optim.niter_max = niter
optim.nls_max = nls
optim.stopping_criterion = 1e-10
optim.n = len(model)
optim_pnlcg = optim.solve(optim, model)

# Gauss-Newton
logger.info(50 * "*", " Gauss-Newton ", 50 * "*")
optim = Optimization("gn")
optim.logger = logger.info
optim.compute_cost_and_grad_and_hess = compute_cost_and_grad_and_hess
optim.is_preco = False
optim.niter_max = niter
optim.nls_max = nls
optim.damping = 0.0
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
i7 = np.arange(len(f7))

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
plt.pcolormesh(x, y, np.log10(rosenbrock([x, y])), zorder=-11, cmap='gray',
               shading='auto')
plt.plot(optim_bfgs.msave[0, :optim_bfgs.current_iter+1],
         optim_bfgs.msave[1, :optim_bfgs.current_iter+1], label="bfgs")
plt.plot(optim_pbfgs.msave[0, :optim_pbfgs.current_iter+1],
         optim_pbfgs.msave[1, :optim_pbfgs.current_iter+1], label="pbfgs")
plt.plot(optim_step.msave[0, :optim_step.current_iter+1],
         optim_step.msave[1, :optim_step.current_iter+1], label="steep")
plt.plot(optim_pstep.msave[0, :optim_pstep.current_iter+1],
         optim_pstep.msave[1, :optim_pstep.current_iter+1], label="psteep")
plt.plot(optim_nlcg.msave[0, :optim_nlcg.current_iter+1],
         optim_nlcg.msave[1, :optim_nlcg.current_iter+1], label="steep")
plt.plot(optim_pnlcg.msave[0, :optim_pnlcg.current_iter+1],
         optim_pnlcg.msave[1, :optim_pnlcg.current_iter+1], label="psteep")
plt.plot(optim_gn.msave[0, :optim_gn.current_iter+1],
         optim_gn.msave[1, :optim_gn.current_iter+1], label="gn")
ax2.set_rasterization_zorder(-10)
# ax2.set_aspect('equazl', 'box')
plt.xlim(0.4, 1.6)
plt.ylim(0.8, 2.4)
plt.legend(loc=3)
plt.title('Model Movement')
plt.show()


# plot_model_history(optim_gn, outfile="./testhist.pdf")
# plot_optimization(optim_gn, outfile="./testoptim.pdf")


# print(optim_gn.msave)

import os
import matplotlib.pyplot as plt
import numpy as np

# Internal
# Internal
import lwsspy.plot as lplot
import lwsspy.utils as lutils
import lwsspy.base as lbase
import lwsspy.inversion as linv

lplot.updaterc()


def x4(r):
    """x4 values"""
    x = r[0]
    f = (x) ** 4 + 0.001 * x ** 2 - 2*x + 8.0
    return f


def x4_prime(r):
    """x4 values"""
    x = r[0]
    f = 4 * (x) ** 3 + 0.002 * x - 2
    return np.array([f])


def x4_pprime(r):
    """x4 values"""
    x = r[0]
    f = 12 * (x) ** 2 + 0.002
    return np.array([[f]])


def cost_and_grad(r):
    """Just takes in and does both"""
    #      Cost & Grad
    return x4(r), x4_prime(r)


def cost_and_grad_and_hess(r):
    """Just takes in and does all"""
    #      Cost & Grad
    return x4(r), x4_prime(r), x4_pprime(r)


def x4_preco(q):
    """x4 Preconditioner
    """
    x = 0.79370053
    h = np.zeros(1)
    h[0] = 12 * (x) ** 2 + 0.002
    # h(3) = -400*x;
    q = q/h
    return q


def x4_preco_2(x):
    """x4 Preconditioner
    """
    h = x4_pprime(x)
    l = 0.1 * np.max(np.abs(np.diag(h))) * np.eye(1)
    print(h+l)
    print(x4_prime(x))
    q = np.linalg.solve(h + l, x4_prime(x))
    return q


# Define initial model
model = np.array([-2.0])
maxiter = 20
maxnls = 5

# Prepare optim bfgs
print(50 * "*", " BFGS ", 54 * "*")
optim = linv.Optimization("bfgs")
optim.compute_cost = x4
optim.compute_grad = x4_prime
optim.apply_preconditioner = x4_preco
optim.is_preco = False
optim.niter_max = maxiter
optim.nls_max = maxnls
optim.stopping_criterion = 1e-10
optim.n = len(model)
optim_bfgs = optim.solve(optim, model)

# BFGS preco
optim = linv.Optimization("bfgs")
optim.compute_cost = x4
optim.compute_grad = x4_prime
optim.apply_preconditioner = x4_preco
optim.is_preco = True
optim.niter_max = maxiter
optim.nls_max = maxnls
optim.stopping_criterion = 1e-10
optim.n = len(model)
optim_pbfgs = optim.solve(optim, model)

print(50 * "*", " Steepest ", 50 * "*")
# Prepare optim steepest
optim = linv.Optimization("steepest")
# optim.compute_cost = x4
# optim.compute_gradient = x4_prime
optim.compute_cost_and_gradient = cost_and_grad
optim.apply_preconditioner = x4_preco
optim.is_preco = False
optim.niter_max = maxiter
optim.nls_max = maxnls
optim.stopping_criterion = 1e-10
optim.n = len(model)
optim_step = optim.solve(optim, model)

# Steepest preco
optim = linv.Optimization("steepest")
optim.compute_cost = x4
optim.compute_gradient = x4_prime
optim.compute_cost_and_gradient = cost_and_grad
optim.apply_preconditioner = x4_preco
optim.is_preco = True
optim.niter_max = maxiter
optim.nls_max = maxnls
optim.stopping_criterion = 1e-10
optim.n = len(model)
optim.alpha = 1
optim_pstep = optim.solve(optim, model)

print(50 * "*", " NLCG ", 54 * "*")
# Prepare optim nlcg
optim = linv.Optimization("nlcg")
optim.compute_cost = x4
optim.compute_gradient = x4_prime
optim.apply_preconditioner = x4_preco
optim.is_preco = False
optim.niter_max = maxiter
optim.nls_max = maxnls
optim.stopping_criterion = 1e-10
optim.n = len(model)
optim_nlcg = optim.solve(optim, model)

# Steepest preco
optim = linv.Optimization("nlcg")
optim.compute_cost = x4
optim.compute_gradient = x4_prime
optim.apply_preconditioner = x4_preco
optim.is_preco = True
optim.niter_max = maxiter
optim.nls_max = maxnls
optim.stopping_criterion = 1e-10
optim.n = len(model)
optim_pnlcg = optim.solve(optim, model)

print(50 * "*", " GN ", 54 * "*")
# Prepare optim nlcg
optim = linv.Optimization("gn")
optim.compute_cost = x4
optim.compute_gradient = x4_prime
optim.compute_cost_and_grad_and_hess = cost_and_grad_and_hess
# optim.apply_preconditioner = x4_preco
optim.is_preco = False
optim.niter_max = maxiter
optim.nls_max = maxnls
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

optim_list = [
    optim_bfgs, optim_pbfgs, optim_step, optim_pstep,
    optim_nlcg, optim_pnlcg, optim_gn
]
labellist = [
    "BFGS", "P-BFGS", "Steep", "P-Steep", "NLCG", "P-NLCG", "GN"
]

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
plt.plot(i7, f7, label="gn")
plt.legend(loc=4)

ax2 = plt.subplot(1, 2, 2)
x = [np.linspace(-3.0, 2.75, 200)]
plt.plot(optim_bfgs.msave[0, :optim_bfgs.current_iter+1], x4(
    [optim_bfgs.msave[0, :optim_bfgs.current_iter+1]]), label="bfgs")
plt.plot(optim_pbfgs.msave[0, :optim_pbfgs.current_iter+1],
         x4([optim_pbfgs.msave[0, :optim_pbfgs.current_iter+1]]), label="pbfgs")
plt.plot(optim_step.msave[0, :optim_step.current_iter+1],
         x4([optim_step.msave[0, :optim_step.current_iter+1]]), label="steep")
plt.plot(optim_pstep.msave[0, :optim_pstep.current_iter+1], x4(
    [optim_pstep.msave[0, :optim_pstep.current_iter+1]]), label="psteep")
plt.plot(optim_nlcg.msave[0, :optim_nlcg.current_iter+1],
         x4([optim_nlcg.msave[0, :optim_nlcg.current_iter+1]]), label="steep")
plt.plot(optim_pnlcg.msave[0, :optim_pnlcg.current_iter+1], x4(
    [optim_pnlcg.msave[0, :optim_pnlcg.current_iter+1]]), label="psteep")
plt.plot(optim_gn.msave[0, :optim_gn.current_iter+1], x4(
    [optim_gn.msave[0, :optim_gn.current_iter+1]]), label="gn")
plt.plot(x[0], x4(x), label=r"$f(x) = x^4 - 4x^2 -2x$")
# ax2.set_aspect('equal', 'box')
plt.legend(loc=1)
plt.title('Model Movement')
linv.plot_single_parameter_optimization(
    optim_list, modellabel="x", labellist=labellist,
    outfile=os.path.join(lbase.DOCFIGURES, "optimization_x4.svg"))
linv.plot_single_parameter_optimization(
    optim_list, modellabel="x", labellist=labellist,
    outfile=os.path.join(lbase.DOCFIGURES, "optimization_x4.pdf"))


plt.show()

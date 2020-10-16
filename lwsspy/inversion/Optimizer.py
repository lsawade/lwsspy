from numpy import np
from copy import deepcopy


def NOP(*args, **kwargs):
    """Does nothing
    """
    pass


class Optimization:

    def __init__(
        self,
        otype: str = 'bfgs',
        fcost_init: float = 0.0,
        fcost: float = 0.0,
        norm_grad_init: float = 0.0,
        norm_grad: float = 0.0,
        stopping_criterion: float = 1e-10,
        niter_max: int = 50,
        qk: float = 0.0,
        q: float = 0.0,
        is_preco: bool = True,
        alpha: float = 0.0,
        n: int = 0,  # number of parameters
        model: list = [],
        grad: list = [],
        descent: = [],
        nsave: int = 0,
        perc: float = 0.025,
        fcost_hist: list = [],
        flag: str = "suceed",
        # for linesearch
        nls_max: int = 20,
        c1: float = 1e-4,
        c2: float = 0.9,
        al: float = 0.,
        ar: float = 0.,
        strong: bool = False,
        factor: float = 10.0,
        # For BFGS
        gsave: list = [],
        msave: list = [],
        # Routine
        compute_cost: callable = NOP,
        compute_gradient: callable = NOP,
        compute_cost_and_gradient: callable = NOP,
        descent_direction: callable = NOP,
        apply_preconditioner: callable = NOP,
        save_model_to_disk: callable = NOP,
        save_model_and_gradient: callable = store_grad_and_model,
        solve: callable = Solve_Optimisation_Problem,
        get_optim_si_yi: callable = get_optim_si_yi,
        compute_beta: callable = fletcher_reeves)

    # Useful things
    self.type = otype
    self.fcost_init = fcost_init
    self.fcost = fcost
    self.norm_grad_init = norm_grad_init
    self.norm_grad = norm_grad
    self.stopping_criterion = stopping_criterion
    self.niter_max = niter_max
    self.qk = qk
    self.q = q
    self.is_preco = is_preco
    self.alpha = alpha
    self.n = n      # number of parameters
    self.model = model
    self.grad = grad
    self.descent = descent
    self.nsave = nsave
    self.perc = perc
    self.fcost_hist = fcost_hist
    self.flag = flag

    # for linesearch
    self.nls_max = nls
    self.c1 = cs
    self.c2 = c2
    self.al = al
    self.ar = ar
    self.strong = strong
    self.factor = factor

    # For BFGS
    self.nb_mem = self.niter_max  # by default
    self.gsave = gsave
    self.msave = msave

    # Routine
    self.compute_cost = compute_cost
    self.compute_gradient = compute_gradient
    self.compute_cost_and_gradient = compute_cost_and_gradient
    self.descent_direction = descent_direction
    self.apply_preconditioner = apply_preconditioner
    self.save_model_to_disk = save_model_to_disk
    self.save_model_and_gradient = save_model_and_gradient
    self.solve = solve
    self.get_optim_si_yi = get_optim_si_yi
    self.compute_beta = compute_beta
    # self.compute_beta  = polak_ribiere # could also be fletcher reeves


def norm_l2(a):
    val = sqrt(sum(a.*a))
    return val


def scalar_product(a, b):
    val = sum(a.*b)
    return val


def Solve_Optimisation_Problem(optim, model):

    if optim.type == "steepest":
        optim.descent_direction = get_steepest_descent_direction
        if optim.is_preco == False:
            optim.apply_preconditioner = NOP

    elseif optim.type == "bfgs":
        optim.descent_direction = get_bfgs_descent_direction
        if (~optim.is_preco):
            optim.apply_preconditioner = preconditioner_nocedal

        optim.ro = zeros(optim.niter_max)
        optim.ai = zeros(optim.niter_max)
    elseif optim.type == "nlcg":
        optim.descent_direction = get_nonlinear_conjugate_gradient_direction
        if optim.is_preco == False:
            optim.apply_preconditioner = NOP

    else:
        raise ValueError("Optimization algorithm not recognized")

    optim.model_ini = deepcopy(model)
    optim.model = model

    optim.fcost_ini = deepcopy(optim.compute_cost(optim.model))

    optim.grad_ini = optim.compute_gradient(optim.model)

    optim.fcost = optim.fcost_ini
    optim.grad = optim.grad_ini

    if False == (optim.alpha > 0):
        # if (~optim.is_preco)

        optim.alpha = optim.perc * \
            np.max(np.abs(optim.model))/np.max(np.abs(optim.grad_ini))

        # else
        #    optim.alpha = optim.perc; % * max(abs.(optim.model));
        #

    if (optim.nb_mem < 1):
        optim.nb_mem = optim.niter_max  # by default

    # optim.alpha = 1.
    optim.msave = np.zeros(optim.n, optim.nb_mem)
    optim.gsave = np.zeros(optim.n, optim.nb_mem)

    # Some attached defs
    optim.bfgs_formula = bfgs_formula         # accessible outside for resolution
    optim.get_si_and_yi = get_optim_si_yi      # could be overwritten
    optim.store_grad_and_model = store_grad_and_model  # could be overwritten

    # Perform optimization
    optim.fcost_hist.append(optim.fcost./optim.fcost_ini)
    for iter in range(optim.niter_max):

        print("\n\nModel: ", optim.model)
        print("Grad:  ", optim.grad, "\n\n")

        # Initialize
        optim.al = 0
        optim.ar = 0
        optim.current_iter = iter

        # Save gradient and model
        optim.save_model_and_gradient(optim)

        # Get descent direction
        optim.descent_direction(optim)

        # Compute product of descent times gradient
        optim.q = scalar_product(optim.descent, optim.grad)

        # Perform linesearch
        perform_linesearch(optim)
        if optim.flag == "fail"
        break

        # If linesearch successful update informations
        optim.fcost = optim.fcost_new
        optim.model = optim.model_new
        if optim.type == "nlcg":
            optim.grad_prev = optim.grad

        optim.grad = optim.grad_new
        optim.q = optim.qnew

        push(optim.fcost_hist, optim.fcost./optim.fcost_ini)

        # Check stopping criteria
        if ((optim.fcost / optim.fcost_ini) < optim.stopping_criterion) or (iter > optim.niter_max):
            optim.save_model_and_gradient(optim)
            print("Optimization algorithm has converged.")
            break

    return deepcopy(optim)


def get_steepest_descent_direction(optim):
    if (optim.is_preco is True):
        optim.norm_grad = norm_l2(optim.grad)
        optim.descent = -optim.apply_preconditioner(optim.grad)
        optim.norm_desc = norm_l2(optim.descent)
        optim.descent = optim.descent * optim.norm_grad / optim.norm_desc
    else:
        optim.descent = -optim.grad


def get_nonlinear_conjugate_gradient_direction(optim):

    # At first iteration perform a steepest descent update
    if (optim.current_iter == 1):
        get_steepest_descent_direction(optim)

    else:
        # Compute beta
        optim.compute_beta(optim)

        # Descent equal the conjugate direction
        optim.descent = optim.beta * optim.descent - optim.grad


def get_bfgs_descent_direction(optim):

    # At first iteration perform a steepest descent update
    if (optim.current_iter == 1):
        get_steepest_descent_direction(optim)
    else:
        # Call BFGS formula
        optim.bfgs_formula(optim)
        optim.descent = -optim.descent
        optim.alpha = 1  # should always be tried first


def bfgs_formula(optim):
    # First loop
    iter = optim.current_iter
    optim.descent = optim.grad
    for i in range(iter, -1, 0):
        optim.get_si_and_yi(optim, i)
        sty = scalar_product(optim.si, optim.yi)
        std = scalar_product(optim.si, optim.descent)
        optim.ro[i] = 1 . / sty
        optim.ai[i] = optim.ro[i] .* std
        optim.descent = optim.descent - optim.ai[i] .* optim.yi

    # Apply preco
    if (optim.is_preco):
        optim.apply_preconditioner(optim.descent)
    else:
        optim.apply_preconditioner(optim)

    # Second loop
    for i in range(0: iter):
        optim.get_si_and_yi(optim, i)
        optim.yitd = scalar_product(optim.yi, optim.descent)
        beta = optim.ro[i] * optim.yitd
        optim.descent = optim.descent + optim.si .* (optim.ai[i] - beta)


def get_optim_si_yi(optim, i):
    optim.si = optim.msave[:, i+1] - optim.msave[:, i]
    optim.yi = optim.gsave[:, i+1] - optim.gsave[:, i]
    # add def to read from disk


def perform_linesearch(optim):

    # Line search
    for ils in range(optim.nls_max):

        # New model and evaluate
        optim.model_new = optim.model + optim.alpha .* optim.descent
        optim.fcost_new = optim.compute_cost(optim.model_new)
        optim.grad_new = optim.compute_gradient(optim.model_new)

        # Safeguard check for inf and nans...
        if isnan(optim.fcost_new) or isinf(optim.fcost_new):
            # assume we've been too far and reduce step
            optim.ar = optim.alpha
            optim.alpha = (optim.al+optim.ar)*0.5
            continue

        # Compute dot product between new gradient and descent
        optim.qnew = scalar_product(optim.grad_new, optim.descent)

        # Check Wolfe conditions for current iteration
        check_wolfe_conditions(optim)

        # Manage alpha in consequence
        if optim.w3 is False:  # not a descent direction... quit
            optim.flag = "fail"
            break
            # error('Not a descent direction... STOP');

        if (optim.w1 is True) and (optim.w2 is True):  # both are satisfied, then terminate
            print(
                f"\niter = {optim.current_iter}, ",
                f"f./fo={optim.fcost_new./optim.fcost_ini:5.4e}, "
                f"nls = {ils}, wolfe1 = {optim.w1} wolfe2 = {optim.w2}, "
                f"a={optim.alpha}, al={optim.al}, ar={optim.ar}\n")
            break

        if optim.w1 is False:               # not a sufficient decrease, we've been too far
            optim.ar = optim.alpha
            optim.alpha = (optim.al+optim.ar)*0.5
        elif (optim.w1 is True) and (optim.w2 is False):
            optim.al = optim.alpha
            if optim.ar > 0:                 # sufficient decrease but too close already backeted decrease in interval
                optim.alpha = (optim.al + optim.ar)*0.5
            else:  # sufficient decrease but too close, then increase a
                optim.alpha = optim.factor * optim.alpha

    # Check linesearch
    if ils == optim.nls_max:  # and optim.w1== False and optim.w2 == False)
        raise ValueError("Linesearch failed, stop optimization")


def store_grad_and_model(optim):
    iter = optim.current_iter
    optim.msave[:, iter] = optim.model[:]
    optim.gsave[:, iter] = optim.grad[:]


def check_wolfe_conditions(optim):

    # Init wolfe boolean
    optim.w1 = False
    optim.w2 = False
    optim.w3 = True

    # Check descent direction
    if optim.q > 0:
        optim.w3 = False
        #error('Not a descent dir');

    # Check first wolfe
    if optim.fcost_new <= optim.fcost + optim.c1 * optim.alpha * optim.q:
        optim.w1 = True

    # Check second wolfe
    if optim.strong == False:
        if (optim.qnew >= optim.c2 * optim.q)
        optim.w2 = True

    else:
        if abs(optim.qnew) >= abs(optim.c2 * optim.q):
            optim.w2 = True


def preconditioner_nocedal(optim):
    iter = optim.current_iter
    optim.get_si_and_yi(optim, iter-1)
    sty = scalar_product(optim.si, optim.yi)
    yty = scalar_product(optim.yi, optim.yi)
    optim.gam = sty / yty
    optim.descent = optim.gam*optim.descent


def fletcher_reeves(optim):
    xtx = scalar_product(optim.grad, optim.grad)
    xtxp = scalar_product(optim.grad_prev, optim.grad_prev)
    optim.beta = xtx / xtxp


def polak_ribiere(optim):
    # not sure what happens with this one
    dgrad = optim.grad - optim.grad_prev
    xtx = scalar_product(optim.grad, dgrad)
    xtxp = scalar_product(optim.grad_prev, optim.grad_prev)
    optim.beta = np.max(0, xtx / xtxp)
    print(optim.beta)
    # pause

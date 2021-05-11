import numpy as np
from copy import deepcopy, copy


def NOP(args):
    """Does nothing
    """
    return args


def norm_l2(a):
    """Computes the L2 norm of a vector

    Parameters
    ----------
    a : numpy.ndarray
        array of arbitrary size, but it makes only sense for a 1d vector

    Returns
    -------
    float
        float witht L2 norm
    """
    val = np.sqrt(np.sum(a*a))
    return val


def scalar_product(a, b):
    """Computes scalar product

    Parameters
    ----------
    a : numpy.ndarray
        array
    b : numpy.ndarray
        array of same size as array a

    Returns
    -------
    float
        containing scalar product
    """
    val = np.sum(a*b)
    return val


def Solve_Optimisation_Problem(optim, model):
    """Takes in class:`lwsspy.Optimization`

    Parameters
    ----------
    optim : lwsspy.Optimization
        Optimization struct
    model : numpy.ndarray
        model vector

    Returns
    -------
    lwsspy.Optimization
        returns a copy of the optimization struct that has been optimized.

    Raises
    ------
    ValueError
        if wrong optimizaiton algorithm is used
    """
    # because python is weird
    optim = deepcopy(optim)

    if optim.type == "steepest":
        optim.descent_direction = get_steepest_descent_direction
        if optim.is_preco is False:
            optim.apply_preconditioner = NOP

    elif optim.type == "bfgs":
        optim.descent_direction = get_bfgs_descent_direction
        if (optim.is_preco is False):
            optim.apply_preconditioner = preconditioner_nocedal
        optim.ro = np.zeros(optim.niter_max)
        optim.ai = np.zeros(optim.niter_max)

    elif optim.type == "nlcg":
        optim.descent_direction = get_nonlinear_conjugate_gradient_direction
        if optim.is_preco is False:
            optim.apply_preconditioner = NOP

    elif optim.type == "gn":
        optim.descent_direction = get_gauss_newton_descent_direction

    else:
        raise ValueError("Optimization algorithm not recognized")

    optim.model_ini = deepcopy(model)
    optim.model = model
    optim.n = len(model)

    if optim.compute_cost_and_grad_and_hess is not None:
        optim.fcost_ini, optim.grad_ini, optim.hess_ini = \
            optim.compute_cost_and_grad_and_hess(optim.model)
        optim.hess = optim.hess_ini
    elif optim.compute_cost_and_gradient is not None:
        optim.fcost_ini, optim.grad_ini = \
            optim.compute_cost_and_gradient(optim.model)
    else:
        optim.fcost_ini = optim.compute_cost(optim.model)
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
    optim.msave = np.zeros((optim.n, optim.nb_mem))
    optim.gsave = np.zeros((optim.n, optim.nb_mem))

    # Some attached defs
    optim.bfgs_formula = bfgs_formula       # accessible outside for resolution
    optim.get_si_and_yi = get_optim_si_yi      # could be overwritten
    # optim.store_grad_and_model = store_grad_and_model  # could be overwritten

    # Perform optimization
    optim.logger(f"    Model: {optim.model}")
    optim.logger(f"    Grad:  {optim.grad}")

    # Start iteration.
    for _iter in range(optim.niter_max):

        # Initialize
        optim.al = 0
        optim.ar = 0
        optim.current_iter = _iter

        # Save model
        optim.fcost_hist.append(optim.fcost)
        optim.save_model_and_gradient(optim)

        # Get descent direction
        optim.descent_direction(optim)

        # Compute product of descent times gradient
        optim.q = scalar_product(optim.descent, optim.grad)

        # Perform linesearch
        perform_linesearch(optim)
        if optim.flag == "fail":
            break

        # Save previous gradient for Non-linear conjugate gradient
        if optim.type == "nlcg":
            optim.grad_prev = optim.grad

        # If linesearch successful update informations
        optim.fcost_prev = optim.fcost
        optim.fcost = optim.fcost_new
        optim.model = optim.model_new
        optim.grad = optim.grad_new
        optim.q = optim.qnew

        # Check stopping criteria
        if (np.abs(optim.fcost - optim.fcost_prev)/optim.fcost_ini
                < optim.stopping_criterion_cost_change):
            optim.logger(
                "Cost function not decreasing enough to justify iteration.")
            # Update the iteration number otherwise the previous one is overwritten
            optim.current_iter = _iter + 1
            optim.fcost_hist.append(optim.fcost/optim.fcost_ini)
            optim.save_model_and_gradient(optim)
            break
        elif (optim.fcost/optim.fcost_ini < optim.stopping_criterion):
            optim.logger("Optimization algorithm has converged.")
            # Update the iteration number otherwise the previous one is overwritten
            optim.current_iter = _iter + 1
            optim.fcost_hist.append(optim.fcost/optim.fcost_ini)
            optim.save_model_and_gradient(optim)
            break
        elif np.max(np.abs((optim.alpha * optim.descent)/optim.model_ini)) \
                < optim.stopping_criterion_model:
            optim.logger("Model is not updating enough anymore.")
            # Update the iteration number otherwise the previous one is overwritten
            optim.current_iter = _iter + 1
            optim.fcost_hist.append(optim.fcost/optim.fcost_ini)
            optim.save_model_and_gradient(optim)
            break

    return optim


def get_steepest_descent_direction(optim):
    """Gets steepest descent direction

    Parameters
    ----------
    optim : Optimization
        optimization class

    """
    if (optim.is_preco is True):
        optim.norm_grad = norm_l2(optim.grad)
        optim.descent = -optim.apply_preconditioner(optim.grad)
        optim.norm_desc = norm_l2(optim.descent)
        optim.descent = optim.descent * optim.norm_grad / optim.norm_desc
    else:
        optim.descent = -optim.grad


def get_nonlinear_conjugate_gradient_direction(optim):
    """Get nonlinear conjugate Gradient descent Direction

    Parameters
    ----------
    optim : Optimization
        optimization class
    """

    # At first iteration perform a steepest descent update
    if (optim.current_iter == 0):
        get_steepest_descent_direction(optim)

    else:
        # Compute beta
        optim.compute_beta(optim)

        # Descent equal the conjugate direction
        optim.descent = optim.beta * optim.descent - optim.grad


def get_bfgs_descent_direction(optim):
    """Gets BFGS descent direction.

    Parameters
    ----------
    optim : Optimization
        optimization class
    """

    # At first iteration perform a steepest descent update
    if (optim.current_iter == 0):
        get_steepest_descent_direction(optim)
    else:
        # Call BFGS formula
        optim.bfgs_formula(optim)
        optim.descent = -optim.descent
        optim.alpha = 1  # should always be tried first


def get_gauss_newton_descent_direction(optim):
    """Gets Gauss-Newton descent direction.

    Parameters
    ----------
    optim : Optimization
        optimization class
    """

    optim.descent = np.linalg.solve(
        optim.hess, -optim.grad)
    optim.alpha = 1  # should always be tried first


def bfgs_formula(optim):
    """Runs BFGS formula to get descent direction

    Parameters
    ----------
    optim : Optimization
        optimization class
    """
    # First loop
    _iter = optim.current_iter
    optim.descent = optim.grad
    for i in range(_iter-1, 1, -1):
        optim.get_si_and_yi(optim, i)
        sty = scalar_product(optim.si, optim.yi)
        std = scalar_product(optim.si, optim.descent)
        optim.ro[i] = 1 / sty
        optim.ai[i] = optim.ro[i] * std
        optim.descent = optim.descent - optim.ai[i] * optim.yi

    # Apply preco
    # Apply preco
    if (optim.is_preco):
        optim.apply_preconditioner(optim.descent)
    else:
        optim.apply_preconditioner(optim)

    # Second loop
    for i in range(_iter):
        optim.get_si_and_yi(optim, i)
        optim.yitd = scalar_product(optim.yi, optim.descent)
        beta = optim.ro[i] * optim.yitd
        optim.descent = optim.descent + optim.si * (optim.ai[i] - beta)


def get_optim_si_yi(optim, i):
    """Gets the si and and yi history for the descent direction.

    Parameters
    ----------
    optim : Optimization
        Optimization class
    i : int
        iteration counter
    """
    optim.si = optim.msave[:, i] - optim.msave[:, i-1]
    optim.yi = optim.gsave[:, i] - optim.gsave[:, i-1]
    # add def to read from disk


def perform_linesearch(optim):
    """Performs linesearch on class:`lwsspy.Optimization`

    Parameters
    ----------
    optim : Optimization
        optimization class

    Raises
    ------
    ValueError
        If linesearch fails (max number of linesearch iterations are reached).
    """

    # Line search
    for ils in range(optim.nls_max):

        # New model and evaluate
        optim.model_new = optim.model + optim.alpha * optim.descent

        # If simultaneous cost and grad computation is defined do that.
        if optim.compute_cost_and_grad_and_hess is not None:
            optim.fcost_new, optim.grad_new, optim.hess_new = optim.compute_cost_and_grad_and_hess(
                optim.model_new)
        elif optim.compute_cost_and_gradient is not None:
            optim.fcost_new, optim.grad_new = optim.compute_cost_and_gradient(
                optim.model_new)
        else:
            optim.fcost_new = optim.compute_cost(optim.model_new)
            optim.grad_new = optim.compute_gradient(optim.model_new)

        optim.logger(
            f"    ils: {ils} -- "
            f"f/fo={optim.fcost_new/optim.fcost_ini:5.4e} -- "
            f"model: {optim.model_new} -- alpha: {optim.alpha}")

        # Safeguard check for inf and nans...
        if np.isnan(optim.fcost_new) or np.isinf(optim.fcost_new):
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
            optim.logger(
                f"iter = {optim.current_iter}, "
                f"f/fo={optim.fcost_new/optim.fcost_ini:5.4e}, "
                f"nls = {ils}, wolfe1 = {optim.w1} wolfe2 = {optim.w2}, "
                f"a={optim.alpha}, al={optim.al}, ar={optim.ar}")
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
    if ils == (optim.nls_max - 1):  # and optim.w1== False and optim.w2 == False)
        optim.logger("Linesearch ended without finding good model candidate.")
        optim.flag = "fail"


def store_grad_and_model(optim):
    """Function to store the model and the gradient.

    Parameters
    ----------
    optim : Optimization
        Optimization class
    """
    _iter = optim.current_iter
    optim.msave[:, _iter] = optim.model[:]
    optim.gsave[:, _iter] = optim.grad[:]


def check_wolfe_conditions(optim):
    """Checks Wolfe conditions

    Parameters
    ----------
    optim : Optimization
        Optimization class
    """

    # Init wolfe boolean
    optim.w1 = False
    optim.w2 = False
    optim.w3 = True

    # Check descent direction
    if optim.q > 0:
        optim.w3 = False
        # error('Not a descent dir');

    # Check first wolfe
    if optim.fcost_new <= optim.fcost + optim.c1 * optim.alpha * optim.q:
        optim.w1 = True

    # Check second wolfe
    if optim.strong is False:
        if optim.qnew >= optim.c2 * optim.q:
            optim.w2 = True

    else:
        if abs(optim.qnew) >= abs(optim.c2 * optim.q):
            optim.w2 = True


def preconditioner_nocedal(optim):
    """Applies Nocedal style preconditioner to class:`Optimization`.

    Parameters
    ----------
    optim : Optimization
        Optimization
    """
    _iter = optim.current_iter
    optim.get_si_and_yi(optim, _iter)
    sty = scalar_product(optim.si, optim.yi)
    yty = scalar_product(optim.yi, optim.yi)
    optim.gam = sty / yty
    optim.descent = optim.gam*optim.descent


def fletcher_reeves(optim):
    """Fletcher-Reeves algorithm to get beta

    Parameters
    ----------
    optim : Optimizatiton
        Optimization class
    """
    xtx = scalar_product(optim.grad, optim.grad)
    xtxp = scalar_product(optim.grad_prev, optim.grad_prev)
    optim.beta = xtx / xtxp


def polak_ribiere(optim):
    """Polak-Ribiere algorithm to get beta

    Parameters
    ----------
    optim : Optimization
        Optimization struct
    """
    # not sure what happens with this one
    dgrad = optim.grad - optim.grad_prev
    xtx = scalar_product(optim.grad, dgrad)
    xtxp = scalar_product(optim.grad_prev, optim.grad_prev)
    optim.beta = np.max(0, xtx / xtxp)
    # pause


class Optimization:

    def __init__(
        self,
        otype: str = 'bfgs',
        fcost_init: float = 0.0,
        fcost: float = 0.0,
        fcost_prev: float = 0.0,
        norm_grad_init: float = 0.0,
        norm_grad: float = 0.0,
        stopping_criterion: float = 1e-10,
        stopping_criterion_cost_change: float = 1e-10,
        stopping_criterion_model: float = 1e-6,
        niter_max: int = 50,
        qk: float = 0.0,
        q: float = 0.0,
        is_preco: bool = True,
        alpha: float = 0.0,
        n: int = 0,  # number of parameters
        model: np.ndarray = np.array([]),
        grad: np.ndarray = np.array([]),
        hess: np.ndarray = np.array([[]]),
        descent: np.ndarray = np.array([]),
        nsave: int = 0,
        perc: float = 0.025,
        damping: float = 0.0,
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
        nb_mem: int = 0,
        # Logger to make your life more beautiful
        logger: callable = print,
        # Routine
        compute_cost: callable = NOP,
        compute_gradient: callable = NOP,
        compute_cost_and_gradient: callable or None = None,
        compute_cost_and_grad_and_hess: callable or None = None,
        descent_direction: callable = NOP,
        apply_preconditioner: callable = NOP,
        save_model_to_disk: callable = NOP,
        save_model_and_gradient: callable = store_grad_and_model,
        solve: callable = Solve_Optimisation_Problem,
        get_optim_si_yi: callable = get_optim_si_yi,
            compute_beta: callable = fletcher_reeves):
        """Optimization class to run optimize a problem with given cost function
        and cost function gradient.

        Parameters
        ----------
        otype : str, optional
            Optimization algorithm, by default 'bfgs'
        fcost_init : float, optional
            initial cost, by default 0.0
        fcost : float, optional
            cost function value, by default 0.0
        norm_grad_init : float, optional
            initial norm of the gradient, by default 0.0
        norm_grad : float, optional
            current norm of the gradient, by default 0.0
        stopping_criterion : float, optional
            value to cost function has to reach, by default 1e-10
        stopping_criterion_model : float, optional
            value to stop at if model isn't updating enough anymore
        niter_max : int, optional
            max iteration (excluding the linesearch), by default 50
        qk : float, optional
            qk, for linesearch, by default 0.0
        q : float, optional
            q, for linesearch, by default 0.0
        is_preco : bool, optional
            proconditioning flag, by default True
        alpha : float, optional
            alpha sttep length to be found by by linesearch, by default 0.0
        n : int, optional
            number of model parameters, by default 0
        grad : np.ndarray, optional
            current gradient, by default np.array([])
        descent : np.ndarray, optional
            current descent direction, by default np.array([])
        nsave : int, optional
            matrix to store model iterations, by default 0
        perc : float, optional
            precentage of step, by default 0.025
        damping : float, optional
            damping the Gauss-Newton step, 0.0 means no damping, by default 0.0
        fcost_hist : list, optional
            cost function history, by default []
        flag : str, optional
            success flag, by default "suceed"
        nls_max : int, optional
            maximum linesearch itterations, by default 20
        c1 : float, optional
            linesearch parameter c1, by default 1e-4
        c2 : float, optional
            linesearch parameter c1, by default 0.9
        al : float, optional
            left alpha boundary, by default 0.
        ar : float, optional
            right alpha boundary, by default 0.
        strong : bool, optional
            strong wolfe condition are abided if True, by default False
        factor : float, optional
            Factor to multiply alpha by, by default 10.0
        gsave : list, optional
            array to save the past gradients, by default []
        msave : list, optional
            array to save past models, by default []
        compute_cost : callable, optional
            function that computes the cost, by default NOP
        compute_gradient : callable, optional
            function that computes the gradient, by default NOP
        compute_cost_and_gradient : callable or None, optional
            function that computes the cost and the gradient at the same time,
            by default None
        descent_direction : callable, optional
            function that computes the descent direction, by default NOP
        apply_preconditioner : callable, optional
            function that applies a preconditioner, by default NOP
        save_model_to_disk : callable, optional
            function that writes the model to disk, not implemented,
            by default NOP
        save_model_and_gradient : callable, optional
            funciton that stores model and gradient within the optimization
            class, by default store_grad_and_model
        solve : callable, optional
            solves the optimization problem,
            by default Solve_Optimisation_Problem
        get_optim_si_yi : callable, optional
            function that gets the optimal si and yi,
            by default get_optim_si_yi
        compute_beta : callable, optional
            function that computes beta, by default fletcher_reeves
        """

        # Useful things
        self.type = otype
        self.fcost_init = fcost_init
        self.fcost = fcost
        self.fcost_prev = fcost
        self.norm_grad_init = norm_grad_init
        self.norm_grad = norm_grad
        self.stopping_criterion = stopping_criterion
        self.stopping_criterion_cost_change = stopping_criterion_cost_change
        self.stopping_criterion_model = stopping_criterion_model
        self.niter_max = niter_max
        self.qk = qk
        self.q = q
        self.is_preco = is_preco
        self.alpha = alpha
        self.n = n      # number of parameters
        self.model = model
        self.grad = grad
        self.hess = hess
        self.descent = descent
        self.nsave = nsave
        self.perc = perc
        self.damping = damping
        self.lam = damping
        self.fcost_hist = fcost_hist
        self.flag = flag

        # for linesearch
        self.nls_max = nls_max
        self.c1 = c1
        self.c2 = c2
        self.al = al
        self.ar = ar
        self.strong = strong
        self.factor = factor

        # For BFGS
        self.nb_mem = nb_mem  # by default
        self.gsave = gsave
        self.msave = msave

        # Logger
        self.logger = logger

        # Routine
        self.compute_cost = compute_cost
        self.compute_gradient = compute_gradient
        self.compute_cost_and_gradient = compute_cost_and_gradient
        self.compute_cost_and_grad_and_hess = compute_cost_and_grad_and_hess
        self.descent_direction = descent_direction
        self.apply_preconditioner = apply_preconditioner
        self.save_model_to_disk = save_model_to_disk
        self.save_model_and_gradient = save_model_and_gradient
        self.solve = solve
        self.get_optim_si_yi = get_optim_si_yi
        self.compute_beta = compute_beta
        # self.compute_beta  = polak_ribiere # could also be fletcher reeves

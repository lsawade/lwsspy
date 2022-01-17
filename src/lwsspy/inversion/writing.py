
import os
from .optimizer import Optimization
from ..math.rosenbrock import rosenbrock, drosenbrock, ddrosenbrock
from .gaussian2d import gaussian2d
import jax.numpy as jnp
from jax.core import Primitive


def checkdir(cdir):
    if not os.path.exists(cdir):
        os.makedirs(cdir)


outdir = "/Users/lucassawade/optimizationdir"
datadir = os.path.join(outdir, "data")
syntdir = os.path.join(outdir, "synt")
costdir = os.path.join(outdir, "cost")
graddir = os.path.join(outdir, "gradients")
hessdir = os.path.join(outdir, "hessians")

checkdir(datadir)
checkdir(syntdir)
checkdir(costdir)
checkdir(graddir)
checkdir(hessdir)

# actual m: amplitude, x0, yo, sigma_x, sigma_y, theta, offset
m_sol = jnp.array([4, 115, 90, 25, 35, 22.5, 5])

# Initial guess m0: amplitude, x0, yo, sigma_x, sigma_y, theta, offset
m0 = jnp.array([1.0, 100, 100, 20, 40, 0, 7.5])

# Create some data
x = jnp.linspace(0, 200, 201)
y = jnp.linspace(0, 200, 201)
X = jnp.meshgrid(x, y)

# Create data
data = gaussian2d(m_sol, X)
data = data + 0.2 * np.random.normal(size=data.shape)


def forward(m, x):
    return gaussian2d(m, x)


def cost(m, x):
    return 1/data.size * 0.5 * jnp.sum((forward_p(m, x)-data)**2)


# def cgh(m)

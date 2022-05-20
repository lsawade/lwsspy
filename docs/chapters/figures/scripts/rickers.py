# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import lwsspy.plot as lplt
from lwsspy.wavelets import ricker
from lwsspy.plot.updaterc import updaterc
import lwsspy.base as lbase
updaterc()

# Set time vector and center time
t = np.linspace(2, 7, 300)

# Set Ricker parameters
t0 = 4.5
f0 = 1
A = 1.25
L = 2

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].plot(t, ricker.ricker(t, t0, f0, A), label='$r$')
axes[0].plot(t, ricker.dr_dt0(t, t0, f0, A),
             label=r'$\frac{\partial r}{\partial t_c}$')
axes[0].plot(t, ricker.dr_df0(t, t0, f0, A),
             label=r'$\frac{\partial r}{\partial f_0}$')
axes[0].plot(t, ricker.dr_dA(t, t0, f0, A),
             label=r'$\frac{\partial r}{\partial A}$')
axes[0].legend()
lplt.plot_label(axes[0], label='A', location=1, box=False)

R = ricker.R(t, L, t0, f0, A)
dRdt0, dRdf0, dRdA = ricker.dRdm(t, L, t0, f0, A)

axes[1].plot(t, R, label='$R$')
axes[1].plot(t, dRdt0, label=r'$\frac{\partial R}{\partial t_c}$')
axes[1].plot(t, dRdf0, label=r'$\frac{\partial R}{\partial f_0}$')
axes[1].plot(t, dRdA, label=r'$\frac{\partial R}{\partial A}$')
axes[1].legend()
lplt.plot_label(axes[1], label='B', location=1, box=False)


outnamesvg = os.path.join(lbase.DOCFIGURES, "rickers.svg")
outnamepdf = os.path.join(lbase.DOCFIGURES, "rickers.pdf")

plt.savefig(outnamesvg, transparent=True)
plt.savefig(outnamepdf, transparent=True)

plt.show()

# %%

import os
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

# Internal
import lwsspy.plot as lplt
import lwsspy.base as lbase
lplt.updaterc()

# Create data
x = np.linspace(0, 5*np.pi, 10000)
y = np.cos(x)
z = -np.sin(x)

# Create Norm
norm = Normalize(vmin=-1, vmax=1)

# Plot figure
plt.figure(figsize=(8, 3))
plt.subplots_adjust(bottom=0.15, top=0.9, left=0.05, right=0.95)
ax = plt.gca()

# Main function
lines, sm = lplt.plot_xyz_line(
    x, y, z,
    linewidths=(np.abs(z)+1)**2.0,
    cmap='seismic', norm=norm)

cbar = plt.colorbar(sm, aspect=40, fraction=0.05, pad=0.025)
cbar.set_label(r'$\frac{\mathrm{d}y}{\mathrm{d}x}$', rotation=0, labelpad=10)
plt.title("Plotting a Line with Adjustable Color and Width")

# For making the figure nicer
minor = lplt.Multiple(12, number=np.pi, latex='\pi')
major = lplt.Multiple(2, number=np.pi, latex='\pi')
ax.xaxis.set_major_locator(major.locator)
ax.xaxis.set_minor_locator(minor.locator)
ax.xaxis.set_major_formatter(major.formatter)


plt.savefig(os.path.join(lbase.DOCFIGURES,
                         "xyz_line.svg"), transparent=True)
plt.show()

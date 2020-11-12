import os
import matplotlib.pyplot as plt
import lwsspy as lpy

lpy.updaterc()

lpy.plot_litho(which='all', parameter='depth', cmap="Spectral")
plt.savefig(os.path.join(lpy.DOCFIGURES, "litho1_depth.svg"), dpi=300)
lpy.plot_litho(which='lower_crust', parameter='depth', cmap="Spectral")
plt.savefig(os.path.join(lpy.DOCFIGURES,
                         "litho1_lower_crust_bottom_depth.svg"), dpi=300)

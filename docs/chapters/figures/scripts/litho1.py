import os
import matplotlib.pyplot as plt
import lwsspy.base as lbase
import lwsspy.plot as lplt
import lwsspy.maps as lmaps

lplt.updaterc()

lmaps.plot_litho(which='all', parameter='depth', cmap="Spectral")
plt.savefig(os.path.join(lbase.DOCFIGURES, "litho1_depth.svg"), dpi=300)
lmaps.plot_litho(which='lower_crust', parameter='depth', cmap="Spectral")
plt.savefig(os.path.join(lbase.DOCFIGURES,
                         "litho1_lower_crust_bottom_depth.svg"), dpi=300)

import matplotlib.pyplot as plt
import numpy as np
from ..plot_util.plot_zoom import PlotZoom
from ..plot_util.pz_figure import pz_figure

xc = np.round(45)
yc = np.round(45)
xwin = 5
ywin = 5


# Create and show image
pz_figure()
ax = plt.axes()
plt.imshow(im)
tit = plt.title("Press any key to start picking values...")
plt.show(block=False)

# Wati for the User to press a key to start
plt.waitforbuttonpress()

# Create zoom window
plt.figure()
zax = plt.axes()
zax.imshow(im)
zax.set_xlim((xc - xwin, xc + xwin))
zax.set_ylim((yc + ywin, yc - ywin))
PZ = PlotZoom(ax, zax, xc, yc, xwin, ywin)
PZ.connect()
plt.show(block=False)

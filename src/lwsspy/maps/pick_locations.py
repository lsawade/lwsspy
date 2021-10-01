import matplotlib.pyplot as plt
import numpy as np
import lwsspy as lpy

xc = np.round(45)
yc = np.round(45)
xwin = 5
ywin = 5

import matplotlib
import matplotlib.pyplot as plt

def move_figure(f, x, y):
    """Move figure's upper left corner to pixel (x, y)"""
    backend = matplotlib.get_backend()
    if backend == 'TkAgg':
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'WXAgg':
        f.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        f.canvas.manager.window.move(x, y)


# Create and show image
fig = pz_figure()
X, Y, dx, dy = fig.canvas.manager.window.geometry().getRect()
ax = plt.axes()
plt.imshow(im)
tit = plt.title("Press any key to start picking values...")
plt.show(block=False)

# Wati for the User to press a key to start
plt.waitforbuttonpress()

# Create zoom window
zfig = plt.figure()
zfig.canvas.managermngr.window.setGeometry(X + dx, Y, dx, dy)
zax = plt.axes()
zax.imshow(im)
zax.set_xlim((xc - xwin, xc + xwin))
zax.set_ylim((yc + ywin, yc - ywin))
PZ = PlotZoom(ax, zax, xc, yc, xwin, ywin)
PZ.connect()
plt.show(block=False)

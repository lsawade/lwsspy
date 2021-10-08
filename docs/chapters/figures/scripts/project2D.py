from matplotlib.patches import ConnectionPatch
import os
import matplotlib.pyplot as plt
import numpy as np

# Internal
import lwsspy.plot as lplt
import lwsspy.math as lmat
import lwsspy.base as lbase

lplt.updaterc()

# Create data
x0 = np.array([0, 2, 2, 0])
y0 = np.array([0, 0, 2, 2])
xn = np.array([0, 4, 5, 2])
yn = np.array([0, 1, 3, 4])
x0q = np.array([0.5, 1.5, 1.5, 0.5])
y0q = np.array([0.5, 0.5, 1.5, 1.5])

xnq, ynq = lmat.project2D(x0, y0, xn, yn, x0q, y0q)

# Plot figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))
plt.subplots_adjust(bottom=0.275, wspace=0.25)
ax1.plot(np.append(x0, x0[0]), np.append(y0, y0[0]),
         'kx-', label="0-Coordinate-System-Basis-Nodes")
ax1.plot(np.append(x0q, x0q[0]), np.append(y0q, y0q[0]),
         'ko-', label="Query-Points in 0-CS")
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax2.plot(np.append(xn, xn[0]), np.append(yn, yn[0]),
         'x-', label="N-Coordinate-System-Basis-Nodes")
ax2.plot(np.append(xnq, xnq[0]), np.append(ynq, ynq[0]),
         'o-', label="Resulting points in N-CS")
ax2.set_xlabel('x')
ax2.set_ylabel('y')

# Show transformation
for _x0, _y0, _xn, _yn in zip(x0q, y0q, xnq, ynq):
    con = ConnectionPatch(xyA=(_x0, _y0), axesA=ax1, coordsA="data",
                          xyB=(_xn, _yn), axesB=ax2, coordsB="data")
    fig.add_artist(con)
fig.legend(loc=8)

# Save figure with rasterization with 300dpi
plt.savefig(os.path.join(lbase.DOCFIGURES, "project2D.svg"), dpi=300)
plt.show()

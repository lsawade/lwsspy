import os
import matplotlib.pyplot as plt
import numpy as np

# Internal
import lwsspy.plot as lplt
import lwsspy.math as lmat
import lwsspy.base as lbase

lplt.updaterc()

# Create data
x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
xx, yy = np.meshgrid(x, y)

# Plot figure
plt.figure()
ax = plt.gca()
ax.set_rasterization_zorder(-10)  # Important line!
plt.plot(x, y**2, 'w')  # Zorder default is 0
plt.pcolormesh(xx, yy, xx ** 2 + yy ** 2, edgecolor=None, zorder=-15,
               shading='auto')


# Save figure with rasterization with 300dpi
plt.savefig(os.path.join(lbase.DOCFIGURES, "test_rasterize.svg"), dpi=300)
plt.show()

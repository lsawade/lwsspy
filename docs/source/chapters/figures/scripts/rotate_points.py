import os
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPoly
import lwsspy as lpy
from cartopy.crs import PlateCarree

# Circle around Z-axis
r = 1
phi = np.arange(0, 2*np.pi, 2*np.pi/100)
theta = 10*np.pi/180

# Circle around c axis in cartesian
x = r*np.sin(theta)*np.cos(phi)
y = r*np.sin(theta)*np.sin(phi)
z = r*np.cos(theta)*np.ones_like(phi)
points = np.vstack((x, y, z))

# Rotate from Z-axis to other points
Rx = lpy.Ra2b((0, 0, 1), (1, 0, 0))
Ry = lpy.Ra2b((0, 0, 1), (0, 1, 0))
R = lpy.Ra2b((0, 0, 1), (1, 1.5, 0.5))
rotations = [Rx, Ry, R]
labels = ["x", "y", "p"]

# Plot circles
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.view_init(20, 50)

# Plot x y z axes
plt.plot((-1, 1), (0, 0), (0, 0), 'k')
plt.plot((0, 0), (-1, 1), (0, 0), 'k')
plt.plot((0, 0), (0, 0), (-1, 1), 'k')
plt.plot(x, y, z, label='Original')

for _rot, _lab in zip(rotations, labels):
    rpoints = _rot @ points
    plt.plot(rpoints[0, :], rpoints[1, :], rpoints[2, :], label=_lab)

plt.savefig(os.path.join(lpy.DOCFIGURES,
                         "rotation_Ra2b.svg"), transparent=True)


# Create random lat and long
x = np.linspace(0, 2, 75)
y = x**2 + np.sin(10*x) * 2 * np.exp(-x**2)

lon = x*45
lat = y*10

# Epicentral distance of the buffer
delta = 4

# Compute buffer
pp, circles = lpy.line_buffer(lat, lon, delta=delta)
# _, circles, _, _ = lpy.plot_line_buffer(lat, lon, delta=delta)

fig = plt.figure(figsize=(10, 5))
ax = plt.subplot(121, projection=PlateCarree())
lpy.plot_map()
plt.plot(lon, lat, 'k-', marker='.')
lpy.remove_ticklabels_topright(ax)
plt.title('Circles used to build polygon')

Ncircles = len(circles)

for _i, coords in enumerate(circles):
    cmap = plt.get_cmap('rainbow')
    plt.plot(coords[0], coords[1], c=cmap(_i/Ncircles))

ax = plt.subplot(122, projection=PlateCarree())
lpy.plot_map()
plt.plot(lon, lat, 'k-', marker='.')
p = MPoly(pp, fc=(0.7, 0.3, 0.9), ec='none', alpha=0.33)
ax.add_patch(p)
lpy.remove_yticklabels(ax)
lpy.remove_ticklabels_topright(ax)
plt.title('Polygon from union of circles')

plt.savefig(os.path.join(lpy.DOCFIGURES,
                         "buffer_from_rotated_circles.svg"), transparent=True)

plt.show(block=True)

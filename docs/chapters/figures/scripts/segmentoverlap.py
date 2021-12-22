from matplotlib.cm import ScalarMappable
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from lwsspy.maps.overlap import overlap
from lwsspy.maps.overlaps import overlaps
from lwsspy.maps.overlapsj import overlapsj
from lwsspy.maps.map_axes import map_axes
from lwsspy.maps.threshproj import threshproj
from lwsspy.maps.plot_map import plot_map
from lwsspy.statistics.even2Dpoints import even2Dpoints
from cartopy.crs import Geodetic, Mollweide
import sys
import time

import jax.numpy as jnp
from jax import jit
from jax.config import config
config.update("jax_log_compiles", 1)

lon, lat = even2Dpoints(10, height=150, width=300, radius=5)

# Creating List of segments
# segments = tuple([(_lat0, _lon0, _lat1, _lon1) for _i, (_lat0, _lon0) in enumerate(zip(lat, lon))
#                   for _lat1, _lon1 in zip(lat[_i+1:], lon[_i+1:])
#                   if (((_lat0-_lat1)+(_lon0-_lon1)) != 0)])

segments = jnp.array([(_lat0, _lon0, _lat1, _lon1) for _i, (_lat0, _lon0) in enumerate(zip(lat, lon))
                     for _lat1, _lon1 in zip(lat[_i+1:], lon[_i+1:])
                     if (((_lat0-_lat1)+(_lon0-_lon1)) != 0)])

# segments = np.array(segments)

# # Creating counter matrix
# idx = np.zeros((len(segments), len(segments)), dtype=bool)


# # Segment overlap
# for _i, x in enumerate(segments):
#     for _j, y in enumerate(segments[_i+1:]):

#         idx[_i, _i+1+_j] = overlap(x[:2], x[2:], y[:2], y[2:])

# idx = np.array(overlaps(segments))

# jitting
# overlaps_ = jit(overlaps, static_argnums=(0,))
overlaps_ = jit(overlapsj)

# # compiling
t1c = time.time()
idx = overlaps_(segments).block_until_ready()
t2c = time.time()

print('Time to compile:', (t2c - t1c))

# # timing the execution
Niter = 10
t1e = time.time()
for i in range(Niter):
    __ = overlaps_(segments).block_until_ready()
t2e = time.time()

print('Time to execute:', (t2e - t1e)/Niter)

# sys.exit()
plt.figure()
ax = plt.axes(projection=threshproj(Mollweide()))
ax.set_global()
plot_map(oceanbg=(0.95, 0.95, 0.95))
count = np.sum(idx + idx.T, axis=0)
norm = Normalize(vmin=np.min(count), vmax=np.max(count))
cmap = plt.get_cmap('hot_r')
pseg = np.array(segments)

pos = np.argsort(count)

for i, (_seg, _c) in enumerate(zip(pseg[pos, :], count[pos])):

    ax.plot(_seg[1::2], _seg[::2],
            c=cmap(norm(_c)), transform=Geodetic())

plt.colorbar(ScalarMappable(cmap=cmap, norm=norm))

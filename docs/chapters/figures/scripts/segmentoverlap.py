import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from lwsspy.maps.overlap import overlap
from lwsspy.maps.map_axes import map_axes
from lwsspy.maps.plot_map import plot_map
from lwsspy.statistics.even2Dpoints import even2Dpoints
from cartopy.crs import PlateCarree


lon, lat = even2Dpoints(10, height=180, width=360, radius=10)

# Creating List of segments
segments = [(_lat0, _lon0, _lat1, _lon1) for _i, (_lat0, _lon0) in enumerate(zip(lat, lon))
            for _lat1, _lon1 in zip(lat[_i+1:], lon[_i+1:])
            if (((_lat0-_lat1)+(_lon0-_lon1)) != 0)]

# Creating counter matrix
idx = np.zeros((len(segments), len(segments)), dtype=bool)

# Segment overlap
for _i, x in enumerate(segments):
    for _j, y in enumerate(segments[_i+1:]):

        idx[_i, _i+1+_j] = overlap(x[:2], x[2:], y[:2], y[2:])

plt.figure()
ax = map_axes()
plot_map()
count = np.sum(idx + idx.T, axis=0)
norm = Normalize(vmin=np.min(count), vmax=np.max(count))
cmap = plt.get_cmap('hot_r')
pseg = np.array(segments)

for i in range(len(pseg)):

    ax.plot(pseg[i, 1::2], pseg[i, ::2],
            c=cmap(norm(count[i])), transform=PlateCarree())

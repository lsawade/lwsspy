# %%
import vtk
import numpy as np
import pyvista as pv
from copy import deepcopy
from pyglimer.ccp.ccp import read_ccp
import lwsspy as lpy


# %%
ccpstack = read_ccp(filename='US_P_0.58_minrad_3D_it_f2.pkl',
                    folder='ccps', fmt=None)

lat = ccpstack.coords_new[0]
lon = ccpstack.coords_new[1]


x, y, z = lpy.geo2cart(lpy.EARTH_RADIUS_KM, lat, lon)

# %%

r = lpy.EARTH_RADIUS_KM * np.ones_like(lat)
points = np.vstack((deepcopy(lon*lpy.DEG2KM).flatten(),
                    deepcopy(lat*lpy.DEG2KM).flatten(),
                    deepcopy(r).flatten())).T

pc = pv.PolyData(points)

mesh = pc.delaunay_2d(alpha=ccpstack.binrad*1.5*lpy.DEG2KM)
# mesh.plot()

# %%
points = deepcopy(mesh.points)
n_points = mesh.n_points
cells = mesh.faces.reshape(mesh.n_cells, 4)
cells[:, 0] = 6
cells = np.hstack((cells, n_points + cells[:, 1:]))
newcells = deepcopy(cells)

zpoints = deepcopy(np.array(points))
zpoints[:, 2] = lpy.EARTH_RADIUS_KM - ccpstack.z[0]
newpoints = np.vstack((points, zpoints))

for _z in ccpstack.z[2:]:

    # Add cells
    extra_cells = cells
    extra_cells[:, 1:] += n_points
    newcells = np.vstack((newcells, extra_cells))

    # Add points
    zpoints = deepcopy(np.array(points))
    zpoints[:, 2] = lpy.EARTH_RADIUS_KM - _z
    newpoints = np.vstack((newpoints, zpoints))

# Define Cell types
newcelltypes = np.array([vtk.VTK_WEDGE] * newcells.shape[0], dtype=np.uint8)

#
grid = pv.UnstructuredGrid(newcells, newcelltypes, newpoints)

grid['RF'] = deepcopy(ccpstack.ccp.T.ravel())
grid['illumination'] = deepcopy(ccpstack.hits.T.ravel())
val = 50
grid['opacity'] = np.where(
    grid['illumination'] > val, 1.0, grid['illumination']/val)


p = pv.Plotter()
p.set_scale(xscale=1.0, yscale=1.0, zscale=-1.0, reset_camera=True)
p.add_orientation_widget(grid)
p.add_mesh(grid, opacity=0.15, cmap='seismic')
slices = grid.slice_orthogonal(
    x=-100*lpy.DEG2KM, y=50*lpy.DEG2KM, z=lpy.EARTH_RADIUS_KM - 30)
p.add_mesh(slices, cmap='seismic')
p.show()

# newcelltypes.append(vtk.VTK_WEDGE)


# newcelltypes = []
# for _dep in ccpstack.z[1:]:
#     for _cell in cellsfilt:
#         _cell[0] = 6
#         _cell.extend([x + n_points for x in _cell[1:]])
#         newcelltypes.append(vtk.VTK_WEDGE)


# %%

# geocoords
x, y, z = lpy.geo2cart(
    newpoints[:, 2], newpoints[:, 1]*lpy.KM2DEG, newpoints[:, 0]*lpy.KM2DEG)

spherepoints = np.vstack((x, y, z)).T
spherepc = pv.PolyData(spherepoints)
spherepc.plot()

# %%
# Compute grid from points
grid = pv.UnstructuredGrid(newcells, newcelltypes, spherepoints)

# Add values to the mesh
grid['RF'] = deepcopy(ccpstack.ccp.T.ravel())


# %%

n_points = mesh.n_points
n_cells = mesh.n_cells

# Get the cells in an ordered fashion
# cells = mesh.cells.reshape((n_cells, 4))


cellsfilt = []
for _i, (_off, _doff) in enumerate(zip(mesh.offset[:-1], np.diff(mesh.offset))):
    addlist = mesh.cells[_off+_i:_off+_i+_doff+1].tolist()
    cellsfilt.append(addlist)

newcelltypes = []
for _cell in cellsfilt:
    if _cell[0] == 3:
        _cell[0] = 6
        _cell.extend([x + n_points for x in _cell[1:]])
        newcelltypes.append(vtk.VTK_WEDGE)
    elif _cell[0] == 2:
        _cell[0] = 4
        _cell.extend([x + n_points for x in _cell[:0:-1]])
        newcelltypes.append(vtk.VTK_QUAD)
    else:
        print("Hi theres something else: ", _cell)

# Flattening uneven list
newcells = np.array([item for sublist in cellsfilt for item in sublist])
newcelltypes = np.array(newcelltypes, dtype=np.uint8)
points = np.array(deepcopy(mesh.points))
newpoints = np.vstack((points, points*0.95))

grid = pv.UnstructuredGrid(newcells, newcelltypes, newpoints)

# %%
n_cells = int(len(cellsfilt)/4)
cellsfilt = np.array(cellsfilt)
cells = np.array(cellsfilt).reshape((n_cells, 4))

# Set celltypes for the
newcelltypes = np.array([vtk.VTK_WEDGE] * n_cells, dtype=np.uint8)

# Manipulate cells to become Wedges
cellsbelow = cells[:, 1:] + n_points

newcells = np.hstack((cells, cellsbelow))

newcells[:, 0] = 6

points = np.array(deepcopy(mesh.points))

newpoints = np.vstack((points, points*0.99))

# %%
grid = pv.UnstructuredGrid(newcells, newcelltypes, newpoints)

import numpy as np
import pyvista as pv
import math
import lwsspy as lpy


def plot_mesh(mesh: pv,
              rotlat: float or None = None,
              rotlon: float or None = None):

    # Initiate Plotter
    p = pv.Plotter()
    xmin, xmax, ymin, ymax, zmax, zmin = mesh.bounds

    if rotlon is not None:
        pv.core.common.axis_rotation(
            mesh.points, rotlon, inplace=True, deg=True, axis='z')
    if rotlat is not None:
        pv.core.common.axis_rotation(
            mesh.points, rotlat, inplace=True, deg=True, axis='y')

    def my_plane_funcx(loc):
        # Compute normal from
        coangle = np.pi/2 - np.arccos(loc)
        loc1x = np.cos(coangle)
        loc1z = np.sin(coangle)
        normal = [loc1x, 0, loc1z]
        normal = normal/np.linalg.norm(normal)

        # Compute Slice
        slc = mesh.slice(normal=normal, origin=[0, 0, 0])
        p.add_mesh(slc, name='Xslice')

    def my_plane_funcy(loc):
        # Compute normal from
        coangle = np.pi/2 - np.arccos(loc)
        loc1x = np.cos(coangle)
        loc1z = np.sin(coangle)
        normal = [0, loc1x, loc1z]
        normal = normal/np.linalg.norm(normal)

        # Compute Slice
        slc = mesh.slice(normal=normal, origin=[0, 0, 0])
        p.add_mesh(slc, name='Yslice')

    def my_plane_funcz(loc):
        slc = mesh.slice(normal=[0, 1, 0], origin=[0, loc, 0])
        p.add_mesh(slc, name='Yslice')

    p.add_mesh(mesh, opacity=0.1)

    p.add_slider_widget(
        my_plane_funcx, value=0, title='X', rng=[xmin, xmax],
        pointa=(.025, .1), pointb=(.31, .1),
        style='modern')
    p.add_slider_widget(
        my_plane_funcy, value=0, title='Y', rng=[ymin, ymax],
        pointa=(.35, .1), pointb=(.64, .1))
    p.add_slider_widget(
        my_plane_funcy, value=0, title='Z', rng=[ymin, ymax],
        pointa=(.68, .1), pointb=(.93, .1))
    p.show_bounds(bounds=mesh.bounds, location='origin')
    # p.show_grid()
    # p.add_axes()
    p.show()


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


class MeshPlot():

    def __init__(self, mesh,
                 rotlat: float or None = None,
                 rotlon: float or None = None,
                 get_mean: bool = False):
        self.mesh = mesh.copy(deep=True)  # Expected PyVista mesh type

        if rotlon is not None:
            pv.core.common.axis_rotation(
                self.mesh.points, rotlon, inplace=True, deg=True, axis='z')
        if rotlat is not None:
            pv.core.common.axis_rotation(
                self.mesh.points, rotlat, inplace=True, deg=True, axis='y')

        # Get bounds
        self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax = \
            self.mesh.bounds

        # Get the vector meanposition of your dataset
        if get_mean:
            self.initpos = np.mean(self.mesh.points, axis=0)
        else:
            # Set initial position and normal
            self.initpos = np.array([1, 0, 0])

        _, self.latitude, self.longitude = lpy.cart2geo(*self.initpos)

        self.center = self.initpos
        self.normaly = np.array([0, 1, 0])
        self.normalz = np.array([0, 0, 1])

        # default parameters
        self.rotation_angle = 0.0
        # self.x = (self.xmax + self.xmin)/2
        # self.y = (self.ymax + self.ymin)/2
        # self.z = (self.zmax + self.zmin)/2
        self.center = np.array([1, 0, 0])

        self.rotmat = np.eye(3)
        self.rotmat_lat = np.eye(3)
        self.rotmat_lon = np.eye(3)
        self.rotmat_slice = np.eye(3)

        # Initiate Plotter
        self.p = pv.Plotter()

        self.p.add_mesh(self.mesh, opacity=0.1)

        # Rotation Widgets
        self.p.add_slider_widget(
            self.rotate_lat, value=self.latitude, title='Lat', rng=[-90, 90],
            pointa=(.025, .35), pointb=(.175, .35))
        self.p.add_slider_widget(
            self.rotate_lon, value=self.longitude, title='Lon', rng=[-180, 180],
            pointa=(.025, .2), pointb=(.175, .2))
        self.p.add_slider_widget(
            self.rotate_slice, value=0, title='Rot', rng=[0, 90],
            pointa=(.025, .05), pointb=(.175, .05))

        # p.add_slider_widget(
        #     self.my_plane_funcy, value=0, title='Z', rng=[ymin, ymax],
        #     pointa=(.68, .1), pointb=(.93, .1))
        self.p.show_bounds(bounds=mesh.bounds, location='origin')

        self.p.show()

    def my_plane_funcy(self):

        # Rotation matrix of the normal
        normal = self.rotmat @ self.normaly

        # Compute Slice
        slc = self.mesh.slice(normal=normal, origin=[0, 0, 0])
        self.p.add_mesh(slc, name='Yslice')

    def my_plane_funcz(self):

        # Rotation matrix of the normal
        normal = self.rotmat @ self.normalz

        # Compute Slice
        slc = self.mesh.slice(normal=normal, origin=[0, 0, 0])
        self.p.add_mesh(slc, name='Zslice')

    def rotate_lat(self, latitude):
        self.latitude = latitude
        self.center = lpy.geo2cart(1.0, self.latitude, self.longitude)
        self.rotmat_lat = rotation_matrix(
            [0, 1, 0], -self.latitude/180.0*np.pi)
        self.update_rotmat()

    def rotate_lon(self, longitude):
        self.longitude = longitude
        self.center = lpy.geo2cart(1.0, self.latitude, self.longitude)
        self.rotmat_lon = rotation_matrix(
            [0, 0, 1], self.longitude/180.0*np.pi)
        self.update_rotmat()

    def rotate_slice(self, angle):
        angle = angle/180.0*np.pi
        self.rotmat_slice = rotation_matrix(list(self.center), angle)
        self.update_rotmat()

    def update_rotmat(self):
        self.rotmat = self.rotmat_lon @ self.rotmat_lat @ self.rotmat_slice
        self.my_plane_funcy()
        self.my_plane_funcz()

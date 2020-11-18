import numpy as np
import pyvista as pv


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
    p.show_bounds(bounds=mesh.bounds, location='back')
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
                 rotlon: float or None = None):
        self.mesh = mesh.copy(deep=True)  # Expected PyVista mesh type

        if rotlon is not None:
            pv.core.common.axis_rotation(
                self.mesh.points, rotlon, inplace=True, deg=True, axis='z')
        if rotlat is not None:
            pv.core.common.axis_rotation(
                self.mesh.points, rotlat, inplace=True, deg=True, axis='y')

        # default parameters
        self.rotation_angle = 0.0
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.normalx = [1, 0, 0]
        self.normaly = [0, 1, 0]
        self.normaly = [0, 0, 1]
        self.rotmat = np.eye(3)
        self.rotmatx = np.eye(3)
        self.rotmaty = np.eye(3)
        self.rotmatz = np.eye(3)

        # Initiate Plotter
        p = pv.Plotter()
        xmin, xmax, ymin, ymax, zmax, zmin = mesh.bounds

        p.add_mesh(mesh, opacity=0.1)

        p.add_slider_widget(
            self.my_plane_funcx, value=0, title='X', rng=[xmin, xmax],
            pointa=(.025, .1), pointb=(.31, .1),
            style='modern')
        p.add_slider_widget(
            self.my_plane_funcy, value=0, title='Y', rng=[ymin, ymax],
            pointa=(.35, .1), pointb=(.64, .1))
        # p.add_slider_widget(
        #     self.my_plane_funcy, value=0, title='Z', rng=[ymin, ymax],
        #     pointa=(.68, .1), pointb=(.93, .1))
        p.show_bounds(bounds=mesh.bounds, location='back')
        

    def my_plane_funcx(self, loc):
        # Compute normal from
        coangle = np.pi/2 - np.arccos(loc)
        loc1x = np.cos(coangle)
        loc1z = np.sin(coangle)
        normal = [loc1x, 0, loc1z]
        normal = normal/np.linalg.norm(normal)

        # Rotation matrix of the normal
        normal = self.rotmat @ normal
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

    def rotatex(self, angle):
        self.rotmatx = rotation_matrix([1, 0, 0], angle)
        self.update_rotmat()

    def update_rotmat(self):
        self.rotmat = self.rotmatx @ self.rotmaty @ self.rotmatz

    

    if rotlon is not None:
        pv.core.common.axis_rotation(
            mesh.points, rotlon, inplace=True, deg=True, axis='z')
    if rotlat is not None:
        pv.core.common.axis_rotation(
            mesh.points, rotlat, inplace=True, deg=True, axis='y')

    def my_plane_funcz(loc):
        slc = mesh.slice(normal=[0, 1, 0], origin=[0, loc, 0])
        p.add_mesh(slc, name='Zslice')

    def rotate(angle):

    
    # p.show_grid()
    # p.add_axes()
    p.show()

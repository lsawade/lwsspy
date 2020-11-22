import numpy as np
import pyvista as pv
import vtkmodules
import math
import lwsspy as lpy
import matplotlib.pyplot as plt


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
        pointa=(.025, .1), pointb=(.31, .15),
        style='modern')
    p.add_slider_widget(
        my_plane_funcy, value=0, title='Y', rng=[ymin, ymax],
        pointa=(.35, .1), pointb=(.64, .15))
    p.add_slider_widget(
        my_plane_funcy, value=0, title='Z', rng=[ymin, ymax],
        pointa=(.68, .1), pointb=(.93, .15))
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


class SetVisibilityCallback:
    """Helper callback to keep a reference to the actor being modified."""

    def __init__(self, actors):
        if type(actors) is list:
            self.actors = actors
        else:
            self.actors = [actors]

    def __call__(self, state):
        for _actor in self.actors:
            if type(_actor) is vtkmodules.vtkInteractionWidgets.vtkSliderWidget:
                pass
            else:
                _actor.SetVisibility(state)


class MeshPlot():

    def __init__(self, mesh,
                 lat: float or None = None,
                 lon: float or None = None,
                 get_mean: bool = False):
        # Init state
        self.not_init_state = False

        # Copy to not modify original mesh when rotating
        self.mesh = mesh.copy(deep=True)

        # Function to get activae scalar
        self.meshname = 'vpv'

        if rotlon is not None:
            pv.core.common.axis_rotation(
                self.mesh.points, rotlon, inplace=True, deg=True, axis='z')
        if rotlat is not None:
            pv.core.common.axis_rotation(
                self.mesh.points, rotlat, inplace=True, deg=True, axis='y')

        # Get bounds
        self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax = \
            self.mesh.bounds

        # Get colorbounds
        self.minM, self.maxM = self.mesh.get_data_range()
        self.clim = [self.minM, self.maxM]
        self.initclim = [self.minM, self.maxM]

        # Get the vector meanposition of your dataset
        if get_mean:
            self.initpos = np.mean(self.mesh.points, axis=0)
        else:
            # Set initial position and normal
            self.initpos = np.array([1, 0, 0])

        # Get starting latitude and longitude values
        _, self.latitude, self.longitude = lpy.cart2geo(*self.initpos)
        self.center = self.initpos

        # Only set so that it doesn't have to be redefined
        self.normaly = np.array([0, 1, 0])
        self.normalz = np.array([0, 0, 1])

        # default parameters
        self.rotmat = np.eye(3)
        self.rotmat_lat = np.eye(3)
        self.rotmat_lon = np.eye(3)
        self.rotmat_slice = np.eye(3)
        self.rotangle = 0.0

        # Initiate Plotter
        self.p = pv.Plotter()

        # Colorbar title
        self.cmapname = "gist_rainbow"
        self.cmap = plt.cm.get_cmap(self.cmapname, 256)
        self.cmapsym = False
        self.scalar_bar_args = dict()

        # Add surrounding volume
        self.volume = self.p.add_mesh(self.mesh, opacity=0.1,
                                      stitle=self.meshname, clim=self.clim,
                                      cmap=self.cmap)

        callback = SetVisibilityCallback(self.volume)
        self.p.add_checkbox_button_widget(
            callback, value=True, size=25, position=(10, 10),
            color_on='white', color_off='grey', background_color='grey')

        # Rotation Widgets
        self.rotlat_slider = self.p.add_slider_widget(
            self.rotate_lat, value=self.latitude, title='Lat', rng=[-90, 90],
            pointa=(.025, .4), pointb=(.175, .4), event_type='always')
        self.rotlon_slider = self.p.add_slider_widget(
            self.rotate_lon, value=self.longitude, title='Lon', rng=[-180, 180],
            pointa=(.025, .25), pointb=(.175, .25), event_type='always')
        self.rotate_slider = self.p.add_slider_widget(
            self.rotate_slice, value=self.rotangle, title='Rot', rng=[0, 90],
            pointa=(.025, .1), pointb=(.175, .1), event_type='always')

        # Colorbar/Scalar Widget
        self.cmin_slider = self.p.add_slider_widget(
            self.cmin_callback, value=self.clim[0], title='cmin',
            rng=[self.minM, self.maxM],
            pointa=(.025, .9), pointb=(.175, .9), event_type='always')
        self.cmax_slider = self.p.add_slider_widget(
            self.cmax_callback, value=self.clim[1], title='cmax',
            rng=[self.minM, self.maxM],
            pointa=(.225, .9), pointb=(.375, .9), event_type='always')
        self.cmax_slider.GetRepresentation().SetValue(9.0)
        # widget_visibility_callback = SetVisibilityCallback(
        #     [self.rotlat_slider, self.rotlon_slider,
        #      self.rotate_slider, self.cmin_slider,
        #      self.cmax_slider])

        # self.p.add_checkbox_button_widget(
        #     widget_visibility_callback, value=True, size=25,
        #     position=(40, 10), color_on='white', color_off='grey',
        #     background_color='grey')

        # p.add_slider_widget(
        #     self.my_plane_funcy, value=0, title='Z', rng=[ymin, ymax],
        #     pointa=(.68, .1), pointb=(.93, .1))
        self.p.show_bounds(bounds=mesh.bounds, location='origin')

        # Init state
        self.not_init_state = True
        self.p.show(use_ipyvtk=True, return_viewer=True)

    def my_plane_funcy(self):

        # Rotation matrix of the normal
        normal = self.rotmat @ self.normaly

        # Compute Slice
        slc = self.mesh.slice(normal=normal, origin=[0, 0, 0])
        self.p.add_mesh(slc, name='Yslice',
                        stitle=self.meshname,
                        clim=self.clim,
                        cmap=self.cmap)

    def my_plane_funcz(self):

        # Rotation matrix of the normal
        normal = self.rotmat @ self.normalz

        # Compute Slice
        slc = self.mesh.slice(normal=normal, origin=[0, 0, 0])
        self.p.add_mesh(slc, name='Zslice',
                        stitle=self.meshname,
                        clim=self.clim,
                        cmap=self.cmap)

    def cmin_callback(self, val):
        if self.cmapsym:
            pass
        else:
            self.clim = [val, self.clim[1]]

        self.p.update_scalar_bar_range(self.clim, name=self.meshname)

        if self.not_init_state:
            self.cmax_slider.GetRepresentation(
            ).SetMinimumValue(self.clim[0])

    def cmax_callback(self, val):
        if self.cmapsym:
            pass
        else:
            self.clim = [self.clim[0], val]

        self.p.update_scalar_bar_range(self.clim, name=self.meshname)

        if self.not_init_state:
            self.cmin_slider.GetRepresentation(
            ).SetMaximumValue(self.clim[1])

    def rotate_lat(self, latitude):
        self.latitude = latitude
        self.center = lpy.geo2cart(1.0, self.latitude, self.longitude)
        self.rotmat_lat = rotation_matrix(
            [0, 1, 0], -self.latitude/180.0*np.pi)
        # Doesn't actually use rotangle in when update mat only is true!
        self.rotate_slice(self.rotangle, update_mat_only=True)
        self.update()

    def rotate_lon(self, longitude):
        self.longitude = longitude
        self.center = lpy.geo2cart(1.0, self.latitude, self.longitude)
        self.rotmat_lon = rotation_matrix(
            [0, 0, 1], self.longitude/180.0*np.pi)
        # Doesn't actually use rotangle in when update mat only is true!
        self.rotate_slice(self.rotangle, update_mat_only=True)
        self.update()

    def rotate_slice(self, angle, update_mat_only=False):
        if update_mat_only is False:
            self.rotangle = angle/180.0*np.pi
        self.rotmat_slice = rotation_matrix(list(self.center), self.rotangle)
        if update_mat_only is False:
            self.update()

    def update(self):
        self.rotmat = self.rotmat_slice @ self.rotmat_lon @ self.rotmat_lat
        self.my_plane_funcy()
        self.my_plane_funcz()
        self.p.update_scalar_bar_range(self.clim, name=self.meshname)

    def reset(self):

        self.clim = [self.minM, self.maxM]
        self.center = self.initpos

        """
        colorbar slider
        sargs = dict(height=0.25, vertical=True,
                     position_x=0.05, position_y=0.05,
                     interactive=True
                     title_font_size=20,
                    label_font_size=16,
                    shadow=True,
                    n_labels=3,
                    italic=True,
                    fmt="%.1f",
                    font_family="arial",)
        annotations = {
            2300: "High",
            805.3: "Cutoff value",
        }
        addmesh(mesh, clim=[1000, 2000],
                below_color='blue',
                above_color='red',
                scalar_bar_args=sargs,
                annotations=annotations)

        update_scalar_bar_range(clim, name=None)

        view vector
        Change the range of a slider!
        widget.GetRepresentation().SetMinimumValue(-1)
        widget.GetRepresentation().SetMaximumValue(2)

        # Reset button

        #
        # SET CAMERA VIEW
        pos = grid_from_sph_coords([180], [90-30], [RADIUS * 10])
        p.set_position(pos.points)
        p.set_viewup((0,0,1))
        """

import numpy as np
import pyvista as pv
import vtkmodules
import math
import lwsspy as lpy
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, LinearTriInterpolator
from copy import deepcopy


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
                 rotlat: float or None = None,
                 rotlon: float or None = None,
                 get_mean: bool = True,
                 opacity: bool = True):
        # Init state
        self.not_init_state = False

        # Copy to not modify original mesh when rotating
        self.mesh = mesh.copy(deep=True)

        # Function to get activae scalar
        self.meshname = 'RF'

        # To rotate data set
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
        self.normalx = np.array([1, 0, 0])
        self.normaly = np.array([0, 1, 0])
        self.normalz = np.array([0, 0, 1])

        # default parameters
        self.rotmat = np.eye(3)
        self.rotmat_lat = np.eye(3)
        self.rotmat_lon = np.eye(3)
        self.rotmat_slice = np.eye(3)
        self.rotangle = 0.0

        # Empty slice parameters
        self.Yslice = dict(name="Yslice", name2D="Yslice2D", slc=None,
                           slc3D=None, slc2D=None,
                           normal=[0, 1, 0], origin=[0, 0, 0])
        self.Zslice = dict(name="Zslice", name2D="Zslice2D", slc=None,
                           slc3D=None, slc2D=None,
                           normal=[0, 1, 0], origin=[0, 0, 0])

        # Get initial rotation matrix
        self.rotate_lat(self.latitude, update_mat_only=True)
        self.rotate_lon(self.longitude, update_mat_only=True)
        self.rotate_slice(self.rotangle, update_mat_only=True)

        # Initiate Plotter
        pv.rcParams['multi_rendering_splitting_position'] = 0.750
        self.p = pv.Plotter(shape='1|3', window_size=(1600, 900))

        # Colorbar title
        self.cmapname = "seismic"
        self.cmap = plt.cm.get_cmap(self.cmapname, 256)
        self.cmapsym = True
        if self.cmapsym:
            self.cabsmax = np.max(np.abs([self.minM, self.maxM]))
            self.minM, self.maxM = [-self.cabsmax, self.cabsmax]
            self.clim = [-0.5*self.cabsmax, 0.5*self.cabsmax]
            self.initclim = [-0.5*self.cabsmax, 0.5*self.cabsmax]
            # print(self.cabsmax)
            # print(self.minM, self.maxM)
            # print(self.clim)
            # print(self.initclim)
            self.cminrange = [self.clim[0], 0.0]
            self.cmaxrange = [0, self.clim[1]]

        else:
            self.clim = [self.minM, self.maxM]
            self.initclim = [self.minM, self.maxM]
            self.cminrange = [self.minM, self.maxM]
            self.cmaxrange = [self.minM, self.maxM]

        self.scalar_bar_args = dict(
            width=0.4, height=0.05, position_x=0.3, position_y=0.05)

        # Add surrounding volume
        if self.mesh['illumination'] is not None and opacity is True:
            self.init_illum = 50
            self.opacity = 'opacity'
            self.mesh['opacity'] = np.where(
                self.mesh['illumination'] >= self.init_illum, 1.0,
                self.mesh['illumination']/self.init_illum)
            self.mesh.set_active_scalars(self.meshname)
        else:
            self.opacity = 1.0

        self.p.subplot(0)
        self.volume = self.p.add_mesh(self.mesh, opacity=0.1,
                                      stitle=self.meshname, clim=self.clim,
                                      cmap=self.cmap,
                                      show_scalar_bar=False)
        self.p.add_scalar_bar(
            title=self.meshname, **self.scalar_bar_args)

        self.p.add_orientation_widget(self.mesh)

        # Rotation Widgets
        self.p.subplot(1)
        self.rotlat_slider = self.p.add_slider_widget(
            self.rotate_lat, value=self.latitude, title='Lat', rng=[-90, 90],
            pointa=(.025, .4), pointb=(.975, .4))  # , event_type='always')
        self.p.subplot(1)
        self.rotlon_slider = self.p.add_slider_widget(
            self.rotate_lon, value=self.longitude, title='Lon', rng=[-180, 180],
            pointa=(.025, .25), pointb=(.975, .25))  # , event_type='always')
        self.p.subplot(1)
        self.rotate_slider = self.p.add_slider_widget(
            self.rotate_slice, value=self.rotangle, title='Rot', rng=[0, 90],
            pointa=(.025, .1), pointb=(.975, .1))  # , event_type='always')

        self.p.subplot(1)
        callback = SetVisibilityCallback(self.volume)
        self.p.add_checkbox_button_widget(
            callback, value=True, size=25, position=(10, 10),
            color_on='white', color_off='grey', background_color='grey')

        # Colorbar/Scalar Widget
        pv.rcParams['slider_style']['classic'].update(
            dict(slider_width=0.01, cap_width=0.01, tube_width=0.001))
        self.cmin_slider = self.p.add_slider_widget(
            self.cmin_callback, value=self.clim[0], title='cmin',
            rng=self.cminrange,
            pointa=(.025, .9), pointb=(.475, .9), event_type='always')
        self.cmax_slider = self.p.add_slider_widget(
            self.cmax_callback, value=self.clim[1], title='cmax',
            rng=self.cmaxrange,
            pointa=(.525, .9), pointb=(.975, .9), event_type='always')

        # Opacity
        if self.mesh['illumination'] is not None and opacity is True:
            self.opacity_slider = self.p.add_slider_widget(
                self.opacity_function, value=self.init_illum,
                title='Illumination', rng=[1., 500.0],
                pointa=(.525, .75), pointb=(.975, .75))  # , event_type='always')

        # self.cmax_slider.GetRepresentation().SetValue(9.0)
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

        # Actually plot bounds
        self.p.subplot(0)
        bounds = self.p.show_bounds(bounds=mesh.bounds, location='back')
        bounds_callback = SetVisibilityCallback(bounds)

        # bounds control button
        self.p.subplot(1)
        self.p.add_checkbox_button_widget(
            bounds_callback, value=True, size=25, position=(40, 10),
            color_on='white', color_off='grey', background_color='grey')

        self.p.add_checkbox_button_widget(
            self.plot_y_slice, value=True, size=25, position=(70, 10),
            color_on='white', color_off='grey', background_color='grey')

        # Init state
        self.not_init_state = True
        self.p.show(use_ipyvtk=True, return_viewer=True)

    def my_plane_funcy(self):

        # Rotation matrix of the normal
        self.normalup = self.rotmat @ self.normalx
        self.Yslice['normal'] = self.rotmat @ self.normaly

        # Compute Slice
        self.Yslice['slc'] = self.mesh.slice(
            normal=self.Yslice['normal'], origin=self.Yslice['origin'])

        # 3D slice
        self.p.subplot(0)
        self.Yslice['slc3D'] = self.p.add_mesh(
            self.Yslice['slc'], name=self.Yslice['name'],
            stitle=self.meshname,
            clim=self.clim,
            cmap=self.cmap,
            opacity=self.opacity)

        # 2D slice
        self.p.subplot(2)
        self.Yslice['slc2D'] = self.p.add_mesh(
            self.Yslice['slc'], name=self.Yslice['name2D'],
            stitle=self.meshname,
            clim=self.clim,
            cmap=self.cmap,
            opacity=self.opacity),
        self.p.view_vector(-self.Yslice['normal'], viewup=self.normalup)

    def my_plane_funcz(self):

        # Rotation matrix of the normal
        self.normalup = self.rotmat @ self.normalx
        self.Zslice["normal"] = self.rotmat @ self.normalz

        # Get slice
        self.Zslice["slc"] = self.mesh.slice(
            normal=self.Zslice['normal'], origin=self.Zslice['origin'])

        # Render 3D slice
        self.p.subplot(0)
        self.Zslice["slc3D"] = self.p.add_mesh(
            self.Zslice['slc'], name=self.Zslice['name'],
            stitle=self.meshname,
            clim=self.clim,
            cmap=self.cmap,
            opacity=self.opacity)

        # Render 2D slice
        self.p.subplot(3)
        self.Zslice["slc2D"] = self.p.add_mesh(
            self.Zslice['slc'], name=self.Zslice['name2D'],
            stitle=self.meshname,
            clim=self.clim,
            cmap=self.cmap,
            opacity=self.opacity)

        self.p.view_vector(-self.Zslice['normal'], viewup=self.normalup)

    def opacity_function(self, val):

        self.mesh['opacity'] = np.where(
            self.mesh['illumination'] >= val, 1.0,
            self.mesh['illumination']/val)
        self.mesh.set_active_scalars(self.meshname)
        self.update()

    def cmin_callback(self, val):
        if self.cmapsym:
            self.clim = [val, -val]
        else:
            self.clim = [val, self.clim[1]]

        self.p.update_scalar_bar_range(self.clim, name=self.meshname)

        if self.not_init_state:
            if self.cmapsym:
                self.cmax_slider.GetRepresentation(
                ).SetValue(-val)
            else:
                self.cmax_slider.GetRepresentation(
                ).SetMinimumValue(self.clim[0])

    def cmax_callback(self, val):
        if self.cmapsym:
            self.clim = [-val, val]
        else:
            self.clim = [self.clim[0], val]

        self.p.update_scalar_bar_range(self.clim, name=self.meshname)

        if self.not_init_state:
            if self.cmapsym:
                self.cmin_slider.GetRepresentation(
                ).SetValue(-val)
            else:
                self.cmin_slider.GetRepresentation(
                ).SetMaximumValue(self.clim[1])

    def rotate_lat(self, latitude, update_mat_only=False):
        self.latitude = latitude
        self.center = lpy.geo2cart(1.0, self.latitude, self.longitude)
        self.rotmat_lat = rotation_matrix(
            [0, 1, 0], -self.latitude/180.0*np.pi)
        # Doesn't actually use rotangle in when update mat only is true!
        self.rotate_slice(self.rotangle, update_mat_only=True)
        if update_mat_only is False:
            self.update()

    def rotate_lon(self, longitude, update_mat_only=False):
        self.longitude = longitude
        self.center = lpy.geo2cart(1.0, self.latitude, self.longitude)
        self.rotmat_lon = rotation_matrix(
            [0, 0, 1], self.longitude/180.0*np.pi)
        # Doesn't actually use rotangle in when update mat only is true!
        self.rotate_slice(self.rotangle, update_mat_only=True)

        if update_mat_only is False:
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

    def plot_y_slice(self, val):

        array = self.slice_to_array(
            self.Yslice["slc"],
            self.Yslice["normal"],
            self.Yslice["origin"],
            self.meshname,
        )
        print(array)
        # plt.figure()
        # plt.matshow(array)
        # plt.show()

    def slice_to_array(self, slc, normal, origin, name, ni=500, nj=500):
        """Converts a PolyData slice to a 2D NumPy array.

        It is crucial to have the true normal and origin of
        the slicing plane

        Parameters
        ----------
        slc : PolyData
            The slice to convert.
        normal : tuple(float)
            the normal of the original slice
        origin : tuple(float)
            the origin of the original slice
        name : str
            The scalar array to fetch from the slice
        ni : int
            The resolution of the array in the i-direction
        nj : int
            The resolution of the array in the j-direction

        """
        # # Make structured grid
        # for out in np.meshgrid(x, y, z):
        #     print(out.shape)
        # plane = pv.StructuredGrid(*np.meshgrid(x, y, z))

        slctemp = slc.copy(deep=True)

        slctemp.points = (self.rotmat.T @ slctemp.points.T).T
        sliceup = self.rotmat.T @ self.normalup

        # Get angles
        # u = sliceup
        # v = [1, 0, 0]
        # c = np.dot(u, v)/np.linalg.norm(u) / np.linalg.norm(v)
        # -> cosine of the angle
        # angle = np.arccos(np.clip(c, -1, 1))  # if you really want the angle
        # print(angle)
        # print(slctemp.bounds)
        # print(slctemp.points)

        # Yet another triangulation
        points = np.vstack(
            (slctemp.points[:, 0],
             slctemp.points[:, 2],
             np.zeros_like(slctemp.points[:, 2]))).T
        pc = pv.PolyData(points)
        mesh = pc.delaunay_2d(alpha=0.5*1.5*lpy.DEG2KM)
        mesh['RF'] = slctemp['RF']

        #  Get triangles
        xy = np.array(mesh.points[:, 0:2])
        r, t = lpy.cart2pol(xy[:, 0], xy[:, 1])
        # t = np.where(t < 0, t + 2*np.pi, t)
        # t += 2 * np.pi
        findlimt = t + 4 * np.pi
        mint = np.min(findlimt) - 4 * np.pi
        maxt = np.max(findlimt) - 4 * np.pi
        cells = mesh.faces.reshape(mesh.n_cells, 4)
        triangles = np.array(cells[:, 1:4])

        print(xy.shape)
        print(triangles.shape)
        print(len(slctemp['RF']))
        print(len(mesh['RF']))

        # Set maximum slice length (makes no sense otherwise)
        if mint < -11.25:
            mint = -11.25
        if maxt > 11.25:
            maxt = 11.25

        plt.figure(figsize=(6.0, 8.0))

        ax = plt.subplot(111, projection='polar')
        ax.set_theta_zero_location("N")
        ax.set_rlim(bottom=np.min(lpy.EARTH_RADIUS_KM - r),
                    top=np.max(lpy.EARTH_RADIUS_KM - r))
        ax.set_rorigin(lpy.EARTH_RADIUS_KM)
        ax.set_thetamin(mint/np.pi*180.0)
        ax.set_thetamax(maxt/np.pi*180.0)
        ax.tick_params(labelbottom=False, labeltop=True,
                       labelleft=True, labelright=True,
                       left=True, right=True,
                       top=True, bottom=True)
        ax.tick_params(axis="x", direction="in", pad=-18)
        caxbound = ax.inset_axes([0.125, 0.01, 0.75, 0.1])
        lpy.remove_ticklabels(caxbound)
        lpy.remove_ticks(caxbound)
        cax = ax.inset_axes([0.15, 0.07, 0.7, 0.025])

        # ax.tick_params()

        #
        dmin = np.min(lpy.EARTH_RADIUS_KM - r)
        dmax = np.max(lpy.EARTH_RADIUS_KM - r)
        dsamp = np.linspace(dmin, dmax, 1000)
        tsamp = np.linspace(mint, maxt, 1000)
        tt, dd = np.meshgrid(tsamp, dsamp)

        # you can add keyword triangles here if you have the triangle array,
        # size [Ntri,3]
        triObj = Triangulation(t, r, triangles=triangles)

        # linear interpolation
        fz = LinearTriInterpolator(triObj, mesh['RF'])
        Z = fz(tt, lpy.EARTH_RADIUS_KM - dd)

        # plt.scatter(t, lpy.EARTH_RADIUS_KM - r)
        mesh = plt.pcolormesh(tt, dd, Z, cmap='seismic',
                              vmin=self.clim[0], vmax=self.clim[1])
        # plt.tripcolor(t/np.pi*180.0, lpy.EARTH_RADIUS_KM - r, mesh['RF'],
        #               triangles=triangles,
        #               cmap='seismic',
        #               vmin=self.clim[0], vmax=self.clim[1])
        # ax.set_xlim(mint, maxt)
        # ax.invert_yaxis()
        plt.colorbar(mesh, cax=cax, orientation='horizontal')
        plt.title(rf"$\leftarrow$ N{360-self.rotangle}$^\circ$")
        plt.show()
        # x = np.linspace(slc.bounds[0], slc.bounds[1], ni)
        # z = np.linspace(slc.bounds[2], slc.bounds[3], nj)

        # rotate and translate grid to be ontop of the slice
        # direction = normal / np.linalg.norm(normal)
        # vx -= vx.dot(direction) * direction
        # vx /= np.linalg.norm(vx)
        # vy = np.cross(direction, vx)
        # rmtx = np.array([vx, vy, direction])
        # plane.points = plane.points.dot(self.rotmat)
        # plane.points -= plane.center
        # plane.points += origin

        # resample the data
        # sampled = plane.sample(slc, tolerance=slc.length*0.5)
        self.p.subplot(0)
        self.p.add_mesh(slctemp)
        # print(sampled)
        # Fill bad data
        # sampled[name][~sampled["vtkValidPointMask"].view(bool)] = np.nan

        # plot the 2D array
        # array = sampled[name].reshape(sampled.dimensions[1:3])
        array = None
        return array

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

import numpy as np
import pyvista as pv
from pyvista import examples


class MeshPlotOpacity():

    def __init__(self, opacity=False):
        # Init state
        self.not_init_state = False

        if opacity is False:
            self.opacity = 1.0
        else:
            self.opacity = 'opacity'

        # Mesh mount saint helens example
        self.meshname = 'TestSlices'
        self.mesh = examples.download_st_helens().warp_by_scalar()
        xmin, xmax, ymin, ymax, zmax, zmin = self.mesh.bounds
        # Get colorbounds
        self.minM, self.maxM = self.mesh.get_data_range()
        # Add scalar array with range (0, 100) that correlates with elevation
        self.mesh['values'] = \
            pv.plotting.normalize(self.mesh['Elevation']) * 100
        # Set opacity
        self.mesh['opacity'] = \
            self.mesh['values']/np.max(self.mesh['values'])

        # Get colormap range
        self.clim = [self.minM, self.maxM]
        self.initclim = [self.minM, self.maxM]
        self.cminrange = [self.minM, self.maxM]
        self.cmaxrange = [self.minM, self.maxM]

        self.p = pv.Plotter()
        self.p.add_orientation_widget(self.mesh)
        self.p.add_mesh(self.mesh, opacity=0.05, stitle=self.meshname)

        self.p.add_slider_widget(
            self.my_plane_funcx, value=np.mean([xmin, xmax]), title='X', rng=[xmin, xmax],
            pointa=(.225, .1), pointb=(.51, .1),)
        self.p.add_slider_widget(
            self.my_plane_funcy, value=np.mean([ymin, ymax]), title='Y', rng=[ymin, ymax],
            pointa=(.55, .1), pointb=(.84, .1))

        self.cmin_slider = self.p.add_slider_widget(
            self.cmin_callback, value=self.clim[0], title='cmin',
            rng=self.cminrange,
            pointa=(.025, .9), pointb=(.475, .9), event_type='always')
        self.cmax_slider = self.p.add_slider_widget(
            self.cmax_callback, value=self.clim[1], title='cmax',
            rng=self.cmaxrange,
            pointa=(.525, .9), pointb=(.975, .9), event_type='always')

        self.p.show_bounds(bounds=self.mesh.bounds, location='back')
        # p.show_grid()
        # p.add_axes()
        self.p.show()

        self.meshname = 'RF'

        # Get bounds
        self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax = \
            self.mesh.bounds

    def cmin_callback(self, val):
        self.clim = [val, self.clim[1]]

        self.p.update_scalar_bar_range(self.clim, name=self.meshname)

        if self.not_init_state:
            self.cmax_slider.GetRepresentation(
            ).SetMinimumValue(self.clim[0])

    def cmax_callback(self, val):
        self.clim = [self.clim[0], val]

        self.p.update_scalar_bar_range(self.clim, name=self.meshname)

        if self.not_init_state:
            self.cmin_slider.GetRepresentation(
            ).SetMaximumValue(self.clim[1])

    def my_plane_funcx(self, loc):
        # Compute Slice
        slc = self.mesh.slice(normal=[1, 0, 0], origin=[loc, 0, 0])
        self.p.add_mesh(slc, name='Xslice', opacity=self.opacity,
                        stitle=self.meshname, clim=self.clim)

    def my_plane_funcy(self, loc):
        # Compute Slice
        slc = self.mesh.slice(normal=[0, 1, 0], origin=[0, loc, 0])
        self.p.add_mesh(slc, name='Yslice', opacity=self.opacity,
                        stitle=self.meshname, clim=self.clim)


MeshPlotOpacity()

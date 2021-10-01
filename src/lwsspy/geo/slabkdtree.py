import os
from ..math.SphericalNN import SphericalNN
from .get_slabs import get_slabs
import numpy as np
import _pickle as pickle
from .. import DOWNLOAD_CACHE
from lwsspy.utils.output import print_action


class SlabKDTree:
    """
    Load and save operations should work seamlessly, so that
    the class only has to be instatiated and the kdtrees will
    be built.

    Parameters
    ----------

    key: str, optional
        which parameter to read, see table below
        =====  ===========
        key    parameter
        =====  ===========
        'dep'  depth
        'dip'  dip
        'str'  strike
        'thk'  thickness
        'unc'  uncertainty
        =====  ===========
    """
    # Get slabs for each parameter
    parameters = ['dep', 'dip', 'str', 'thk', 'unc']
    filename = os.path.join(DOWNLOAD_CACHE, 'slabkdtree.pkl')

    def __init__(self, rebuild=False, verbose=False):

        self.verbose = verbose

        # If kdtree exists do load function
        if os.path.exists(self.filename) and (rebuild is False):
            self.load()
        else:
            # Else build the damn thing.
            self.build_kdtrees()

    def build_kdtrees(self):

        # Initialize empty data dictionary
        self.dss_dict = dict()

        # Populate with all data
        if self.verbose:
            print_action('Loading data from SLAB2.0')
        for _param in self.parameters:
            self.dss_dict[_param] = get_slabs(_param)

        # Plot contours
        self.skdtrees = []
        self.lat_list = []
        self.lon_list = []
        self.data_list = []
        self.bbox_list = []

        if self.verbose:
            print_action('Adding data to class')

        for _i, _dsz in enumerate(self.dss_dict['dep']):

            lon = _dsz['x'][:].data
            lon = np.where(lon > 180.0, lon - 360.0, lon)
            lat = _dsz['y'][:].data
            llon, llat = np.meshgrid(lon, lat)

            # Add lat, lon points to the lat lon list
            self.lat_list.append(llat)
            self.lon_list.append(llon)
            self.bbox_list.append(
                [lon[0], lat[0], lon[-1], lat[-1]])

            # Add the depth data of this dataset to the list
            data = np.ma.filled(_dsz['z'][:, :].data, np.nan)
            self.data_list.append(dict())
            self.data_list[_i]['dep'] = data

            # Add other parameters depending on data's size
            for _param in self.parameters:
                for _paramds in self.dss_dict[_param]:
                    if _paramds['z'][:, :].data.shape == data.shape:
                        self.data_list[_i][_param] = \
                            np.ma.filled(_paramds['z'][:, :].data, np.nan)

        # Create kdtrees for each dataset
        N = len(self.lat_list)
        for _i, (_lats, _lons) in enumerate(zip(self.lat_list, self.lon_list)):

            if self.verbose:
                print_action(f'Creating KDTree {_i:2}/{N}')
            self.skdtrees.append(SphericalNN(_lats.flatten(), _lons.flatten()))

        # Delete dss so that the object is picklable
        del self.dss_dict

        # Save Class
        self.save()

    def interpolators(
            self, qlat, qlon, maximum_distance=0.1, **kwargs):

        # Build interpolators
        self.interps = []
        N = len(self.skdtrees)

        if self.verbose:
            print_action(f'Creating Interpolators...')
        for _i, _kdtree in enumerate(self.skdtrees):

            # Checkif both
            inidx = self.checkifinbbox(self.bbox_list[_i], qlat, qlon)

            if np.any(inidx):

                self.interps.append(
                    _kdtree.interpolator(
                        qlat, qlon, k=5,
                        maximum_distance=maximum_distance,
                        p=2.0,
                        ** kwargs)
                )
            else:
                self.interps.append(None)

    def interpolate(self, qlat, qlon):

        # Create interpolators if they haven't been created
        self.interpolators(qlat, qlon)

        qdata_list = []

        for _i, (_interpolator, _data_dict) in \
                enumerate(zip(self.interps, self.data_list)):

            # Add empty interpolated data dictionary
            if _interpolator is None:
                qdata_list.append(None)
                continue
            else:
                qdata_list.append(dict())

            # Interpolate for each parameter
            for _param, _data in _data_dict.items():
                qdata_list[_i][_param] = _interpolator(_data.flatten())

        return self.interps, qdata_list

    def save(self):
        if self.verbose:
            print_action(f'Saving self')

        with open(self.filename, 'wb') as f:
            pickle.dump(self.__dict__, f, 2)

        if self.verbose:
            print(f'Saving done.')

    @staticmethod
    def checkifinbbox(bbox, lats, lons):

        # For checking the coordinate range
        lonrange = bbox[0::2]
        latrange = bbox[1::2]

        # Longitude around +/-180 range
        if (lonrange[0] > lonrange[1]):
            inlonrange = np.logical_or(
                lons >= lonrange[0], lons <= lonrange[1])
        else:
            inlonrange = np.logical_and(
                lons >= lonrange[0], lons <= lonrange[1])

        # Check if in latrange
        inlatrange = np.logical_and(
            lats >= latrange[0], lats <= latrange[1])

        # Checkif both
        inidx = np.logical_and(inlatrange, inlonrange)

        return inidx

    def load(self):
        if self.verbose:
            print_action(f'Loading Slab KDTrees.')

        with open(self.filename, 'rb') as f:
            self.__dict__.update(pickle.load(f))

        if hasattr(self, 'interps'):
            self.__delattr__('interps')

        if self.verbose:
            print(f'Loading done.')

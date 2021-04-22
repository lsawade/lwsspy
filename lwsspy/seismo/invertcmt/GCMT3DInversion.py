"""
This is the first script to enable depth iterations for the
CMTSOLUTTION depth.
"""
# %% Create inversion directory

# Internal
import lwsspy as lpy

# External
from typing import Callable, Union, Optional, List
import os
import shutil
import datetime
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pdf import PdfPages
from itertools import repeat
from obspy import read, read_events, Stream, Trace
import multiprocessing.pool as mpp
import _pickle as cPickle
from .process_classifier import ProcessParams

lpy.updaterc(rebuild=False)


# Main parameters
window_dict = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "body.window.yml")

SPECFEM = "/scratch/gpfs/lsawade/MagicScripts/specfem3d_globe"
specfem_dict = {
    "bin": "link",
    "DATA": {
        "Par_file": "file",
    },
    "DATABASES_MPI": "link",
    "OUTPUT_FILES": "dir"
}
invdir = '/home/lsawade/lwsspy/invdir_real'
datadir = os.path.join(invdir, "Data")
scriptdir = os.path.dirname(os.path.abspath(__file__))

# %% Get Model CMT
processdict = lpy.read_yaml_file(os.path.join(scriptdir, "process.yml"))


download_dict = dict(
    network=",".join(['CU', 'G', 'GE', 'IC', 'II', 'IU', 'MN']),
    channel="BH*",
    location="00",
)

conda_activation = "source /usr/licensed/anaconda3/2020.7/etc/profile.d/conda.sh && conda activate lwsspy"
compute_node_login = "lsawade@traverse.princeton.edu"
bash_escape = "source ~/.bash_profile"
parameter_check_list = ['depth_in_m', "time_shift", 'latitude', 'longitude',
                        "m_rr", "m_tt", "m_pp", "m_rt", "m_rp", "m_tp"]
nosimpars = ["time_shift", "half_duration"]
mt_params = ["m_rr", "m_tt", "m_pp", "m_rt", "m_rp", "m_tp"]

pardict = dict(
    # m_rr=dict(scale=None, pert=1e23),
    # m_tt=dict(scale=None, pert=1e23),
    # m_pp=dict(scale=None, pert=1e23),
    # m_rt=dict(scale=None, pert=1e23),
    # m_rp=dict(scale=None, pert=1e23),
    # m_tp=dict(scale=None, pert=1e23),
    # latitude=dict(scale=1.0, pert=None),
    # longitude=dict(scale=1.0, pert=None),
    time_shift=dict(scale=1.0, pert=None),
    depth_in_m=dict(scale=1000.0, pert=None)
)
# pardict = dict(
#     depth_in_m=dict(scale=1000.0, pert=None),
#     time_shift=dict(scale=1.0, pert=None)
# )


class GCMT3DInversion:

    # parameter_check_list: list = [
    #     'm_rr', 'm_tt', 'm_pp', 'm_rt', 'm_rp', 'm_tp',
    #     'latitude', 'longitude', 'depth_in_m', 'time_shift', 'hdur'
    # ]
    parameter_check_list: list = parameter_check_list

    nosimpars: list = nosimpars

    def __init__(
            self,
            cmtsolutionfile: str,
            databasedir: str,
            specfemdir: str,
            processdict: dict = processdict,
            pardict: dict = pardict,
            zero_trace: bool = False,
            duration: float = 10800.0,
            starttime_offset: float = -50.0,
            endtime_offset: float = 50.0,
            download_data: bool = True,
            node_login: Optional[str] = None,
            conda_activation: str = conda_activation,
            bash_escape: str = bash_escape,
            download_dict: dict = download_dict,
            damping: float = 0.001,
            weighting: bool = True,
            normalize: bool = True,
            overwrite: bool = False,
            launch_method: str = "srun -n6 --gpus-per-task=1",
            process_func: Callable = lpy.process_stream,
            window_func: Callable = lpy.window_on_stream,
            multiprocesses: int = 20,
            debug: bool = False):

        # CMTSource
        self.cmtsource = lpy.CMTSource.from_CMTSOLUTION_file(cmtsolutionfile)
        self.cmt_out = deepcopy(self.cmtsource)
        self.xml_event = read_events(cmtsolutionfile)[0]

        # File locations
        self.databasedir = os.path.abspath(databasedir)
        self.cmtdir = os.path.join(self.databasedir, self.cmtsource.eventname)
        self.cmt_in_db = os.path.join(self.cmtdir, self.cmtsource.eventname)
        self.overwrite: bool = overwrite
        self.download_data = download_data

        # Simulation stuff
        self.specfemdir = specfemdir
        self.specfem_dict = specfem_dict
        self.launch_method = launch_method.split()

        # Processing parameters
        self.processdict = processdict
        self.process_func = process_func
        self.window_func = window_func
        self.duration = duration
        self.duration_in_m = np.ceil(duration/60.0)
        self.simulation_duration = np.round(self.duration_in_m * 1.02)
        self.multiprocesses = multiprocesses
        self.sumfunc = lambda results: Stream(results)

        # Inversion dictionary
        self.pardict = pardict

        # Download parameters
        self.starttime_offset = starttime_offset
        self.endtime_offset = endtime_offset
        self.download_dict = download_dict

        # Compute Node does not have internet
        self.conda_activation = conda_activation
        self.node_login = node_login
        self.bash_escape = bash_escape

        # Inversion parameters:
        self.nsim = 1
        self.__get_number_of_forward_simulations__()
        self.not_windowed_yet = True
        self.zero_trace = zero_trace
        self.damping = damping
        self.normalize = normalize
        self.weighting = weighting
        self.weights_rtz = dict(R=1.0, T=1.0, Z=1.0)

        # Initialize data dictionaries
        self.data_dict: dict = dict()
        self.synt_dict: dict = dict()
        self.zero_window_removal_dict: dict = dict()

        # Other
        self.debug = debug

        # Basic Checks
        self.__basic_check__()

        # Fix process dict
        self.adapt_processdict()

        # Initialize
        self.init()

        # Set iteration number
        self.iteration = 0

    def __basic_check__(self):

        # Check Parameter dict for wrong parameters
        for _par in self.pardict.keys():
            if _par not in self.parameter_check_list:
                raise ValueError(
                    f"{_par} not supported at this point. \n"
                    f"Available parameters are {self.parameter_check_list}")

        # If one moment tensor parameter is given all must be given.
        if any([_par in self.pardict for _par in mt_params]):
            checklist = [_par for _par in mt_params if _par in self.pardict]
            print(checklist)
            if not all([_par in checklist for _par in mt_params]):
                raise ValueError("If one moment tensor parameter is to be "
                                 "inverted. All must be inverted.\n"
                                 "Update your pardict")
            else:
                self.moment_tensor_inv = True
        else:
            self.moment_tensor_inv = False

        # Check zero trace condition
        if self.zero_trace:
            if self.moment_tensor_inv is False:
                raise ValueError("Can only use Zero Trace condition "
                                 "if inverting for Moment Tensor.\n"
                                 "Update your pardict.")

    def adapt_processdict(self):

        # Get Process parameters
        PP = ProcessParams(
            self.cmtsource.moment_magnitude, self.cmtsource.depth_in_m)
        proc_params = PP.determine_all()

        # Adjust the process dictionary
        for _wave, _process_dict in proc_params.items():
            if _wave in self.processdict:
                # Adjust weight or drop wave altogether
                if _process_dict['weight'] == 0.0 \
                        or _process_dict['weight'] is None:
                    self.processdict.popitem(_wave)
                    continue

                else:
                    self.processdict[_wave]['weight'] = _process_dict["weight"]

                # Adjust pre_filt
                self.processdict[_wave]['process']['pre_filt'] = \
                    [1.0/x for x in _process_dict["filter"]]

                # Adjust windowing config
                for _windict in self.processdict[_wave]["window"]:
                    _windict["config"]["min_period"] = _process_dict["filter"][2]
                    _windict["config"]["max_period"] = _process_dict["filter"][1]

        # Remove unnecessary wavetypes
        popkeys = []
        for _wave in self.processdict.keys():
            if _wave not in proc_params:
                popkeys.append(_wave)
        for _key in popkeys:
            self.processdict.pop(_key, None)

        # Dump the processing file in the cmt directory
        lpy.write_yaml_file(
            self.processdict, os.path.join(self.cmtdir, "process.yml"))

    def init(self):

        # Initialize directory
        self.__initialize_dir__()
        self.__initialize_waveform_dictionaries__()

        # Get observed data and process data
        if self.download_data:
            with lpy.Timer():
                self.__download_data__()

        # Initialize model vector
        self.__init_model_and_scale__()

    def __initialize_dir__(self):

        # Subdirectories
        self.datadir = os.path.join(self.cmtdir, "data")
        self.waveformdir = os.path.join(self.datadir, "waveforms")
        self.stationdir = os.path.join(self.datadir, "stations")
        self.syntdir = os.path.join(self.cmtdir, "synt")
        self.windows = os.path.join(self.cmtdir, "windows")

        # Create subsynthetic directories
        self.synt_syntdir = os.path.join(self.syntdir, "cmt")
        self.synt_pardirs = dict()
        for _par in self.pardict.keys():
            self.synt_pardirs[_par] = os.path.join(self.syntdir, _par)

        # Create database directory if doesn't exist
        self.__create_dir__(self.databasedir)

        # Create entry directory
        self.__create_dir__(self.cmtdir, overwrite=self.overwrite)

        # Create CMT solution
        if os.path.exists(self.cmt_in_db) is False:
            self.cmtsource.write_CMTSOLUTION_file(self.cmt_in_db)
        else:
            check_cmt = lpy.CMTSource.from_CMTSOLUTION_file(self.cmt_in_db)
            if check_cmt != self.cmtsource:
                raise ValueError('Already have a CMTSOLUTION, '
                                 'but it is different from the input one.')

        # Create data directory
        self.__create_dir__(self.datadir)

        # Simulation directory are created as part of the prep simulations
        # routine

    def __init_model_and_scale__(self):

        # Update the scale parameter for the moment tensor inversion
        # depending on the original size of the moment tensor
        if self.moment_tensor_inv:
            for _par, _dict in self.pardict.items():
                if _par in mt_params:
                    _dict["scale"] = self.cmtsource.M0

        # Check whether Mrr, Mtt, Mpp are there for zero trace condition
        if self.zero_trace:

            self.zero_trace_array = np.array([1.0 if _par in ['m_rr', 'm_tt', 'm_pp'] else 0.0
                                              for _par in self.pardict.keys()])
            self.zero_trace_index_array = np.where(
                self.zero_trace_array == 1.)[0]
            self.zero_trace_array = np.append(self.zero_trace_array, 0.0)

        # Get the model vector given the parameters to invert for
        self.model = np.array(
            [getattr(self.cmtsource, _par) for _par in self.pardict.keys()])
        self.init_model = 1.0 * self.model
        self.pars = [_par for _par in self.pardict.keys()]

        # Create scaling vector
        self.scale = np.array([10**lpy.magnitude(getattr(self.cmtsource, _par))
                               if _par not in mt_params else _dict['scale']
                               for _par, _dict in self.pardict.items()])

        self.scaled_model = self.model/self.scale
        self.init_scaled_model = 1.0 * self.scaled_model

    def __initialize_waveform_dictionaries__(self):

        for _wtype in self.processdict.keys():
            self.data_dict[_wtype] = Stream()
            self.synt_dict[_wtype] = dict()
            self.synt_dict[_wtype]["synt"] = Stream()

            for _par in self.pardict.keys():
                self.synt_dict[_wtype][_par] = Stream()

    def process_data(self):
        lpy.print_section("Loading and processing the data")

        with lpy.Timer():
            self.__load_data__()
        with lpy.Timer():
            self.__process_data__()

    def process_synt(self):
        lpy.print_section("Loading and processing the modeled data")

        with lpy.Timer():
            self.__load_synt__()
        with lpy.Timer():
            self.__process_synt__()

    def get_windows(self):

        self.__prep_simulations__()
        self.__write_sources__()
        with lpy.Timer():
            self.__run_simulations__()
        self.process_all_synt()
        with lpy.Timer():
            self.__window__()
        with lpy.Timer():
            # self.__remove_zero_window_traces__()
            # self.__remove_zero_windows_on_synt__()
            self.__prep_simulations__()
        self.not_windowed_yet = False

    def __compute_weights__(self):

        # Weight dictionary
        self.weights = dict()
        self.weights["event"] = [
            self.cmtsource.latitude, self.cmtsource.longitude]

        waveweightdict = dict()
        for _i, (_wtype, _stream) in enumerate(self.data_dict.items()):

            # Dictionary to keep track of the sum in each wave type.
            waveweightdict[_wtype] = 0

            # Get wave type weight from process.yml
            self.weights[_wtype] = dict()
            waveweight = self.processdict[_wtype]["weight"]
            self.weights[_wtype]["weight"] = deepcopy(waveweight)

            # Create dict to access traces
            RTZ_traces = dict()
            for _component, _cweight in self.weights_rtz.items():

                # Copy compnent weight to dictionary
                self.weights[_wtype][_component] = dict()
                self.weights[_wtype][_component]["weight"] = deepcopy(_cweight)

                # Create reference
                RTZ_traces[_component] = []

                # Only add ttraces that have windows.
                for _tr in _stream:
                    if _tr.stats.component == _component \
                            and len(_tr.stats.windows) > 0:
                        RTZ_traces[_component].append(_tr)

                # Get locations
                latitudes = []
                longitudes = []
                for _tr in RTZ_traces[_component]:
                    latitudes.append(_tr.stats.latitude)
                    longitudes.append(_tr.stats.longitude)
                latitudes = np.array(latitudes)
                longitudes = np.array(longitudes)

                # Save locations into dict
                self.weights[_wtype][_component]["lat"] = deepcopy(latitudes)
                self.weights[_wtype][_component]["lon"] = deepcopy(longitudes)

                # Get azimuthal weights for the traces of each component
                if len(latitudes) != 0 and len(longitudes) != 0:
                    azi_weights = lpy.azi_weights(
                        self.cmtsource.latitude,
                        self.cmtsource.longitude,
                        latitudes, longitudes, nbins=12, p=0.5)

                    # Save azi weights into dict
                    self.weights[_wtype][_component]["azimuthal"] = deepcopy(
                        azi_weights)

                    # Get Geographical weights
                    gw = lpy.GeoWeights(latitudes, longitudes)
                    _, _, ref, _ = gw.get_condition()
                    geo_weights = gw.get_weights(ref)

                    # Save geo weights into dict
                    self.weights[_wtype][_component]["geographical"] = deepcopy(
                        geo_weights)

                    # Compute Combination weights.
                    weights = (azi_weights * geo_weights)
                    weights /= np.sum(weights)/len(weights)
                    self.weights[_wtype][_component]["combination"] = deepcopy(
                        weights)
                else:
                    self.weights[_wtype][_component]["azimuthal"] = []
                    self.weights[_wtype][_component]["geographical"] = []
                    self.weights[_wtype][_component]["combination"] = []

                # Add weights to traces
                for _tr, _weight in zip(RTZ_traces[_component], weights):
                    _tr.stats.weights = _cweight * _weight
                    waveweightdict[_wtype] += np.sum(_cweight * _weight)

        # Normalize by component and aximuthal weights
        for _i, (_wtype, _stream) in enumerate(self.data_dict.items()):
            # Create dict to access traces
            RTZ_traces = dict()

            for _component, _cweight in self.weights_rtz.items():
                RTZ_traces[_component] = []
                for _tr in _stream:
                    if _tr.stats.component == _component \
                            and "weights" in _tr.stats:
                        RTZ_traces[_component].append(_tr)

                self.weights[_wtype][_component]["final"] = []
                for _tr in RTZ_traces[_component]:
                    _tr.stats.weights /= waveweightdict[_wtype]

                    self.weights[_wtype][_component]["final"].append(
                        deepcopy(_tr.stats.weights))

        with open(os.path.join(self.cmtdir, "weights.pkl"), "wb") as f:
            cPickle.dump(deepcopy(self.weights), f)

    # def __remove_unrotatable__(self):
    #     """Removes the traces from the data_dict wavetype streams and inventory
    #     that are not rotatable for whatever reason.
    #     """

    #     lpy.print_action("Removing traces that couldn't be rotated ...")
    #     checklist = ["1", "2", "N", "E"]
    #     rotate_removal_list = []
    #     for _wtype, _stream in self.data_dict.items():
    #         for _tr in _stream:
    #             net = _tr.stats.network
    #             sta = _tr.stats.station
    #             loc = _tr.stats.location
    #             cha = _tr.stats.channel
    #             if cha[-1] in checklist:
    #                 rotate_removal_list.append((net, sta, loc, cha))

    #     # Create set.
    #     self.rotate_removal_list = set(rotate_removal_list)

    #     # Remove stations
    #     for _i, _wtype in enumerate(self.data_dict.keys()):
    #         for (net, sta, loc, cha) in self.rotate_removal_list:
    #             # Remove channels from inventory
    #             self.stations = self.stations.remove(
    #                 network=net, station=sta, location=loc, channel=cha)

    #             # Remove Traces from Streams
    #             st = self.data_dict[_wtype].select(
    #                 network=net, station=sta, location=loc, channel=cha)
    #             for tr in st:
    #                 self.data_dict[_wtype].remove(tr)

    # def __remove_zero_window_traces__(self):
    #     """Removes the traces from the data_dict wavetype streams, and
    #     creates list with stations for each trace to be used for removal
    #     prior to processing of the synthetics.

    #     Not removing stuff from inventory, because negligible
    #     """

    #     # Process each wavetype.
    #     self.zero_window_removal_dict = dict()
    #     lpy.print_action("Removing traces without windows...")
    #     for _wtype, _stream in self.data_dict.items():

    #         lpy.print_action(f"    for {_wtype}")
    #         self.zero_window_removal_dict[_wtype] = []
    #         for _tr in _stream:
    #             if len(_tr.stats.windows) == 0:
    #                 net = _tr.stats.network
    #                 sta = _tr.stats.station
    #                 loc = _tr.stats.location
    #                 cha = _tr.stats.channel
    #                 self.zero_window_removal_dict[_wtype].append(
    #                     (net, sta, loc, cha))

    #     # # Create list of all traces that do not have to be simulated anymore
    #     # for _i, _wtype in enumerate(self.data_dict.keys()):
    #     #     if _i == 0:
    #     #         channel_removal_set = set(zero_window_removal_dict[_wtype])
    #     #     else:
    #     #         channel_removal_set.intersection(
    #     #             set(zero_window_removal_dict[_wtype]))

    #     # Remove the set from the window removal dicts
    #     for _i, _wtype in enumerate(self.data_dict.keys()):
    #         for (net, sta, loc, cha) in self.zero_window_removal_dict[_wtype]:
    #             tr = self.data_dict[_wtype].select(
    #                 network=net, station=sta, location=loc, channel=cha)[0]
    #             self.data_dict[_wtype].remove(tr)

    #     # Remove the set from the window removal dicts
    #     # for _i, _wtype in enumerate(self.data_dict.keys()):
    #     #     for (net, sta, loc, cha) in channel_removal_set:
    #     #         tr = self.data_dict[_wtype].select(
    #     #             network=net, station=sta, location=loc, channel=cha)[0]
    #     #         self.data_dict[_wtype].remove(tr)

    # def __remove_zero_windows_on_synt__(self):

    #     # Remove the set from the window removal dicts
    #     for _i, (_wtype, _pardict) in enumerate(self.synt_dict.items()):
    #         for _par, _stream in _pardict.items():
    #             print(_par)
    #             for (net, sta, loc, cha) in self.zero_window_removal_dict[_wtype]:
    #                 print(net, sta, loc, cha)
    #                 tr = _stream.select(
    #                     network=net, station=sta, component=cha[-1])[0]
    #                 _stream.remove(tr)

    def process_all_synt(self):
        lpy.print_section("Loading and processing all modeled data")

        with lpy.Timer():
            self.__load_synt__()
            self.__load_synt_par__()
            # self.__remove_zero_windows_on_synt__()

        with lpy.Timer():
            self.__process_synt__()
            self.__process_synt_par__()

    def __get_number_of_forward_simulations__(self):

        # For normal forward synthetics
        self.nsim = 1

        # Add one for each parameters that requires a forward simulation
        for _par in self.pardict.keys():
            if _par not in self.nosimpars:
                self.nsim += 1

    def __download_data__(self):

        # Setup download times depending on input...
        # Maybe get from process dict?
        starttime = self.cmtsource.cmt_time + self.starttime_offset
        endtime = self.cmtsource.cmt_time + self.duration \
            + self.endtime_offset

        lpy.print_bar("Data Download")

        if self.node_login is None:
            lpy.download_waveforms_to_storage(
                self.datadir, starttime=starttime, endtime=endtime,
                **self.download_dict)

        else:
            from subprocess import Popen, PIPE
            download_cmd = (
                f"download-data "
                f"-d {self.datadir} "
                f"-s {starttime} "
                f"-e {endtime} "
                f"-N {self.download_dict['network']} "
                f"-C {self.download_dict['channel']} "
                f"-L {self.download_dict['location']}"
            )

            login_cmd = ["ssh", "-T", self.node_login]
            comcmd = f"""
            {self.conda_activation}
            {download_cmd}
            """

            lpy.print_action(
                f"Logging into {' '.join(login_cmd)} and downloading")
            print(f"Command: \n{comcmd}\n")

            with Popen(["ssh", "-T", self.node_login],
                       stdin=PIPE, stdout=PIPE, stderr=PIPE,
                       universal_newlines=True) as p:
                output, error = p.communicate(comcmd)

            if p.returncode != 0:
                print(output)
                print(error)
                print(p.returncode)
                raise ValueError("Download not successful.")

    def __load_data__(self):
        lpy.print_action("Loading the data")

        # Load Station data
        self.stations = lpy.read_inventory(
            os.path.join(self.stationdir, "*.xml"))

        # Load seismic data
        self.data = read(os.path.join(self.waveformdir, "*.mseed"))
        self.raw_data = self.data.copy()
        # Populate the data dictionary.
        for _wtype, _stream in self.data_dict.items():
            self.data_dict[_wtype] = self.data.copy()

    def __process_data__(self):

        # Process each wavetype.
        for _wtype, _stream in self.data_dict.items():
            lpy.print_action(f"Processing data for {_wtype}")

            # Call processing function and processing dictionary
            starttime = self.cmtsource.cmt_time \
                + self.processdict[_wtype]["process"]["relative_starttime"]
            endtime = self.cmtsource.cmt_time \
                + self.processdict[_wtype]["process"]["relative_endtime"]

            # Process dict
            processdict = deepcopy(self.processdict[_wtype]["process"])

            processdict.pop("relative_starttime")
            processdict.pop("relative_endtime")
            processdict["starttime"] = starttime
            processdict["endtime"] = endtime
            processdict["inventory"] = self.stations
            processdict.update(dict(
                remove_response_flag=True,
                event_latitude=self.cmtsource.latitude,
                event_longitude=self.cmtsource.longitude,
                geodata=True)
            )

            if self.multiprocesses < 1:
                self.data_dict[_wtype] = self.process_func(
                    _stream, **processdict)
            else:
                lpy.print_action(
                    f"Processing in parallel using {self.multiprocesses} cores")
                self.data_dict[_wtype] = lpy.multiprocess_stream(
                    _stream, processdict)

    def __load_synt__(self):

        # if self.specfemdir is not None:
        # Load forward data
        lpy.print_action("Loading forward synthetics")
        temp_synt = read(os.path.join(
            self.synt_syntdir, "OUTPUT_FILES", "*.sac"))

        for _wtype in self.processdict.keys():
            self.synt_dict[_wtype]["synt"] = temp_synt.copy()

    def __load_synt_par__(self):
        # Load frechet data
        lpy.print_action("Loading parameter synthetics")
        for _par, _pardirs in self.synt_pardirs.items():
            lpy.print_action(f"    {_par}")

            if _par in self.nosimpars:
                temp_synt = read(os.path.join(
                    self.synt_syntdir, "OUTPUT_FILES", "*.sac"))
            else:
                # Load foward/perturbed data
                temp_synt = read(os.path.join(
                    _pardirs, "OUTPUT_FILES", "*.sac"))

            # Populate the wavetype Streams.
            for _wtype, _ in self.data_dict.items():
                self.synt_dict[_wtype][_par] = temp_synt.copy()

        del temp_synt

    def __process_synt__(self, no_grad=False):

        if self.multiprocesses > 1:
            parallel = True
            p = mpp.Pool(processes=self.multiprocesses)
            lpy.print_action(
                f"Processing in parallel using {self.multiprocesses} cores")
        else:
            parallel = False

        for _wtype in self.processdict.keys():
            lpy.print_action(f"Processing synt for {_wtype}")

            # Call processing function and processing dictionary
            starttime = self.cmtsource.cmt_time \
                + self.processdict[_wtype]["process"]["relative_starttime"]
            endtime = self.cmtsource.cmt_time \
                + self.processdict[_wtype]["process"]["relative_endtime"]

            # Process dict
            processdict = deepcopy(self.processdict[_wtype]["process"])
            processdict.pop("relative_starttime")
            processdict.pop("relative_endtime")
            processdict["starttime"] = starttime
            processdict["endtime"] = endtime
            processdict["inventory"] = self.stations
            processdict.update(dict(
                remove_response_flag=False,
                event_latitude=self.cmtsource.latitude,
                event_longitude=self.cmtsource.longitude)
            )
            print(f"Stream {_wtype}/synt: ",
                  len(self.synt_dict[_wtype]["synt"]))

            if parallel:
                self.synt_dict[_wtype]["synt"] = lpy.multiprocess_stream(
                    self.synt_dict[_wtype]["synt"], processdict)
            else:
                self.synt_dict[_wtype]["synt"] = self.process_func(
                    self.synt_dict[_wtype]["synt"], self.stations,
                    **processdict)

        if parallel:
            p.close()

    def __process_synt_par__(self):

        if self.multiprocesses > 1:
            parallel = True
            p = mpp.Pool(processes=self.multiprocesses)
            lpy.print_action(
                f"Processing in parallel using {self.multiprocesses} cores")
        else:
            parallel = False

        for _wtype in self.processdict.keys():
            lpy.print_action(f"Processing synt for {_wtype}")

            # Call processing function and processing dictionary
            starttime = self.cmtsource.cmt_time \
                + self.processdict[_wtype]["process"]["relative_starttime"]
            endtime = self.cmtsource.cmt_time \
                + self.processdict[_wtype]["process"]["relative_endtime"]

            # Process dict
            processdict = deepcopy(self.processdict[_wtype]["process"])
            processdict.pop("relative_starttime")
            processdict.pop("relative_endtime")
            processdict["starttime"] = starttime
            processdict["endtime"] = endtime
            processdict.update(dict(
                remove_response_flag=False,
                event_latitude=self.cmtsource.latitude,
                event_longitude=self.cmtsource.longitude)
            )

            # Process each wavetype.
            for _par, _parsubdict in self.pardict.items():
                print(f"Stream {_wtype}/{_par}: ",
                      len(self.synt_dict[_wtype][_par]))
                if _par in self.nosimpars:
                    self.synt_dict[_wtype][_par] = self.synt_dict[_wtype]["synt"].copy(
                    )
                else:
                    if parallel:
                        self.synt_dict[_wtype][_par] = self.sumfunc(
                            lpy.starmap_with_kwargs(
                                p, self.process_func,
                                zip(self.synt_dict[_wtype]
                                    [_par], repeat(self.stations)),
                                repeat(processdict),
                                len(self.synt_dict[_wtype][_par]))).copy()
                    else:
                        self.synt_dict[_wtype][_par] = self.process_func(
                            self.synt_dict[_wtype][_par], self.stations,
                            **processdict)
                    # divide by perturbation value and scale by scale length
                if _parsubdict["pert"] is not None:
                    if _parsubdict["pert"] != 1.0:
                        lpy.stream_multiply(
                            self.synt_dict[_wtype][_par],
                            1.0/_parsubdict["pert"])

                # Compute frechet derivative with respect to time
                if _par == "time_shift":
                    self.synt_dict[_wtype][_par].differentiate(
                        method='gradient')
                    lpy.stream_multiply(self.synt_dict[_wtype][_par], -1.0)
                if _par == "depth_in_m":
                    lpy.stream_multiply(
                        self.synt_dict[_wtype][_par], 1.0/1000.0)
        if parallel:
            p.close()

    def __window__(self):

        for _wtype in self.processdict.keys():
            lpy.print_action(f"Windowing {_wtype}")

            for window_dict in self.processdict[_wtype]["window"]:

                # Wrap window dictionary
                wrapwindowdict = dict(
                    station=self.stations,
                    event=self.xml_event,
                    config_dict=window_dict,
                    _verbose=self.debug
                )

                # Serial or Multiprocessing
                if self.multiprocesses <= 1:
                    self.window_func(
                        self.data_dict[_wtype],
                        self.synt_dict[_wtype]["synt"],
                        **wrapwindowdict)
                else:

                    self.data_dict[_wtype] = lpy.multiwindow_stream(
                        self.data_dict[_wtype],
                        self.synt_dict[_wtype]["synt"],
                        wrapwindowdict, nprocs=self.multiprocesses)

            # After each trace has windows attached continue
            lpy.add_tapers(self.data_dict[_wtype], taper_type="tukey",
                           alpha=0.25, verbose=self.debug)

            # Some traces aren't even iterated over..
            for _tr in self.data_dict[_wtype]:
                if "windows" not in _tr.stats:
                    _tr.stats.windows = []

    def forward(self):
        pass

    def optimize(self, optim: lpy.Optimization):

        try:
            if self.zero_trace:
                model = np.append(deepcopy(self.scaled_model), 1.0)
            else:
                model = deepcopy(self.scaled_model)
            optim_out = optim.solve(optim, model)
            self.model = deepcopy(optim.model)
            return optim_out
        except Exception as e:
            print(e)
            return optim

    def __prep_simulations__(self):

        lpy.print_action("Prepping simulations")
        # Create forward directory
        if self.specfemdir is not None:
            lpy.createsimdir(self.specfemdir, self.synt_syntdir,
                             specfem_dict=self.specfem_dict)
        else:
            self.__create_dir__(self.syntdir)

        # Create one directory synthetics and each parameter
        for _par, _pardir in self.synt_pardirs.items():
            if _par not in self.nosimpars:
                if self.specfemdir is not None:
                    lpy.createsimdir(self.specfemdir, _pardir,
                                     specfem_dict=self.specfem_dict)
                else:
                    self.__create_dir__(_pardir)

        # Write stations file
        lpy.inv2STATIONS(
            self.stations, os.path.join(self.synt_syntdir, "DATA", "STATIONS"))

        # Update Par_file depending on the parameter.
        syn_parfile = os.path.join(self.synt_syntdir, "DATA", "Par_file")
        syn_pars = lpy.read_parfile(syn_parfile)
        syn_pars["USE_SOURCE_DERIVATIVE"] = False

        # Adapt duration
        syn_pars["RECORD_LENGTH_IN_MINUTES"] = self.simulation_duration

        # Write Stuff to Par_file
        lpy.write_parfile(syn_pars, syn_parfile)

        # Do the same for the parameters to invert for.
        for _par, _pardir in self.synt_pardirs.items():

            # Half duration an time-shift don't need extra simulations
            if _par not in self.nosimpars:

                # Write stations file
                lpy.inv2STATIONS(
                    self.stations, os.path.join(_pardir, "DATA", "STATIONS"))

                # Update Par_file depending on the parameter.
                dsyn_parfile = os.path.join(_pardir, "DATA", "Par_file")
                dsyn_pars = lpy.read_parfile(dsyn_parfile)

                # Set data parameters and  write new parfiles
                locations = ["latitude", "longitude", "depth_in_m"]
                if _par in locations:
                    dsyn_pars["USE_SOURCE_DERIVATIVE"] = True
                    if _par == "depth_in_m":
                        # 1 for depth
                        dsyn_pars["USE_SOURCE_DERIVATIVE_DIRECTION"] = 1
                    elif _par == "latitude":
                        # 2 for latitude
                        dsyn_pars["USE_SOURCE_DERIVATIVE_DIRECTION"] = 2
                    else:
                        # 3 for longitude
                        dsyn_pars["USE_SOURCE_DERIVATIVE_DIRECTION"] = 3
                else:
                    dsyn_pars["USE_SOURCE_DERIVATIVE"] = False

                # Adapt duration
                dsyn_pars["RECORD_LENGTH_IN_MINUTES"] = self.simulation_duration

                # Write Stuff to Par_file
                lpy.write_parfile(dsyn_pars, dsyn_parfile)

    def __update_cmt__(self, model):
        cmt = deepcopy(self.cmtsource)
        for _par, _modelval in zip(self.pars, model * self.scale):
            setattr(cmt, _par, _modelval)
        self.cmt_out = cmt

    def __write_sources__(self):

        # Update cmt solution with new model values
        cmt = deepcopy(self.cmtsource)
        for _par, _modelval in zip(self.pars, self.model):
            setattr(cmt, _par, _modelval)

        # Writing synthetic CMT solution
        lpy.print_action("Writing Synthetic CMTSOLUTION")
        cmt.write_CMTSOLUTION_file(os.path.join(
            self.synt_syntdir, "DATA", "CMTSOLUTION"))

        # For the perturbations it's slightly more complicated.
        for _par, _pardir in self.synt_pardirs.items():

            if _par not in ["time_shift", "half_duration"]:
                # Write source to the directory of simulation
                lpy.print_action(f"Writing Frechet CMTSOLUTION for {_par}")
                if self.pardict[_par]["pert"] is not None:
                    # Perturb source at parameter
                    cmt_pert = deepcopy(cmt)

                    # If parameter a part of the tensor elements then set the
                    # rest of the parameters to 0.
                    tensorlist = ['m_rr', 'm_tt', 'm_pp',
                                  'm_rt', 'm_rp', 'm_tp']
                    if _par in tensorlist:
                        for _tensor_el in tensorlist:
                            if _tensor_el != _par:
                                setattr(cmt_pert, _tensor_el, 0.0)
                            else:
                                setattr(cmt_pert, _tensor_el,
                                        self.pardict[_par]["pert"])
                    else:
                        # Get the parameter to be perturbed
                        to_be_perturbed = getattr(cmt_pert, _par)

                        # Perturb the parameter
                        to_be_perturbed += self.pardict[_par]["pert"]

                        # Set the perturb
                        setattr(cmt_pert, _par, to_be_perturbed)

                    cmt_pert.write_CMTSOLUTION_file(os.path.join(
                        _pardir, "DATA", "CMTSOLUTION"))
                else:
                    cmt.write_CMTSOLUTION_file(os.path.join(
                        _pardir, "DATA", "CMTSOLUTION"))

    def __run_simulations__(self):

        lpy.print_action("Submitting all simulations")
        # Initialize necessary commands
        cmd_list = self.nsim * [[*self.launch_method, './bin/xspecfem3D']]

        cwdlist = [self.synt_syntdir]
        cwdlist.extend(
            [_pardir for _par, _pardir in self.synt_pardirs.items()
             if _par not in self.nosimpars])
        lpy.run_cmds_parallel(cmd_list, cwdlist=cwdlist)

    def __run_forward_only__(self):

        # Initialize necessary commands
        lpy.print_action("Submitting forward simulation")
        cmd_list = [[*self.launch_method, './bin/xspecfem3D']]
        cwdlist = [self.synt_syntdir]
        lpy.run_cmds_parallel(cmd_list, cwdlist=cwdlist)

    def __run_parameters_only__(self):

        # Initialize necessary commands
        lpy.print_action("Submitting parameter simulations")
        cmd_list = (self.nsim - 1) * \
            [[*self.launch_method, './bin/xspecfem3D']]

        cwdlist = []
        cwdlist.extend(
            [_pardir for _par, _pardir in self.synt_pardirs.items()
             if _par not in self.nosimpars])
        lpy.run_cmds_parallel(cmd_list, cwdlist=cwdlist)

    def compute_cost_gradient(self, model):

        # Update model
        self.model = model * self.scale
        self.scaled_model = model

        # Write sources for next iteration
        self.__write_sources__()

        # Run the simulations
        with lpy.Timer():
            self.__run_simulations__()

        # Get streams
        self.process_all_synt()

        # Window Data
        if self.not_windowed_yet:
            self.__window__()
            self.not_windowed_yet = False

        return self.__compute_cost__(), self.__compute_gradient__() * self.scale

    def compute_cost_gradient_hessian(self, model):

        # Update model
        if self.zero_trace:
            mu = model[-1]
            self.model = model[:-1] * self.scale
            self.scaled_model = model[:-1]
        else:
            self.model = model * self.scale
            self.scaled_model = model

        # Write sources for next iteration
        self.__write_sources__()

        # Run the simulations
        if self.iteration == 0:
            # First simulation and processing was done during windowing stage.
            pass
            self.iteration = 1
        else:
            with lpy.Timer():
                self.__run_simulations__()

            # Get streams
            self.process_all_synt()

        # Window Data
        if self.not_windowed_yet:
            self.__window__()
            self.not_windowed_yet = False

        # Evaluate
        cost = self.__compute_cost__()
        g, h = self.__compute_gradient_and_hessian__()

        if self.debug:
            print("Raw")
            print("C:", cost)
            print("G:")
            print(g)
            print("H")
            print(h)

        # Scaling of the cost function
        g *= self.scale
        h = np.diag(self.scale) @ h @ np.diag(self.scale)

        if self.debug:
            print("Scaled")
            print("C:", cost)
            print("G:")
            print(g)
            print("H")
            print(h)

        if self.damping > 0.0:
            factor = self.damping * np.max(np.abs((np.diag(h))))
            print("f", factor)
            modelres = self.scaled_model - self.init_scaled_model
            print("modelres:", modelres)
            print("Costbef:", cost)
            # cost += factor/2 * np.sum(modelres**2)
            print("Costaft:", cost)
            g += factor * modelres
            h += factor * np.eye(len(self.model))

        if self.debug:
            print("Damped")
            print("C:", cost)
            print("G:")
            print(g)
            print("H")
            print(h)

        # Add zero trace condition
        if self.zero_trace:
            m, n = h.shape
            hz = np.zeros((m+1, n+1))
            hz[:-1, :-1] = h
            hz[:, -1] = self.zero_trace_array
            hz[-1, :] = self.zero_trace_array
            h = hz
            g = np.append(g, 0.0)
            g[-1] = np.sum(self.scaled_model[self.zero_trace_index_array])

        if self.debug:
            print("Zero_traced")
            print("C:", cost)
            print("G:")
            print(g)
            print("H")
            print(h)

        return cost, g, h

    def __compute_cost__(self):

        cost = 0
        for _wtype in self.processdict.keys():

            cgh = lpy.CostGradHess(
                data=self.data_dict[_wtype],
                synt=self.synt_dict[_wtype]["synt"],
                verbose=self.debug,
                normalize=self.normalize,
                weight=self.weighting)
            cost += cgh.cost() * self.processdict[_wtype]["weight"]
        return cost

    def __compute_residuals__(self):

        residuals = dict()
        for _wtype in self.processdict.keys():

            cgh = lpy.CostGradHess(
                data=self.data_dict[_wtype],
                synt=self.synt_dict[_wtype]["synt"],
                verbose=self.debug,
                normalize=self.normalize,
                weight=False)
            residuals[_wtype] = cgh.residuals()

        with open(os.path.join(self.cmtdir, "residuals.pkl"), "wb") as f:
            cPickle.dump(deepcopy(residuals), f)

        return residuals

    def __compute_gradient__(self):

        gradient = np.zeros_like(self.model)

        for _wtype in self.processdict.keys():
            # Get all perturbations
            dsyn = list()
            for _i, _par in enumerate(self.pardict.keys()):
                dsyn.append(self.synt_dict[_wtype][_par])

            # Create costgradhess class to computte gradient
            cgh = lpy.CostGradHess(
                data=self.data_dict[_wtype],
                synt=self.synt_dict[_wtype]["synt"],
                dsyn=dsyn,
                verbose=self.debug,
                normalize=self.normalize,
                weight=self.weighting)

            gradient += cgh.grad() * self.processdict[_wtype]["weight"]

        return gradient

    def __compute_gradient_and_hessian__(self):

        gradient = np.zeros_like(self.model)
        hessian = np.zeros((len(self.model), len(self.model)))
        print(self.model)
        print(gradient)
        print(hessian)
        for _wtype in self.processdict.keys():

            # Get all perturbations
            dsyn = list()
            for _i, _par in enumerate(self.pardict.keys()):
                dsyn.append(self.synt_dict[_wtype][_par])
            print(len(dsyn))
            # Create costgradhess class to computte gradient
            cgh = lpy.CostGradHess(
                data=self.data_dict[_wtype],
                synt=self.synt_dict[_wtype]["synt"],
                dsyn=dsyn,
                verbose=self.debug,
                normalize=self.normalize,
                weight=self.weighting)

            tmp_g, tmp_h = cgh.grad_and_hess()
            print(tmp_g, tmp_h)
            gradient += tmp_g * self.processdict[_wtype]["weight"]
            hessian += tmp_h * self.processdict[_wtype]["weight"]

        return gradient, hessian

    def misfit_walk_depth(self):

        # Start the walk
        lpy.print_bar("Misfit walk: Depth")

        scaled_depths = np.arange(self.cmtsource.depth_in_m - 10000,
                                  self.cmtsource.depth_in_m + 10100, 1000)/1000.0
        cost = np.zeros_like(scaled_depths)
        grad = np.zeros((*scaled_depths.shape, 1))
        hess = np.zeros((*scaled_depths.shape, 1, 1))
        dm = np.zeros((*scaled_depths.shape, 1))

        for _i, _dep in enumerate(scaled_depths):

            lpy.print_section(f"Computing CgH for: {_dep} km")
            with lpy.Timer():
                c, g, h = self.compute_cost_gradient_hessian(
                    np.array([_dep]))
                print(f"\n     Iteration for {_dep} km done.")
            cost[_i] = c
            grad[_i, :] = g
            hess[_i, :, :] = h

        # Get the Gauss newton step
        for _i in range(len(scaled_depths)):
            dm[_i, :] = np.linalg.solve(
                hess[_i, :, :], -grad[_i, :])

        plt.switch_backend("pdf")
        plt.figure(figsize=(12, 4))
        # Cost function
        ax = plt.subplot(141)
        plt.plot(cost, scaled_depths, label="Cost")
        plt.legend(frameon=False, loc='upper right')
        plt.xlabel("Cost")
        plt.ylabel("Depth [km]")

        ax = plt.subplot(142, sharey=ax)
        plt.plot(np.squeeze(grad), scaled_depths, label="Grad")
        plt.legend(frameon=False, loc='upper right')
        plt.xlabel("Gradient")
        ax.tick_params(labelleft=False, labelright=False)

        ax = plt.subplot(143, sharey=ax)
        plt.plot(np.squeeze(hess), scaled_depths, label="Hess")
        plt.legend(frameon=False, loc='upper right')
        plt.xlabel("G.-N. Hessian")
        ax.tick_params(labelleft=False, labelright=False)

        ax = plt.subplot(144, sharey=ax)
        plt.plot(np.squeeze(dm), scaled_depths, label="Step")
        plt.legend(frameon=False, loc='upper right')
        plt.xlabel("$\Delta$m [km]")
        ax.tick_params(labelleft=False, labelright=False)

        plt.savefig("misfit_walk_depth.pdf")

        # Start the walk
        lpy.print_bar("DONE.")

    def misfit_walk_depth_times(self):
        """Pardict containing an array of the walk parameters.
        Then we walk entirely around the parameter space."""

        # if len(pardict) > 2:
        #     raise ValueError("Only two parameters at a time.")

        # depths = np.arange(self.cmtsource.depth_in_m - 10000,
        #                    self.cmtsource.depth_in_m + 10100, 1000)
        # times = np.arange(-10.0, 10.1, 1.0)
        depths = np.arange(self.cmtsource.depth_in_m - 5000,
                           self.cmtsource.depth_in_m + 5100, 1000)
        times = np.arange(self.cmtsource.time_shift - 5.0,
                          self.cmtsource.time_shift + 5.1, 1.0)
        t, z = np.meshgrid(times, depths)
        cost = np.zeros(z.shape)
        grad = np.zeros((*z.shape, 2))
        hess = np.zeros((*z.shape, 2, 2))
        dm = np.zeros((*z.shape, 2))

        for _i, _dep in enumerate(depths):
            for _j, _time in enumerate(times):

                c, g, h = self.compute_cost_gradient_hessian(
                    np.array([_dep, _time]))
                cost[_i, _j] = c
                grad[_i, _j, :] = g
                hess[_i, _j, :, :] = h

        # Get the Gauss newton step
        damp = 0.001
        for _i in range(z.shape[0]):
            for _j in range(z.shape[1]):
                dm[_i, _j, :] = np.linalg.solve(
                    hess[_i, _j, :, :] + damp * np.diag(np.ones(2)), - grad[_i, _j, :])
        plt.switch_backend("pdf")
        extent = [np.min(t), np.max(t), np.min(z), np.max(z)]
        aspect = (np.max(t) - np.min(t))/(np.max(z) - np.min(z))
        plt.figure(figsize=(11, 6.5))

        # Get minimum
        ind = np.unravel_index(np.argmin(cost, axis=None), cost.shape)

        # Cost
        ax1 = plt.subplot(3, 4, 9)
        plt.imshow(cost, interpolation=None, extent=extent, aspect=aspect)
        lpy.plot_label(ax1, r"$\mathcal{C}$", dist=0)
        plt.plot(times[ind[0]], depths[ind[1]], "*")
        c1 = plt.colorbar()
        c1.ax.tick_params(labelsize=7)
        c1.ax.yaxis.offsetText.set_fontsize(7)
        ax1.axes.invert_yaxis()
        plt.ylabel(r'$z$')
        plt.xlabel(r'$t$')

        # Gradient
        ax2 = plt.subplot(3, 4, 6, sharey=ax1)
        plt.imshow(grad[:, :, 1], interpolation=None,
                   extent=extent, aspect=aspect)
        c2 = plt.colorbar()
        c2.ax.tick_params(labelsize=7)
        c2.ax.yaxis.offsetText.set_fontsize(7)
        ax2.tick_params(labelbottom=False)
        lpy.plot_label(ax2, r"$g_{\Delta t}$", dist=0)

        ax3 = plt.subplot(3, 4, 10, sharey=ax1)
        plt.imshow(grad[:, :, 0], interpolation=None,
                   extent=extent, aspect=aspect)
        c3 = plt.colorbar()
        c3.ax.tick_params(labelsize=7)
        c3.ax.yaxis.offsetText.set_fontsize(7)
        ax3.tick_params(labelleft=False)
        lpy.plot_label(ax3, r"$g_z$", dist=0)
        plt.xlabel(r'$\Delta t$')

        # Hessian
        ax4 = plt.subplot(3, 4, 3, sharey=ax1)
        plt.imshow(hess[:, :, 0, 1], interpolation=None,
                   extent=extent, aspect=aspect)
        c4 = plt.colorbar()
        c4.ax.tick_params(labelsize=7)
        c4.ax.yaxis.offsetText.set_fontsize(7)
        ax4.tick_params(labelbottom=False)
        lpy.plot_label(ax4, r"$\mathcal{H}_{z,\Delta t}$", dist=0)

        ax5 = plt.subplot(3, 4, 7, sharey=ax1)
        plt.imshow(hess[:, :, 1, 1], interpolation=None,
                   extent=extent, aspect=aspect)
        c5 = plt.colorbar()
        c5.ax.tick_params(labelsize=7)
        c5.ax.yaxis.offsetText.set_fontsize(7)
        ax5.tick_params(labelleft=False, labelbottom=False)
        lpy.plot_label(ax5, r"$\mathcal{H}_{\Delta t,\Delta t}$", dist=0)

        ax6 = plt.subplot(3, 4, 11, sharey=ax1)
        plt.imshow(hess[:, :, 0, 0], interpolation=None,
                   extent=extent, aspect=aspect)
        c6 = plt.colorbar()
        c6.ax.tick_params(labelsize=7)
        c6.ax.yaxis.offsetText.set_fontsize(7)
        ax6.tick_params(labelleft=False)
        lpy.plot_label(ax6, r"$\mathcal{H}_{z,z}$", dist=0)
        plt.xlabel(r'$\Delta t$')

        # Gradient/Hessian
        ax7 = plt.subplot(3, 4, 8, sharey=ax1)
        plt.imshow(dm[:, :, 1], interpolation=None,
                   extent=extent, aspect=aspect)
        c7 = plt.colorbar()
        c7.ax.tick_params(labelsize=7)
        c7.ax.yaxis.offsetText.set_fontsize(7)
        ax7.tick_params(labelleft=False, labelbottom=False)
        lpy.plot_label(ax7, r"$\mathrm{d}\Delta$", dist=0)

        ax8 = plt.subplot(3, 4, 12, sharey=ax1)
        plt.imshow(dm[:, :, 0], interpolation=None,
                   extent=extent, aspect=aspect)
        c8 = plt.colorbar()
        c8.ax.tick_params(labelsize=7)
        c8.ax.yaxis.offsetText.set_fontsize(7)
        ax8.tick_params(labelleft=False)
        lpy.plot_label(ax8, r"$\mathrm{d}z$", dist=0)
        plt.xlabel(r'$\Delta t$')

        plt.subplots_adjust(hspace=0.2, wspace=0.15)
        plt.savefig("SyntheticCostGradHess.pdf")

    def plot_data(self, outputdir="."):
        plt.switch_backend("pdf")
        for _wtype in self.processdict.keys():
            with PdfPages(os.path.join(outputdir, f"data_{_wtype}.pdf")) as pdf:
                for obsd_tr in self.data_dict[_wtype]:
                    fig = plot_seismograms(obsd_tr, cmtsource=self.cmtsource,
                                           tag=_wtype)
                    pdf.savefig()  # saves the current figure into a pdf page
                    plt.close(fig)

                    # We can also set the file's metadata via the PdfPages object:
                d = pdf.infodict()
                d['Title'] = f"{_wtype.capitalize()}-Wave-Data-PDF"
                d['Author'] = 'Lucas Sawade'
                d['Subject'] = 'Trace comparison in one pdf'
                d['Keywords'] = 'seismology, moment tensor inversion'
                d['CreationDate'] = datetime.datetime.today()
                d['ModDate'] = datetime.datetime.today()

    def plot_windows(self, outputdir="."):
        plt.switch_backend("pdf")
        for _wtype in self.processdict.keys():
            with PdfPages(os.path.join(outputdir, f"windows_{_wtype}.pdf")) as pdf:
                for obsd_tr in self.data_dict[_wtype]:
                    try:
                        synt_tr = self.synt_dict[_wtype]["synt"].select(
                            station=obsd_tr.stats.station,
                            network=obsd_tr.stats.network,
                            component=obsd_tr.stats.channel[-1])[0]
                    except Exception as err:
                        print("Couldn't find corresponding synt for obsd trace(%s):"
                              "%s" % (obsd_tr.id, err))
                        continue

                    fig = plot_seismograms(obsd_tr, synt_tr, self.cmtsource,
                                           tag=_wtype)
                    pdf.savefig()  # saves the current figure into a pdf page
                    plt.close(fig)

                    # We can also set the file's metadata via the PdfPages object:
                d = pdf.infodict()
                d['Title'] = f"{_wtype.capitalize()}-Wave-PDF"
                d['Author'] = 'Lucas Sawade'
                d['Subject'] = 'Trace comparison in one pdf'
                d['Keywords'] = 'seismology, moment tensor inversion'
                d['CreationDate'] = datetime.datetime.today()
                d['ModDate'] = datetime.datetime.today()

    def plot_station(self, network: str, station: str, outputdir="."):
        plt.switch_backend("pdf")
        # Get station data
        for _wtype in self.processdict.keys():
            try:
                obsd = self.data_dict[_wtype].select(
                    network=network, station=station)
                synt = self.synt_dict[_wtype]["synt"].select(
                    network=network, station=station)
            except Exception as e:
                print(f"Could load station {network}{station} -- {e}")
            # Plot PDF for each wtype
            with PdfPages(os.path.join(outputdir, f"{network}.{station}_{_wtype}.pdf")) as pdf:
                for component in ["Z", "R", "T"]:
                    try:
                        obsd_tr = obsd.select(
                            station=station, network=network,
                            component=component)[0]
                        synt_tr = synt.select(
                            station=station, network=network,
                            component=component)[0]
                    except Exception as err:
                        print(f"Couldn't find obs or syn for NET.STA.COMP:"
                              f" {network}.{station}.{component} -- {err}")
                        continue

                    fig = plot_seismograms(obsd_tr, synt_tr, self.cmtsource,
                                           tag=_wtype)
                    pdf.savefig()  # saves the current figure into a pdf page
                    plt.close(fig)

                    # We can also set the file's metadata via the PdfPages object:
                d = pdf.infodict()
                d['Title'] = f"{_wtype.capitalize()}-Wave-PDF"
                d['Author'] = 'Lucas Sawade'
                d['Subject'] = 'Trace comparison in one pdf'
                d['Keywords'] = 'seismology, moment tensor inversion'
                d['CreationDate'] = datetime.datetime.today()
                d['ModDate'] = datetime.datetime.today()

    def plot_station_der(self, network: str, station: str, outputdir="."):
        plt.switch_backend("pdf")
        # Get station data
        for _wtype in self.processdict.keys():
            # Plot PDF for each wtype
            with PdfPages(os.path.join(
                    outputdir,
                    f"{network}.{station}_{_wtype}_derivatives.pdf")) as pdf:
                for _par in self.synt_dict[_wtype].keys():
                    if _par != "synt":
                        try:
                            synt = self.synt_dict[_wtype][_par].select(
                                network=network, station=station)
                        except Exception as e:
                            print(f"Could load station "
                                  f"{network}{station} -- {e}")
                        for component in ["Z", "R", "T"]:
                            try:
                                synt_tr = synt.select(
                                    station=station, network=network,
                                    component=component)[0]
                            except Exception as err:
                                print(f"Couldn't find obs or syn "
                                      f"for NET.STA.COMP:"
                                      f" {network}.{station}.{component} "
                                      f"-- {err}")
                                continue

                            fig = plot_seismograms(
                                synt_tr, cmtsource=self.cmtsource,
                                tag=f"{_wtype.capitalize()}-{_par.capitalize()}")
                            pdf.savefig()  # saves the current figure into a pdf page
                            plt.close(fig)

                    # We can also set the file's metadata via the PdfPages object:
                d = pdf.infodict()
                d['Title'] = f"{_wtype.capitalize()}-Wave-PDF"
                d['Author'] = 'Lucas Sawade'
                d['Subject'] = 'Trace comparison in one pdf'
                d['Keywords'] = 'seismology, moment tensor inversion'
                d['CreationDate'] = datetime.datetime.today()
                d['ModDate'] = datetime.datetime.today()

    def plot_windows(self, outputdir="."):
        plt.switch_backend("pdf")
        for _wtype in self.processdict.keys():
            with PdfPages(os.path.join(outputdir, f"windows_{_wtype}.pdf")) as pdf:
                for obsd_tr in self.data_dict[_wtype]:
                    try:
                        synt_tr = self.synt_dict[_wtype]["synt"].select(
                            station=obsd_tr.stats.station,
                            network=obsd_tr.stats.network,
                            component=obsd_tr.stats.channel[-1])[0]
                    except Exception as err:
                        print("Couldn't find corresponding synt for obsd trace(%s):"
                              "%s" % (obsd_tr.id, err))
                        continue

                    fig = plot_seismograms(obsd_tr, synt_tr, self.cmtsource,
                                           tag=_wtype)
                    pdf.savefig()  # saves the current figure into a pdf page
                    plt.close(fig)

                    # We can also set the file's metadata via the PdfPages object:
                d = pdf.infodict()
                d['Title'] = f"{_wtype.capitalize()}-Wave-PDF"
                d['Author'] = 'Lucas Sawade'
                d['Subject'] = 'Trace comparison in one pdf'
                d['Keywords'] = 'seismology, moment tensor inversion'
                d['CreationDate'] = datetime.datetime.today()
                d['ModDate'] = datetime.datetime.today()

    @ staticmethod
    def __create_dir__(dir, overwrite=False):
        if os.path.exists(dir) is False:
            os.mkdir(dir)
        else:
            if overwrite:
                shutil.rmtree(dir)
                os.mkdir(dir)
            else:
                pass


def plot_seismograms(obsd: Trace, synt: Union[Trace, None] = None,
                     cmtsource: Union[lpy.CMTSource, None] = None,
                     tag: Union[str, None] = None):
    station = obsd.stats.station
    network = obsd.stats.network
    channel = obsd.stats.channel
    location = obsd.stats.location

    trace_id = f"{network}.{station}.{location}.{channel}"

    # Times and offsets computed individually, since the grid search applies
    # a timeshift which changes the times of the traces.
    if cmtsource is None:
        offset = 0
    else:
        offset = obsd.stats.starttime - cmtsource.cmt_time
        if isinstance(synt, Trace):
            offset_synt = synt.stats.starttime - cmtsource.cmt_time

    times = [offset + obsd.stats.delta * i for i in range(obsd.stats.npts)]
    if isinstance(synt, Trace):
        times_synt = [offset_synt + synt.stats.delta * i
                      for i in range(synt.stats.npts)]

    # Figure Setup
    fig = plt.figure(figsize=(15, 5))
    ax1 = plt.subplot(211)
    plt.subplots_adjust(left=0.03, right=0.97, top=0.95)

    ax1.plot(times, obsd.data, color="black", linewidth=0.75,
             label="Observed")
    if isinstance(synt, Trace):
        ax1.plot(times_synt, synt.data, color="red", linewidth=0.75,
                 label="Synthetic")
    ax1.set_xlim(times[0], times[-1])
    ax1.legend(loc='upper right', frameon=False, ncol=3, prop={'size': 11})
    ax1.tick_params(labelbottom=False, labeltop=False)

    # Setting top left corner text manually
    if isinstance(tag, str):
        label = f"{trace_id}\n{tag.capitalize()}"
    else:
        label = f"{trace_id}"
    lpy.plot_label(ax1, label, location=1, dist=0.005, box=False)

    # plot envelope
    ax2 = plt.subplot(212)
    ax2.plot(times, lpy.envelope(obsd.data), color="black",
             linewidth=1.0, label="Observed")
    if isinstance(synt, Trace):
        ax2.plot(times, lpy.envelope(synt.data), color="red", linewidth=1,
                 label="Synthetic")
    ax2.set_xlim(times[0], times[-1])
    ax2.set_xlabel("Time [s]", fontsize=13)
    lpy.plot_label(ax2, "Envelope", location=1, dist=0.005, box=False)
    if isinstance(synt, Trace):
        try:
            for win in obsd.stats.windows:
                left = times[win.left]
                right = times[win.right]
                re1 = Rectangle((left, ax1.get_ylim()[0]), right - left,
                                ax1.get_ylim()[1] - ax1.get_ylim()[0],
                                color="blue", alpha=0.25, zorder=-1)
                ax1.add_patch(re1)
                re2 = Rectangle((left, ax2.get_ylim()[0]), right - left,
                                ax2.get_ylim()[1] - ax2.get_ylim()[0],
                                color="blue", alpha=0.25, zorder=-1)
                ax2.add_patch(re2)
        except Exception as e:
            print(e)

    return fig


def bin():

    # Inputs
    event = "C201711191509A"
    database = "/gpfs/alpine/geo111/scratch/lsawade/testdatabase"
    specfemdir = "/gpfs/alpine/geo111/scratch/lsawade/SpecfemMagic/specfem3d_globe"
    launch_method = "jsrun -n 6 -a 4 -c 4 -g 1"

    gcmt3d = GCMT3DInversion(event, database, specfemdir, pardict=pardict,
                             download_data=True,
                             overwrite=False, launch_method=launch_method,
                             damping=0.001)
    gcmt3d.init()
    gcmt3d.process_data()
    gcmt3d.get_windows()
    gcmt3d.__compute_weights__()

    # print(50 * "-", "Cost, Grad, Hess", 50 * "_")

    # # gcmt3d.misfit_walk_depth()
    optim_list = []

    max_iter = 5
    max_nls = 4

    with lpy.Timer():

        # Gauss Newton Optimization Structure
        lpy.print_bar("GN")
        optim_gn = lpy.Optimization("gn")
        optim_gn.compute_cost_and_grad_and_hess = gcmt3d.compute_cost_gradient_hessian
        optim_gn.is_preco = False
        optim_gn.niter_max = max_iter
        optim_gn.nls_max = max_nls
        optim_gn.alpha = 1.0
        optim_gn.stopping_criterion = 9.0e-1

        # Run optimization
        with lpy.Timer():
            optim_out = gcmt3d.optimize(optim_gn)
            lpy.print_action("DONE with Gauss-Newton.")

        # Update model and write model
        gcmt3d.__update_cmt__(optim_out.model)
        gcmt3d.cmt_out.write_CMTSOLUTION_file(
            f"{gcmt3d.cmtdir}/{gcmt3d.cmt_out.eventname}_GN")

        optim_list.append(deepcopy(optim_out))

        # # BFGS
        # gcmt3d.__init_model_and_scale__()
        # lpy.print_bar("BFGS")
        # optim_bfgs = lpy.Optimization("bfgs")
        # optim_bfgs.compute_cost_and_gradient = gcmt3d.compute_cost_gradient
        # optim_bfgs.is_preco = False
        # optim_bfgs.niter_max = max_iter
        # optim_bfgs.nls_max = max_nls
        # optim_bfgs.stopping_criterion = 9.5e-1
        # optim_bfgs.n = len(gcmt3d.model)

        # # Run optimization
        # optim_out = gcmt3d.optimize(optim_bfgs)

        # # Update model and write model
        # gcmt3d.__update_cmt__(optim_out.model)
        # gcmt3d.cmt_out.write_CMTSOLUTION_file(
        #     f"{gcmt3d.cmtdir}/{gcmt3d.cmt_out.eventname}_BFGS")

        # optim_list. append(deepcopy(optim_out))

    #     # # Regularized Gauss Newton
    #     gcmt3d.damping = 0.1
    #     gcmt3d.__init_model_and_scale__()
    #     lpy.print_bar("Gauss-Newton Regularized")
    #     optim_gnr = lpy.Optimization("gn")
    #     optim_gnr.compute_cost_and_grad_and_hess = gcmt3d.compute_cost_gradient_hessian
    #     optim_gnr.is_preco = False
    #     optim_gnr.niter_max = max_iter
    #     optim_gnr.nls_max = max_nls
    #     optim_gnr.alpha = 1.0
    #     optim_gnr.stopping_criterion = 9.5e-1
    #     optim_gnr.n = len(gcmt3d.model)

    #     # Run optimization
    #     with lpy.Timer():
    #         optim_out = gcmt3d.optimize(optim_gnr)
    #         lpy.print_action("DONE with Regularized Gauss-Newton")

    #     # Update model and write model
    #     gcmt3d.__update_cmt__(optim_out.model)
    #     gcmt3d.cmt_out.write_CMTSOLUTION_file(
    #         f"{gcmt3d.cmtdir}/{gcmt3d.cmt_out.eventname}_GNR")

    #     optim_list. append(deepcopy(optim_out))

    # # Write PDF
    plt.switch_backend("pdf")
    lpy.plot_model_history(
        optim_list,
        list(pardict.keys()),  # "BFGS-R" "BFGS",
        outfile=f"{gcmt3d.cmtdir}/InversionHistory_2params.pdf")
    print(gcmt3d.scale)
    lpy.plot_optimization(
        optim_list,
        outfile=f"{gcmt3d.cmtdir}/misfit_reduction_history.pdf")

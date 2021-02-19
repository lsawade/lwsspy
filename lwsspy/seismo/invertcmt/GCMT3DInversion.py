"""
This is the first script to enable depth iterations for the
CMTSOLUTTION depth.
"""
# %% Create inversion directory

# Internal
import lwsspy as lpy

# External
from typing import Callable, Union
import os
import shutil
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from itertools import repeat
from obspy import read, read_events, Stream
import multiprocessing.pool as mpp

lpy.updaterc()


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


class GCMT3DInversion:

    def __init__(
            self,
            cmtsolutionfile: str,
            databasedir: str,
            specfemdir: str,
            processdict: dict = processdict,
            pardict: dict = dict(depth_in_m=dict(scale=1000.0, pert=None),
                                 time_shift=dict(scale=1.0, pert=None)),
            zero_trace: bool = False,
            duration: float = 7200.0,
            starttime_offset: float = -300.0,
            endtime_offset: float = 0.0,
            download_data: bool = False,
            node_login: Union[str, None] = None,
            conda_activation: str = conda_activation,
            bash_escape: str = bash_escape,
            download_dict: dict = download_dict,
            overwrite: bool = False,
            launch_method: str = "mpirun -n 6",
            process_func: Callable = lpy.process_stream,
            window_func: Callable = lpy.window_on_stream,
            multiprocesses: int = 0):

        # CMTSource
        self.cmtsource = lpy.CMTSource.from_CMTSOLUTION_file(cmtsolutionfile)
        self.xml_event = read_events(cmtsolutionfile)[0]

        # File locations
        self.databasedir = os.path.abspath(databasedir)
        self.cmtdir = os.path.join(self.databasedir, self.cmtsource.eventname)
        self.cmt_in_db = os.path.join(self.cmtdir, self.cmtsource.eventname)
        self.overwrite: bool = overwrite
        self.download_data = download_data
        self.specfemdir = specfemdir
        self.specfem_dict = specfem_dict

        # Processing parameters
        self.process_func = process_func
        self.processdict = processdict
        self.duration = duration
        self.multiprocesses = multiprocesses
        self.sumfunc = lambda results: Stream(results)

        # Inversion dictionary
        self.pardict = pardict
        # Parameter checking
        # self.parameter_check_list = [
        #     'm_rr', 'm_tt', 'm_pp', 'm_rt', 'm_rp', 'm_tp',
        #     'latitude', 'longitude', 'depth_in_m', 'cmt_time', 'hdur'
        # ]
        self.parameter_check_list = [
            'depth_in_m', 'time_shift'
        ]

        # Check Parameter dict for wrong parameters
        for _par in self.pardict.keys():
            if _par not in self.parameter_check_list:
                raise ValueError(
                    f"{_par} not supported at this point. \n"
                    f"Available parameters are {self.parameter_check_list}")

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

        # Initialize data dictionaries
        self.data_dict: dict = dict()
        self.synt_dict: dict = dict()

    def init(self):

        # Initialize directory
        self.__initialize_dir__()
        self.__initialize_waveform_dictionaries__()

        # Get observed data and process data
        if self.download_data:
            with lpy.Timer():
                self.__download_data__()

    def process_data(self):
        lpy.print_bar("PREPPING DATA")

        with lpy.Timer():
            self.__load_data__()
        with lpy.Timer():
            self.__process_data__()

    def __get_number_of_forward_simulations__(self):

        # For normal forward synthetics
        self.nsim = 1

        # Add one for each parameters that requires a forward simulation
        for _par in self.pardict.keys():
            if _par not in ["time_shift", "half_duration"]:
                self.nsim += 1

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

        # Create forward directory
        if self.specfemdir is not None:
            lpy.createsimdir(self.specfemdir, self.synt_syntdir,
                             specfem_dict=self.specfem_dict)
        else:
            self.__create_dir__(self.syntdir)

        # Create one directory synthetics and each parameter
        for _pardir in self.synt_pardirs.values():
            if self.specfemdir is not None:
                lpy.createsimdir(self.specfemdir, _pardir,
                                 specfem_dict=self.specfem_dict)
            else:
                self.__create_dir__(_pardir)

    def __init_model_and_scale__(self):

        # Get the model vector given the parameters to invert for
        self.model = np.array(
            [getattr(self.cmtsource, _par) for _par in self.pardict.keys()])

        self.pars = [_par for _par in self.pardict.keys()]

        # Create scaling vector
        self.scale = np.array([_dict["scale"]
                               for _, _dict in self.pardict.items()])

    def __initialize_waveform_dictionaries__(self):

        for _wtype in self.processdict.keys():
            self.data_dict[_wtype] = Stream()
            self.synt_dict[_wtype] = dict()

            self.synt_dict[_wtype]["synt"] = Stream()

            for _par in self.pardict.keys():
                self.synt_dict[_wtype][_par] = Stream()

    def __download_data__(self):

        # Setup download times depending on input...
        # Maybe get from process dict?
        starttime = self.cmtsource.origin_time + self.starttime_offset
        endtime = self.cmtsource.origin_time + self.duration \
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

        # Populate the data dictionary.
        for _wtype, _stream in self.data_dict.items():
            self.data_dict[_wtype] += deepcopy(self.data)

    def __process_data__(self):

        # Process each wavetype.
        for _wtype, _stream in self.data_dict.items():
            lpy.print_action(f"Processing data for {_wtype}")

            # Call processing function and processing dictionary
            starttime = self.cmtsource.origin_time \
                + self.processdict[_wtype]["process"]["relative_starttime"]
            endtime = self.cmtsource.origin_time \
                + self.processdict[_wtype]["process"]["relative_endtime"]

            # Process dict
            processdict = deepcopy(self.processdict[_wtype]["process"])
            processdict.pop("relative_starttime")
            processdict.pop("relative_endtime")
            processdict["starttime"] = starttime
            processdict["endtime"] = endtime
            processdict.update(dict(
                remove_response_flag=True,
                event_latitude=self.cmtsource.latitude,
                event_longitude=self.cmtsource.longitude)
            )

            if self.multiprocesses < 1:
                self.data_dict[_wtype] = self.process_func(
                    _stream, self.stations, **processdict)
            else:
                lpy.print_action(
                    f"Processing in parallel using {self.multiprocesses} cores")
                with mpp.Pool(processes=self.multiprocesses) as p:
                    self.data_dict[_wtype] = self.sumfunc(
                        lpy.starmap_with_kwargs(
                            p, self.process_func,
                            zip(_stream, repeat(self.stations)),
                            repeat(processdict), len(_stream))
                    )

    def __load_synt__(self):

        # Load forward data
        lpy.print_action("Processing synthetics")
        temp_synt = read(os.path.join(
            self.synt_syntdir, "OUTPUT_FILES", "*.sac"))

        for _wtype, _ in self.data_dict.items():
            self.synt_dict["synt"][_wtype] += deepcopy(temp_synt)

        # Populate the data dictionary.
        for _par, _pardirs in self.synt_pardirs.items():

            # Load foward/perturbed data
            lpy.print_action("Processing synthetics")
            temp_synt = read(os.path.join(
                _pardirs, "OUTPUT_FILES", "*.sac"))

            # Create empty wavetype dict for each parameter
            self.synt_dict[_par] = dict()

            # Populate the wavetype Streams.
            for _wtype, _ in self.data_dict.items():
                self.synt_dict[_par][_wtype] += deepcopy(temp_synt)

        del temp_synt

    def __process_synt__(self):

        for _wtype, _stream in self.synt_dict['synt'].items():
            lpy.print_action("Processing synt for {_wtype}")

            # Call processing function and processing dictionary
            starttime = self.cmtsource.origin_time \
                + self.processdict[_wtype]["process"]["relative_starttime"]
            endtime = self.cmtsource.origin_time \
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

            if self.multiprocesses < 1:
                self.synt_dict["synt"][_wtype] = self.process_func(
                    _stream, **processdict)
            else:
                lpy.print_action(
                    f"Processing in parallel using {self.multiprocesses} cores")
                with mpp.Pool(processes=self.multiprocesses) as p:
                    self.synt_dict["synt"][_wtype] = lpy.starmap_with_kwargs(
                        p, self.process_func,
                        zip(_stream, repeat(self.stations)),
                        repeat(processdict), len(_stream)
                    )

        # Process each wavetype.
        for _par, _parsubdict in self.pardict.items():
            for _wtype, _stream in self.data_dict.items():
                lpy.print_action("Processing {_par} for {_wtype}")

                # Call processing function and processing dictionary
                starttime = self.cmtsource.origin_time \
                    + self.processdict[_wtype]["process"]["relative_starttime"]
                endtime = self.cmtsource.origin_time \
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

                if self.multiprocesses < 1:
                    self.synt_dict[_par][_wtype] = self.process_func(
                        _stream, self.stations, **processdict)
                else:
                    with mpp.Pool(processes=self.multiprocesses) as p:
                        self.synt_dict[_par][_wtype] = lpy.starmap_with_kwargs(
                            p, self.process_func,
                            zip(_stream, repeat(self.stations)),
                            repeat(processdict), len(_stream))

                # divide by perturbation value and scale by scale length
                if _parsubdict["pert"] is not None:
                    if 1.0/_parsubdict["pert"] * _parsubdict["scale"] != 1.0:
                        lpy.stream_multiply(
                            self.synt_dict[_par][_wtype],
                            1.0/_parsubdict["pert"] * _parsubdict["scale"])
                else:
                    if _parsubdict["scale"] != 1.0:
                        lpy.stream_multiply(
                            self.synt_dict[_par][_wtype],
                            1.0/_parsubdict["pert"] * _parsubdict["scale"])

    def __window__(self):

        for _wtype in self.processdict.keys():
            lpy.print_action("Windowing {_wtype}")
            self.window_func(self.data_dict[_wtype],
                             self.synt_dict["synt"][_wtype],
                             self.processdict[_wtype]["window"],
                             station=self.stations, event=self.xml_event)
            lpy.add_tapers(self.data_dict[_wtype],
                           taper_type="tukey", alpha=0.25)

        self.not_windowed_yet = False

    def forward(self):
        pass

    def optimize(self, optim: lpy.Optimization):
        pass

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

                if self.pardict[_par]["pert"] is not None:
                    # Perturb source at parameter
                    cmt_pert = deepcopy(cmt)

                    # Get the parameter to be perturbed
                    to_be_perturbed = getattr(cmt_pert, _par)

                    # Perturb the parameter
                    to_be_perturbed += self.pardict[_par]["pert"]

                    # Set the perturb
                    setattr(cmt_pert, _par, to_be_perturbed)

                    # If parameter a part of the tensor elements then set the
                    # rest of the parameters to 0.
                    tensorlist = ['m_rr', 'm_tt', 'm_pp',
                                  'm_rt', 'm_rp', 'm_tp']
                    if _par in tensorlist:
                        for _tensor_el in tensorlist:
                            if _tensor_el != _par:
                                setattr(cmt_pert, _tensor_el, 0.0)

                # Write source to the directory of simulation
                lpy.print_action(f"Writing Frechet CMTSOLUTION for {_par}")
                cmt.write_CMTSOLUTION_file(os.path.join(
                    _pardir, "DATA", "CMTSOLUTION"))

    def __prep_simulations__(self):

        # Create  synthetics simulation
        lpy.createsimdir(self.specfemdir, self.synt_syntdir,
                         specfem_dict=self.specfem_dict)

        # Write stations file
        lpy.inv2STATIONS(
            self.stations, os.path.join(self.synt_syntdir, "DATA", "STATIONS"))

        # Update Par_file depending on the parameter.
        syn_parfile = os.path.join(self.synt_syntdir, "DATA", "Par_file")
        syn_pars = lpy.read_parfile(syn_parfile)
        syn_pars["USE_SOURCE_DERIVATIVE"] = False

        # Write Stuff to Par_file
        lpy.write_parfile(syn_pars, syn_parfile)

        # Do the same for the parameters to invert for.
        for _par, _pardir in self.synt_pardirs.items():

            # Half duration an time-shift don't need extra simulations
            if _par not in ["time_shift", "half_duration"]:

                # Create base simulation dir
                lpy.createsimdir(self.specfemdir, _pardir,
                                 specfem_dict=self.specfem_dict)

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

                # Write Stuff to Par_file
                lpy.write_parfile(dsyn_pars, dsyn_parfile)

    def __run_simulations__(self):

        # Initialize necessary commands
        cmd_list = [[self.launch_method, './bin/xspecfem3D']]
        cwdlist = [self.synt_syntdir]
        cwdlist.extend([_pardir for _pardir in self.pardirs.values()])

        lpy.print_action("Submitting simulations")
        lpy.run_cmds_parallel(cmd_list, cwdlist=cwdlist)

    def compute_cost_and_gradient(self, model):

        # Update model
        for _i, _scale, _new_model \
                in enumerate(zip(self.scale, model)):
            self.model[_i] = _new_model * _scale

        # Write sources for next iteration
        self.__write_sources__()

        # Run the simulations
        self.__run_simulations__()

        # Get streams
        self.__load_synt__()
        self.__process_synt__()

        # Window Data
        self.__window__()

        return self.__compute_cost__(), self.__compute_gradient__()

    def __compute_cost__(self):

        cost = 0
        for _wtype in self.processdict.keys():

            cost += lpy.stream_cost_win(self.data_dict,
                                        self.synt_dict["synt"][_wtype])

        return cost

    def __compute_gradient__(self):

        gradient = np.zeros_like(self.model)

        for _i, _par in enumerate(self.pardict.keys()):
            for _wtype in self.processdict.keys():

                gradient[_i] += lpy.stream_grad_frechet_win(
                    self.data_dict[_wtype], self.synt_dict["synt"][_wtype],
                    self.synt_dict[_par][_wtype])

        return gradient

    def misfit_walk(self, pardict: dict):
        """Pardict containing an array of the walk parameters.
        Then we walk entirely around the parameter space."""
        pass

    def plot_data(self):
        print("Not working right now")
        return
        for network in self.stations:
            for station in network:
                for _wtype in self.processdict.keys():
                    streamplot = self.data_dict[_wtype].select(
                        network=network.code,
                        station=station.code)
                    N = len(streamplot)
                    plt.figure(figsize=(2*N, 10))
                    streamplot.plot(block=False)
                    plt.suptitle(_wtype.capitalize())

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


# %% Create inversion directory and simulation directories
# if os.path.exists(invdir) is False:
#     os.mkdir(invdir)

# if os.path.exists(datadir) is False:
#     download_waveforms_cmt2storage(
#         CMTSOLUTION, datadir,
#         duration=1200, stationxml="invdir_real/station2_filtered.xml")

# station_xml = os.path.join(datadir, "stations", "*.xml")
# waveforms = os.path.join(datadir, "waveforms", "*.mseed")

# # %%
# # Creating forward simulation directories for the synthetic
# # and the derivate simulations
# syntsimdir = os.path.join(invdir, 'Synt')
# dsynsimdir = os.path.join(invdir, 'Dsyn')
# syntstations = os.path.join(syntsimdir, 'DATA', 'STATIONS')
# dsynstations = os.path.join(dsynsimdir, 'DATA', 'STATIONS')
# synt_parfile = os.path.join(syntsimdir, "DATA", "Par_file")
# dsyn_parfile = os.path.join(dsynsimdir, "DATA", "Par_file")
# synt_cmt = os.path.join(syntsimdir, "DATA", "CMTSOLUTION")
# dsyn_cmt = os.path.join(dsynsimdir, "DATA", "CMTSOLUTION")

# compute_synt = True
# if compute_synt:
#     createsimdir(SPECFEM, syntsimdir, specfem_dict=specfem_dict)
#     createsimdir(SPECFEM, dsynsimdir, specfem_dict=specfem_dict)

#     #  Fix Stations!
#     stationxml2STATIONS(station_xml, syntstations)
#     stationxml2STATIONS(station_xml, dsynstations)

#     # Get Par_file Dictionaries
#     synt_pars = read_parfile(synt_parfile)
#     dsyn_pars = read_parfile(dsyn_parfile)

#     # Set data parameters and  write new parfiles
#     synt_pars["USE_SOURCE_DERIVATIVE"] = False
#     dsyn_pars["USE_SOURCE_DERIVATIVE"] = True
#     dsyn_pars["USE_SOURCE_DERIVATIVE_DIRECTION"] = 1  # For depth
#     write_parfile(synt_pars, synt_parfile)
#     write_parfile(dsyn_pars, dsyn_parfile)

# print_bar("Starting the inversion")

# # Create dictionary for processing
# rawdata = read(waveforms)
# inv = read_inventory(station_xml)
# print_action("Processing the observed data")
# processparams = read_yaml_file(os.path.join(scriptdir, "process.body.yml"))
# data = process_wrapper(rawdata, cmt_init, processparams, inv=inv)

# # %% Loading and Processing the data


# def compute_stream_cost(data, synt):

#     x = 0.0

#     for tr in data:
#         network, station, location, channel, component = (
#             tr.stats.network, tr.stats.station, tr.stats.location,
#             tr.stats.channel, tr.stats.component)
#         # Get the trace sampling time
#         dt = tr.stats.delta
#         d = tr.data

#         try:
#             s = synt.select(network=network, station=station,
#                             component=component)[0].data

#             for win, tap in zip(tr.stats.windows, tr.stats.tapers):
#                 ws = s[win.left:win.right]
#                 wo = d[win.left:win.right]
#                 x += 0.5 * np.sum(tap * (ws - wo) ** 2) * dt

#         except Exception as e:
#             print(
#                 f"Error - gradient - "
#                 f"{network}.{station}.{location}.{channel}: {e}")

#     return x


# def compute_stream_grad_depth(data, synt, dsyn):

#     x = 0.0

#     for tr in data:
#         network, station, location, channel, component = (
#             tr.stats.network, tr.stats.station, tr.stats.location,
#             tr.stats.channel, tr.stats.component)

#         # Get the trace sampling time
#         dt = tr.stats.delta
#         d = tr.data

#         try:
#             s = synt.select(network=network, station=station,
#                             component=component)[0].data
#             dsdz = dsyn.select(network=network, station=station,
#                                component=component)[0].data

#             for win, tap in zip(tr.stats.windows, tr.stats.tapers):
#                 wsyn = s[win.left:win.right]
#                 wobs = d[win.left:win.right]
#                 wdsdz = dsdz[win.left:win.right]
#                 x += np.sum((wsyn - wobs) * wdsdz * tap) * dt

#         except Exception as e:
#             print(
#                 f"Error - gradient - "
#                 f"{network}.{station}.{location}.{channel}: {e}")

#     return x


# def compute_stream_grad_and_hess(data, synt, dsyn):

#     g = 0.0
#     h = 0.0
#     for tr in data:
#         network, station, location, channel, component = (
#             tr.stats.network, tr.stats.station, tr.stats.location,
#             tr.stats.channel, tr.stats.component)

#         # Get the trace sampling time
#         dt = tr.stats.delta
#         d = tr.data

#         try:
#             s = synt.select(network=network, station=station,
#                             component=component)[0].data
#             dsdz = dsyn.select(network=network, station=station,
#                                component=component)[0].data

#             for win, tap in zip(tr.stats.windows, tr.stats.tapers):
#                 wsyn = s[win.left:win.right]
#                 wobs = d[win.left:win.right]
#                 wdsdz = dsdz[win.left:win.right]
#                 g += np.sum((wsyn - wobs) * wdsdz * tap) * dt
#                 h += np.sum(wdsdz ** 2 * tap) * dt

#         except Exception as e:
#             print(
#                 f"Error - gradient/hess - "
#                 f"{network}.{station}.{location}.{channel}: {e}")

#     return g, h


# iterwindow = 0


# def compute_cost_and_gradient(model):
#     global iterwindow, data
#     # Populate the simulation directories
#     cmt = deepcopy(cmt_init)
#     cmt.depth_in_m = model[0] * 1000  # Gradient is in km!
#     cmt.write_CMTSOLUTION_file(synt_cmt)
#     cmt.write_CMTSOLUTION_file(dsyn_cmt)

#     # Run the simulations
#     cmd_list = 2 * [['mpiexec', '-n', '1', './bin/xspecfem3D']]
#     cwdlist = [syntsimdir, dsynsimdir]
#     print_action("Submitting simulations")
#     if compute_synt:
#         run_cmds_parallel(cmd_list, cwdlist=cwdlist)
#     print()

#     # Get streams
#     synt = read(os.path.join(syntsimdir, "OUTPUT_FILES", "*.sac"))
#     dsyn = read(os.path.join(dsynsimdir, "OUTPUT_FILES", "*.sac"))
#     print_action("Processing Synthetic")
#     # with nostdout():
#     synt = process_wrapper(synt, cmt_init, processparams,
#                            inv=inv, observed=False)

#     print_action("Processing Fréchet")
#     # with nostdout():
#     dsyn = process_wrapper(dsyn, cmt_init, processparams,
#                            inv=inv, observed=False)

#     # After the first forward modeling window the data
#     if iterwindow == 0:
#         print_action("Windowing")
#         window_config = read_yaml_file(
#             os.path.join(scriptdir, "window.body.yml"))
#         print("Data")
#         print(data)
#         print("Synt")
#         print(synt)
#         print("Inv")
#         print(inv)
#         window_on_stream(data, synt, window_config,
#                          station=inv, event=xml_event)
#         add_tapers(data, taper_type="tukey", alpha=0.25)
#         iterwindow += 1

#     cost = stream_cost_win(data, synt)
#     grad = np.array([stream_grad_frechet_win(data, synt, dsyn)])

#     return cost, grad


# def compute_cost_and_gradient_hessian(model):
#     global iterwindow, data
#     # Populate the simulation directories
#     cmt = deepcopy(cmt_init)
#     cmt.depth_in_m = model[0] * 1000.0  # Gradient is in km!
#     cmt.write_CMTSOLUTION_file(synt_cmt)
#     cmt.write_CMTSOLUTION_file(dsyn_cmt)

#     # Run the simulations
#     cmd_list = 2 * [['mpiexec', '-n', '1', './bin/xspecfem3D']]
#     cwdlist = [syntsimdir, dsynsimdir]
#     print_action("Submitting simulations")
#     if compute_synt:
#         run_cmds_parallel(cmd_list, cwdlist=cwdlist)
#     print()

#     # Get streams
#     synt = read(os.path.join(syntsimdir, "OUTPUT_FILES", "*.sac"))
#     dsyn = read(os.path.join(dsynsimdir, "OUTPUT_FILES", "*.sac"))
#     print_action("Processing Synthetic")
#     # with nostdout():
#     synt = process_wrapper(synt, cmt_init, processparams,
#                            inv=inv, observed=False)

#     print_action("Processing Fréchet")
#     # with nostdout():
#     dsyn = process_wrapper(dsyn, cmt_init, processparams,
#                            inv=inv, observed=False)

#     # After the first forward modeling window the data
#     if iterwindow == 0:
#         print_action("Windowing")
#         window_config = read_yaml_file(
#             os.path.join(scriptdir, "window.body.yml"))
#         window_on_stream(data, synt, window_config,
#                          station=inv, event=xml_event)
#         add_tapers(data, taper_type="tukey", alpha=0.25)
#         iterwindow += 1

#     cost = stream_cost_win(data, synt)
#     grad, hess = stream_grad_and_hess_win(data, synt, dsyn)

#     return cost, np.array([grad]), np.array([[hess]])


# # %% Computing the depth range
# depths = np.arange(cmt_init.depth_in_m - 10000,
#                    cmt_init.depth_in_m + 11000, 1000) / 1000
# cost = []
# grad = []
# hess = []
# for _dep in depths:
#     print_action(f"Computing depth: {_dep}")
#     c, g, h = compute_cost_and_gradient_hessian(np.array([_dep]))
#     cost.append(c)
#     grad.append(g[0])
#     hess.append(h[0, 0])

# plt.figure(figsize=(11, 4))
# ax1 = plt.subplot(1, 4, 1)
# plt.plot(cost, depths)
# plt.ylabel('z')
# plt.xlabel('Cost')
# plt.gca().invert_yaxis()
# ax2 = plt.subplot(1, 4, 2, sharey=ax1)
# plt.plot(grad, depths)
# plt.xlabel('Gradient')
# ax2.tick_params(labelleft=False)
# ax3 = plt.subplot(1, 4, 3, sharey=ax1)
# plt.plot(hess, depths)
# plt.xlabel('Hessian')
# ax3.tick_params(labelleft=False)
# ax4 = plt.subplot(1, 4, 4, sharey=ax1)
# plt.plot(np.array(grad)/np.array(hess), depths)
# plt.xlabel('Gradient/Hessian')
# ax4.tick_params(labelleft=False)

# plt.subplots_adjust(hspace=0.125, wspace=0.125)
# plt.savefig("DataCostGradHess.pdf")


# ax = plot_station_xml(station_xml)
# ax.add_collection(
#     beach(cmt_init.tensor, xy=(cmt_init.longitude, cmt_init.latitude),
#           width=5, size=100, linewidth=1.0))
# plt.savefig("depth_inversion_map.pdf")

# # Define initial model that is 10km off
# model = np.array([cmt_init.depth_in_m]) / 1000.0

# print_section("BFGS")
# # Prepare optim steepest
# optim = Optimization("bfgs")
# optim.compute_cost_and_gradient = compute_cost_and_gradient
# optim.is_preco = False
# optim.niter_max = 7
# optim.nls_max = 3
# optim.stopping_criterion = 1e-8
# optim.n = len(model)
# optim_bfgs = optim.solve(optim, model)

# plot_optimization(
#     optim_bfgs, outfile="depth_inversion_misfit_reduction.pdf")
# plot_model_history(optim_bfgs, labellist=['depth'],
#                    outfile="depth_inversion_model_history.pdf")

# print_section("GN")
# # Prepare optim steepest
# optim = Optimization("gn")
# optim.compute_cost_and_grad_and_hess = compute_cost_and_gradient_hessian
# optim.is_preco = False
# optim.niter_max = 7
# optim.damping = 0.0
# optim.stopping_criterion = 1e-8
# optim.n = len(model)
# optim_gn = optim.solve(optim, model)

# plot_optimization(
#     [optim_bfgs, optim_gn], outfile="depth_inversion_misfit_reduction_comp.pdf")
# plot_model_history([optim_bfgs, optim_gn], labellist=['depth'],
#                    outfile="depth_inversion_model_history_comp.pdf")

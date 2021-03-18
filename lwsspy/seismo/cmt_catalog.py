from __future__ import annotations
from typing import List, Union, Iterable
from .source import CMTSource
from . import sourcedecomposition
from obspy import Catalog
from obspy.core.event import Event
from glob import glob
import numpy as np
import inspect
import _pickle as cPickle


class CMTCatalog:

    cmts: List[CMTSource]

    # Attributes and properties of a CMTSource
    attributes: list = [a for a in vars(CMTSource()).keys()]
    attributes += [
        a for a, _ in inspect.getmembers(
            CMTSource, lambda x: not inspect.isroutine(x)) if "__" not in a]
    specialpropertylist: list = ['tbp', 'tbp_norm']

    # Just for printing
    uniqueatt = set(attributes + specialpropertylist)

    # Get possible decomposition types
    dtypes = [func for func, _ in inspect.getmembers(
        sourcedecomposition, inspect.isfunction)]

    def __init__(self, cmts: Union[List[CMTSource], None] = None):
        """Creates instance of CMTCatalog

        Parameters
        ----------
        cmts : Union[List[CMTSource], None], optional
            List of CMTSource's, by default None
        """

        if cmts is not None:
            self.cmts = cmts
        else:
            self.cmts = []

    def save(self, filename: str):
        """Saves the catalog as Pickle in bytes"""
        with open(filename, "wb") as output_file:
            cPickle.dump(self, output_file)

    @classmethod
    def load(cls, filename: str):
        """Loads the ``catalog.pkl`` in bytes"""
        with open(filename, "rb") as input_file:
            cls = cPickle.load(input_file)
        return cls

    @ classmethod
    def from_obspy_catalog(cls, cat: Catalog):
        """Converts obspy catalog to CMTCatalog"""

        cmts = [CMTSource.from_event(ev) for ev in cat]

        return cls(cmts)

    @ classmethod
    def from_file_list(self, file: Union[str, list]):
        """Takes in filestring, globstring list of files, or list of glob 
        strings. Then converts each file to a cmtsolution, from which a
        catalog class will be generated."""

        if isinstance(file, str):
            filelist = [file]
        elif isinstance(file, list):
            if isinstance(file[0], str):
                filelist = file
            else:
                raise ValueError(
                    "List type not supported. Must be list of strings.")

        cmtfilelist = []
        for _file in filelist:
            cmtfilelist.extend(glob(_file))

        cmtlist = []
        for _cmtfile in cmtfilelist:
            cmtlist.append(CMTSource.from_CMTSOLUTION_file(_cmtfile))

        return self(cmtlist)

    def getvals(self, vtype="tensor", dtype: Union[str, None] = None):
        """This function is a bit more elaborate, but it gives access to each
        of the CMT parameters in form of ndarrays.

        Parameters
        ----------
        vtype : str
            String of attribute of CMT solutions, default ``tensor``
            if ``decomp`` (decomposition) is chosen one needs to specify the 
            type of decomposition ``dtype``
        dtype : str
            Decomposition type. decompositions are defined in module
            ``sourcedecomposition``, default None
            Needs to be specified if ``vtype`` is chosen to be ``decomp``.

        Return
        ------
        arraylike
            parameter array depending on the chosen parameter
        """

        # If very normal attribute type
        if vtype in self.attributes and vtype not in self.specialpropertylist:
            vals = []
            for _cmt in self.cmts:
                vals.append(getattr(_cmt, vtype))
            return np.array(vals)

        # If very special attribute typem but still attribute
        elif vtype in self.attributes and vtype in self.specialpropertylist:
            lb = []
            ev = []
            for _cmt in self.cmts:
                lb, ev = getattr(_cmt, vtype)
            return np.array(lb), np.array(ev)

        elif vtype == "decomp":
            if dtype is not None and dtype in self.dtypes:

                vals = []
                for _cmt in self.cmts:
                    vals.append(getattr(_cmt, vtype)(dtype))
                return np.array(vals)
            else:
                raise ValueError(
                    f"Method for decomposition must be given and"
                    f"in {self.dtypes}")
        else:
            raise ValueError(
                f"Value {vtype} not implemented, choose from"
                f"{self.uniqueatt}")

    def add(self, cmt: Union[List[Union[CMTSource, Event]],
                             CMTSource, Event, Catalog, CMTCatalog]):
        """Adds an event a

        Parameters
        ----------
        cmt : Union[List[Union[CMTSource, Event]], CMTSource, Event, Catalog]
            [description]
        """

        if isinstance(cmt, CMTSource):
            self.cmts.append(cmt)
        elif isinstance(cmt, Event):
            self.cmts.append(CMTSource.from_event(cmt))
        elif isinstance(cmt, list) or isinstance(cmt, Catalog) \
                or isinstance(cmt, CMTCatalog):
            for _cmt in cmt:
                if isinstance(_cmt, CMTSource):
                    self.cmts.append(_cmt)
                elif isinstance(_cmt, Event):
                    self.cmts.append(CMTSource.from_event(_cmt))
        else:
            ValueError(
                f"Type {type(cmt)} is not supported to be added to the catalog.")

    def get_event(self, eventname: str):

        # If eventid is a string
        for _i, _cmt in enumerate(self.cmts):
            if _cmt.eventname == eventname:
                return _cmt

        raise ValueError(f"No event in catalog for {eventname}")

    def pop(self, eventname: Union[List[str], str, List[int], int]):

        # If eventid is a string
        if isinstance(eventname, str):
            popindices = [_i for _i, _cmt in enumerate(self.cmts)
                          if _cmt.eventname == eventname]

        # If eventid is a list
        elif isinstance(eventname, list):
            popindices = []

            for _ev in eventname:
                if isinstance(_ev, str):
                    popindices.append(
                        [_i for _i, _cmt in enumerate(self.cmts)
                            if _cmt.eventname == _ev][0])
                elif isinstance(_ev, int):
                    popindices.append(_ev)
                else:
                    raise ValueError(
                        f"Type {eventname[0]} for event "
                        f"popping is not supported.")
        else:
            raise ValueError(
                f"Type {eventid} for popping is not supported.")

        # Pop indeces in reverse order to not mess up the list.
        for _popindex in reversed(sorted(popindices)):
            self.cmts.pop(_popindex)

    def __len__(self):
        """Returns the lengths of the catalog"""
        return len(self.cmts)

    def __iter__(self):
        """ Returns the Iterator object. """
        return iter(self.cmts)

    def __getitem__(self, index: Union[int, Iterable[int], slice]):
        """Returns index from cmt list"""
        if isinstance(index, int):
            return self.cmts[index]
        elif isinstance(index, slice):
            return CMTCatalog(self.cmts[index])
        elif isinstance(index, Iterable):
            retlist = []
            for _i in index:
                retlist.append(self.cmts[int(_i)])
            return CMTCatalog(retlist)
        else:
            raise ValueError("Index type not supported.")

    def sort(self, key="origin_time"):
        """Sorts the loaded CMT solutions after key that is given.

        Parameters
        ----------
        key : str, optional
            Key could be any attribute of a cm solution, 
            by default "origin_time"

        Raises
        ------
        ValueError
            If key is not supported.
        """
        if key in self.attributes:
            vals = self.getvals(key)
            indeces = vals.argsort().astype(int)
            self.cmts = self[indeces].cmts
        else:
            raise ValueError(
                f"{key} is not a valid sorting value.\n"
                f"Use {self.attributes}.")

    def check_ids(self, other: CMTCatalog, verbose: bool = False):
        """Takes in another catalog and returns a tuple of self and other
        that are contain only common eventnames and that are sorted.

        Parameters
        ----------
        other : CMTCatalog
            Another catalog
        verbose : bool
            Print events that were only found in one catalog.
        """

        cmtself = []
        cmtother = []
        for _cmt in self.cmts:
            try:
                _cmtother = other.get_event(_cmt.eventname)
                cmtself.append(_cmt)
                cmtother.append(_cmtother)
            except ValueError as e:
                if verbose:
                    print(
                        f"Didn't find corresponding events "
                        f"for {_cmt.eventname}")

        return CMTCatalog(cmtself), CMTCatalog(cmtother)

from typing import List, Union
from .source import CMTSource
from obspy import Catalog, Event


class CMTCatalog:

    cmts: List[CMTSource]

    def __init__(self, cmts: Union[List[CMTSource], None] = None):

        if cmts is not None:
            self.cmts = cmts

    @classmethod
    def from_obspy_catalog(cls, cat: Catalog):

        cmts = [CMTSource.from_event(ev) for ev in cat]

        return cls(cmts)

    def add(self, cmt: Union[List[Union[CMTSource, Event]],
                             CMTSource, Event, Catalog]):

        if isinstance(cmt, CMTSource):
            self.cmts.append(cmt)
        elif isinstance(cmt, Event):
            self.cmts.append(CMTSource.from_event(cmt))
        elif isinstance(cmt, list) or isinstance(cmt, Catalog):
            for _cmt in cmt:
                if isinstance(_cmt, CMTSource):
                    self.cmts.append(_cmt)
                elif isinstance(_cmt, Event):
                    self.cmts.append(CMTSource.from_event(_cmt))
        else:
            ValueError(
                f"Type {type(cmt)} is not supported to be added to the catalog.")

    def pop(self, eventid: Union[List[str], str, List[int], int]):

        if isinstance(eventid, str):
            popindices = [_i for _i, _cmt in enumerate(self.cmts)
                          if _cmt.eventname == 'C201711191509A']
        elif isinstance(eventid, list):
            popindices = []

            for _ev in eventid:
                if isinstance(_ev, str):
                    popindices.append(
                        [_i for _i, _cmt in enumerate(self.cmts)
                            if _cmt.eventname == _ev][0])
                elif isinstance(_ev, int):
                    popindices.append(_ev)
            else:
                raise ValueError(
                    f"Type {eventid[0]} for event popping is not supported.")
        else:
            raise ValueError(
                f"Type {eventid} for popping is not supported.")

        # Pop indeces in reverse order to not mess up the list.
        for _popindex in reversed(sorted(popindices)):
            self.cmts.pop(_popindex)

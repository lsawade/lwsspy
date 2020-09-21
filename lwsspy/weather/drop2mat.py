from typing import Union
import os.path as p
from .w

def drop2mat(filname: str, outfilename: Union[str or None] = None
             default_run: bool = False):

    if default_run:
        filename = p.join(p.dirname(p.abspath(__file__)), 'data', )


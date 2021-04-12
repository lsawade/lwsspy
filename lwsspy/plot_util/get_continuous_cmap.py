import matplotlib.colors as mcolors
import numpy as np
from .rgb2dec import rgb_to_dec
from .hex2rgb import hex_to_rgb

cdict = {'red':   [(0.0,  0.0, 0.0),
                   (0.5,  1.0, 1.0),
                   (1.0,  1.0, 1.0)],

         'green': [(0.0,  0.0, 0.0),
                   (0.25, 0.0, 0.0),
                   (0.75, 1.0, 1.0),
                   (1.0,  1.0, 1.0)],

         'blue':  [(0.0,  0.0, 0.0),
                   (0.5,  0.0, 0.0),
                   (1.0,  1.0, 1.0)]}


def get_continuous_cmap(clist, float_list=None, ctype: str = 'dec'):
    '''Creates and returns a color map that can be used in heat map figures.
    If float_list is not provided, colour map graduates linearly between each 
    color in clist. If float_list is provided, each color in clist is 
    mapped to the respective location in float_list. 

    Parameters
    ----------
    clist:
        list of colors
    float_list: 
        list of floats between 0 and 1, same length as clist. 
        Must start with 0 and end with 1.
    type: str
        'rgb', 'hex', 'dec', Default 'dec'

    Returns
    ----------
    colour map

    '''
    # Convert hex to decimal rgb and 255rgb to decimal rgb
    if ctype == 'hex':
        rgb_list = [rgb_to_dec(hex_to_rgb(_hexcolor)) for _hexcolor in clist]
    elif ctype == 'rgb':
        rgb_list = [rgb_to_dec(_color) for _color in clist]
    else:
        rgb_list = clist

    # Get distances
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0, 1, len(rgb_list)))

    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]]
                    for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = mcolors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp

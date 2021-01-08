
import numpy as np


def reduce_fontsize(fontsize):

    if type(fontsize) == int or type(fontsize) == float:
        newfontsize = int(np.round(0.7*fontsize))
    else:

        fontsizelist = ['xx-small', 'x-small', 'small',
                        'medium', 'large', 'x-large', 'xx-large']

        for _i, _fs in enumerate(fontsizelist):
            if fontsize == _fs:
                if _i == 0:
                    print("Warning, fontsize not changed xx-small is the "
                          "smallest font size")
                    newfontsize = fontsize
                else:
                    newfontsize = fontsizelist[_i - 1]

    return newfontsize



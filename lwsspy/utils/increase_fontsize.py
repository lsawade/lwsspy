import numpy as np


def increase_fontsize(fontsize):

    if type(fontsize) == int or type(fontsize) == float:
        newfontsize = int(np.round(1.3*fontsize))
    else:

        fontsizelist = ['xx-small', 'x-small', 'small',
                        'medium', 'large', 'x-large', 'xx-large']

        for _i, _fs in enumerate(fontsizelist):
            if fontsize == _fs:
                if _i == len(fontsizelist) - 1:
                    print("Warning, fontsize not changed xx-large is the "
                          "largest font size")
                    newfontsize = fontsize
                else:
                    newfontsize = fontsizelist[_i + 1]

    return newfontsize

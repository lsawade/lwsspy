import matplotlib
import matplotlib.pyplot as plt


def right_align_legend(legend: matplotlib.legend.Legend):
    """Does as the title suggests. Takes in a legend and right aligns the text.
    Puts markers to the right that is. and right aligns the text.

    Parameters
    ----------
    legend : matplotlib.legend.Legend
        A legend to be fixed.
    """

    vpc = legend._legend_box._children[-1]._children[:]
    for vp in vpc:
        for c in vp._children:
            c._children.reverse()
        vp.align = "right"

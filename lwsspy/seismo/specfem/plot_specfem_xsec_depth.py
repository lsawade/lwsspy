import matplotlib.pyplot as plt
import cartopy
import numpy as np
import lwsspy as lpy


def plot_specfem_xsec_depth(infile, outfile, label):
    """Takes in a Specfem depthslice and creates a map from it.

    Parameters
    ----------
    infile : str
        CSV file that contains the depth slice
    outfile : str
        name to save the figure to. Exports PDFs only and export a name 
        depending on depth of the slice.
    label : str
        label of the colorbar

    :Author:
        Lucas Sawade (lsawade@princeton.edu)

    :Last Modified:
        2020.12.23 16.30

    """

    # Define title:
    if label == 'rho':
        cbartitle = r"$\rho$"
    elif label == 'vpv':
        cbartitle = r"$v_{P_v}$"
    elif label == 'vsv':
        cbartitle = r"$v_{S_v}$"
    else:
        raise ValueError(f"Label {label} is not implemented.")

    # Load data from file
    llon, llat, rad, val, _, _, _ = lpy.read_specfem_xsec_depth(infile)

    # Get Depth of slice
    depth = lpy.EARTH_RADIUS_KM - np.mean(rad)

    # Create Figure
    lpy.updaterc()
    plt.figure(figsize=(9, 4))
    ax = plt.axes(projection=cartopy.crs.PlateCarree())
    ax.set_rasterization_zorder(-10)
    lpy.plot_map(fill=False, zorder=1)

    pmesh = plt.pcolormesh(llon, llat, val,
                           transform=cartopy.crs.PlateCarree(), zorder=-15)
    lpy.plot_label(ax, f"{depth:.1f} km", aspect=2.0,
                   location=1, dist=0.025, box=True)

    c = plt.colorbar(pmesh, fraction=0.05, pad=0.075)
    c.set_label(cbartitle, rotation=0, labelpad=10)
    # plt.show()
    plt.savefig(f"{outfile}_{int(depth):d}km.pdf", dpi=300)


def bin():

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', dest='infile',
                        help='CSV file containing the slice info',
                        required=True, type=str)
    parser.add_argument('-o', dest='outfile',
                        help='Output filename to save the pdf plot to',
                        required=True, type=str)
    parser.add_argument('-l', dest='label',
                        help='label: rho, vpv, ...',
                        required=True, type=str)
    args = parser.parse_args()

    # Run
    plot_specfem_xsec_depth(args.infile, args.outfile, args.label)

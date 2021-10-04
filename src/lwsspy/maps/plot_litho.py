import matplotlib
import matplotlib.pyplot as plt
from cartopy.crs import PlateCarree
import numpy as np
import lwsspy as lpy


def plot_litho(which="all", parameter="depth", print_keys: bool = False,
               cmap: str = 'Spectral'):
    # These are the parameters in the model Litho
    parameters = {
        'depth': r"z [km]",
        'density': r"kg/m^3",
        'vp': r"v_P [km/s]",
        'vs': r"v_S [km/s]",
        'qkappa': r"Q_{\kappa}",
        'qmu': r"Q_{\mu}",
        'vp2': r"v_{P_2} [km/s]",
        'vs2': r"v_{S_2} [km/s]",
        'eta': r"\eta"
    }

    # CHeck whether correct parameters are given
    if parameter not in parameters.keys():
        raise KeyError(f"{parameter} not in {parameters}")

    # Available boundaries for plotting
    boundaries = {
        'all': 'All',
        'asthenospheric_mantle': "Asthenosphere",
        'lid':  "Lid",
        'lower_crust': "Lower Crust",
        'middle_crust': "Middle Crust",
        'upper_crust': "Upper Crust",
        'lower_sediments': "Lower Sediments",
        'middle_sediments': "Middle Sediments",
        'upper_sediments': "Upper Sediments",
        'ice': "Ice",
        'water': "Water",
    }

    # Give keys if asked for them
    if print_keys:
        print(f"Possible keys: {boundaries.keys()}")
        return

    # Check whether requested parameter is in the crustal model
    if which not in boundaries.keys():
        raise KeyError(f"{which} not in {boundaries.keys()}")

    # Create modifiers
    _parameter = "_" + parameter
    mods = {"_top": "Top", "_bottom": "Bottom"}

    # Read the Litho1.0 file
    litho = lpy.maps.read_litho()

    # Get extent of the dataset
    minlat, maxlat = np.min(litho.latitude), np.max(litho.latitude)
    minlon, maxlon = np.min(litho.longitude), np.max(litho.longitude)
    extent = [minlon, maxlon, minlat, maxlat]

    # Figures size
    size = 2

    if which == 'all':
        # Remove 'all' key from key list!
        boundaries.pop('all')

        # Number of keys in dict
        Nkey = len(boundaries.keys())

        # Prep fig and axes
        fig, axs = plt.subplots(Nkey, 2, figsize=[4*1.1*size, 1.0*Nkey*size],
                                subplot_kw={'projection': PlateCarree()})
        plt.subplots_adjust(wspace=0.1, left=0.1,
                            right=1.05, bottom=0.05, top=0.95)

        # Define list of keys
        bkeys = list(boundaries.keys())

        # Loop over keys
        for _i, _key in enumerate(bkeys[::-1]):

            # Get the right color norm for each row
            vmin, vmax = [], []
            for _j, _v in enumerate(mods.keys()):
                # Escape for missing data
                if (_key == 'asthenospheric_mantle') and (_v == '_bottom'):
                    axs[_i, _j].axis('off')
                    continue
                m = getattr(litho, _key + _v + _parameter).values
                vmin.append(np.nanmin(m))
                vmax.append(np.nanmax(m))
            vmin, vmax = np.nanmin(vmin), np.nanmax(vmax)
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

            # Populate the subplots
            for _j, _v in enumerate(mods.keys()):

                # Escape for non exisiting bottom in athenosphere
                if (_key == 'asthenospheric_mantle') and (_v == '_bottom'):
                    axs[_i, _j].axis('off')
                    continue

                # Plot the map
                plt.sca(axs[_i, _j])
                # Plot coastlines
                lpy.maps.plot_map(fill=False)
                # Plot surface
                axs[_i, _j].imshow(
                    getattr(litho, _key + _v + _parameter)[::-1, :],
                    extent=extent, cmap=cmap, norm=norm)
                # Make rasterizeable
                axs[_i, _j].set_rasterization_zorder(-10)  # Important line!
                lpy.plot.plot_label(
                    axs[_i, _j],
                    f"{boundaries[_key]} {mods[_v]} {parameter.capitalize()}",
                    aspect=2.0, location=1, dist=-0.05)
                # Remove ticklabels
                lpy.plot.remove_ticklabels(axs[_i, _j])

            # Add colorbar for the row
            cbar = lpy.plot.nice_colorbar(
                matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm), ax=axs[_i, :])
            if parameter == 'depth':
                cbar.ax.invert_yaxis()
            cbar.set_label(parameters[parameter])

    else:
        # Separate function for the asthenospheric mantle since it's got no
        # bottom
        if which == 'asthenospheric_mantle':
            # Create subplots
            fig, axs = plt.subplots(
                1, 1, figsize=[2*1.3*size, size],
                subplot_kw={'projection': PlateCarree()})
            plt.sca(axs)
            axs.set_rasterization_zorder(-10)  # Important line!
            lpy.maps.plot_map(fill=False)
            im = axs.imshow(
                getattr(litho,  which + "_top" + _parameter)[::-1, :],
                extent=extent, zorder=-20)
            lpy.plot.remove_ticklabels(axs)
            lpy.plot.plot_label(axs,
                                f"{boundaries[which]} {mods['_top']} {parameter}",
                                aspect=2.0, location=1, dist=-0.05)
            cbar = lpy.plot.nice_colorbar(im)
            if parameter == 'depth':
                cbar.ax.invert_yaxis()
            cbar.set_label("z [km]")

        else:
            # Setting up subplots
            fig, axs = plt.subplots(
                1, 2, figsize=[4*1.0*size, size],
                subplot_kw={'projection': PlateCarree()})
            plt.subplots_adjust(wspace=0.05, left=0.1,
                                right=1.05, bottom=0.15, top=0.85)

            # Getting the right color norm
            vmin, vmax = [], []
            for _j, _v in enumerate(mods.keys()):
                m = getattr(litho, which + _v + _parameter).values
                vmin.append(np.min(m))
                vmax.append(np.max(m))
            norm = matplotlib.colors.Normalize(vmin=np.min(vmin),
                                               vmax=np.max(vmax))

            # Populating the subplots
            for _j, _v in enumerate(mods.keys()):
                plt.sca(axs[_j])
                lpy.maps.plot_map(fill=False)
                axs[_j].imshow(
                    getattr(litho, which + _v + _parameter)[::-1, :],
                    extent=extent, cmap=cmap, norm=norm)
                axs[_j].set_rasterization_zorder(-10)  # Important line!
                lpy.plot.plot_label(
                    axs[_j],
                    f"{boundaries[which]} {mods[_v]} {parameter.capitalize()}",
                    aspect=2.0, location=1, dist=-0.05)

                lpy.plot.remove_ticklabels(axs[_j])

            # Add colorbar for the row
            cbar = lpy.plot.nice_colorbar(
                matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm), ax=axs)
            if parameter == 'depth':
                cbar.ax.invert_yaxis()
            cbar.set_label("z [km]")

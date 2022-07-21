import imp

from matplotlib.cm import ScalarMappable
from . import ot
import matplotlib.pyplot as plt
import numpy as np
from .. import plot as lplt


def plot_OTWaveform(otw: ot.Waveform):
    _, axes = plt.subplots(2, 2, figsize=(8, 5))
    plt.subplots_adjust(wspace=0.1, hspace=0.3)

    extension = 0.05
    utmin, utmax = np.min(otw.ut), np.max(otw.ut)
    usmin, usmax = np.min(otw.us), np.max(otw.us)
    umin, umax = np.minimum(utmin, usmin), np.maximum(utmax, usmax)
    du = umax - umin
    ulims = umin - 0.5 * extension * du, umax + 0.5 * extension * du

    # Plot observed
    axes[0, 0].plot([np.min(otw.tt), np.max(otw.tt)], [0, 0], 'k:', lw=0.75)
    axes[0, 0].plot(otw.tt, otw.ut)
    axes[0, 0].set_xlim(np.min(otw.tt), np.max(otw.tt))
    axes[0, 0].set_ylim(ulims)
    axes[0, 0].set_xlabel(f'$t$ [s]')
    axes[0, 0].set_ylabel(f'Amplitude $u$')
    axes[0, 0].set_title('Target: Observed')

    # Plot predicted
    axes[0, 1].plot([np.min(otw.ts), np.max(otw.ts)], [0, 0], 'k:', lw=0.75)
    axes[0, 1].plot(otw.ts, otw.us)
    axes[0, 1].set_xlim(np.min(otw.ts), np.max(otw.ts))
    axes[0, 1].set_ylim(ulims)
    axes[0, 1].set_xlabel(f'$t$ [s]')
    axes[0, 1].set_title('Source: Synthetic')

    baseline = (0 - otw.unorm[0])/(otw.unorm[1]-otw.unorm[0])

    # Plot observed
    axes[1, 0].plot([np.min(otw.ttn), np.max(otw.ttn)],
                    2*[baseline], 'k:', lw=0.75)
    axes[1, 0].plot(otw.ttn, otw.utn)
    axes[1, 0].set_xlim(np.min(otw.ttn), np.max(otw.ttn))
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_xlabel(f'$t\prime$')
    axes[1, 0].set_ylabel(f'Amplitude $u\prime$')

    # Plot predicted
    axes[1, 1].plot([np.min(otw.tsn), np.max(otw.tsn)],
                    2*[baseline], 'k:', lw=0.75)
    axes[1, 1].plot(otw.tsn, otw.usn)
    axes[1, 1].set_xlim(np.min(otw.tsn), np.max(otw.tsn))
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_xlabel(f'$t\prime$')

    # Remove ticks on RHS plot
    axes[0, 1].tick_params(labelleft=False)
    axes[1, 1].tick_params(labelleft=False)


def plot_OTW_FP(otw: ot.Waveform):

    if otw.fingerprint_computed is False:
        otw.fingerprints()

    # Create figure
    _, axes = plt.subplots(1, 2, figsize=(8, 4))
    # subplots_adjust(wspace=0.4)

    # #### BOTTOM ROW scaled and fingerprinted
    # Observed
    # axes[0].plot([np.min(tn0), np.max(tn0)], [baseline, baseline], 'k:', lw=0.75)
    axes[0].plot(otw.ttn, otw.utn, 'k')
    axes[0].set_xlim(np.min(otw.ttn), np.max(otw.ttn))
    axes[0].set_title('Observed/Target')

    # Predicted
    # axes[1].plot([np.min(tn1), np.max(tn1)], [baseline, baseline], 'k:', lw=0.75)
    axes[1].plot(otw.tsn, otw.usn, 'k')
    axes[1].set_xlim(np.min(otw.tsn), np.max(otw.tsn))
    axes[1].set_title('Synthetic/Source')

    # Set limits
    axes[0].set_ylim(0, 1)
    axes[1].set_ylim(0, 1)

    # Setting ticklabels
    axes[1].tick_params(labelleft=False, labelright=True)
    axes[0].tick_params(labelleft=False)
    axes[1].tick_params(labelleft=False)

    # Setting axes labels
    axes[0].set_xlabel('Time $t\prime$ [s]')
    axes[1].set_xlabel('Time $t\prime$ [s]')
    axes[0].set_ylabel('Amplitude $u\prime$')

    # Plot contours
    axes[0].contour(otw.dtt, otw.dut, otw.dt.T, colors='k',
                    linewidths=0.25, levels=30)
    axes[1].contour(otw.dts, otw.dus, otw.ds.T,
                    colors='k', linewidths=0.25, levels=30)


def plot_OTW_2D_PDF(otw: ot.Waveform):

    if otw.fingerprint_computed is False:
        otw.fingerprints()

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    # subplots_adjust(wspace=0.4)

    # #### BOTTOM ROW scaled and fingerprinted
    # Observed
    # axes[0].plot([np.min(tn0), np.max(tn0)], [baseline, baseline], 'k:', lw=0.75)
    axes[0].plot(otw.ttn, otw.utn, 'k')
    axes[0].set_xlim(np.min(otw.ttn), np.max(otw.ttn))
    axes[0].set_title('Observed/Target')

    # Predicted
    # axes[1].plot([np.min(tn1), np.max(tn1)], [baseline, baseline], 'k:', lw=0.75)
    axes[1].plot(otw.tsn, otw.usn, 'k')
    axes[1].set_xlim(np.min(otw.tsn), np.max(otw.tsn))
    axes[1].set_title('Synthetic/Source')

    # Set limits
    axes[0].set_ylim(0, 1)
    axes[1].set_ylim(0, 1)

    # Setting ticklabels
    axes[1].tick_params(labelleft=False, labelright=True)
    axes[0].tick_params(labelleft=False)
    axes[1].tick_params(labelleft=False)

    # Setting axes labels
    axes[0].set_xlabel('Time $t\prime$ [s]')
    axes[1].set_xlabel('Time $t\prime$ [s]')
    axes[0].set_ylabel('Amplitude $u\prime$')

    # Plot contours
    pc = axes[0].pcolormesh(
        otw.dtt, otw.dut, otw.ptn.T, cmap='rainbow', vmin=0, vmax=np.max(otw.ptn))
    axes[1].pcolormesh(
        otw.dts, otw.dus, otw.psn.T, cmap='rainbow', vmin=0, vmax=np.max(otw.ptn))
    fig.colorbar(
        ScalarMappable(norm=pc.norm, cmap=pc.cmap), ax=axes,
        orientation='horizontal', fraction=0.05, shrink=0.6, aspect=40, pad=0.2)


def plot_OTW_FP_MARG(otw: ot.Waveform):

    # Create figure
    _, axes = plt.subplots(2, 2, figsize=(8, 6))
    plt.subplots_adjust(hspace=0.3, bottom=0.15,
                        left=0.15, right=0.85, top=0.95)

    # #### TOP ROW unscaled #####
    axes[0, 0].plot(otw.tt, otw.ut, 'k')
    axes[0, 0].set_xlim(np.min(otw.tt), np.max(otw.tt))
    axes[0, 0].set_title('Observed/Target')

    # Predicted
    # axes[1].plot([np.min(tn1), np.max(tn1)], [baseline, baseline], 'k:', lw=0.75)
    axes[0, 1].plot(otw.ts, otw.us, 'r')
    axes[0, 1].set_xlim(np.min(otw.ts), np.max(otw.ts))
    axes[0, 1].set_title('Synthetic/Source')

    # Set limits
    axes[0, 0].set_ylim(otw.unorm)
    axes[0, 1].set_ylim(otw.unorm)

    # #### BOTTOM ROW scaled and fingerprinted
    # Observed
    axes[1, 0].plot(otw.ttn, otw.utn, 'k')
    axes[1, 0].set_xlim(np.min(otw.ttn), np.max(otw.ttn))

    # Predicted
    # axes[1].plot([np.min(tn1), np.max(tn1)], [baseline, baseline], 'k:', lw=0.75)
    axes[1, 1].plot(otw.tsn, otw.usn, 'r')
    axes[1, 1].set_xlim(np.min(otw.tsn), np.max(otw.tsn))

    # Set limits
    axes[1, 0].set_ylim(0, 1)
    axes[1, 1].set_ylim(0, 1)

    # Setting ticklabels
    axes[0, 1].tick_params(labelleft=False, labelright=True)
    axes[1, 0].tick_params(labelleft=False, labelbottom=False)
    axes[1, 1].tick_params(labelleft=False, labelbottom=False)

    # Setting axes labels
    axes[0, 1].set_xlabel('Time $t$ [s]')
    axes[0, 0].set_xlabel('Time $t$ [s]')
    axes[0, 0].set_ylabel('Amplitude $u$')

    # Plot contours
    axes[1, 0].contour(otw.dtt, otw.dut, otw.dt.T, colors='k',
                       linewidths=0.25, levels=30)
    axes[1, 1].contour(otw.dts, otw.dus, otw.ds.T,
                       colors='k', linewidths=0.25, levels=30)

    # Create Axes for the marginal distributions
    iP = 99090
    targetxax = lplt.axes_from_axes(
        axes[1, 0], iP+1, extent=[0, -0.2, 1.0, 0.15])
    targetyax = lplt.axes_from_axes(
        axes[1, 0], iP+2, extent=[-0.2, 0.0, 0.15, 1.0])
    sourcexax = lplt.axes_from_axes(
        axes[1, 1], iP+3, extent=[0, -0.2, 1.0, 0.15])
    sourceyax = lplt.axes_from_axes(
        axes[1, 1], iP+4, extent=[1.05, 0.0, 0.15, 1.0])

    # Set limits
    targetyax.set_ylim(0, 1)
    targetyax.set_xlim(0, np.max(otw.MargU.gn)*1.5)
    sourceyax.set_ylim(0, 1)
    sourceyax.set_xlim(0, np.max(otw.MargU.fn)*1.5)
    targetxax.set_xlim(np.min(otw.ttn), np.max(otw.ttn))
    sourcexax.set_xlim(np.min(otw.tsn), np.max(otw.tsn))

    # Special figure settings and formatting
    targetyax.invert_xaxis()
    targetxax.tick_params(labelleft=False, which='both',
                          left=False, top=False, right=False)
    sourcexax.tick_params(labelleft=False, which='both',
                          left=False, top=False, right=False)
    targetyax.tick_params(labelbottom=False, bottom=False,
                          which='both', top=False, right=False, direction='in')
    sourceyax.tick_params(labelbottom=False, labelleft=False, labelright=True, bottom=False,
                          which='both', top=False, left=False, right=True, direction='in')

    # Fix spines
    targetxax.spines.right.set_visible(False)
    targetxax.spines.left.set_visible(False)
    targetxax.spines.top.set_visible(False)
    targetyax.spines.right.set_visible(False)
    targetyax.spines.bottom.set_visible(False)
    targetyax.spines.top.set_visible(False)
    sourcexax.spines.right.set_visible(False)
    sourcexax.spines.left.set_visible(False)
    sourcexax.spines.top.set_visible(False)
    sourceyax.spines.left.set_visible(False)
    sourceyax.spines.bottom.set_visible(False)
    sourceyax.spines.top.set_visible(False)

    # Set labels
    targetxax.set_xlabel("Time $t\prime$")
    sourcexax.set_xlabel("Time $t\prime$")
    targetyax.set_ylabel('Amplitude $u\prime$')

    # Plot marginal distributions
    targetxax.fill_between(otw.MargT.y, otw.MargT.gn, color=(0.8, 0.8, 0.8))
    targetyax.fill_betweenx(otw.MargU.y, otw.MargU.gn, color=(0.8, 0.8, 0.8))
    sourcexax.fill_between(otw.MargT.x, otw.MargT.fn, color=(0.9, 0.7, 0.7))
    sourceyax.fill_betweenx(otw.MargU.x, otw.MargU.fn, color=(0.9, 0.7, 0.7))

    lplt.plot_label(axes[0, 0], 'a)', location=6, box=False)
    lplt.plot_label(axes[0, 1], 'b)', location=6, box=False)
    lplt.plot_label(axes[1, 0], 'c)', location=6, box=False)
    lplt.plot_label(axes[1, 1], 'd)', location=6, box=False)


def plot_OTW_FP_MARG_COMP(otw: ot.Waveform, noFP=True):

    # Create figure
    _, axes = plt.subplots(2, 1, figsize=(8, 6))
    plt.subplots_adjust(left=0.2, right=0.8, hspace=0.3, bottom=0.15)

    # #### TOP ROW unscaled #####
    axes[0].plot(otw.tt, otw.ut, 'k', label='Observed')
    axes[0].set_xlim(np.min(otw.tt), np.max(otw.tt))
    # axes[0].set_title('Target (black) and Source(red) ')

    # Predicted
    # axes[1].plot([np.min(tn1), np.max(tn1)], [baseline, baseline], 'k:', lw=0.75)
    axes[0].plot(otw.ts, otw.us, 'r', label='Synthetic')
    axes[0].set_xlim(np.min(otw.ts), np.max(otw.ts))

    # Set limits
    axes[0].set_ylim(otw.unorm)

    # Setting axes labels
    axes[0].set_xlabel('Time $t$ [s]')
    axes[0].set_ylabel('Amplitude $u$')

    # #### BOTTOM ROW scaled and fingerprinted
    # Observed
    axes[1].plot(otw.ttn, otw.utn, 'k', label='Target/Obs')
    axes[1].set_xlim(np.min(otw.ttn), np.max(otw.ttn))

    # Predicted
    # axes[1].plot([np.min(tn1), np.max(tn1)], [baseline, baseline], 'k:', lw=0.75)
    axes[1].plot(otw.tsn, otw.usn, 'r', label='Source/Syn')
    axes[1].set_xlim(np.min(otw.tsn), np.max(otw.tsn))
    axes[1].legend(loc='upper left', frameon=False)
    # Set limits
    axes[1].set_ylim(0, 1)

    # Setting ticklabels
    axes[1].tick_params(labelleft=False, labelbottom=False)

    # Plot contours
    if noFP is False:
        axes[1].contour(otw.dtt, otw.dut, otw.dt.T, colors='k',
                        linewidths=0.25, levels=30, alpha=0.5)
        axes[1].contour(otw.dts, otw.dus, otw.ds.T,
                        colors='r', linewidths=0.25, levels=30, alpha=0.5)

    # Create Axes for the marginal distributions
    iP = 99090
    pxax = lplt.axes_from_axes(
        axes[1], iP+1, extent=[0, -0.2, 1.0, 0.15])
    pyax = lplt.axes_from_axes(
        axes[1], iP+2, extent=[-0.15, 0.0, 0.125, 1.0])

    # Set limits
    pyax.set_ylim(0, 1)
    pyax.set_xlim(0, np.max(otw.MargU.gn)*1.25)
    pxax.set_xlim(np.min(otw.ttn), np.max(otw.ttn))

    # Special figure settings and formatting
    pxax.tick_params(labelleft=False, which='both',
                     left=False, top=False, right=False, direction='in')
    pyax.tick_params(labelbottom=False, bottom=False,
                     which='both', top=False, right=False, direction='in')
    # pxax.tick_params(labelbottom=False, bottom=True,
    #                  which='both', top=False, right=False, direction='in')

    # Fix spines
    pxax.spines.right.set_visible(False)
    pxax.spines.left.set_visible(False)
    pxax.spines.top.set_visible(False)
    pyax.spines.right.set_visible(False)
    pyax.spines.bottom.set_visible(False)
    pyax.spines.top.set_visible(False)

    # Set labels
    pxax.set_xlabel("Time $t\prime$")
    pyax.set_ylabel('Amplitude $u\prime$')

    # Plot marginal distributions
    pxax.fill_between(otw.MargT.y, otw.MargT.gn, color=(0.8, 0.8, 0.8))
    pxax.plot(otw.MargT.x, otw.MargT.fn, 'r')
    pyax.fill_betweenx(otw.MargU.y, otw.MargU.gn, color=(0.8, 0.8, 0.8))
    pyax.plot(otw.MargU.fn, otw.MargU.x, 'r')


def plot_OTW_FP_D(otw: ot.Waveform):

    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    # subplots_adjust(wspace=0.4)

    for i in range(3):
        if i > 0:
            lw = 1.0
            ls = (0, (3, 7, 1, 7))
        else:
            lw = 1.0
            ls = '-'
        axes[i].plot(otw.tsn, otw.usn, 'k', lw=lw, ls=ls)
        axes[i].set_xlim(np.min(otw.tsn), np.max(otw.tsn))
        axes[i].set_ylim(0, 1)
        axes[i].set_xlabel('Time $t\prime$ [s]')

    axes[0].set_ylabel('Amplitude $u\prime$')
    # Setting ticklabels
    # axes[1].tick_params(labelleft=False, labelright=True)
    # axes[0].tick_params(labelleft=False)
    # axes[1].tick_params(labelleft=False)

    # Setting axes labels
    axes[3].set_xlabel('Time $t\prime$')

    # Setting titles
    axes[0].set_title('$d_{ij}$')
    axes[1].set_title(r'$\frac{\partial d_{ij}}{\partial u_k}$')
    axes[2].set_title(r'$\frac{\partial d_{ij}}{\partial u_{k+1}}$')
    axes[3].set_title(r'$k$')

    # Plot contours
    axes[0].contour(otw.dts, otw.dus, otw.ds.T, colors='k',
                    linewidths=0.25, levels=30)
    p1 = axes[1].pcolormesh(
        otw.dts, otw.dus, otw.dd_duk.T, cmap='rainbow', linewidths=0.0,
        vmin=np.mean(otw.dd_duk.T) - np.std(otw.dd_duk.T),
        vmax=np.mean(otw.dd_duk.T) + np.std(otw.dd_duk.T))
    # plt.colorbar(p1, ax=axes[1], orientation='horizontal')
    p2 = axes[2].pcolormesh(
        otw.dts, otw.dus, otw.dd_dukp1.T, cmap='rainbow', linewidths=0.0,
        vmin=np.mean(otw.dd_dukp1) - np.std(otw.dd_dukp1),
        vmax=np.mean(otw.dd_dukp1) + np.std(otw.dd_dukp1))
    # plt.colorbar(p2, ax=axes[2], orientation='horizontal')
    axes[3].pcolormesh(
        otw.dts, otw.dus, otw.idk.T, cmap='rainbow', linewidths=0.0)


def plot_rays(otw: ot.Waveform, npts: int = 250):

    # Create figure
    # fig, axes = subplots(1,4, figsize=(12, 3))
    # subplots_adjust(wspace=0.4)
    plt.figure(figsize=(8, 6))
    ax = plt.axes()

    lw = 1.0
    ls = '-'
    x, = ax.plot(otw.tsn, otw.usn, 'k', lw=lw, ls=ls,
                 label=r'$\mathbf{x}(\lambda, \mathbf{x}_k, \mathbf{x}_{k+1})$')
    ax.set_ylim(0, 1)
    ax.set_xlabel('Time $t\prime$')
    ax.set_ylabel('Amplitude $u\prime$')

    # Get random rays
    lin_idx = np.random.randint(0, otw.dts.size * otw.dus.size, npts)
    ids, jds = np.unravel_index(lin_idx, otw.ds.shape)

    # Get segments lengths
    dt = np.diff(otw.tsn)
    du = np.diff(otw.usn)

    # Get ks
    ks = otw.idk[ids, jds].flatten()
    lmds = otw.lambdas[ids, jds].flatten()
    ulam = (otw.usn[:-1])[ks] + du[ks]*lmds
    tlam = (otw.tsn[:-1])[ks] + dt[ks]*lmds

    # Get locations in the distance matrix
    ptsk = otw.dts[ids].flatten()
    pusk = otw.dus[jds].flatten()

    #
    pl = ax.plot(np.vstack((ptsk, tlam)), np.vstack(
        (pusk, ulam)), c=(0.8, 0.2, 0.2), lw=0.5, label='Distances')
    ps = ax.scatter(ptsk, pusk, c='k', s=.5, zorder=10, label=r'$\mathbf{p}$')
    ax.scatter(tlam, ulam, c='k', s=.5, zorder=10)
    ax.set_xlim(np.min(otw.tsn), np.max(otw.tsn))
    plt.legend(handles=(pl[0], ps, x), loc='lower left')


def plot_lambdas(otw: ot.Waveform):

    # Create figure
    # fig, axes = subplots(1,4, figsize=(12, 3))
    # subplots_adjust(wspace=0.4)
    plt.figure(figsize=(8, 6))
    ax = plt.axes()

    lw = 1.0
    ls = '-'
    ax.plot(otw.tsn, otw.usn, 'k', lw=lw, ls=ls)
    ax.set_xlim(np.min(otw.tsn), np.max(otw.tsn))
    ax.set_ylim(0, 1)
    ax.set_xlabel('Time $t\prime$')
    ax.set_ylabel('Amplitude $u\prime$')

    p = ax.pcolormesh(
        otw.dts, otw.dus, otw.lambdas.T, cmap='rainbow', linewidths=0.0,
        vmin=0,
        vmax=1, zorder=-1)
    plt.colorbar(p)

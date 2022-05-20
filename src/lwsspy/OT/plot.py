def plot_fingerprint(tn0, dn, tn1, sn, pt0, pu0, ddij, pt1, pu1, dsij, baseline):
    
    # Create figure
    fig, axes = subplots(1,2, figsize=(8, 4))
    # subplots_adjust(wspace=0.4)
    
    # #### BOTTOM ROW scaled and fingerprinted
    # Observed
    # axes[0].plot([np.min(tn0), np.max(tn0)], [baseline, baseline], 'k:', lw=0.75)
    axes[0].plot(tn0, dn, 'k')
    axes[0].set_xlim(np.min(tn0), np.max(tn0))

    # Predicted
    # axes[1].plot([np.min(tn1), np.max(tn1)], [baseline, baseline], 'k:', lw=0.75)
    axes[1].plot(tn1, sn, 'k')
    axes[1].set_xlim(np.min(tn1), np.max(tn1))
    
    # Set limits
    axes[0].set_ylim(0,1)
    axes[1].set_ylim(0,1)
    
    # Setting ticklabels
    axes[1].tick_params(labelleft=False, labelright=True)
    axes[0].tick_params(labelleft=False)
    axes[1].tick_params(labelleft=False)
    
    # Setting axes labels
    axes[0].set_xlabel('Time $t\prime$ [s]')
    axes[1].set_xlabel('Time $t\prime$ [s]')
    axes[0].set_ylabel('Amplitude $u\prime$')

    # Plot contours
    axes[0].contour(pt0, pu0, ddij.T, colors='k', linewidths=0.25, levels=30)
    axes[1].contour(pt1, pu1, dsij.T, colors='k', linewidths=0.25, levels=30)
    


def plot_OTWaveform(otw: OTWaveform, scale=True, baseline=0.0, transformed=False):
    fig, axes = plt.subplots(2,2, figsize=(10, 6))
    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    
    # Plot observed
    axes[0,0].plot([np.min(otw.tt), np.max(otw.tt)], [0,0], 'k:', lw=0.75)
    axes[0,0].plot(otw.tt, otw.ut)
    axes[0,0].set_xlim(np.min(otw.tt), np.max(otw.tt))
    axes[0,0].set_ylim(otw.unorm[0],otw.unorm[1])
    axes[0,0].set_xlabel(f'$t$ [s]')
    axes[0,0].set_ylabel(f'Amplitude $u$')
    axes[0,0].set_title('Source: Synthetic')

    
    # Plot predicted
    axes[0,1].plot([np.min(otw.ts), np.max(otw.ts)], [0,0], 'k:', lw=0.75)
    axes[0,1].plot(otw.ts, otw.us)
    axes[0,1].set_xlim(np.min(otw.ts), np.max(otw.ts))
    axes[0,1].set_ylim(otw.unorm[0],otw.unorm[1])
    axes[0,1].set_xlabel(f'$t$ [s]')
    axes[0,1].set_title('Target: Observed')
    
    baseline =  (0 - otw.unorm[0])/(otw.unorm[1]-otw.unorm[0])
    
    # Plot observed
    axes[1,0].plot([np.min(otw.ttn), np.max(otw.ttn)], 2*[baseline], 'k:', lw=0.75)
    axes[1,0].plot(otw.ttn, otw.utn)
    axes[1,0].set_xlim(np.min(otw.ttn), np.max(otw.ttn))
    axes[1,0].set_ylim(0,1)
    axes[1,0].set_xlabel(f'$t\prime$')
    axes[1,0].set_ylabel(f'Amplitude $u\prime$')
    
    # Plot predicted
    axes[1,1].plot([np.min(otw.ts), np.max(otw.ts)], 2*[baseline], 'k:', lw=0.75)
    axes[1,1].plot(otw.tsn, otw.usn)
    axes[1,1].set_xlim(np.min(otw.tsn), np.max(otw.tsn))
    axes[1,1].set_ylim(0,1)
    axes[1,1].set_xlabel(f'$t\prime$')
    
    # Remove ticks on RHS plot
    axes[0,1].tick_params(labelleft=False)
    axes[1,1].tick_params(labelleft=False)
    

def plot_ds_fp(
    t0, d, t1, s, tn0, dn, tn1, sn, baseline,
    pt0, pu0, ddij, pt1, pu1, dsij,
    ):
    
    # Create figure
    fig, axes = subplots(2,2, figsize=(8, 8))
    # subplots_adjust(wspace=0.4)
    
    # #### TOP ROW unscaled #####
    # Plot observed
    axes[0,0].plot([np.min(t0), np.max(t0)], [0,0], 'k:', lw=0.75)
    axes[0,0].plot(t0, d, 'k')
    axes[0,0].set_xlim(np.min(t0), np.max(t0))

    # Plot predicted
    axes[0,1].plot([np.min(t1), np.max(t1)], [0,0], 'k:', lw=0.75)
    axes[0,1].plot(t1, s, 'k')
    axes[0,1].set_xlim(np.min(t1), np.max(t1))
        
    # Set limits
    extension = 0.2
    amin, amax  = np.min(np.hstack((d,s))) ,np.max(np.hstack((d,s)))
    deltaa = amax - amin
    u0, u1 = amin - 0.5 * extension * deltaa, amax + 0.5 * extension * deltaa
    axes[0,0].set_ylim(u0,u1)
    axes[0,1].set_ylim(u0,u1)
    
    # #### BOTTOM ROW scaled and fingerprinted
    # Observed
    axes[1,0].plot([np.min(tn0), np.max(tn0)], [baseline, baseline], 'k:', lw=0.75)
    axes[1,0].plot(tn0, dn, 'k')
    axes[1,0].set_xlim(np.min(tn0), np.max(tn0))

    # Predicted
    axes[1,1].plot([np.min(tn1), np.max(tn1)], [baseline, baseline], 'k:', lw=0.75)
    axes[1,1].plot(tn1, sn, 'k')
    axes[1,1].set_xlim(np.min(tn1), np.max(tn1))
    
    # Set limits
    axes[1,0].set_ylim(0,1)
    axes[1,1].set_ylim(0,1)
    
    # Setting ticklabels
    axes[0,1].tick_params(labelleft=False, labelright=True)
    axes[1,0].tick_params(labelleft=False, labelbottom=False)
    axes[1,1].tick_params(labelleft=False, labelbottom=False)
    
    # Setting axes labels
    axes[0,1].set_xlabel('Time $t$ [s]')
    axes[0,0].set_xlabel('Time $t$ [s]')
    axes[0,0].set_ylabel('Amplitude $u$')
    
    # Probabilities
    scaling=.05
    pdij = np.exp(-ddij/scaling)
    psij = np.exp(-dsij/scaling)

    # Normalizing factors
    npdij = np.sum(pdij)
    npsij = np.sum(psij)

    # Normalize
    pdij_norm = pdij/npdij
    psij_norm = psij/npsij
    
    # 1D Probabilities
    dx = np.sum(pdij_norm, axis=1)
    dy = np.sum(pdij_norm, axis=0)
    sx = np.sum(psij_norm, axis=1)
    sy = np.sum(psij_norm, axis=0)
    
    
    # Plot contours
    axes[1,0].contour(pt0, pu0, ddij.T, colors='k', linewidths=0.25, levels=30)
    axes[1,1].contour(pt1, pu1, dsij.T, colors='k', linewidths=0.25, levels=30)
    
    # Create Axes for the marginal distributions
    iP = 99090
    dxax = lplt.axes_from_axes(axes[1,0], iP+1, extent= [0, -0.2, 1.0, 0.15])
    dyax = lplt.axes_from_axes(axes[1,0], iP+2, extent= [-0.2, 0.0, 0.15, 1.0])
    sxax = lplt.axes_from_axes(axes[1,1], iP+3, extent= [0, -0.2, 1.0, 0.15])
    syax = lplt.axes_from_axes(axes[1,1], iP+4, extent= [1.05, 0.0, 0.15, 1.0])

    # Set limits
    dyax.set_ylim(0, 1)
    dyax.set_xlim(0, np.max(dy)*1.5)
    syax.set_ylim(0, 1)
    syax.set_xlim(0, np.max(sy)*1.5)
    dxax.set_xlim(np.min(tn0), np.max(tn0))
    sxax.set_xlim(np.min(tn1), np.max(tn1))

    # Special figure settings and formatting
    dyax.invert_xaxis()
    dxax.tick_params(labelleft=False, which='both', left=False, top=False, right=False)
    sxax.tick_params(labelleft=False, which='both', left=False, top=False, right=False)
    dyax.tick_params(labelbottom=False, bottom=False, which='both', top=False, right=False, direction='in')
    syax.tick_params(labelbottom=False, labelleft=False, labelright=True, bottom=False,
                     which='both', top=False, left=False, right=True,direction='in')

    # Fix spines
    dxax.spines.right.set_visible(False)
    dxax.spines.left.set_visible(False)
    dxax.spines.top.set_visible(False)
    dyax.spines.right.set_visible(False)
    dyax.spines.bottom.set_visible(False)
    dyax.spines.top.set_visible(False)
    sxax.spines.right.set_visible(False)
    sxax.spines.left.set_visible(False)
    sxax.spines.top.set_visible(False)
    syax.spines.left.set_visible(False)
    syax.spines.bottom.set_visible(False)
    syax.spines.top.set_visible(False)
    
    # Set labels
    dxax.set_xlabel("Time $t\prime$ [s]")
    sxax.set_xlabel("Time $t\prime$ [s]")
    dyax.set_ylabel('Amplitude $u\prime$')

    # Plot marginal distributions
    dxax.fill_between(pt0, dx, color=(0.8, 0.8, 0.8))
    dyax.fill_betweenx(pu0, dy, color=(0.8, 0.8, 0.8))
    sxax.fill_between(pt1, sx, color=(0.8, 0.8, 0.8))
    syax.fill_betweenx(pu1, sy, color=(0.8, 0.8, 0.8))
    
    
def plot_fingerprint_diff(t, d, pt, pu, dij, dij_k, dij_kp1, idk):
    
    # Create figure
    fig, axes = subplots(1,4, figsize=(12, 3))
    # subplots_adjust(wspace=0.4)
    
    for i in range(3):
        if i >0:
            lw = 1.0
            ls = (0, (3, 7, 1, 7))
        else:
            lw = 1.0
            ls = '-'
        axes[i].plot(t, d, 'k', lw=lw, ls=ls)
        axes[i].set_xlim(np.min(t), np.max(t))
        axes[i].set_ylim(0,1)
        axes[i].set_xlabel('Time $t\prime$ [s]')
        
    axes[0].set_ylabel('Amplitude $u\prime$')
    # Setting ticklabels
    # axes[1].tick_params(labelleft=False, labelright=True)
    # axes[0].tick_params(labelleft=False)
    # axes[1].tick_params(labelleft=False)
    
    # Setting axes labels
    axes[3].set_xlabel('Time $t\prime$ [s]')

    # Setting titles
    axes[0].set_title('$d_{ij}$')
    axes[1].set_title(r'$\frac{\partial d_{ij}}{\partial u_k}$')
    axes[2].set_title(r'$\frac{\partial d_{ij}}{\partial u_{k+1}}$')
    axes[3].set_title(r'$k$')

    # Plot contours
    axes[0].contour(pt, pu, dij.T, colors='k', linewidths=0.25, levels=30)
    p1 = axes[1].pcolormesh(
        pt, pu, dij_k.T, cmap='rainbow', linewidths=0.0,
        vmin=np.mean(dij_k) - np.std(dij_k), 
        vmax=np.mean(dij_k) + np.std(dij_k))
    colorbar(p1,ax=axes[1], orientation='horizontal')
    p2 = axes[2].pcolormesh(
        pt, pu, dij_kp1.T, cmap='rainbow', linewidths=0.0,
        vmin=np.mean(dij_kp1) - np.std(dij_kp1), 
        vmax=np.mean(dij_kp1) + np.std(dij_kp1))
    colorbar(p2,ax=axes[2], orientation='horizontal')
    axes[3].pcolormesh(
        pt, pu, idk.T, cmap='rainbow', linewidths=0.0)
    
def plot_rays(t, d, pt, pu, dij, lambdas, idk, npts):
    
    # Create figure
    # fig, axes = subplots(1,4, figsize=(12, 3))
    # subplots_adjust(wspace=0.4)
    figure()
    ax = axes()

    lw = 1.0
    ls = '-'
    ax.plot(t, d, 'k', lw=lw, ls=ls)
    ax.set_ylim(0,1)
    ax.set_xlabel('Time $t\prime$ [s]')        
    ax.set_ylabel('Amplitude $u\prime$')

    # Get random rays
    lin_idx = np.random.randint(0, pt.size * pu.size,npts)
    ids, jds = np.unravel_index(lin_idx, dij.shape)
    
    # Get segments lengths
    dt = np.diff(t)
    du = np.diff(d)
    
    # Get ks 
    ks = idk[ids, jds].flatten()
    lmds = lambdas[ids, jds].flatten()
    ulam = (d[:-1])[ks] + du[ks]*lmds
    tlam = (t[:-1])[ks] + dt[ks]*lmds
            
    # Get locations in the distance matrix
    ptsk = pt[ids].flatten()
    pusk = pu[jds].flatten()
    
    #
    ax.plot(np.vstack((ptsk, tlam)), np.vstack((pusk, ulam)), c=(0.8, 0.2, 0.2), lw=0.5)
    ax.scatter(ptsk, pusk, c='k', s=.5, zorder=10)
    # mt, mu = np.meshgrid(pt, pu)
    # ax.scatter(ptsk, pusk, c='k', s=5, zorder=10)
    # ax.scatter(mt.flatten(), mu.flatten(), c='k', s=1, alpha=0.25)
    ax.scatter(tlam, ulam, c='k', s=.5, zorder=10)
    
    ax.axis('equal')
    ax.set_xlim(np.min(t), np.max(t))
    

def plot_lambdas(t, d, pt, pu, lambdas):
    
    # Create figure
    # fig, axes = subplots(1,4, figsize=(12, 3))
    # subplots_adjust(wspace=0.4)
    figure()
    ax = axes()

    lw = 1.0
    ls = '-'
    ax.plot(t, d, 'k', lw=lw, ls=ls)
    ax.set_xlim(np.min(t), np.max(t))
    ax.set_ylim(0,1)
    ax.set_xlabel('Time $t\prime$ [s]')        
    ax.set_ylabel('Amplitude $u\prime$')
    
    p = ax.pcolormesh(
        pt, pu, lambdas.T, cmap='rainbow', linewidths=0.0, 
        vmin=0, 
        vmax=1, zorder=-1)
    colorbar(p)
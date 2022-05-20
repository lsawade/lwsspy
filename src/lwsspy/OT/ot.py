import numpy as np
import matplotlib.pyplot as plt


class OTWaveform:
    
    def __init__(self, t_target, u_target, t_source, u_source, p=2, alpha=0.5, s=0.04, extension=0.2):
        
        """Class that contains all relevant functions to compare two waveforms 
        using optimal transport.
        
        .normalize() - Adds normalized waveforms 
                        .ttn, .utn, 
                        .tsn, .usn, and
                      the scaling bounds for t 
                        .tnorm, and u 
                        .unorm
        """

        # Waveform parameters
        self.tt = t_target
        self.ut = u_target
        self.ts = t_source
        self.us = u_source
        
        # Parameters for Wasserstein computation
        self.p = p
        self.alpha = alpha
        
        # Parameters for probability density function
        self.s = s
        
        # Normalization parameters
        self.extension = extension
        
        # Normalize
        self.normalize()
        
    def normalize(self):
        """Adds normalize waveforms .ttn, .utn, .tsn, .usn, and
        the scaling bounds for t .tnorm, and u .unorm"""
        
        # Set scaling boundaries
        self.tnorm = np.min(self.tt) , np.max(self.tt)

        # Scale time vectors
        self.ttn = (self.tt - self.tnorm[0])/(self.tnorm[1]-self.tnorm[0])
        self.tsn = (self.ts - self.tnorm[0])/(self.tnorm[1]-self.tnorm[0])
        
        # Get scaling values
        amin = np.min(np.hstack((self.ut,self.us)))
        amax = np.max(np.hstack((self.ut,self.us)))
        
        # Extend the scaling region slightly
        deltaa = amax - amin
        self.unorm = amin - 0.5 * self.extension * deltaa, amax + 0.5 * self.extension * deltaa
        
        # Scale the waveforms
        self.utn = (self.ut - self.unorm[0])/(self.unorm[1]-self.unorm[0])
        self.usn = (self.us - self.unorm[0])/(self.unorm[1]-self.unorm[0])

        


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
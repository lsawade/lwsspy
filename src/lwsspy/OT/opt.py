import numpy as np
from lwsspy.wavelets import ricker
from lwsspy.OT.ot import Waveform
from scipy.signal import convolve
from scipy.signal.windows import tukey


def create_OT_Waveform():
    """Creates some data and synthetics and puts them in the OTW wasserform."""

    # General waveform parameters
    N = 500
    L = 2

    # Observed/Target
    tc0 = 0
    f00 = 1.6
    A0 = 1.0
    tt = np.linspace(-2, 2, N)

    # Real model
    m_real = [tc0, f00, A0]
    u = ricker.R(tt, L, *m_real)

    # Make some noiiiise
    noise = 0.25*np.max(np.abs(u))*np.random.normal(loc=0, scale=1.0, size=N)
    smoothnoise = convolve(noise, tukey(31)/np.sum(tukey(31)), 'same')

    # Target waveform amplitude
    ut = u + smoothnoise

    # Synthetic/Source
    # Shift relative to the observed data
    dt = 5.0

    ts = tt + dt
    tc1 = tc0 + dt
    f01 = 1.0
    A1 = 3.0

    m_init = [tc1, f01, A1]
    us = ricker.R(ts, L, *m_init)

    # Create waveform object
    otw = Waveform(tt, ut, ts, us)

    return otw


def optfunc(x, data):
    '''
        Routine to act as an interface with scipy.minimize(). Actions
            - takes model parameters and builds rickerwavelet (forward problem) plus derivatives
            - takes rickerwavelet and builds fingerprint waveform object and OT object of 2D fingerprint density function.
            - takes OT object and calculates Wasserstein misfits between time and amplitude marginals together with derivatives.
            - Combines derivatives using the chain rule and returns average Wasserstein misfit for time-amplitude marginals plus derivatives.

    '''
    # Get input data
    [tt, ut, L0, L1, fc, tc, p, tangentnorm] = data  # get data block

    # Create model vector
    tc1, A1 = x
    us = ricker.R(tt, L=L1, t0=tc1, f0=fc, A=A1)
    dudm = ricker.dRdm(tt, L=L1, t0=tc1, f0=fc, A=A1)
    dudm = (dudm[0], dudm[2])

    # Optimal Transport waveform comparison tool
    otw = Waveform(tt, ut, tt, us, p=p, tangentnorm=tangentnorm)
    otw.fingerprints()
    otw.probabilities()
    W = otw.wasser()
    otw.dWduk()
    deriv = otw.compute_dWdm(dudm)

    return W[0], deriv


def optfunc_LSQ(x, data):
    '''
        Routine to act as an interface with scipy.minimize(). Actions
            - takes model parameters and builds rickerwavelet (forward problem) plus derivatives
            - outputs L2 norm and gradient
    '''
    # Get input data
    [tt, ut, L0, L1, fc, tc, p, _] = data  # get data block
    dt = np.diff(tt)[0]
    # Create model vector
    tc1, A1 = x
    us = ricker.R(tt, L=L1, t0=tc1, f0=fc, A=A1)
    dudm = ricker.dRdm(tt, L=L1, t0=tc1, f0=fc, A=A1)
    dudm = (dudm[0], dudm[2])

    # Cost
    C = 0.5*np.sum((us-ut)**2*dt)

    # Gradient
    G = np.array([np.sum(dt*(us-ut) * _dudm) for _dudm in dudm])

    return C, G


def optfunc_full(x, data):
    '''
        Routine to act as an interface with scipy.minimize(). Actions
            - takes model parameters and builds rickerwavelet (forward problem) plus derivatives
            - takes rickerwavelet and builds fingerprint waveform object and OT object of 2D fingerprint density function.
            - takes OT object and calculates Wasserstein misfits between time and amplitude marginals together with derivatives.
            - Combines derivatives using the chain rule and returns average Wasserstein misfit for time-amplitude marginals plus derivatives.

    '''
    # Get input data
    [tt, ut, p, s, alpha, Nt, Nu, tangentnorm] = data  # get data block

    # Create model vector
    tc1, fc1, A1, L1 = x
    us = ricker.R(tt, L=L1, t0=tc1, f0=fc1, A=A1)
    dudm = ricker.dRdm(tt, L=L1, t0=tc1, f0=fc1, A=A1)

    # Optimal Transport waveform comparison tool
    otw = Waveform(tt, ut, tt, us, p=p, s=s, alpha=alpha,
                   Nt=Nt, Nu=Nu, tangentnorm=tangentnorm)
    otw.fingerprints()
    otw.probabilities()
    W = otw.wasser()
    otw.dWduk()
    deriv = otw.compute_dWdm(dudm)

    return W[0], deriv


def optfunc_LSQ_full(x, data):
    '''
        Routine to act as an interface with scipy.minimize(). Actions
            - takes model parameters and builds rickerwavelet (forward problem) plus derivatives
            - outputs L2 norm and gradient
    '''
    # Get input data
    [tt, ut, _, _, _, _, _, _] = data  # get data block
    dt = np.diff(tt)[0]

    # Create model vector
    tc1, fc1, A1, L1 = x
    us = ricker.R(tt, L=L1, t0=tc1, f0=fc1, A=A1)
    dudm = ricker.dRdm(tt, L=L1, t0=tc1, f0=fc1, A=A1)

    # Cost
    C = 0.5*np.sum((us-ut)**2*dt)

    # Gradient
    G = np.array([np.sum(dt*(us-ut) * _dudm) for _dudm in dudm])

    return C, G

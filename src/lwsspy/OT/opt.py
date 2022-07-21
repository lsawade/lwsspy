from scipy.signal import hilbert as analytic_signal
import numpy as np
from lwsspy.wavelets import ricker
from lwsspy.OT.ot import Waveform
from scipy.signal import convolve
from scipy.signal.windows import tukey


def create_OT_Waveform():
    """Creates some data and synthetics and puts them in the OTW wasserform."""

    # General waveform parameters
    N = 500

    # Observed/Target
    tc0 = 0
    fc0 = 1.6
    L0 = 2.0
    A0 = 1.0
    tt = np.linspace(-2, 2, N)

    # Real model
    m_real_d = dict(L=L0, t0=tc0, f0=fc0, A=A0)
    u = ricker.R(tt, **m_real_d)

    # Make some noiiiise
    noise = 0.25*np.max(np.abs(u))*np.random.normal(loc=0, scale=1.0, size=N)
    smoothnoise = convolve(noise, tukey(31)/np.sum(tukey(31)), 'same')

    # Target waveform amplitude
    ut = u + smoothnoise

    # Synthetic/Source
    # Shift relative to the observed data
    dt = 2.0

    ts = tt + dt
    tc1 = tc0 + dt
    fc1 = 1.0
    A1 = 3.0
    L1 = 1.9

    m_init_d = dict(L=L1, t0=tc1, f0=fc1, A=A1)
    us = ricker.R(ts, **m_init_d)

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


def envelope(array):
    """Is the absolute value of the analytic signal, which is the
    signal + j*hilbert(signal). So
    envelope = sqrt[signal^2 + hilbert(signal)^2]. """
    return np.abs(analytic_signal(array))


def hilbert(array):
    """Scipy.signal.hilbert actually computes the analytical signal. The
    imaginary parrt of which is the hilbert transform"""
    return np.imag(analytic_signal)


def optfunc_LSQ_ENV(x, data):
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

    # Compute envelopes
    eus = envelope(us)
    hus = hilbert(us)
    hdudm = (hilbert(_dudm) for _dudm in dudm)

    edudm = (
        1/eus * (us * _dudm + hus * _hdudm)
        for _dudm, _hdudm in zip(dudm, hdudm)
    )

    # Cost
    Cwf = 0.5*np.sum((us-ut)**2*dt)
    Cenv = 0.5 * np.sum((envelope(us)-envelope(ut))**2*dt)
    C = Cwf + Cenv

    # Gradient
    Gwf = np.array([np.sum(dt*(us-ut) * _dudm) for _dudm in dudm])
    Genv = np.array([
        dt * np.sum((envelope(us)-envelope(ut)) * _edudm)
        for _edudm in edudm])
    G = Gwf + Genv

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


def optfunc_sambridge(x, data):
    '''
        Routine to act as an interface with scipy.minimize(). Actions
            - takes model parameters and builds rickerwavelet (forward problem) plus derivatives
            - takes rickerwavelet and builds fingerprint waveform object and OT object of 2D fingerprint density function.
            - takes OT object and calculates Wasserstein misfits between time and amplitude marginals together with derivatives.
            - Combines derivatives using the chain rule and returns average Wasserstein misfit for time-amplitude marginals plus derivatives.

    '''
    # Get input data
    [tt, ut, L1, p, s, alpha, Nt, Nu, tangentnorm] = data  # get data block

    # Create model vector
    tc1, fc1, A1 = x
    us = ricker.R(tt, L=L1, t0=tc1, f0=fc1, A=A1)
    dudm = ricker.dRdm(tt, L=L1, t0=tc1, f0=fc1, A=A1)[0:3]

    # Optimal Transport waveform comparison tool
    otw = Waveform(tt, ut, tt, us, p=p, s=s, alpha=alpha,
                   Nt=Nt, Nu=Nu, tangentnorm=tangentnorm)
    otw.fingerprints()
    otw.probabilities()
    W = otw.wasser()
    otw.dWduk()
    deriv = otw.compute_dWdm(dudm)

    return W[0], deriv


def optfunc_LSQ_sambridge(x, data):
    '''
        Routine to act as an interface with scipy.minimize(). Actions
            - takes model parameters and builds rickerwavelet (forward problem) plus derivatives
            - outputs L2 norm and gradient
    '''
    # Get input data
    [tt, ut, L1, _, _, _, _, _, _] = data  # get data block
    dt = np.diff(tt)[0]

    # Create model vector
    tc1, fc1, A1 = x
    us = ricker.R(tt, L=L1, t0=tc1, f0=fc1, A=A1)
    dudm = ricker.dRdm(tt, L=L1, t0=tc1, f0=fc1, A=A1)[0:3]

    # Cost
    C = 0.5*np.sum((us-ut)**2*dt)

    # Gradient
    G = np.array([np.sum(dt*(us-ut) * _dudm) for _dudm in dudm])

    return C, G

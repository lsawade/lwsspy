from typing import Iterable
import numpy as np
import matplotlib.pyplot as plt
from .fingerprint import fingerprint, fingerprint_and_diff
from .wasserstein import compute_dWdp


class Marginal:

    def __init__(
            self, x: np.ndarray, f: np.ndarray, y: np.ndarray, g: np.ndarray):

        # X locations for the marginal probabilities along t
        self.x = x
        self.y = y

        # Distribtutions
        self.f = f
        self.g = g

        #
        # Normalization Factors
        self.Nf = np.sum(self.f)
        self.Ng = np.sum(self.g)

        # Normalize
        self.fn = self.f/self.Nf
        self.gn = self.g/self.Ng

        # Cumulative density function
        self.G = np.cumsum(self.gn)
        self.F = np.cumsum(self.fn)

        # Fix normalization error
        self.G /= self.G[-1]
        self.F /= self.F[-1]

        # Since we are using this class exclusively for optimal transport,
        # we can always automatically compute the values important for the
        # Wasserstein distance.
        self._get_wasserstein_values()

    def _get_wasserstein_values(self):
        """This method populates the class with values needed to compute the
        Wasserstein difference """

        # Create tk vector to sample the inverse CDFs
        a = np.append(self.F[:-1], self.G)
        self.tkarg = np.argsort(a)
        self.tk = a[self.tkarg]

        # Get indeces for each CDF
        self.iF = np.searchsorted(
            self.F, self.tk, side='left', sorter=None)
        self.jG = np.searchsorted(
            self.G, self.tk, side='left', sorter=None)

        # Get the dt0 vector
        self.dtk = np.insert(self.tk[1:] - self.tk[:-1], 0, self.tk[0])

        # Compute difference
        self.zf = self.x[self.iF]
        self.zg = self.y[self.jG]

        # compute absolute difference between the z values
        self.dz = np.abs(self.zf-self.zg)

    def wasser(self, p=2):
        return np.sum(self.dz**p * self.dtk)

    def dW_dp(self, p=2):
        """Do NOT feed the normalized PDF. This function assumes that the
        PDF was not yet normalized."""
        return compute_dWdp(self.tk, self.tkarg, self.f, self.iF, self.dz, p=p)

    # def d2W_d2p(self, p=2)


class Waveform:

    normalized: bool = False
    fingerprint_computed: bool = False
    probabilities_computed: bool = False

    def __init__(
            self,
            t_target: np.ndarray, u_target: np.ndarray,
            t_source: np.ndarray, u_source: np.ndarray,
            p: int = 2, alpha: float = 0.5,
            s: float = 0.04, extension: float = 0.025,
            Nt: int = 150, Nu: int = 100, tangentnorm: bool = True):
        """Class that contains all relevant functions to compare two waveforms
        using optimal transport.

        .normalize() - Adds scaled waveforms
                       .ttn, .utn  - scaled Target waveform
                       .tsn, .usn  - scaled Source waveform
                       and the scaling factors
                       .tnorm, and u
                       .unorm

        .fingerprint() - Adds fingerprints for both source and target waveform
                         and derivatives for the source waveform
                         target:
                            .dtt, .dut, .dt

                         source:
                            .dts, .dus, .ds    -  t, u, distance matrix
                            .dd_duk, .dd_dukp1 -  derivatives with respect to u
                            .lambdas           -  (2D matrix) with lambdas
                                                  corresponding to distance
                                                  matrix and segment
                            .idk               -  (2D matrix) indeces of closest
                                                  (distance matrix) segment

        .probabilites() - Adds 2D and marginal 1D probabilites, discretization
                          is the same as for the distance matrix as the density
                          matrix is just a derived measure of the distance
                          .ps, .pt    - 2D densities of source and target
                          .psn, .ptn  - normalized 2D densities of
                                        source and target
                          .psN, .ptN  - normalization factors
                          .MargT     - Marginal class for t distribution
                                        contains subclasses
                                .g, .f   -  Target (g) and Source (f)
                                            marginal probablities
                                .x, .y,  -  location of Source (x -> f(x)) and
                                                        Target (g -> g(y))
                                            marginal probablities along t
                                .Nf, .Ng - normalization factors
                                .fn, .gn - normalized Source (f) and Target (g)
                                           marginal probablities
                                .F , .G  - Source (f) and Target (g)
                                           cumulative density functions
                                .tkarg, .tk - Indeces and values that sort the
                                              combined CDF location
                                .iF, .jG - indeces for the respective CDF where
                                           tk is location
                                .dtk     - tk - t_{k-1}
                                .zf, zg  - Values of the inverse CDFs
                                           corresponding to t_k values
                                .dz      - absolute difference between zg and
                                           zf

                          .MargU     - Marginal class for u distribution
                          .ug, .uf    - Target (g) and Source (f)
                                        marginal probablities along u
                          .uxg, .uxf  - location of Target (g) and Source (f)
                                        marginal probablities along u

                          .ugn, .ufn  - normalized  Target (g) and Source (f)
                                        marginal probablities along u
                          .tgN, .tfN  - normalization factors for t
                          .ugN, .ufN  - normalization factors for u




        """

        # Waveform parameters
        self.tt = t_target
        self.ut = u_target
        self.ts = t_source
        self.us = u_source

        # Parameters for the discrtization of the distance field
        self.Nt = Nt
        self.Nu = Nu

        # Parameters for Wasserstein computation
        self.p = p
        self.alpha = alpha

        # Parameters for probability density function
        self.s = s

        # Normalization parameters
        self.extension = extension
        self.tangentnorm = tangentnorm

        # Normalize
        self.normalize()

    def normalize(self):
        """Adds normalize waveforms .ttn, .utn, .tsn, .usn, and
        the scaling bounds for t .tnorm, and u .unorm"""

        # Set scaling boundaries
        self.tnorm = np.min(self.tt), np.max(self.tt)

        # Scale time vectors
        self.ttn = (self.tt - self.tnorm[0])/(self.tnorm[1]-self.tnorm[0])
        self.tsn = (self.ts - self.tnorm[0])/(self.tnorm[1]-self.tnorm[0])

        # Get scaling values
        if self.tangentnorm:

            # Use only target distribution for the normalization
            amin = np.min(self.ut)
            amax = np.max(self.ut)

            # Extend the scaling region slightly
            deltaa = amax - amin

            # Get extended boundaries
            self.unorm = amin - 0.5 * self.extension * \
                deltaa, amax + 0.5 * self.extension * deltaa

            # Get total range
            du = np.diff(self.unorm)

            # Compute ubar
            ubart = 1/du * (2*self.ut - self.unorm[0] - self.unorm[1])
            ubars = 1/du * (2*self.us - self.unorm[0] - self.unorm[1])

            # Scale the waveforms
            self.utn = 0.5 + 1/np.pi * np.arctan(ubart)
            self.usn = 0.5 + 1/np.pi * np.arctan(ubars)

            # Compute Jacobian
            self.dusndu = 2/(np.pi * du) * 1/(1+ubars**2)

        else:
            amin = np.min(self.ut)
            amax = np.max(self.ut)

            # Don't do this it leads to bad convergence
            # amin = np.min(np.hstack((self.ut, self.us)))
            # amax = np.max(np.hstack((self.ut, self.us)))

            # Extend the scaling region slightly
            deltaa = amax - amin
            self.unorm = amin - 0.5 * self.extension * \
                deltaa, amax + 0.5 * self.extension * deltaa

            # Scale the waveforms
            self.utn = (self.ut - self.unorm[0])/(self.unorm[1]-self.unorm[0])
            self.usn = (self.us - self.unorm[0])/(self.unorm[1]-self.unorm[0])

            self.dusndu = 1/(self.unorm[1]-self.unorm[0])

        # Set normalized flag
        self.normalized = True

    def fingerprints(self, deriv=True):
        """Computes fingerprints for both the source, and the target,
        and for the source (synthetic) it also computes the derivatives
        of the distance with respect to waveform amplitude k"""

        # Check if waveforms have been normalized
        if self.normalized is False:
            self.normalize()

        # For the target distribution we only ever need the distributions
        # and not the
        self.dtt, self.dut, self.dt = fingerprint(
            self.ttn, self.utn, self.Nt, self.Nu)

        # For the source distribution we also need the derivatives of the
        # distance field with respe
        self.dts, self.dus, self.ds, \
            self.lambdas, dd_dunk, dd_dunkp1, self.idk\
            = fingerprint_and_diff(self.tsn, self.usn, self.Nt, self.Nu)

        # Since we input the normalized waveform amplitudes we have to
        # "unnormlized" gradients by multiplying with the
        if self.tangentnorm:
            self.dd_duk = dd_dunk * self.dusndu[self.idk]
            self.dd_dukp1 = dd_dunkp1 * self.dusndu[self.idk + 1]
        else:
            self.dd_duk = dd_dunk * self.dusndu
            self.dd_dukp1 = dd_dunkp1 * self.dusndu

        # Set flag that the fingerprint has been computed
        self.fingerprint_computed = True

    def probabilities(self):

        # Check if fingerprint has been computed, if not, computed
        if self.fingerprint_computed is False:
            self.fingerprints()

        # Compute
        self.pt = np.exp(-self.dt/self.s)
        self.ps = np.exp(-self.ds/self.s)

        # Normalizing factors
        self.ptN = np.sum(self.pt)
        self.psN = np.sum(self.ps)

        # Normalize
        self.ptn = self.pt/self.ptN
        self.psn = self.ps/self.psN

        # Marginal 1D Probabilities for t
        tf = np.sum(self.psn, axis=1)
        tg = np.sum(self.ptn, axis=1)

        # Computed marginal probabilites for u
        uf = np.sum(self.psn, axis=0)
        ug = np.sum(self.ptn, axis=0)

        # Compute marginal 1D probablity density function for t
        self.MargT = Marginal(self.dts, tf, self.dtt, tg)

        # Compute marginal 1D probablity density function for u
        self.MargU = Marginal(self.dus, uf, self.dut, ug)

        # Set probabilites computed flag
        self.probabilities_computed = True

    def wasser(self):

        if self.probabilities_computed is False:
            self.probabilities()

        W_t = self.MargT.wasser(p=self.p)
        W_u = self.MargU.wasser(p=self.p)
        W = self.alpha * W_t + (1-self.alpha) * W_u

        return W, W_t, W_u

    def dWduk(self):
        """Here both the derivatives of W, W_t, and W_u with respect to the
        height of the waveforms are going to be computed."""

        if self.probabilities_computed is False:
            self.probabilities()

        # Compute derivatives with respect to the probability density function
        self.dWtdp = self.MargT.dW_dp(self.p)
        dWtdp_tile = np.tile(self.dWtdp, (self.dus.size, 1)).T

        self.dWudp = self.MargU.dW_dp(self.p)
        dWudp_tile = np.tile(self.MargU.dW_dp(self.p), (self.dts.size, 1))

        # Compute unnormalized gradient with respect to the
        dWtdp = (dWtdp_tile - np.sum(dWtdp_tile*self.psn)) / self.psN
        dWudp = (dWudp_tile - np.sum(dWudp_tile*self.psn)) / self.psN

        # Compute the derivatives with respect to uk and ukp1
        self.dWtduk = dWtdp * -self.ps/self.s * self.dd_duk
        self.dWtdukp1 = dWtdp * -self.ps/self.s * self.dd_dukp1
        self.dWuduk = dWudp * -self.ps/self.s * self.dd_duk
        self.dWudukp1 = dWudp * -self.ps/self.s * self.dd_dukp1

    def compute_dWdm(self, dudm):

        Nm = len(dudm)
        self.dWdm = np.zeros(Nm)

        self.dWtdm = np.zeros(Nm)
        self.dWudm = np.zeros(Nm)

        for i, dudmi in enumerate(dudm):

            self.dWudm[i] = np.sum(
                (self.dWuduk * dudmi[self.idk]) +
                self.dWudukp1 * dudmi[self.idk+1])

            self.dWtdm[i] = np.sum(
                self.dWtduk * dudmi[self.idk] +
                self.dWtdukp1 * dudmi[self.idk+1])

        self.dWdm = self.alpha * self.dWtdm + (1-self.alpha) * self.dWudm

        return self.dWdm

    def compute_dWdu(self):

        # Computing the derivatives with respect to amplitude
        self.dWtdu = np.zeros_like(self.us)
        self.dWudu = np.zeros_like(self.us)

        # Get indeces
        unq, ids = np.unique(self.idk.flatten(), return_inverse=True)
        unqkp1, idskp1 = np.unique(self.idk.flatten() + 1, return_inverse=True)

        # Compute bincount
        self.dWtdu[unq] = np.bincount(ids, self.dWtduk.flatten())
        self.dWtdu[unqkp1] += np.bincount(idskp1, self.dWtdukp1.flatten())
        self.dWudu[unq] = np.bincount(ids, self.dWuduk.flatten())
        self.dWudu[unqkp1] += np.bincount(idskp1, self.dWudukp1.flatten())

        self.dWdu = self.alpha * self.dWtdu + (1-self.alpha) * self.dWudu

        return self.dWdu, self.dWtdu, self.dWudu

    def __compute_dWdu_loop__(self):

        # Computing the derivatives with respect to amplitude
        self.dWtdu = np.zeros_like(self.us)
        self.dWudu = np.zeros_like(self.us)

        # Get indeces
        # unq, ids = np.unique(self.idk-1, return_inverse=True)
        # print('idk', np.min(self.idk), np.max(self.idk))
        # unq, ids = np.unique(self.idk.flatten(), return_inverse=True)
        # unqkp1, idskp1 = np.unique(self.idk.flatten()+1, return_inverse=True)

        # print('unq', np.min(unq), np.max(unq))
        # print('unqp1', np.min(unqkp1), np.max(unqkp1))
        # print('ids', np.min(ids), np.max(ids))
        # print('idkp1', np.min(idskp1), np.max(idskp1))

        # # Compute bincount
        # self.dWtdu[unq] = np.bincount(ids, self.dWtduk.flatten())
        # self.dWtdu[unqkp1] += np.bincount(idskp1, self.dWtdukp1.flatten())
        # self.dWudu[unq] = np.bincount(ids, self.dWtduk.flatten())
        # self.dWudu[unqkp1] += np.bincount(idskp1, self.dWtdukp1.flatten())
        from tqdm import tqdm
        for i in tqdm(range(len(self.us))):
            self.dWtdu[i] = np.sum(self.dWtduk[self.idk == i])
            self.dWtdu[i] += np.sum(self.dWtdukp1[self.idk == i - 1])
            self.dWudu[i] = np.sum(self.dWuduk[self.idk == i])
            self.dWudu[i] += np.sum(self.dWudukp1[self.idk == i - 1])

        self.dWdu = self.alpha * self.dWtdu + (1-self.alpha) * self.dWudu

        return self.dWdu, self.dWtdu, self.dWudu

    def dWduk_FD(self):
        """Here both the derivatives of W, W_t, and W_u with respect to the
        height of the waveforms are going to be computed."""

        if self.probabilities_computed is False:
            self.probabilities()

        # Compute derivatives with respect to the probability density function
        self.dWtdp = self.MargT.dW_dp(self.p)

        dWtdp_tile = np.tile(self.dWtdp, (self.dus.size, 1)).T
        self.dWudp = self.MargU.dW_dp(self.p)
        dWudp_tile = np.tile(self.MargU.dW_dp(self.p), (self.dts.size, 1))

        # Compute unnormalized gradient with respect to the
        dWtdp = (dWtdp_tile - np.sum(dWtdp_tile*self.psn)) / self.psN
        dWudp = (dWudp_tile - np.sum(dWudp_tile*self.psn)) / self.psN

        pert = 0.001

        W_t = self.MargT.wasser(p=self.p)
        W_u = self.MargU.wasser(p=self.p)
        W = self.alpha * W_t + (1-self.alpha) * W_u

        self.dWdu = np.zeros_like(self.us)
        self.dWtdu = np.zeros_like(self.us)
        self.dWudu = np.zeros_like(self.us)

        for k in range(self.us.size):
            us1 = self.us.copy()
            us1[k] += pert
            otw1 = Waveform(self.tt, self.ut, self.ts, us1)
            otw1.fingerprints()
            otw1.probabilities()
            dW, dWt, dWu = otw1.wasser()

            self.dWdu[k] = (dW-W)/pert
            self.dWtdu[k] = (dWt-W_t)/pert
            self.dWudu[k] = (dWu-W_u)/pert

    def compute_dWdm_FD(self, dudm):

        Nm = len(dudm)
        self.dWdm = np.zeros(Nm)

        self.dWtdm = np.zeros(Nm)
        self.dWudm = np.zeros(Nm)

        for i, dudmi in enumerate(dudm):
            self.dWdm[i] = np.sum(self.dWdu * dudmi)
            self.dWtdm[i] = np.sum(self.dWtdu * dudmi)
            self.dWudm[i] = np.sum(self.dWudu * dudmi)
        # +
            # self.dWtdukp1 * dudmi[self.idk+1])
            # self.dWudm[i] = np.sum(
            #     self.dWuduk * dudmi[self.idk] +
            #     self.dWudukp1 * dudmi[self.idk+1])

            # self.dWdm = self.alpha * self.dWtdm + \
            #     (1-self.alpha) * self.dWudm

        return self.dWdm

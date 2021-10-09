# %%
from matplotlib.colors import Normalize
import lwsspy.statistics as lstat
import lwsspy.math as lmat
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def bimodal_gaussian(t, amp1=2.5, to1=20, sig1=5, amp2=1.75, to2=32, sig2=3):
    """ Creates bimodal gaussian from gaussian function.

    :param t: time vector
    :param amp1: amplitude of first gaussian
    :param to1: center time of first gaussian
    :param sig1: standard deviation of first gaussian
    :param amp2: amplitude of second gaussian
    :param to2: center time of second gaussian
    :param sig2: standard deviation of second gaussian
    :return:
    """

    g1 = lstat.gaussian(t, to1, sig1, amp1)
    g2 = lstat.gaussian(t, to2, sig2, amp2)

    return g1 + g2

# %%
# def create_traces(wavelet):


# return alpha, zc

# %%# %%
# def logo():
# Create wavelet
amp1 = 1.0
to1 = 0.0  # 50
sig1 = 0.050
amp2 = 0.0
to2 = 0.050  # -50
sig2 = 0.03
t = np.linspace(-0.500, 0.500, 1000)
wavelet = bimodal_gaussian(t, amp1, to1, sig1, amp2, to2, sig2)/2


# Create S shape in 2D
def f(x):
    return 2.25*(x)**3 - 1.5*x


y = np.linspace(-1.25, 1.25, 1500)
x = f(-y)

# Create meshgrid
xx, yy = np.meshgrid(y, y)

# Compute distances to points
_, idx = lmat.compute_dist2D(xx.flatten(), yy.flatten(), x, y)

# Compute distance from center
rr = np.sqrt(xx**2 + yy**2)


# %%
# Fill array with zeros in S location
zz = np.zeros_like(xx)
uidx = np.unravel_index(idx-1, zz.shape)
zz[uidx] = 1.0

# Add factor to make wave 'fade-out'
factor = 1 - (y/np.max(np.abs(y)))**2
zzf = zz * factor[:, np.newaxis]
zzf = zzf/np.max(zzf)

# Convolve with wavelet
zc = np.apply_along_axis(
    lambda m: np.convolve(m, wavelet, 'same'),
    axis=1, arr=zzf)
zc = zc/np.max(zc)


# %%
# alpha, zz = create_traces(wavelet)
# Use logistic function to compute alpha
alpha = lmat.logistic(rr, k=0.3, x0=0.85, t=0.0, b=1.0)

norm = Normalize(vmin=0.0, vmax=1.0)
plt.figure()
plt.imshow(1.0*np.ones_like(zc), cmap='gray_r', alpha=zc*alpha, norm=norm)
plt.axis('off')
plt.savefig('logo.png', transparent=True)
# plt.close()
# %%
plt.figure(facecolor='k')
norm = Normalize(vmin=0.0, vmax=1.0)
plt.imshow(1.0*np.ones_like(zc), cmap='gray', alpha=zc*alpha, norm=norm)
plt.axis('off')
plt.savefig('logo_dark.png', transparent=True)
# plt.close()
# plt.show(block=True)

# %%
# def logo_traces():

#     # Create wavelet
#     amp1 = 1.25
#     to1 = 0.0  # 50
#     sig1 = 0.050
#     amp2 = 0.0
#     to2 = 0.050  # -50
#     sig2 = 0.03
#     t = np.linspace(-0.500, 0.500, 1000)
#     wavelet = bimodal_gaussian(t, amp1, to1, sig1, amp2, to2, sig2)/2

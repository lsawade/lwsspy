# %%
import numpy as np
import matplotlib.pyplot as plt


# %%
# Radius theta and phi
rad = np.linspace(0, 1, 101)
the = np.linspace(-np.pi/2, np.pi/2, 91)
phi = np.linspace(-np.pi, np.pi, 181)

R, T, P = np.meshgrid(rad, the, phi)

Z = np.cos(6*np.pi*R)*np.sin(2*P)*np.sin(2*T)

# %% For all R and theta get phi slices
deg = 50
idr1 = np.argmin(np.abs(phi-deg/180*np.pi))
idr2 = np.argmin(np.abs(phi-(deg/180*np.pi - np.pi)))

# %% Get slices
sl1 = Z[:, :, idr1]
sl2 = Z[:, :, idr2]

# %% new meshgrids
r1, t1 = np.meshgrid(rad, the+np.pi/2)
r2, t2 = np.meshgrid(rad, the-np.pi/2)

# %%
plt.figure()
plt.polar()

plt.pcolormesh(t1, r1, sl1)
plt.pcolormesh(t2, r2, sl2)

import os
import numpy as np
import matplotlib.pyplot as plt

import lwsspy as lpy

plt.figure(figsize=(8, 4))
plt.subplots_adjust(bottom=0.1, top=0.9, left=0.05, right=0.95, hspace=0.1)
ax = plt.subplot(211)
x = np.linspace(-np.pi, 3*np.pi, 500)
plt.plot(x, np.cos(x))
plt.title(r'Multiples of $\pi$ - 2 ways')
ax.grid(True)
ax.set_aspect(1.0)
ax.axhline(0, color='black', lw=2)
ax.axvline(0, color='black', lw=2)

# Define Locators/formatters more or less by hand
ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
ax.xaxis.set_major_formatter(plt.FuncFormatter(
    lpy.multiple_formatter(2, number=np.pi, latex='\pi')))

ax = plt.subplot(212)

# Create Multiple class containing locators, and formatters
minor = lpy.Multiple(12, number=np.pi, latex='\pi')
major = lpy.Multiple(2, number=np.pi, latex='\pi')

# Plot
x = np.linspace(-np.pi, 3*np.pi, 500)
plt.plot(x, np.cos(x))
ax.grid(True)
ax.set_aspect(1.0)
ax.axhline(0, color='black', lw=2)
ax.axvline(0, color='black', lw=2)
ax.xaxis.set_major_locator(major.locator)
ax.xaxis.set_minor_locator(minor.locator)
ax.xaxis.set_major_formatter(major.formatter)

plt.savefig(os.path.join(lpy.DOCFIGURES,
                         "multiple_locator.svg"), transparent=True)
plt.show()

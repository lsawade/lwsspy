import os
import matplotlib.pyplot as plt
import numpy as np

# Internal
from lwsspy import DOCFIGURES  # Location to store figure
from lwsspy import updaterc    # Makes figure pretty in my opinion
updaterc()

# Create vector
x = np.linspace(-1, 1, 100)

for _i in range(4):

    # Create subplots
    if _i == 0:
        ax = plt.subplot(221+_i)
    else:
        ax = plt.subplot(221+_i, sharex=ax, sharey=ax)

    # Plot line
    plt.plot(x, x ** (_i+1), label=f"$y = x^{_i + 1}$")
    ax.grid(True)
    ax.autoscale(enable=True, axis='both', tight=True)

    # Fix labels
    if _i in [0, 2]:
        plt.ylabel("y")
    if _i in [0, 1]:
        ax.tick_params(labelbottom=False)  # The important line
    if _i in [2, 3]:
        plt.xlabel("x")
    if _i in [1, 3]:
        ax.tick_params(labelleft=False)  # The important line

    # Generate legend
    plt.legend(loc=4)

# Adjusting the plots
plt.subplots_adjust(hspace=0.125, wspace=0.125)

# Saving 
plt.savefig(os.path.join(DOCFIGURES, 'remove_labels.svg'))

# And showing for good measure
plt.show()

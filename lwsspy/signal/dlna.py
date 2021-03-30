
import numpy as np

def dlnA(d, s):
    return 0.5 * np.log(np.sum(d ** 2) / np.sum(s ** 2))
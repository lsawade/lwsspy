import numpy as np
import re
from matplotlib.dates import datestr2num
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


class Trendline:

    def __init__(self, x, y, pred_type='2'):

        self.x = x
        self.y = y
        self.pred_type = pred_type
        self.set_callable()

    def set_callable(self):

        if self.pred_type == '2':
            def func(x, a, b, c):
                return a * x**2 + b*x + c

            def dfunc(x, a, b, c):
                return np.vstack((x**2, x, np.ones_like(x))).T

        elif self.pred_type == 'log':
            def func(x, a, b):
                return a + b * np.log(x)

            def dfunc(x, a, b):
                return np.vstack((np.ones_like(x), np.log(x))).T

        elif self.pred_type == 'log10':
            def func(x, a, b):
                return a + b * np.log10(x)

            def dfunc(x, a, b):
                return np.vstack((np.ones_like(x), np.log(x))).T

        elif self.pred_type == 'exp':
            def func(x, a, b):
                return a + np.exp(b * x)

            def dfunc(x, a, b):
                return np.vstack((np.ones_like(x), x * np.exp(b * x))).T

        elif self.pred_type == 'linear':
            def func(x, a, b):
                return a + b * x

            def dfunc(x, a, b):
                return np.vstack((x, np.ones_like(x))).T
        else:
            raise ValueError("Prediction type not implemented")

        self.func = func
        self.dfunc = dfunc

    def fit(self, xi=None):

        # Fit line
        pos = ~np.isnan(self.x) & ~np.isnan(self.y)
        self.poly, self.cov = curve_fit(
            self.func, self.x[pos], self.y[pos], jac=self.dfunc)

        # Fitted data
        f = np.polyval(self.poly, self.x)

        # Rsquared
        self.R2 = 1 - np.sum((self.y[pos] - self.model(self.x[pos]))**2) / \
            np.sum((self.y[pos] - np.mean(self.y[pos]))**2)

        if xi is not None:
            f = np.polyval(self.poly, xi)

    def model(self, x):
        return self.func(x, *self.poly)

    def confidence_band(self, x):

        # Get Gradient at the point
        grady = self.dfunc(x, *self.poly)

        # Get diagonal covariance
        var = np.diag(grady @ self.cov @ grady.T)

        # Get 95 confidence interval
        self.conf = 1.96 * np.sqrt(var)

    def plot_trendline(self, x, *args, **kwargs):

        # Get axes
        ax = plt.gca()

        # Poly line
        ax.plot(x, self.model(x), *args, **kwargs)

    def plot_confidence_band(self, x, **kwargs):

        # Get axes
        ax = plt.gca()

        # Compute confidence band
        self.confidence_band(x)

        # Compute predictions
        y = self.model(x)

        # compute upper and lower band
        std_p = y + self.conf
        std_m = y - self.conf

        # Plot upper and lower band
        poly = ax.fill_between(x, std_p, y2=std_m,  **kwargs)

        return poly

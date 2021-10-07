import numpy as np
import matplotlib.pyplot as plt


def multiple_formatter(denominator=2, number=np.pi, latex='\pi'):
    """Creates a formatter for multiples of a certain number. Convenient for
    functions of radians e.g.

    Originally taken from `SOF`_.

    .. _SOF:: https://stackoverflow.com/questions/40642061/how-to-set-axis-ticks-in-multiples-of-pi-python-matplotlib

    Parameters
    ----------
    denominator : int, optional
        Denominator, by default 2
    number : float, optional
        some number, by default np.pi
    latex : str, optional
        Latex number, by default '\pi'

    Returns
    -------
    a formatter for ticklables

    Example
    -------

    >>> x = np.linspace(-np.pi, 3*np.pi,500)
    >>> plt.plot(x, np.cos(x))
    >>> plt.title(r'Multiples of $\pi$')
    >>> ax = plt.gca()
    >>> ax.grid(True)
    >>> ax.set_aspect(1.0)
    >>> ax.axhline(0, color='black', lw=2)
    >>> ax.axvline(0, color='black', lw=2)
    >>> ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    >>> ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
    >>> ax.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
    >>> plt.show()


    Notes
    -----

    :Author:
        Lucas Sawade (lsawade@princeton.edu)

    :Last Modified:
        2021.04.09 12.50

    """
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def _multiple_formatter(x, pos):
        den = denominator
        num = np.int(np.rint(den*x/number))
        com = gcd(num, den)
        (num, den) = (int(num/com), int(den/com))
        if den == 1:
            if num == 0:
                return r'$0$'
            if num == 1:
                return r'$%s$' % latex
            elif num == -1:
                return r'$-%s$' % latex
            else:
                return r'$%s%s$' % (num, latex)
        else:
            if num == 1:
                return r'$\frac{%s}{%s}$' % (latex, den)
            elif num == -1:
                return r'$\frac{-%s}{%s}$' % (latex, den)
            else:
                return r'$\frac{%s%s}{%s}$' % (num, latex, den)
    return _multiple_formatter


class Multiple:
    def __init__(self, denominator=2, number=np.pi, latex='\pi'):
        """Creates a formatter and locator for multiples of a certain number. 
        Convenient for functions of radians e.g. More sophisticated than the
        formatter itself.

        Originally taken from `SOF`_.

        .. _SOF:: https://stackoverflow.com/questions/40642061/how-to-set-axis-ticks-in-multiples-of-pi-python-matplotlib

        Parameters
        ----------
        denominator : int, optional
            Denominator, by default 2
        number : float, optional
            some number, by default np.pi
        latex : str, optional
            Latex number, by default '\pi'

        Returns
        -------
        a formatter for ticklables

        Example
        -------

        >>> tau = np.pi*2
        >>> den = 60
        >>> major = Multiple(den, tau, r'\tau')
        >>> minor = Multiple(den*4, tau, r'\tau')
        >>> x = np.linspace(-tau/60, tau*8/60,500)
        >>> plt.plot(x, np.exp(-x)*np.cos(60*x))
        >>> plt.title(r'Multiples of $\tau$')
        >>> ax = plt.gca()
        >>> ax.grid(True)
        >>> ax.axhline(0, color='black', lw=2)
        >>> ax.axvline(0, color='black', lw=2)
        >>> ax.xaxis.set_major_locator(major.locator())
        >>> ax.xaxis.set_minor_locator(minor.locator())
        >>> ax.xaxis.set_major_formatter(major.formatter())
        >>> plt.show()


        Notes
        -----

        :Author:
            Lucas Sawade (lsawade@princeton.edu)

        :Last Modified:
            2021.04.09 12.50

        """

        self.denominator = denominator
        self.number = number
        self.latex = latex

    @property
    def locator(self):
        return plt.MultipleLocator(self.number / self.denominator)

    @property
    def formatter(self):
        return plt.FuncFormatter(multiple_formatter(self.denominator, self.number, self.latex))

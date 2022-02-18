from astropy import units as u
import numpy as np
from scipy import special as sc

from .core import Profile
from .helpers import math
from .helpers.decorators import array, inMpc


class Einasto(Profile):
    """Einasto (1965) profile.

    .. math::

        \rho(r) = \rho_\mathrm{s}
            \exp\left(-\frac2\alpha
                        \left[
                            \left(\frac{r}{r_\mathrm{s}}\right)^\alpha - 1
                        \right]
                \right)

    with :math:`\rho_\mathrm{s}`, :math:`r_\mathrm{s}`, and
    :math:`\alpha` all strictly positive quantities.

    Parameters
    ----------
    rho_s : float, ndarray or astropy Quantity
        characteristic density, in Msun/Mpc^3 (if not an astropy
        Quantity)
    r_s : float, ndarray or astropy Quantity
        characteristic radius where the density profile has a slope of
        -2. If not an astropy Quantity, assumed to be in Mpc
    alpha : float or ndarray
        slope steepening

    .. warning:
        The ``mdelta`` method has not yet been implemented
    """
    def __init__(self, rho_s, r_s, alpha, z=0, **kwargs):
        if isinstance(rho_s, u.Quantity):
            rho_s = rho_s.to(u.Msun/u.Mpc**3).value
        if isinstance(r_s, u.Quantity):
            r_s = r_s.to(u.Mpc).value
        self._set_shape(rho_s*r_s*alpha)
        super().__init__(z=z, **kwargs)
        self.rho_s = rho_s
        self.r_s = r_s
        self.alpha = alpha

    ### attributes ###

    @property
    def total_mass(self):
        return 4*np.pi*self.rho_s*self.r_s**3/self.alpha \
            / (2/self.alpha)**(3/self.alpha) * sc.gamma(3/self.alpha) \
            * np.exp(2/self.alpha)

    ### methods ###

    @array
    @inMpc
    def profile(self, r):
        return self.rho_s / np.exp(2/self.alpha * ((r/self.r_s)**self.alpha-1))

    @array
    @inMpc
    def mass_profile(self, r):
        """Mass enclosed in a radius r"""
        x = r / self.r_s
        a = 3 / self.alpha
        return self.total_mass / sc.gamma(a) \
            * (sc.gamma(a) - math.gamma(a, 2*x**self.alpha/self.alpha))

    def _mdelta(self, overdensity, background='c', err=1e-3, n_guess_rng=1000,
               max_iter=50):
        """Iteratively estimate the mass within a spherical overdensity
        radius

        Parameters
        ----------
        overdensity : int or float
            spherical overdensity within which to calculate the mass

        .. warning::
            Not implemented
        """
        self._assert_background(background)
        self._assert_overdensity(overdensity)
        
        return


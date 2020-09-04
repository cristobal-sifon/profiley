from astropy import units as u
from astropy.cosmology import Planck15
import numpy as np
from scipy.special import sici

from .core import Profile
from .helpers.decorators import array, inMpc
from .helpers.lensing import BaseLensing
from .helpers.spherical import mass_from_radius, radius_from_mass


class BaseNFW(BaseLensing, Profile):

    def __init__(self, mass, c, z, overdensity=200, background='c',
                 cosmo=Planck15, frame='comoving', numeric_kwargs={}):
        self._assert_background(background)
        if isinstance(mass, u.Quantity):
            mass = mass.to(u.Msun).value
        if not np.iterable(mass):
            mass = np.array([mass])
        self.mass = mass
        self._shape = self.mass.shape
        self.z = self._define_array(z)
        super().__init__(self.z, cosmo=cosmo, frame=frame, **numeric_kwargs)
        self.background = background
        self.c = self._define_array(c)
        self.overdensity = overdensity
        self._delta_c = None
        self._rs = None
        self._radius = None
        self._sigma_s = None

    ### attributes ###

    @property
    def delta_c(self):
        if self._delta_c is None:
            self._delta_c = self._f_delta_c(self.c, self.overdensity)
        return self._delta_c

    @property
    def rs(self):
        if self._rs is None:
            self._rs = self.radius / self.c
        return self._rs

    @property
    def radius(self):
        if self._radius is None:
            self._radius = radius_from_mass(
                self.mass, self.overdensity, self.rho_bg)
        return self._radius

    @property
    def sigma_s(self):
        if self._sigma_s is None:
            self._sigma_s = self.rs * self.delta_c * self.rho_bg
        return self._sigma_s

    ### hidden methods ###

    def _assert_background(self, background):
        if background not in 'cm':
            msg = "background must be either 'c' (critical) or 'm' (mean);" \
                  f' received {background}'
            raise ValueError(msg)

    def _f_delta_c(self, c, overdensity):
        return (overdensity*c**3/3) / (np.log(1+c) - c/(1+c))

    ### methods ###

    '''
    def mdelta(self, overdensity, background='c'):
        """Calculate mass at any overdensity from the original mass
        definition"""
        if overdensity == self.overdensity \
                and background == self.background:
            return self.mass
        c_guess = (self.overdensity/overdensity)**0.5 * self.c
        if overdensity > self.overdensity:
            c_rng = np.linspace(0.5*c_guess, c_guess, 100)
        else:
            c_rng = np.linspace(c_guess, 1.5*c_guess, 100)
        delta_c_rng = self._f_delta_c(c_rng, overdensity)
        delta_c_diff = np.abs(delta_c_rng - np.expand_dims(self.delta_c, -1))
    '''


class GNFW(BaseNFW):
    """Generalized NFW profile

    Density profile described by

    .. math::

        \rho(r) = \frac{\delta_\mathrm{c}\rho_\mathrm{bg}}
                    {(r/r_\mathrm{s})^\gamma
                        \left[1+(r/r_\mathrm{s})^\alpha\right]^(\beta-\gamma)/\alpha

    A common but more restriced GNFW profile can be recovered by setting
    alpha=1, beta=3.

    """

    def __init__(self, mass, c, z, alpha, beta, gamma, **kwargs):
        super().__init__(mass, c, z, **kwargs)
        self.alpha = self._define_array(alpha)
        self.beta = self._define_array(beta)
        self.gamma = self._define_array(gamma)

    ### main methods ###

    @inMpc
    @array
    def density(self, r):
        exp = (self.beta-self.gamma) / self.alpha
        return self.delta_c * self.rho_bg \
            / ((r/self.rs)**self.gamma * (1+(r/self.rs)**alpha)**exp)


class NFW(BaseNFW):
    """Navarro-Frenk-White profile (Navarro et al. 1995)

    Density profile:

    .. math::

        \rho(r) = \frac{\delta_\mathrm{c}\rho_\mathrm{bg}}
                    {(r/r_\mathrm{s})(1+r/r_\mathrm{s})^2}
    """

    def __init__(self, mass, c, z, **kwargs):
        super(NFW, self).__init__(mass, c, z, **kwargs)

    ### main methods ###

    @inMpc
    @array
    def density(self, r):
        """Three-dimensional density profile"""
        return self.delta_c * self.rho_bg / (r/self.rs * (1+r/self.rs)**2)

    @inMpc
    @array
    def surface_density(self, R):
        """Surface density at distance(s) R"""
        x = R / self.rs
        s = np.ones_like(x) / 3
        s[x == 0] = 0
        j = (x > 0) & (x < 1)
        s[j] = (1 - 2*np.arctanh(((1-x[j]) / (1+x[j]))**0.5) / (1 - x[j]**2)**0.5) \
            / (x[j]**2 - 1)
        j = x > 1
        s[j] = (1 - 2*np.arctan(((x[j]-1) / (1+x[j]))**0.5) / (x[j]**2 - 1)**0.5) \
            / (x[j]**2 - 1)
        return 2 * self.sigma_s * s

    @inMpc
    @array
    def enclosed_surface_density(self, R):
        """Surface density enclosed within R"""
        x = R / self.rs
        s = np.ones_like(x) + np.log(0.5)
        s[x == 0] = 0
        j = (x > 0) & (x < 1)
        s[j] = (np.log(0.5*x[j]) \
                + 2 * np.arctanh(((1 - x[j])/(1 + x[j]))**0.5) \
                  / (1 - x[j]**2)**0.5) \
               / x[j]**2
        j = x > 1
        s[j] = (2 * np.arctan(((x[j] - 1)/(1 + x[j]))**0.5) / (x[j]**2-1)**0.5 \
                + np.log(0.5*x[j])) / x[j]**2
        return 4 * self.sigma_s * s

    @array
    def fourier(self, k):
        """Fourier transform"""
        ki = k * self.rs
        bs, bc = sici(ki)
        asi, ac = sici((1+self.c)*ki)
        return 4 * np.pi * self.rho_bg * self.delta_c * self.rs**3 / self.mass \
            * (np.sin(ki)*(asi-bs) - (np.sin(self.c*ki) / ((1+self.c)*ki)) \
               + np.cos(ki)*(ac-bc))

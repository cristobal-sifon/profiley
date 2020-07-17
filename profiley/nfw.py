from astropy.cosmology import Planck15
import numpy as np
from scipy.special import sici

from .helpers.decorators import array, inMpc
from .profiles import DensityProfile


class BaseNFW(DensityProfile):

    def __init__(self, mass, c, z, overdensity=200, background='c', cosmo=Planck15):
        assert background in 'cm', \
            "background must be either 'c' (critical) or 'm' (mean)"
        super().__init__(mass, z, cosmo=cosmo)
        self._background = background
        self._concentration = self._define_array(c)
        self._overdensity = overdensity
        self._deltac = None
        self._rs = None
        self._radius = None
        self._sigma_s = None

    ### attributes ###

    @property
    def background(self):
        return self._background

    @property
    def c(self):
        return self._concentration

    @property
    def deltac(self):
        if self._deltac is None:
            self._deltac = (self.overdensity * self.c**3 / 3) \
                / (np.log(1+self.c) - self.c/(1+self.c))
        return self._deltac

    @property
    def overdensity(self):
        return self._overdensity

    @property
    def rs(self):
        if self._rs is None:
            self._rs = self.radius / self.c
        return self._rs

    @property
    def radius(self):
        if self._radius is None:
            self._radius = \
                (self.mass / (4*np.pi/3) / (self.overdensity*self.rho_bg))**(1/3)
        return self._radius

    @property
    def sigma_s(self):
        if self._sigma_s is None:
            self._sigma_s = self.rs * self.deltac * self.rho_bg
        return self._sigma_s


class gNFW(BaseNFW):

    def __init__(self, mass, c, alpha, z, **kwargs):
        super(gNFW, self).__init__(mass, c, z, **kwargs)
        self.alpha = self._define_array(alpha)

    ### main methods ###

    @inMpc
    @array
    def density(self, R):
        return self.deltac * self.rho_bg \
            / ((R/self.rs)**self.alpha * (1+R/self.rs)**(3-self.alpha))


class NFW(BaseNFW):
    """Navarro-Frenk-White profile"""

    def __init__(self, mass, c, z, **kwargs):
        super(NFW, self).__init__(mass, c, z, **kwargs)

    ### main methods ###

    @inMpc
    @array
    def density(self, R):
        return self.deltac * self.rho_bg / (R/self.rs * (1+R/self.rs)**2)

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
        return 4 * np.pi * self.rho_bg * self.deltac * self.rs**3 / self.mass \
            * (np.sin(ki)*(asi-bs) - (np.sin(self.c*ki) / ((1+self.c)*ki)) \
               + np.cos(ki)*(ac-bc))

from astropy import constants as ct, units as u
from astropy.cosmology import FLRW, Planck15


class BaseCosmo(object):

    def __init__(self, cosmo=Planck15, frame='comoving'):
        assert isinstance(cosmo, FLRW), \
            'argument `cosmo` must be an `astropy.cosmology.FLRW` object'
        self.cosmo = cosmo
        assert frame in ('comoving', 'proper')
        self.frame = frame
        self._background = None
        self._critical_density = None
        self._hmf = None
        self._mean_density = None
        self._overdensity = None
        self._rho_bg = None
        self.__c = None
        self.__G = None

    @property
    def _c(self):
        """Speed of light in Mpc/s"""
        if self.__c is None:
            self.__c = ct.c.to(u.Mpc/u.s).value
        return self.__c

    @property
    def _G(self):
        """Gravitational constant in Mpc^3/Msun/s^2"""
        if self.__G is None:
            self.__G = ct.G.to(u.Mpc**3/u.Msun/u.s**2).value
        return self.__G

    @property
    def critical_density(self):
        """Critical density in Msun/Mpc^3"""
        if self._critical_density is None:
            self._critical_density = \
                self.cosmo.critical_density(self.z).to(u.Msun/u.Mpc**3).value
            #if self.frame == 'comoving':
                #self._critical_density = self._critical_density / (1+self.z)**3
        return self._critical_density

    @property
    def mean_density(self):
        """Mean density in Msun/Mpc^3"""
        if self._mean_density is None:
            self._mean_density = \
                (self.cosmo.critical_density0 * self.cosmo.Om0 * (1+self.z)**3).to(
                    u.Msun/u.Mpc**3).value
        return self._mean_density

    @property
    def rho_bg(self):
        """Mean or critical density depending on ``self.background``"""
        if self.background == 'm':
            return self.mean_density
        return self.critical_density


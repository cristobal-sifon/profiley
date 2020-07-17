from astropy import units as u
from astropy.cosmology import Planck15
import numpy as np

from ._cosmology import BaseCosmo
from .decorators import float_args


class BaseLensing(BaseCosmo):

    def __init__(self, zl, zs=None, cosmo=Planck15):
        """Base lensing calculations

        Parameters
        ----------
        zl : float or array of floats
            lens redshift
        zs : float or array of floats, optional
            source redshift
        cosmo : astropy.cosmology.FLRW object
            cosmology

        Notes
        -----
        - If both zl and zs are arrays, their shapes must be such that
          they can be combined in numpy operations; this will not be
          corrected automatically.
        """
        try:
            zl / 1
        except TypeError as e:
            msg = 'argument `zl` must be a float or an array of floats'
            raise TypeError(msg) from e
        if zs is not None:
            try:
                zs / 1
            except TypeError as e:
                msg = 'argument `zs` must be a float or an array of floats'
                raise TypeError(msg) from e
        super(BaseLensing, self).__init__(cosmo=cosmo)
        self.zl = zl
        self._zs = zs
        self.cosmo = cosmo
        self._Dl = None
        self._Dls = None
        self._Ds = None
        self.__c = None
        self.__G = None
        return

    ### attributes ###

    @property
    def zs(self):
        return self._zs

    #@float_args
    @zs.setter
    def zs(self, zs):
        self._zs = zs

    ### aliases ###

    @property
    def Dl(self):
        if self._Dl is None:
            self._Dl = self.cosmo.angular_diameter_distance(self.zl).to(
                u.Mpc).value
        return self._Dl

    @property
    def Dls(self):
        if self._Dls is None:
            self._Dls = self.cosmo.angular_diameter_distance_z1z2(
                self.zl, self.zs).to(u.Mpc).value
        return self._Dls

    @property
    def Ds(self):
        if self._Ds is None:
            self._Ds = self.cosmo.angular_diameter_distance(self.zs).to(
                u.Mpc).value
        return self._Ds

    ### methods ###

    #@float_args
    def beta(self, zs=None):
        """max(0, Dls/Ds)"""
        if zs is None:
            b = self.Dls / self.Ds
        else:
            #b = float(self.cosmo.angular_diameter_distance_z1z2(self.zl, zs) \
                      #/ self.cosmo.angular_diameter_distance(zs))
            b = (self.cosmo.angular_diameter_distance_z1z2(self.zl, zs).to(u.Mpc) \
                 / self.cosmo.angular_diameter_distance(zs).to(u.Mpc)).value
        return np.max([np.zeros_like(b), b])

    #@float_args
    def excess_surface_density(self, R):
        return self.enclosed_surface_density(R) - self.surface_density(R)

    #@float_args
    def sigma_crit(self, zs=None): #, frame='comoving'):
        """Critical surface density, in Msun/Mpc^2"""
        #assert frame in ('comoving', 'physical','proper')
        A = self._c**2 / (4*np.pi*self._G)
        #if frame == 'comoving':
            #A = A / (1+self.zl)**2
        return A / (self.Dl*self.beta(zs=zs))

    #@float_args
    def convergence(self, R, zs=None, **kwargs):
        return self.surface_density(R) / self.sigma_crit(zs=zs, **kwargs)


class Lens(BaseCosmo):
    """Lens object class"""
    # should move some of what's in BaseLensing into here
    # then calling BaseLensing (perhaps just Lensing) should
    # create (or take) both a Lens and a Source objects
    def __init__(self, zl, cosmo=Planck15):
        super(Lens, self).__init__(cosmo=cosmo)
        self.zl = zl
        self._chi = None
        self._Dl = None
        self._Wk = None

    @property
    def chi(self):
        if self._chi is None:
            self._chi = self.cosmo.comoving_distance(self.zl)
        return self._chi

    @property
    def Dl(self):
        if self._Ds is None:
            self._Ds = self.cosmo.angular_diameter_distance(self.zl)
        return self._Ds

    ### methods ###

    #@float_args
    def Dls(self, zs):
        return self.cosmo.angular_diameter_distance_z1z2(self.zl, zs)

    #@float_args
    def sigma_crit(self, zs, frame='comoving'):
        assert frame in ('comoving', 'physical','proper')
        A = self._c**2 / (4*np.pi*self._G)
        if frame == 'comoving':
            A = A / (1+self.zl)**2
        return A / (self.Dl*self.beta(zs=zs))

    #@float_args
    def lensing_kernel(self, zs):
        chizs = self.cosmo.comoving_distance(zs)
        return (3/2 * self.cosmo.Om0 * self.cosmo.H0**2 \
                * (1+zs) / self.cosmo.H(zs) * self.chi / self._c \
                * (chizs-self.chi) / chizs).to(u.Mpc/u.s).value


class Source(BaseCosmo):

    """Lensed source class"""

    def __init__(self, zs, cosmo=Planck15):
        super(Source, self).__init__(cosmo=cosmo)
        self.zs = zs
        self._chi = None
        self._Ds = None
        self._Wk = None

    @property
    def chi(self):
        if self._chi is None:
            self._chi = self.cosmo.comoving_distance(self.zs)
        return self._chi

    @property
    def Ds(self):
        if self._Ds is None:
            self._Ds = self.cosmo.angular_diameter_distance(self.zs)
        return self._Ds

    ### methods ###

    def Dls(self, zl):
        return self.cosmo.angular_diameter_distance_z1z2(zl, self.zs)

    def sigma_crit(self, zl, frame='comoving'):
        assert frame in ('comoving', 'physical','proper')
        A = self._c**2 / (4*np.pi*self._G)
        if frame == 'comoving':
            A = A / (1+zl)**2
        return A / (self.angular_diameter_distance(zl)*self.beta(zs=zs))

    def lensing_kernel(self, zl):
        chizl = self.cosmo.comoving_distance(zl)
        return (3/2 * self.cosmo.Om0 * self.cosmo.H0**2 \
                * (1+self.zs) / self.cosmo.H(self.zs) * chizl / self._c \
                * (self.chi-chizl) / self.chi).to(u.Mpc/u.s).value

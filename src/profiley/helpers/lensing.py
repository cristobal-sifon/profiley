from astropy import units as u
from astropy.cosmology import Planck15
import numpy as np

from .cosmology import BaseCosmo
from .decorators import float_args


class BaseLensing(BaseCosmo):

    def __init__(self, z, z_s=None, cosmo=Planck15, frame='comoving',
                 **kwargs):
        """Base lensing calculations for Profile objects

        Parameters
        ----------
        z : float or array of floats
            lens redshift
        z_s : float or array of floats, optional
            source redshift
        cosmo : astropy.cosmology.FLRW object
            cosmology

        Notes
        -----
        - If both z and z_s are arrays, their shapes must be such that
          they can be combined in numpy operations; this will not be
          corrected automatically.
        """
        try:
            z / 1
        except TypeError as e:
            msg = 'argument `z` must be a float or an array of floats'
            raise TypeError(msg) from e
        if z_s is not None:
            try:
                z_s / 1
            except TypeError as e:
                msg = 'argument `z_s` must be a float or an array of floats'
                raise TypeError(msg) from e
        super().__init__(cosmo=cosmo, frame=frame, **kwargs)
        self.z = z
        self._z_s = z_s
        self.cosmo = cosmo
        self._Dl = None
        self._Dls = None
        self._Ds = None
        self.__c = None
        self.__G = None
        return

    ### attributes ###

    @property
    def z_s(self):
        return self._z_s

    #@float_args
    @z_s.setter
    def z_s(self, z_s):
        self._z_s = z_s

    ### aliases ###

    @property
    def Dl(self):
        if self._Dl is None:
            self._Dl = self.cosmo.angular_diameter_distance(self.z).to(
                u.Mpc).value
        return self._Dl

    @property
    def Dls(self):
        if self._Dls is None:
            self._Dls = self.cosmo.angular_diameter_distance_z1z2(
                self.z, self.z_s).to(u.Mpc).value
        return self._Dls

    @property
    def Ds(self):
        if self._Ds is None:
            self._Ds = self.cosmo.angular_diameter_distance(self.z_s).to(
                u.Mpc).value
        return self._Ds

    ### methods ###

    #@float_args
    def beta(self, z_s=None):
        """max(0, Dls/Ds)"""
        if z_s is None:
            b = self.Dls / self.Ds
        else:
            b = (self.cosmo.angular_diameter_distance_z1z2(self.z, z_s).to(u.Mpc) \
                 / self.cosmo.angular_diameter_distance(z_s).to(u.Mpc)).value
        return np.max([np.zeros_like(b), b])

    #@float_args
    def sigma_crit(self, z_s=None):
        """Critical surface density, in Msun/Mpc^2"""
        A = self._c**2 / (4*np.pi*self._G)
        if self.frame == 'comoving':
            A = A / (1+self.z)**2
        return A / (self.Dl*self.beta(z_s=z_s))

    #@float_args
    def convergence(self, R, z_s=None, Roff=None, **kwargs):
        if Roff is None:
            s = self.projected(R)
        else:
            s = self.offset_projected(R, Roff)
        if z_s is None:
            z_s = self.z_s
        return s / self.sigma_crit(z_s=z_s, **kwargs)

    def offset_convergence(self, R, Roff, z_s=None, **kwargs):
        if z_s is None:
            z_s = self.z_s
        return self.offset_projected(R, Roff) \
            / self.sigma_crit(z_s=z_s, **kwargs)


class Lens(BaseCosmo):
    """Lens object class"""
    # should move some of what's in BaseLensing into here
    # then calling BaseLensing (perhaps just Lensing) should
    # create (or take) both a Lens and a Source objects
    def __init__(self, z, cosmo=Planck15):
        super(Lens, self).__init__(cosmo=cosmo)
        self.z = z
        self._chi = None
        self._Dl = None
        self._Wk = None

    @property
    def chi(self):
        if self._chi is None:
            self._chi = self.cosmo.comoving_distance(self.z)
        return self._chi

    @property
    def Dl(self):
        if self._Ds is None:
            self._Ds = self.cosmo.angular_diameter_distance(self.z)
        return self._Ds

    ### methods ###

    #@float_args
    def Dls(self, z_s):
        return self.cosmo.angular_diameter_distance_z1z2(self.z, z_s)

    #@float_args
    def sigma_crit(self, z_s, frame='comoving'):
        assert frame in ('comoving', 'physical','proper')
        A = self._c**2 / (4*np.pi*self._G)
        if frame == 'comoving':
            A = A / (1+self.z)**2
        return A / (self.Dl*self.beta(z_s=z_s))

    #@float_args
    def lensing_kernel(self, z_s):
        chi_z_s = self.cosmo.comoving_distance(z_s)
        return (3/2 * self.cosmo.Om0 * self.cosmo.H0**2 \
                * (1+z_s) / self.cosmo.H(z_s) * self.chi / self._c \
                * (chi_z_s-self.chi) / chi_z_s).to(u.Mpc/u.s).value


class Source(BaseCosmo):

    """Lensed source class"""

    def __init__(self, z_s, cosmo=Planck15):
        super(Source, self).__init__(cosmo=cosmo)
        self.z_s = z_s
        self._chi = None
        self._Ds = None
        self._Wk = None

    @property
    def chi(self):
        if self._chi is None:
            self._chi = self.cosmo.comoving_distance(self.z_s)
        return self._chi

    @property
    def Ds(self):
        if self._Ds is None:
            self._Ds = self.cosmo.angular_diameter_distance(self.z_s)
        return self._Ds

    ### methods ###

    def Dls(self, z):
        return self.cosmo.angular_diameter_distance_z1z2(z, self.z_s)

    def sigma_crit(self, z, frame='comoving'):
        assert frame in ('comoving', 'physical','proper')
        A = self._c**2 / (4*np.pi*self._G)
        if frame == 'comoving':
            A = A / (1+z)**2
        return A / (self.angular_diameter_distance(z)*self.beta(z_s=z_s))

    def lensing_kernel(self, z):
        chi_z = self.cosmo.comoving_distance(z)
        return (3/2 * self.cosmo.Om0 * self.cosmo.H0**2 \
                * (1+self.z_s) / self.cosmo.H(self.z_s) * chi_z / self._c \
                * (self.chi-chi_z) / self.chi).to(u.Mpc/u.s).value

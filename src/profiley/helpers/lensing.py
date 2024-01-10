from astropy import units as u
from astropy.cosmology import Planck15
import numpy as np

from .cosmology import BaseCosmo
from .decorators import float_args


class Lens(BaseCosmo):
    """Lens object class

    Parameters
    ----------
    z : float or np.ndarray
        redshift

    Optional parameters
    -------------------
    cosmo : astropy.cosmology.FLRW object (default ``Planck15``)
        cosmology
    frame : str
        comoving or physical frame for distances
    kwargs : dict
        additional arguments passed to ``BaseCosmo``
    """

    def __init__(self, z, cosmo=Planck15, frame="comoving", **kwargs):
        super().__init__(cosmo=cosmo, frame=frame, **kwargs)
        self.z = z
        self._chi = None
        self._Dl = None
        self._Wk = None

    @property
    def chi(self):
        """Comoving distance to ``self``"""
        if self._chi is None:
            self._chi = self.cosmo.comoving_distance(self.z)
        return self._chi

    @property
    def Dl(self):
        """Angular diameter distance from the observer to ``self``"""
        if self._Ds is None:
            self._Ds = self.cosmo.angular_diameter_distance(self.z)
        return self._Ds

    ### methods ###

    # @float_args
    def Dls(self, z_s):
        """Angular diameter distance from ``self`` to a source at redshift ``z_s``"""
        return self.cosmo.angular_diameter_distance_z1z2(self.z, z_s).to(u.Mpc).value

    # @float_args
    def Dls_over_Ds(self, z_s):
        """max(0, Dls/Ds)"""
        b = self.Dls(z_s) / self.cosmo.angular_diameter_distance(z_s).to(u.Mpc).value
        return np.max([np.zeros_like(b), b], axis=0)

    # @float_args
    def sigma_crit(self, z_s, frame="comoving"):
        """Critical surface density for a source at redshift ``z_s``"""
        assert frame in ("comoving", "physical", "proper")
        A = self._c**2 / (4 * np.pi * self._G)
        if frame == "comoving":
            A = A / (1 + self.z) ** 2
        return A / (self.Dl * self.Dls_over_Ds(z_s=z_s))

    # @float_args
    def convergence(self, R, z_s=None, Roff=None, **kwargs):
        if Roff is None:
            s = self.projected(R, **kwargs)
        else:
            s = self.offset_projected(R, Roff, **kwargs)
        if z_s is None:
            z_s = self.z_s
        return s / self.sigma_crit(z_s=z_s)

    def offset_convergence(self, R, Roff, z_s=None, **kwargs):
        if z_s is None:
            z_s = self.z_s
        return self.offset_projected(R, Roff) / self.sigma_crit(z_s=z_s, **kwargs)

    # @float_args
    def lensing_kernel(self, z_s):
        """Lensing kernel for a source at redshift ``z_s``"""
        chi_z_s = self.cosmo.comoving_distance(z_s)
        return (
            (
                3
                / 2
                * self.cosmo.Om0
                * self.cosmo.H0**2
                * (1 + z_s)
                / self.cosmo.H(z_s)
                * self.chi
                / self._c
                * (chi_z_s - self.chi)
                / chi_z_s
            )
            .to(u.Mpc / u.s)
            .value
        )


class Source(BaseCosmo):

    """Lensed source class

    Parameters
    ----------
    z_s : float or np.ndarray
        source redshift

    Optional parameters
    -------------------
    cosmo : astropy.cosmology.FLRW object
        cosmology
    """

    def __init__(self, z_s, cosmo=Planck15):
        super(Source, self).__init__(cosmo=cosmo)
        self.z_s = z_s
        self._chi = None
        self._Ds = None
        self._Wk = None

    @property
    def chi(self):
        """Comoving distance to ``self``"""
        if self._chi is None:
            self._chi = self.cosmo.comoving_distance(self.z_s)
        return self._chi

    @property
    def Ds(self):
        """Angular diameter distance from the observer to ``self``"""
        if self._Ds is None:
            self._Ds = self.cosmo.angular_diameter_distance(self.z_s)
        return self._Ds

    ### methods ###

    def Dls(self, z_lens):
        """Angular diameter distance from a lens at redshift ``z_lens`` to ``self``"""
        return self.cosmo.angular_diameter_distance_z1z2(z_lens, self.z_s)

    def sigma_crit(self, z_lens, frame="comoving"):
        """Critical surface density for a lens at redshift ``z_lens``"""
        assert frame in ("comoving", "physical", "proper")
        A = self._c**2 / (4 * np.pi * self._G)
        if frame == "comoving":
            A = A / (1 + z_lens) ** 2
        return A / (
            self.angular_diameter_distance(z_lens) * self.Dls_over_Ds(z_s=self.z_s)
        )

    def lensing_kernel(self, z_lens):
        """Lensing kernel for a lens at redshift ``z_lens``"""
        chi_z = self.cosmo.comoving_distance(z_lens)
        return (
            (
                3
                / 2
                * self.cosmo.Om0
                * self.cosmo.H0**2
                * (1 + self.z_s)
                / self.cosmo.H(self.z_s)
                * chi_z
                / self._c
                * (self.chi - chi_z)
                / self.chi
            )
            .to(u.Mpc / u.s)
            .value
        )

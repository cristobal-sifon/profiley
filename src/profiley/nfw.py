import astropy
from astropy import units as u
from astropy.cosmology import Planck15
import numpy as np
from scipy.special import sici
import warnings

from .core import Profile
from .helpers.decorators import array, inMpc
from .helpers.lensing import BaseLensing
from .helpers.spherical import mass_from_radius, radius_from_mass


class BaseNFW(Profile):

    def __init__(self, mass, c, z, overdensity: float=500,
                 background: str='c', cosmo: astropy.cosmology.FLRW=Planck15,
                 frame: str='comoving', numeric_kwargs={}):
        # check overdensity and background
        self._assert_background(background)
        if overdensity <= 0:
            raise ValueError(
                f'overdensity must be positive; received {overdensity}')

        if isinstance(mass, u.Quantity):
            mass = mass.to(u.Msun).value
        if not np.iterable(mass):
            mass = np.array([mass])
        self._set_shape(mass*c*z)
        self.mass = mass
        self.c = c
        self.background = background
        self.overdensity = overdensity
        # additional NFW convenience attributes
        self._delta_c = None
        self._rs = None
        self._radius = None
        self._sigma_s = None
        super().__init__(
            z, cosmo=cosmo, frame=frame, **numeric_kwargs)

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

    def mdelta(self, overdensity: float, background: str='c', err: float=1e-3,
               n_guess_rng: int=1000, max_iter: int=50):
        """Calculate mass at any overdensity from the original mass
        definition

        Parameters
        ----------
        overdensity : float
            overdensity at which the mass should be calculated

        Optional parameters
        -------------------
        background : one of ('c','m')
            background density as a reference for ``overdensity``.
            WARNING: currently only the same background as used in
            defining this object is implemented
        err: float
            maximum error on ``delta_c`` that can be tolerated to claim
            convergence
        n_guess_rng : int, optional
            how many samples of ``c`` to obtain in each iteration. See
            Notes below.
        max_iter : int, optional
            maximum number of iterations

        Returns
        -------
        mdelta, cdelta : ndarray, shape ``self.c.shape``
            mass and concentrations calculated at the requested
            overdensity
        """
        self._assert_background(background)
        if overdensity == self.overdensity \
                and background == self.background:
            return self.mass
        # do we need to convert the background density?
        if background == self.background:
            bgfactor = 1
        else:
            # this is m to c
            bgfactor = self.mean_density / self.critical_density
            # reciprocal for c to m
            if background == 'm':
                bgfactor = 1 / bgfactor
        # to handle arbitrary dimensions
        c_shape = self.c.shape
        self_c = self.c.reshape(-1)
        self_delta_c = self.delta_c.reshape(-1)
        # I found that this guess is good to within 20% typically
        c_guess = (self.overdensity/overdensity)**0.5 * self_c
        c_rng = np.linspace(0.5*c_guess, 1.5*c_guess, n_guess_rng)
        delta_c_rng = self._f_delta_c(c_rng, overdensity)
        delta_c_diff = np.abs(delta_c_rng/self_delta_c - 1)
        argmins = np.argmin(delta_c_diff, axis=0)
        # without the copy I am not allowed to write into this array
        cdelta = np.diagonal(c_rng[argmins], axis1=-2, axis2=-1).copy()
        # delta_c_err are the minima of delta_c_diff
        delta_c_err = np.diagonal(
            delta_c_diff[argmins], axis1=-2, axis2=-1).copy()
        i = 0
        while np.any(delta_c_err > err):
            k = (delta_c_err > err)
            # if our best guess is at the edge then we don't want to
            # shrink the search range, but if we don't shrink it
            # progressively otherwise then we'll never improve our answer
            if np.any(argmins == 0) or np.any(argmins == n_guess_rng-1):
                width = 0.1
            else:
                width = 0.1 / (i+1)
            c_rng = np.linspace(
                (1-width)*cdelta[k], (1+width)*cdelta[k], n_guess_rng)
            delta_c_diff = np.abs(
                self._f_delta_c(c_rng, overdensity)/self_delta_c[k]-1)
            argmins = np.argmin(delta_c_diff, axis=0)
            delta_c_err[k] = np.diagonal(
                delta_c_diff[argmins], axis1=-2, axis2=-1)
            if (delta_c_err[k] <= err).sum():
                j = (delta_c_err <= err)
                cdelta[k & j] = np.diagonal(
                    c_rng[argmins], axis1=-2, axis2=-1)[j[k]]
            i += 1
            if i == max_iter:
                warn = f'Did not achieve convergence after {max_iter}' \
                       f' iterations; error on delta_c =' \
                       f' {delta_c_err[k].mean():.2e} +/-' \
                       f' {delta_c_err[k].std():.2e}' \
                       f' (max err = {delta_c_err[k].max():.2e})'
                warnings.warn(warn)
                break
        # back to the original shape, also correcting for different
        # background, if applicable
        cdelta = bgfactor**(1/3) * cdelta.reshape(c_shape)
        # calculate mass from the relation between mass, c, and overdensity
        mfactor = (overdensity/self.overdensity) * (cdelta/self.c)**3
        return mfactor*self.mass, cdelta


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

    def __init__(self, mass, c, z, alpha, beta, gamma, overdensity=500,
                 background='c', frame='comoving', cosmo=Planck15, **kwargs):
        super().__init__(
            mass, c, z, overdensity=overdensity, background=background,
            frame=frame, cosmo=cosmo, **kwargs)
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

    Parameters
    ----------
    mass, c, z : float or np.ndarray
        mass, concentration, and redshift defining the NFW profile.
        Their shapes are arbitrary but they must be such that they can
        be multiplied together as they come

    Optional parameters
    -------------------
    overdensity : float
        overdensity with respect to the background density
    background : str
        'c' (critical) or 'm' (mean) background density
    """

    def __init__(self, mass, c, z, overdensity=500, background='c',
                 frame='comoving', cosmo=Planck15, **kwargs):
        super(NFW, self).__init__(
            mass, c, z, overdensity=overdensity, background=background,
            frame=frame, cosmo=cosmo, **kwargs)

    def __repr__(self):
        msg = f'NFW profile object containing {np.prod(self._shape)}' \
              f' profiles. shape: {self._shape}'
        od_msg = f'overdensity: {self.overdensity}{self.background}'
        if np.iterable(self.mass) and self.mass.min() < self.mass.max():
            mass_msg = 'log10 mass/Msun range =' \
                       f' {np.log10(self.mass.min()):.2f}' \
                       f'-{np.log10(self.mass.max()):.2f}'
        else:
            mass_msg = 'log10 mass/Msun =' \
                       f' {np.log10(np.unique(self.mass)[0]):.2f}'
        if np.iterable(self.c) and self.c.min() < self.c.max():
            c_msg = 'concentration range =' \
                    f' {self.c.min():.2f}-{self.c.max():.2f}'
        else:
            c_msg = f'concentration = {np.unique(self.c)[0]:.2f}'
        if np.iterable(self.z) and self.z.min() < self.z.max():
            z_msg = f'redshift range = {self.z.min():.2f}-{self.z.max():.2f}'
        else:
            z_msg = f'redshift = {np.unique(self.z)[0]:.2f}'
        #return f'{msg}\n  {mass_msg}\n  {c_msg}\n  {z_msg}'
        return '\n  '.join([msg, od_msg, mass_msg, c_msg, z_msg])

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
        s[j] = (1 - 2*np.arctanh(((1-x[j]) / (1+x[j]))**0.5)
                     / (1-x[j]**2)**0.5) \
               / (x[j]**2 - 1)
        j = x > 1
        s[j] = (1 - 2*np.arctan(((x[j]-1) / (1+x[j]))**0.5) \
                    / (x[j]**2-1)**0.5) \
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

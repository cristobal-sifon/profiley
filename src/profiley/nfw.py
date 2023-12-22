import astropy
from astropy import units as u
from astropy.cosmology import Planck15
import numpy as np
from scipy.integrate import simps
from scipy.special import sici
import warnings

from .core import Profile
from .helpers.decorators import array, inMpc
from .helpers.lensing import BaseLensing
from .helpers.spherical import mass_from_radius, radius_from_mass


class BaseNFW(Profile):
    """Base class for NFW-like density profiles"""

    def __init__(
        self,
        mass,
        c,
        z,
        overdensity: float = 500,
        background: str = "c",
        cosmo: astropy.cosmology.FLRW = Planck15,
        frame: str = "comoving",
        **numeric_kwargs,
    ):
        if isinstance(mass, u.Quantity):
            mass = mass.to(u.Msun).value
        if not np.iterable(mass):
            mass = np.array([mass])
        if not np.iterable(c):
            c = np.array([c])
        self.mass = mass
        self.c = c
        # additional NFW convenience attributes
        self._delta_c = None
        self._rs = None
        self._radius = None
        self._sigma_s = None
        super().__init__(
            z,
            overdensity=overdensity,
            cosmo=cosmo,
            background=background,
            frame=frame,
            **numeric_kwargs,
        )

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
            self._radius = radius_from_mass(self.mass, self.overdensity, self.rho_bg)
        return self._radius

    @property
    def sigma_s(self):
        if self._sigma_s is None:
            self._sigma_s = self.rs * self.delta_c * self.rho_bg
        return self._sigma_s

    ### hidden methods ###

    def _f_delta_c(self, c, overdensity):
        return (overdensity * c**3 / 3) / (np.log(1 + c) - c / (1 + c))

    def density(self, *args, **kwargs):
        """Alias for ``self.profile``"""
        return self.profile(*args, **kwargs)


class GNFW(BaseNFW):
    """Generalized NFW profile

    Density profile:

    .. math::

        \\rho(r) = \\frac{\\delta_\\mathrm{c}\\rho_\\mathrm{bg}}
                    {(r/r_\\mathrm{s})^\\gamma
                        \\left[1+(r/r_\\mathrm{s})^\\alpha\\right]^(\\beta-\\gamma)/\\alpha

    Parameters
    ----------
    mass, c, z : float or np.ndarray
        mass, concentration, and redshift defining the NFW profile.
        Their shapes are arbitrary but they must be such that they can
        be multiplied together as they come

    Optional parameters
    -------------------
    alpha : float or np.ndarray
        sharpness of the transition between inner and outer slope
        around the scale radius. A larger value produces a sharper
        transition.
    beta : float or np.ndarray
        slope of the density profile at large radii
    gamma : float or np.ndarray
        slope of the density profile at small radii

    For additional optional parameters see ``NFW``

    Notes
    -----
    - A common but more restriced GNFW profile can be recovered by setting
        alpha=1, beta=3 and varying gamma alone
    - The default parameters (alpha=1, beta=3, gamma=1) correspond to the
        regular NFW profile
    """

    def __init__(
        self,
        mass,
        c,
        z,
        alpha=1,
        beta=3,
        gamma=1,
        overdensity=500,
        background="c",
        frame="comoving",
        cosmo=Planck15,
        **kwargs,
    ):
        self._set_shape(mass * c * z * alpha * beta * gamma)
        super().__init__(
            mass,
            c,
            z,
            overdensity=overdensity,
            background=background,
            frame=frame,
            cosmo=cosmo,
            **kwargs,
        )
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    ### main methods ###

    @inMpc
    @array
    def profile(self, r):
        exp = (self.beta - self.gamma) / self.alpha
        return (
            self.delta_c
            * self.rho_bg
            / ((r / self.rs) ** self.gamma * (1 + (r / self.rs) ** self.alpha) ** exp)
        )


class NFW(BaseNFW):
    """Navarro-Frenk-White profile (Navarro et al. 1995)

    Density profile:

    .. math::

        \\rho(r) = \\frac{\\delta_\\mathrm{c}\\rho_\\mathrm{bg}}
                    {(r/r_\\mathrm{s})(1+r/r_\\mathrm{s})^2}

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
    cosmo : Astropy.cosmology.FLRW
        cosmology object
    """

    def __init__(
        self,
        mass,
        c,
        z,
        overdensity=500,
        background="c",
        frame="comoving",
        cosmo=Planck15,
        **kwargs,
    ):
        self._set_shape(mass * c * z)
        super(NFW, self).__init__(
            mass,
            c,
            z,
            overdensity=overdensity,
            background=background,
            frame=frame,
            cosmo=cosmo,
            **kwargs,
        )

    def __repr__(self):
        msg = (
            f"NFW profile object containing {np.prod(self._shape)}"
            f" profiles\nshape: {self._shape}"
        )
        od_msg = f"overdensity: {self.overdensity}{self.background}"
        if np.iterable(self.mass) and self.mass.min() < self.mass.max():
            mass_msg = (
                "log10 mass/Msun range ="
                f" {np.log10(self.mass.min()):.2f}"
                f"-{np.log10(self.mass.max()):.2f}"
            )
        else:
            mass_msg = "log10 mass/Msun =" f" {np.log10(np.unique(self.mass)[0]):.2f}"
        if np.iterable(self.c) and self.c.min() < self.c.max():
            c_msg = "concentration range =" f" {self.c.min():.2f}-{self.c.max():.2f}"
        else:
            c_msg = f"concentration = {np.unique(self.c)[0]:.2f}"
        if np.iterable(self.z) and self.z.min() < self.z.max():
            z_msg = f"redshift range = {self.z.min():.2f}-{self.z.max():.2f}"
        else:
            z_msg = f"redshift = {np.unique(self.z)[0]:.2f}"
        return "\n  ".join([msg, od_msg, mass_msg, c_msg, z_msg])

    ### main methods ###

    @inMpc
    @array
    def profile(self, r):
        """Three-dimensional density profile"""
        return self.delta_c * self.rho_bg / (r / self.rs * (1 + r / self.rs) ** 2)

    @inMpc
    @array
    def projected(self, R, **kwargs):
        """Analytical projected NFW at distance(s) R"""
        x = R / self.rs
        s = np.ones(x.shape) / 3
        s[x == 0] = 0
        j = (x > 0) & (x < 1)
        s[j] = (
            1
            - 2 * np.arctanh(((1 - x[j]) / (1 + x[j])) ** 0.5) / (1 - x[j] ** 2) ** 0.5
        ) / (x[j] ** 2 - 1)
        j = x > 1
        s[j] = (
            1 - 2 * np.arctan(((x[j] - 1) / (1 + x[j])) ** 0.5) / (x[j] ** 2 - 1) ** 0.5
        ) / (x[j] ** 2 - 1)
        return 2 * self.sigma_s * s

    @inMpc
    @array
    def projected_cumulative(self, R, **kwargs):
        """Analytical cumulative projected NFW profile"""
        x = R / self.rs
        s = np.ones(x.shape) + np.log(0.5)
        s[x == 0] = 0
        j = (x > 0) & (x < 1)
        s[j] = (
            np.log(0.5 * x[j])
            + 2 * np.arctanh(((1 - x[j]) / (1 + x[j])) ** 0.5) / (1 - x[j] ** 2) ** 0.5
        ) / x[j] ** 2
        j = x > 1
        s[j] = (
            2 * np.arctan(((x[j] - 1) / (1 + x[j])) ** 0.5) / (x[j] ** 2 - 1) ** 0.5
            + np.log(0.5 * x[j])
        ) / x[j] ** 2
        return 4 * self.sigma_s * s

    @array
    def fourier(self, k):
        """Fourier transform"""
        ki = k * self.rs
        bs, bc = sici(ki)
        asi, ac = sici((1 + self.c) * ki)
        return (
            4
            * np.pi
            * self.rho_bg
            * self.delta_c
            * self.rs**3
            / self.mass
            * (
                np.sin(ki) * (asi - bs)
                - (np.sin(self.c * ki) / ((1 + self.c) * ki))
                + np.cos(ki) * (ac - bc)
            )
        )

    # @array
    # def enclosed_mass(self, r):
    #     x = r / self.rs
    #     return (
    #         4
    #         * np.pi
    #         * self.rs**3
    #         * self.delta_c
    #         * self.rho_bg
    #         * (np.log(1 + x) - x / (1 + x))
    #     )

    ### more methods ###

    def mdelta(
        self,
        overdensity: float,
        background: str = "c",
        *,
        rtol: float = 0.001,
        trial_range: float = 0.1,
        trial_size: int = 99,
        max_trials: int = 100,
        log_rmin: float = -10,
        integral_samples: int = 1000,
    ):
        """This works at least for 1d but can reorganize quite a bit.

        To do:
            - Only calculate for bad entries, i.e., do not continue calculating
                entries for which we already found a solution - DONE
            - Remove unnecesary code: with the new while condition there's no
                need to repeat the calculation of enclosed and so on, which means
                by sorting what's in the loop we can remove most of what's outside - DONE
            - Generalize to more than one dimension (see original mdelta)
            - This function should be rdelta, and then mdelta should be a simple thing
                calculating rdelta from here and then just get the mass
        """
        # first integrate out to self.radius to see when we need a larger
        # or smaller radius
        rho_bg = self.get_rho_bg(background)
        target = overdensity * rho_bg
        # print("target =", target)
        Rdelta = np.zeros(self.radius.shape)
        # print("Rdelta =", Rdelta)
        # need this to be able to update Rdelta instead of just getting a view
        rng = np.arange(Rdelta.size, dtype=int)
        # will need to generalize this for when self.radius has Ndim > 1
        R = np.logspace(-1, 1, trial_size)[:, None] * self.radius
        i = 0
        while np.any(Rdelta == 0):
            # print("i =", i)
            remaining = Rdelta == 0
            # print("remaining =", remaining)
            # print("R =", R, R.shape)
            # should probably mask R in here too so we reduce the number of
            # integrals
            enclosed = self.enclosed(R, log_rmin, integral_samples)[:, remaining]
            err = np.abs(enclosed - target[remaining]) / target[remaining]
            # print("enclosed =", enclosed, enclosed.shape)
            # print("err =", err, err.shape)
            # all of this should happen only with those that have err > rtol
            jmin = np.argmin(err, axis=0)
            # print("jmin =", jmin, jmin.shape)
            minerr = np.min(err, axis=0)
            # print("minerr =", minerr)
            minmask = err == minerr
            # print("minmask =", minmask, minmask.shape)
            # a dirty trick - just applying the mask sorts with the last
            # dimension instead of the first one
            Rbest = np.sum(R[:, remaining] * minmask, axis=0)
            # print("Rbest =", Rbest)
            good = minerr < rtol
            # print("good =", good)
            # print("Rbest[good] =", Rbest[good])
            Rdelta[rng[remaining][good]] = Rbest[good]
            # print("Rdelta =", Rdelta)
            # print()
            # ideally the range here would depend on the range in the previous iteration
            # but I don't think that will be easy. However if it doesn't then this thing
            # can get stuck where the resolution provided by trial_size and trial_range
            # simply aren't good enough to achieve rtol.
            R[:, rng[remaining][~good]] = (
                np.linspace(1 - trial_range, 1 + trial_range, trial_size)[:, None]
                * Rbest[~good]
            )
            i += 1
            if i == max_trials:
                wrn = "Could not converge to within rtol after {max_trials} iterations."
                warnings.warn(wrn, RuntimeWarning)
                break
        mdelta = 4 * np.pi * Rdelta**3 * target / 3
        return mdelta, Rdelta

    def _mdelta(
        self,
        overdensity: float,
        background: str = "c",
        err: float = 1e-3,
        n_guess_rng: int = 1000,
        max_iter: int = 50,
    ):
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
        if overdensity == self.overdensity and background == self.background:
            return self.mass, self.c
        # do we need to convert the background density?
        if background == self.background:
            bgfactor = 1
        else:
            # this is m to c
            bgfactor = self.critical_density / self.mean_density
            # reciprocal for c to m
            if background == "m":
                bgfactor = 1 / bgfactor
        if overdensity == self.overdensity:
            mass = self.mass * bgfactor
            c = (mass / self.mass * bgfactor) ** (1 / 3) * self.c
            return mass, c
        # to handle arbitrary dimensions
        c_shape = self.c.shape
        self_c = self.c.reshape(-1)
        self_delta_c = self.delta_c.reshape(-1)
        # I found that this guess is good to within 20% typically
        c_guess = (self.overdensity / overdensity) ** 0.5 * self_c
        # so we create a +/-50% search range to be sure
        c_rng = np.linspace(0.5 * c_guess, 1.5 * c_guess, n_guess_rng)
        delta_c_rng = self._f_delta_c(c_rng, overdensity)
        delta_c_diff = np.abs(delta_c_rng / self_delta_c - 1)
        argmins = np.argmin(delta_c_diff, axis=0)
        # without the copy I am not allowed to write into this array
        cdelta = np.diagonal(c_rng[argmins], axis1=-2, axis2=-1).copy()
        # delta_c_err are the minima of delta_c_diff
        delta_c_err = np.diagonal(delta_c_diff[argmins], axis1=-2, axis2=-1).copy()
        i = 0
        while np.any(delta_c_err > err):
            # k = delta_c_err > err
            k = np.argmin(delta_c_err)
            # if our best guess is at the edge then we don't want to
            # shrink the search range, but if we don't shrink it
            # progressively otherwise then we'll never improve our answer
            # width=0.1 refers to a 10% search range
            if np.any(argmins == 0) or np.any(argmins == n_guess_rng - 1):
                width = 0.1
            else:
                width = 0.1 / (i + 1)
            c_rng = np.linspace(
                (1 - width) * cdelta[k], (1 + width) * cdelta[k], n_guess_rng
            )
            delta_c_diff = np.abs(
                self._f_delta_c(c_rng, overdensity) / self_delta_c[k] - 1
            )
            argmins = np.argmin(delta_c_diff, axis=0)
            delta_c_err[k] = np.diagonal(delta_c_diff[argmins], axis1=-2, axis2=-1)
            if (delta_c_err[k] <= err).sum():
                j = delta_c_err <= err
                cdelta[k & j] = np.diagonal(c_rng[argmins], axis1=-2, axis2=-1)[j[k]]
            i += 1
            if i == max_iter:
                warn = (
                    f"Did not achieve convergence after {max_iter}"
                    f" iterations; error on delta_c ="
                    f" {delta_c_err[k].mean():.2e} +/-"
                    f" {delta_c_err[k].std():.2e}"
                    f" (max err = {delta_c_err[k].max():.2e})"
                )
                warnings.warn(warn)
                break
        # back to the original shape, also correcting for different
        # background, if applicable
        cdelta = bgfactor ** (1 / 3) * cdelta.reshape(c_shape)
        # calculate mass from the relation between mass, c, and overdensity
        mfactor = (overdensity / self.overdensity) * (cdelta / self.c) ** 3
        return mfactor * self.mass, cdelta

    def rdelta(self, overdensity: float, background: str = "c", **kwargs):
        """Calculate the radius enclosing a given overdensity

        Parameters
        ----------
        overdensity : float
            overdensity at which the radius should be calculated

        Optional parameters
        -------------------
        background : one of ('c','m')
            background density as a reference for ``overdensity``.
        kwargs : dict
            passed to ``self.mdelta``

        Returns
        -------
        rdelta : ndarray, shape ``self.c.shape``
            radius enclosing the requested overdensity
        """
        mdelta, cdelta = self.mdelta(overdensity, background, **kwargs)
        rho = self.mean_density if background == "m" else self.critical_density
        return (mdelta / (overdensity * rho * 4 * np.pi / 3)) ** (1 / 3)


class TNFW(BaseNFW):
    """Truncated NFW profile

    The density profile is given by

    .. math::

        \\rho(r) = \\frac{\\delta_\\mathrm{c}\\rho_\mathrm{bg}}{x(1+x)^2}
                    \\left(\\frac{\\tau^2}{\\tau^2+x^2}\\right)^{\\mathrm{\\eta}}

    with

    .. math::

        x = r/r_\\mathrm{s}

    and

    .. math::

        \\tau = r_\\mathrm{t}/r_\\mathrm{s}

    the truncation radius in units of the scale radius.

    Analytical expressions for projected profiles have been derived by
    Baltz, Marshall & Oguri for the cases of ``eta={1,2}``. Here the
    projected profiles are calculated numerically.

    Parameters
    ----------
    mass, c, z : float or np.ndarray
        mass, concentration, and redshift defining the NFW profile.
        Their shapes are arbitrary but they must be such that they can
        be multiplied together as they come

    Optional parameters
    -------------------
    tau : float or np.ndarray
        truncation radius, in units of the scale radius
    eta : float or np.ndarray
        exponent of the decay beyond the truncation radius. Set to zero
        to recover regular NFW

    For additional optional parameters see ``NFW``
    """

    def __init__(self, mass, c, z, tau=1, eta=1, **kwargs):
        self._set_shape(mass * c * z * tau * eta)
        super().__init__(mass, c, z, **kwargs)
        self.tau = tau
        self.eta = eta

    ### main methods ###

    @inMpc
    @array
    def profile(self, r):
        x = r / self.rs
        return (
            self.delta_c
            * self.rho_bg
            / (x * (1 + x) ** 2)
            * (self.tau**2 / (self.tau**2 + x**2)) ** self.eta
        )


class Hernquist(Profile):
    """Hernquist (1990) profile.

    .. math::

        \\rho(r) = \\frac{\\delta_\\mathrm{c}\\rho_\\mathrm{bg}}
                    {(r/r_\\mathrm{s})\\left(1+r/r_\\mathrm{s}\\right)^3}

    """

    # if this is not an NFW-like then it does not have delta_c or any of those

    def __init__(self, mass, c, z, **kwargs):
        self._set_shape(mass * c * z)
        super().__init__(z, **kwargs)
        self.mass = mass
        self.c = c

    def profile(self, r: np.ndarray):
        return

    def mass_enclosed(self, r: np.ndarray, **kwargs):
        """The mass enclosed within a radius :math:`r` is

        .. math::

            M(r) = 2\\pi\\delta_\\mathrm{c}\\rho_\\mathrm{bg}r_\\mathrm{s}^3\\frac{r^2}{(r+r_\\mathrm{s})^2}
        """
        return (
            2
            * np.pi
            * self.rho_bg
            * self.delta_c
            * self.rs**3
            * r**2
            / (r + self.rs) ** 2
        )


class WebskyNFW(BaseNFW):
    """Modified NFW profile as adopted in the Websky simulations (Stein et al. 2020)

    The density profile is given by

    .. math::

       \\rho(r) = \\begin{cases}
            \\rho_\\mathrm{NFW}(r) & r < r_\\mathrm{200m} \\\\
            \\rho_\\mathrm{NFW}(r)\\left(\\frac{r}{r_\\mathrm{200m}}\\right)^{-\\alpha}  & r_\\mathrm{200m} < r < 2r_\\mathrm{200m} \\\\
            0 & r > r_\\mathrm{200m}
        \\end{cases}

    By default this profile is initiated with ``overdensity=200`` and
    ``background="m"``. If different parameters are specified this means
    masses and concentrations are defined as such, but the profile is
    still defined with respect to r200m.

    The Websky simulations adopt a concentration c200m=7 fixed at all
    masses and redshifts, but the concentration does not have a default
    value in this implementation (and will be defined with respect to
    the overdensity used when initializing the object).
    """

    def __init__(self, mass, c, z, alpha=2, **kwargs):
        self._set_shape(mass * c * z * alpha)
        super().__init__(mass, c, z, **kwargs)
        self.alpha = alpha

    ### main methods ###

    @inMpc
    @array
    def profile(self, r):
        """Three-dimensional density profile"""
        x = r / self.rs
        rho_200m = self.mean_density
        if self.overdensity == 200 and self.background == "m":
            r200m = self.radius
            delta_c = self.delta_c
        else:
            r200m = self.rdelta(200, "m")
            c200m = r200m / self.rs
            delta_c = self._f_delta_c(c200m, 200)
        return (
            rho_200m
            * delta_c
            / (x * (1 + x) ** 2)
            * (r / r200m) ** (-self.alpha * ((r200m < r) & (r < 2 * r200m)))
            * (r < 2 * r200m)
        )

from astropy import units as u
from astropy.cosmology import Planck15
import numpy as np
from scipy.integrate import cumtrapz, quad, simps, trapz
from time import time
import warnings

try:
    import pyccl as ccl

    has_ccl = True
except ImportError:
    has_ccl = False

from .helpers.cosmology import BaseCosmo
from .helpers.decorators import array, deprecated, inMpc
from .helpers.lensing import Lens


warnings.simplefilter("once", category=DeprecationWarning)


class Profile(Lens):
    """Profile object

    All profiles should inherit from ``Profile``

    Defining your own profile is very simple. As an example, let's define a simple power-law profile with two free parameters, the normalization and the slope:

    .. math ::

        f(r | a, b) = a \\, r^b

    The definition would look like this:

    .. code-block ::

        class PowerLaw(Profile):

            def __init__(self, norm, slope, **kwargs):
                self._set_shape(norm*slope)
                super().__init__(**kwargs)
                self.norm = norm
                self.slope = slope

            @array
            def profile(self, r):
                return self.norm * r**self.slope

    That's it! The ``__init__()`` method needs only two lines of code (in addition to attribute definitions). The last line is necessary to allow ``profiley`` to automatically handle arbitrary shapes, through the definition of a ``shape`` attribute. Note that ``_set_shape`` takes only one argument (besides ``self``) - the *product* of the class arguments. That is, if the arguments are arrays, their dimensions must be such that a product can be carried out without any manipulation.

    Profile projections
    -------------------

    If the projection of this profile is analytical, any or all of the following methods can also be specified: ::

        projected(self, R)
        projected_cumulative(self, R)
        projected_excess(self, R)
        offset_profile(self, R, Roff)
        offset_projected(self, R, Roff)
        offset_projected_cumulative(self, R, Roff)
        offset_projected_excess(self, R, Roff)

    If it does not have analytical expressions, these methods will also exist, but they will be calculated numerically, so they may be somewhat slower depending on the precision required.

    Cosmology
    ---------

    All ``Profile`` objects contain all cosmological information with which they have been initialized through the ``self.cosmo`` attribute, which can be any ``astropy.cosmology.FLRW`` object.

    """

    def __init__(
        self,
        z=0,
        overdensity=500,
        los_loglimit=6,
        nsamples_los=200,
        resampling=20,
        logleft=-10,
        left_samples=100,
        **kwargs,
    ):
        """Initialize a profile object

        Optional arguments
        ------------------
        z : float or ndarray of floats
            redshift
        overdensity : int or float
            overdensity with respect to the background (does not apply to all Profile children; see each specific class for details)

        """
        super().__init__(z, **kwargs)
        # check overdensity
        self._assert_overdensity(overdensity)
        self.overdensity = overdensity
        # for numerical integration -- perhaps these could be passed
        # in a single dictionary
        self.los_loglimit = los_loglimit
        self.nsamples_los = nsamples_los
        self.resampling = resampling
        self.logleft = logleft
        self.left_samples = left_samples
        # empty init
        self.__dimensions = None

    @property
    def _one(self):
        if self.__one is None:
            self.__one = u.dimensionless_unscaled
        return self.__one

    @property
    def _dimensions(self):
        if self.__dimensions is None:
            self.__dimensions = tuple([1] * len(self.shape))
        return self.__dimensions

    @property
    def shape(self):
        if not hasattr(self, "_shape"):
            msg = (
                "attribute shape does not exist. Please make sure"
                " to call ``self._set_shape`` in ``__init__``"
            )
            raise AttributeError(msg)
        return self._shape

    ### private methods ###

    def _assert_overdensity(self, overdensity):
        assert not np.iterable(overdensity), "parameter overdensity must be a scalar"
        try:
            overdensity / 1
        except TypeError as err:
            raise TypeError("parameter overdensity must be a number") from err
        if overdensity <= 0:
            raise ValueError(f"overdensity must be positive; received {overdensity}")
        return

    def _expand_dims(self, x):
        """Add the appropriate number of dimensions to the end of x"""
        return np.expand_dims(x, tuple(range(-len(self.shape), 0)))

    def _set_shape(self, args_product):
        if hasattr(self, "shape"):
            msg = "attribute shape already set, cannot be overwritten"
            raise ValueError(msg)
        if np.iterable(args_product):
            self._shape = args_product.shape
        else:
            self._shape = (0,)

    ### methods ###

    def mdelta(self, overdensity: float, background: str, **kwargs):
        """Calculate the mass out to a radius containing a specified overdensity

        See ``rdelta`` for help with parameters

        Returns
        -------
        mdelta : np.ndarray
            masses within spheres of radius ``rdelta``
        rdelta : np.ndarray
            radii enclosing the specified overdensity
        density_err : np.ndarray, optional
            fractional error in the density at the returned radii. Only returned if ``return_errors==True``
        """
        r = self.rdelta(overdensity, background, **kwargs)
        if kwargs.get("return_errors", False):
            r, err = r
        m = 4 * np.pi * r**3 * overdensity * self.get_rho_bg(background) / 3
        if kwargs.get("return_errors", False):
            return m, r, err
        return m, r

    def rdelta(
        self,
        overdensity: float,
        background: str,
        *,
        rtol: float = 0.001,
        trial_range: float = 0.1,
        trial_size: int = 20,
        max_trials: int = 100,
        log_rmin: float = -10,
        integral_samples: int = 1000,
        return_errors: bool = False,
    ):
        """
        Calculate the radius within which the density equals a specified overdensity

        Parameters
        ----------
        overdensity : float
            The desired overdensity
        background : 'c' or 'm'
            Whether to use the critical or mean density as a reference
        rtol : float, optional
            The relative tolerance for convergence. The default is 0.001
        trial_range : float, optional
            The fractional range to explore around the best radius in each iteration. Use a smaller number for smaller tolerances. The default is 0.1, corresponding to a 10% search in each iteration
        trial_size : int, optional
            The number of radii to try in each iteration. The default is 20
        max_trials : int, optional
            The maximum number of trials to perform. If convergence is not reached a ``RuntimeWarning`` is raised. The default is 100
        log_rmin : float, optional
            The minimum log10 radius to use for profile integration. See ``profile`` for details. The default is -10
        integral_samples : int, optional
            number of samples to use in the integration. See ``profile`` for details. The default is 1000
        return_errors : bool, optional
            whether to return fractional errors in the density at the returned radii. Default is False

        Returns
        -------
        rdelta : np.ndarray
            The radius containing the specified overdenstiy
        density_err : np.ndarray, optional
            fractional error in the density at the returned radii. Only returned if ``return_errors==True``

        Each returned array has a shape equal to ``self.shape``.

        """
        if overdensity == self.overdensity and background == self.background:
            if return_errors:
                return self.mass, self.c, np.zeros(self.shape)
            return self.mass, self.c
        # first integrate out to self.radius to see when we need a larger
        # or smaller radius
        rho_bg = self.get_rho_bg(background) * np.ones(self.shape)
        target = overdensity * rho_bg
        Rdelta = np.zeros(self.radius.shape)
        # need this to be able to update Rdelta instead of just getting a view
        rng = np.arange(Rdelta.size, dtype=int)
        dims_to_expand = tuple(range(1, len(self.shape) + 1))
        R = np.expand_dims(np.logspace(-1, 1, trial_size), dims_to_expand) * self.radius
        i = 0
        while np.any(Rdelta == 0):
            remaining = Rdelta == 0
            # should probably mask R in here too so we reduce the number of
            # integrals
            cumul = self.cumulative(
                R, log_rmin=log_rmin, integral_samples=integral_samples
            )
            err = np.abs(cumul - target) / target
            # all of this should happen only with those that have err > rtol
            minerr = np.min(err, axis=0)
            minmask = err == minerr
            # set everything except the best radii to zero
            Rbest = np.sum(R * minmask, axis=0)
            good = minerr < rtol
            Rdelta = Rdelta + Rbest * remaining * good
            # update R to the closest match found so far
            R = R * good + Rbest * (~good) * np.expand_dims(
                np.linspace(1 - trial_range, 1 + trial_range, trial_size),
                dims_to_expand,
            )
            i += 1
            if i == max_trials:
                # this to return the best radius we found, even if it does not
                # meet rtol
                Rdelta = Rdelta + Rbest * remaining
                wrn = (
                    f"Could not converge to within rtol after {max_trials} iterations."
                )
                warnings.warn(wrn, RuntimeWarning)
                break
        Rdelta = Rdelta.reshape(self.shape)
        if return_errors:
            return Rdelta, minerr
        return Rdelta

    @inMpc
    @array
    def cumulative(
        self, r: np.ndarray, *, log_rmin: float = -10, integral_samples: int = 1000
    ) -> np.ndarray:
        """Mean value of the profile within a radius r,

        .. math::

            \\bar\\rho(R) = \\frac3{R^3} \\int_0^R r^2 \\rho(r)\\,dr

        Parameters
        ----------
        r : np.ndarray
            positions at which to calculate the cumulative profile

        Optional arguments
        ------------------
        log_rmin : float
            log10 lower limit to integrate the profile
        integral_samples : int
            number of samples to generate for Simpson-rule integration

        """
        return self.mass_cumulative(
            r, log_rmin=log_rmin, integral_samples=integral_samples
        ) / (4 / 3 * np.pi * r**3)

    @inMpc
    @array
    def mass_cumulative(
        self, r: np.ndarray, *, log_rmin: float = -10, integral_samples: int = 1000
    ) -> np.ndarray:
        """Spherical integral within a radius r,

        .. math::

            M(<R) =  4\\pi \\int_0^R r^2 \\rho(r)\\,dr

        Parameters
        ----------
        r : np.ndarray
            positions at which to calculate the cumulative profile

        Optional arguments
        ------------------
        log_rmin : float
            log10 lower limit to integrate the profile
        integral_samples : int
            number of samples to generate for Simpson-rule integration

        """
        R_int = np.logspace(log_rmin, np.log10(r), integral_samples)
        return 4 * np.pi * simps(R_int**2 * self.profile(R_int), R_int, axis=0)

    @inMpc
    def projected(
        self,
        R: np.ndarray,
        *,
        log_rmin: float = -10,
        log_rmax: float = 6,
        integral_samples: int = 200,
    ) -> np.ndarray:
        """Line of sight projected profile, calculated numerically

        Parameters
        ----------
        R : np.ndarray
            positions at which to calculate the projected profile

        Optional arguments
        ------------------
        log_rmin, log_rmax : float
            lower and upper limits for logspace resampling for integration
        integral_samples : int
            number of samples to generate for Simpson-rule integration of the projected profile


        .. note ::

            The default values for the integration parameters give numerical errors well below 0.1% over the range R=[1e-5,100] Mpc, when comparing the numerical and analytical implementations for an NFW profile (the former can be obtained by defining a GNFW profile with default kwargs)

        """
        assert log_rmin < log_rmax, (
            "argument log_rmin must be larger than log_rmax, received"
            f" {log_rmin,log_rmax}"
        )
        assert integral_samples // 1 == integral_samples, (
            "argument integral_samples must be int, received"
            f" {integral_samples} ({type(integral_samples)})"
        )
        R_los = np.logspace(log_rmin, log_rmax, integral_samples)
        # R = (R_los**2 + R[..., None] ** 2) ** 0.5
        # return 2 * simps(self.profile(R), R_los, axis=len(R.shape) - 1)
        # we must add dimensions to R_los rather than R so that @array
        # works as expected. It can be quite slow though
        R = (np.expand_dims(R_los, tuple(range(1, R.ndim + 1))) ** 2 + R**2) ** 0.5
        return 2 * simps(self.profile(R), R_los, axis=0)

    @inMpc
    def projected_cumulative(
        self,
        R: np.ndarray,
        *,
        log_rmin: float = -10,
        left_samples: int = 100,
        resampling: int = 20,
        **kwargs,
    ) -> np.ndarray:
        """Cumulative projected profile within R, calculated numerically

        Parameters
        ----------
        R : np.ndarray
            positions at which to calculate the projected profile

        Optional arguments
        ------------------
        log_rmin : float
            lower limit for logspace resampling for integration. The same value will be passed to ``self.projected``
        resampling : int
            number of samples into which each R-interval in the data will be re-sampled. For instance, if two adjacent data points are at ``Rbin=0.1,0.2`` then for the integration they will be replaced by ::

                newRbin = np.logspace(
                    np.log10(0.1), np.log10(0.2), resampling, endpoint=False)

            (the endpoint will be added when sampling the following bin)
        left_samples : int
            number of samples to use between log_rmin and the first value of R, with a logarithmic sampling

        Additional arguments will be passed to ``self.projected``

        .. note ::

            The default values for the integration parameters give numerical errors well below 0.1% over the range :math:`R=[10^{-5},100]\\,\\mathrm{Mpc}`, when comparing the numerical and analytical implementations for an NFW profile (the former can be obtained by defining a GNFW profile with default kwargs)

        """
        assert isinstance(left_samples, (int, np.integer)), (
            "argument left_samples must be int, received"
            f" {left_samples} ({type(left_samples)})"
        )
        assert isinstance(resampling, (int, np.integer)), (
            "argument resampling must be int, received"
            f" {resampling} ({type(resampling)})"
        )
        logR = np.log10(R)
        # resample R with left_samples to the left and resampling values between each
        # this should therefore be a 1d array; it should be easy to the take specifically
        # the input radii from the resulting integral
        Ro = np.hstack(
            [
                [0],
                np.logspace(log_rmin, logR[0], left_samples, endpoint=False),
                np.hstack(
                    [
                        np.logspace(logR[i - 1], logR[i], resampling, endpoint=False)
                        for i in range(1, R.shape[0])
                    ]
                ),
                [R.max()],
            ]
        )
        # these correspond to the indices where the original radii are stored
        j = np.arange(1 + left_samples, Ro.shape[0], resampling)
        integ = cumtrapz(
            self._expand_dims(Ro) * self.projected(Ro, log_rmin=log_rmin, **kwargs),
            Ro,
            initial=0,
            axis=0,
        )
        return 2 * integ[j] / self._expand_dims(R) ** 2

    def projected_excess(
        self,
        R: np.ndarray,
        log_rmin: float = -10,
        log_rmax: float = 6,
        integral_samples: int = 200,
        left_samples: int = 100,
        resampling: int = 20,
    ) -> np.ndarray:
        """Cumulative projected profile file excess at projected distance(s) R, defined as ::

            projected_excess(R) = projected_cumulative(R) - projected(R)

        This profile is most commonly used as the galaxy weak lensing *shear* observable, :math:`\\gamma` where the projected excess is referred to as the *excess surface density* (ESD or :math:`\\Delta\\Sigma`),

        .. math::

            \\Delta\\Sigma(R) = \\gamma\\Sigma_\\mathrm{c}

        where :math:`\\Sigma_\\mathrm{c}` is the critical surface density

        Parameters
        ----------
        R : float or array of float
            projected distance(s)

        Optional arguments are passed to either ``self.projected`` or ``self.projected_cumulative``

        """
        s1 = self.projected_cumulative(
            R,
            log_rmin=log_rmin,
            left_samples=left_samples,
            resampling=resampling,
            log_rmax=log_rmax,
            integral_samples=integral_samples,
        )
        s2 = self.projected(
            R, log_rmin=log_rmin, log_rmax=log_rmax, integral_samples=integral_samples
        )
        return s1 - s2

    def offset(
        self,
        func: callable,
        R: np.ndarray,
        Roff: np.ndarray,
        theta_samples: int = 360,
        weights: np.ndarray = None,
        **kwargs,
    ) -> np.ndarray:
        """Calcuate any profile with a reference point different from its center

        .. note ::

            We recommend using the significantly faster ``profiley.numeric.offset`` instead of this method, and will merge that implementation into this function in the future.

        Parameters
        ----------
        func : callable
            the funcion to calculate
        R : np.ndarray, shape (N,)
            radii at which to calculate the offset surface density
        Roff : np.ndarray, shape (M,)
            offsets with respect to the profile center

        Optional parameters
        -------------------
        theta_samples : int
            number of samples for the angular integral from 0 to 2*pi
        weights : array of floats, shape (M,)
            weights to apply to each profile corresponding to every value of ``Roff``. See ``Returns`` below
        kwargs : dict
            arguments to pass to ``func``

        Returns
        -------
        offset : np.ndarray,
            offset profile. The shape of the array depends on whether the ``weights`` argument is specified: if *not* specified (default), then

            .. code-block ::

                shape: (M,N,*self.shape)

            if ``weights`` is provided, then the first axis will be weight-averaged over so that

            .. code-block ::

                shape: (N,*self.shape)

        """
        if not isinstance(theta_samples, (int, np.integer)):
            raise TypeError(
                "argument theta_samples must be int, received"
                f" {theta_samples} ({type(theta_samples)})"
            )
        if not np.iterable(Roff):
            Roff = np.array([Roff])
        assert len(Roff.shape) == 1, "argument Roff must be 1d"
        if weights is not None:
            if weights.size != Roff.size:
                msg = (
                    "weights must have the same size as Roff,"
                    f" received {weights.size}, {Roff.size},"
                    " respectively."
                )
                raise ValueError(msg)

        # can't get this to work using the @array decorator
        R = R.reshape((R.size, *self._dimensions))
        Roff = Roff.reshape((Roff.size, *self._dimensions, 1, 1))
        theta = np.linspace(0, 2 * np.pi, theta_samples)
        theta1 = theta.reshape((theta_samples, *self._dimensions, 1))
        x = (Roff**2 + R**2 + 2 * R * Roff * np.cos(theta1)) ** 0.5
        off = np.array([trapz(func(i, **kwargs), theta, axis=0) for i in x])

        if weights is not None:
            # create a slice so we can multiply by weights
            # along the first axis
            s_ = [None] * off.ndim
            s_[0] = slice(None)
            Roff = np.squeeze(Roff)
            off = trapz(weights[tuple(s_)] * off, Roff, axis=0) / trapz(weights, Roff)
        return off / (2 * np.pi)

    def offset_profile(self, R, Roff, **kwargs):
        """Alias for ``offset(profile, R, Roff, **kwargs)``"""
        return self.offset(self.profile, R, Roff)

    def offset_projected(self, R, Roff, **kwargs):
        """Alias for ``offset(projected, R, Roff, **kwargs)``"""
        return self.offset(self.projected, R, Roff, **kwargs)

    def offset_projected_cumulative(self, R, Roff, **kwargs):
        """Alias for ``offset(projected_cumulative, R, Roff, **kwargs)``"""
        return self.offset(self.projected_cumulative, R, Roff, **kwargs)

    def offset_projected_excess(self, R, Roff, **kwargs):
        """Alias for ``offset(projected_excess, R, Roff, **kwargs)``"""
        return self.offset(self.projected_excess, R, Roff, **kwargs)

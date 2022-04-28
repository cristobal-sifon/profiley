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
from .helpers.lensing import BaseLensing


warnings.simplefilter('once', category=DeprecationWarning)


class Profile(BaseLensing):
    """Profile object

    All profiles should inherit from ``Profile``

    Defining your own profile is very simple. As an example, let's
    define a simple power-law profile with two free parameters, the
    normalization and the slope:

    .. math::

        f(r, a, b) = a * r**b

    ::

        class PowerLaw(Profile):

            def __init__(self, norm, slope, **kwargs):
                self._set_shape(norm*slope)
                super().__init__(**kwargs)
                self.norm = norm
                self.slope = slope

            @array
            def profile(self, r):
                return self.norm * r**self.slope

    That's it! The ``__init__()`` method needs only two lines of code
    (in addition to attribute definitions). The last line is necessary
    to allow ``profiley`` to automatically handle arbitrary shapes,
    through the definition of a ``_shape`` attribute. Note that
    ``_set_shape`` takes only one argument (besides ``self``) - the
    *product* of the class arguments. That is, if the arguments are
    arrays, their dimensions must be such that a product can be carried
    out without any manipulation.

    Profile projections
    -------------------

    If the projection of this profile is analytical, any or all of the
    following methods can also be specified: ::

        projected(self, R)
        projected_cumulative(self, R)
        projected_excess(self, R)
        offset_profile(self, R, Roff)
        offset_projected(self, R, Roff)
        offset_projected_cumulative(self, R, Roff)
        offset_projected_excess(self, R, Roff)

    If it does not have analytical expressions, these methods will also
    exist, but they will be calculated numerically, so they may be
    somewhat slower depending on the precision required.

    Deprecated methods
    ..................

    The following methods are deprecated as of ``v1.3.0`` and will be
    removed in a future version. They should be replaced as follows:

        surface_density --> projected
        enclosed_surface_density --> projected_cumulative
        excess_surface_density --> projected_excess

    and analogously with ``offset`` methods.

    Cosmology
    ---------

    All ``Profile`` objects contain all cosmological information with
    they have been initialized through the ``self.cosmo`` attribute,
    which can be any ``astropy.cosmology.FLRW`` object.

    """
    def __init__(self, z=0, overdensity=500,
                 los_loglimit=6, nsamples_los=200, resampling=20,
                 logleft=-10, left_samples=100, **kwargs):
        """Initialize a profile object

        Optional arguments
        ------------------
        z : float or ndarray of floats
            redshift
        overdensity : int or float
            overdensity with respect to the background (does not apply
            to all Profile children; see each specific class for
            details)
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
        if not hasattr(self, '_shape'):
            msg = 'attribute shape does not exist. Please make sure' \
                  ' to call ``self._set_shape`` in ``__init__``'
            raise AttributeError(msg)
        return self._shape

    ### private methods ###

    def _assert_overdensity(self, overdensity):
        assert not np.iterable(overdensity), \
            'parameter overdensity must be a scalar'
        try:
            overdensity / 1
        except TypeError as err:
            raise TypeError('parameter overdensity must be a number') from err
        if overdensity <= 0:
            raise ValueError(
                f'overdensity must be positive; received {overdensity}')
        return

    def _set_shape(self, args_product):
        if hasattr(self, 'shape'):
            msg = 'attribute shape already set, cannot be overwritten'
            raise ValueError(msg)
        if np.iterable(args_product):
            self._shape = args_product.shape
        else:
            self._shape = (1,)

    ### methods ###

    @inMpc
    @array
    def projected(self, R: np.ndarray, log_rmin=-10, log_rmax=6,
                  integral_samples=200):
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
            number of samples to generate for Simpson-rule integration
            of the projected profile


        Notes on numerical integration
        ------------------------------
         -The default values for the integration parameters give
            numerical errors well below 0.1% over the range
            R=[1e-5,100] Mpc, when comparing the numerical and
            analytical implementations for an NFW profile (the
            former can be obtained by defining a GNFW profile with
            default kwargs)
        """
        assert log_rmin < log_rmax, \
            'argument log_rmin must be larger than log_rmax, received' \
            f' {log_rmin,log_rmax}'
        assert integral_samples // 1 == integral_samples, \
            'argument integral_samples must be int, received' \
            f' {integral_samples} ({type(integral_samples)})'
        R_los = np.logspace(log_rmin, log_rmax, integral_samples)[:,None]
        R = np.transpose(
            [np.hypot(*np.meshgrid(R_los[:,i], R[:,0]))
             for i in range(R_los.shape[1])],
            axes=(1,2,0))
        return 2 * simps(self.profile(R), R_los[None], axis=1)

    @inMpc
    @array
    def projected_cumulative(self, R: np.ndarray, log_rmin: float=-10,
                             left_samples: int=100, resampling: int=20,
                             **kwargs):
        """Cumulative projected profile within R, calculated
        numerically

        Parameters
        ----------
        R : np.ndarray
            positions at which to calculate the projected profile

        Optional arguments
        ------------------
        log_rmin : float
            lower limit for logspace resampling for integration. The
            same value will be passed to ``self.projected``
        resampling : int
            number of samples into which each R-interval in the
            data will be re-sampled. For instance, if two adjacent
            data points are at Rbin=0.1,0.2 then for the integration
            they will be replaced by
                newRbin = np.logspace(np.log10(0.1), np.log10(0.2),
                                      resampling, endpoint=False)
            (the endpoint will be added when sampling the following bin)
        left_samples : int
            number of samples to use between log_rmin and the first
            value of R, with a logarithmic sampling

        Additional arguments will be passed to ``self.projected``

        Notes on numerical integration
        ------------------------------
         -The default values for the integration parameters give
            numerical errors well below 0.1% over the range
            R=[1e-5,100] Mpc, when comparing the numerical and
            analytical implementations for an NFW profile (the
            former can be obtained by defining a GNFW profile with
            default kwargs)
        """
        assert isinstance(left_samples, (int,np.integer)), \
            'argument left_samples must be int, received' \
            f' {left_samples} ({type(left_samples)})'
        assert isinstance(resampling, (int,np.integer)), \
            'argument resampling must be int, received' \
            f' {resampling} ({type(resampling)})'
        # for convenience
        logR = np.log10(R[:,0])
        # resample R
        Ro = np.vstack(
            [np.zeros(R.shape[1]),
             np.logspace(log_rmin, logR[0], left_samples, endpoint=False)[:,None],
             np.concatenate(
                [np.logspace(logR[i-1], logR[i], resampling, endpoint=False)
                 for i in range(1, R.shape[0])])[:,None],
             R.max()*np.ones(R.shape[1])]
            )
        j = np.arange(1+left_samples, Ro.shape[0], resampling)
        integ = cumtrapz(
            Ro*self.projected(Ro, log_rmin=log_rmin, **kwargs),
            Ro, initial=0, axis=0)
        return 2 * integ[j] / R**2

    def projected_excess(self, R: np.ndarray, log_rmin=-10, log_rmax=6,
                         integral_samples=200,
                         left_samples=100, resampling=20):
        """Cumulative projected profile file excess at projected
        distance(s) R, defined as

            projected_excess(R) = projected_cumulative(R) - projected(R)

        This profile is most commonly used as the galaxy weak lensing
        *shear* observable, :math:`\gamma` where the projected excess
        is referred to as the *excess surface density* (ESD or
        :math:`\Delta\Sigma`),

        .. math::

            \Delta\Sigma(R) = \gamma\Sigma_\mathrm{c}

        where :math:`\Sigma_\mathrm{c}` is the critical surface density

        Parameters
        ----------
        R : float or array of float
            projected distance(s)

        Optional arguments are passed to either ``self.projected`` or
        ``self.projected_cumulative``
        """
        s1 = self.projected_cumulative(
            R, log_rmin=log_rmin, left_samples=left_samples,
            resampling=resampling, log_rmax=log_rmax,
            integral_samples=integral_samples)
        s2 = self.projected(R, log_rmin=log_rmin, log_rmax=log_rmax,
            integral_samples=integral_samples)
        return s1 - s2

    def offset(self, func, R, Roff, theta_samples=360, weights=None,
               **kwargs):
        """Calcuate any profile with a reference point different
        from its center

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
            weights to apply to each profile corresponding to every
            value of ``Roff``. See ``Returns`` below
        kwargs : dict
            arguments to pass to ``func``

        Returns
        -------
        offset : np.ndarray,
            offset profile. The shape of the array depends on whether
            the ``weights`` argument is specified: if *not* specified
            (default), then
            .. code-block::

                shape: (M,N,*self.shape)

            if ``weights`` is provided, then the first axis will be
            weight-averaged over so that
            .. code-block::

                shape: (N,*self.shape)

        """
        if not isinstance(theta_samples, (int,np.integer)):
            raise TypeError(
                'argument theta_samples must be int, received' \
                f' {theta_samples} ({type(theta_samples)})')
        if not np.iterable(Roff):
            Roff = np.array([Roff])
        assert len(Roff.shape) == 1, 'argument Roff must be 1d'
        if weights is not None:
            if weights.size != Roff.size:
                msg = 'weights must have the same size as Roff,' \
                      f' received {weights.size}, {Roff.size},' \
                      ' respectively.'
                raise ValueError(msg)

        # can't get this to work using the @array decorator
        R = R.reshape((R.size,*self._dimensions))
        Roff = Roff.reshape((Roff.size,*self._dimensions,1,1))
        theta = np.linspace(0, 2*np.pi, theta_samples)
        theta1 = theta.reshape((theta_samples,*self._dimensions,1))
        x = (Roff**2 + R**2 + 2*R*Roff*np.cos(theta1))**0.5
        off = np.array([trapz(func(i, **kwargs), theta, axis=0) for i in x])

        if weights is not None:
            # create a slice so we can multiply by weights
            # along the first axis
            s_ = [None] * off.ndim
            s_[0] = slice(None)
            Roff = np.squeeze(Roff)
            off = trapz(weights[tuple(s_)]*off, Roff, axis=0) \
                / trapz(weights, Roff)
        return off / (2*np.pi)

    def offset_profile(self, R, Roff, **kwargs):
        """Alias for ``offset(profile, R, Roff, **kwargs)``"""
        return self.offset(self.profile, R, Roff)

    def offset_projected(self, R, Roff, **kwargs):
        """Alias for ``offset(projected, R, Roff, **kwargs)``"""
        return self.offset(self.projected, R, Roff, **kwargs)

    def offset_projected_cumulative(self, R, Roff, **kwargs):
        """Alias for ``offset(projected_cumulative, R, Roff,
        **kwargs)``"""
        return self.offset(self.projected_cumulative, R, Roff, **kwargs)

    def offset_projected_excess(self, R, Roff, **kwargs):
        """Alias for ``offset(projected_excess, R, Roff,
        **kwargs)``"""
        return self.offset(self.projected_excess, R, Roff, **kwargs)

    def _fourier(self, rmax=10, dr=0.1):
        """This is not working yet! Might just need to fall back to quad"""
        r = np.arange(dr, rmax, dr)
        f = self.profile(r)
        # compute Fourier transform by numpy's FFT function
        g = np.fft.fft(f)
        print('g =', g.shape)
        # frequency normalization factor is 2*np.pi/dt
        k = np.fft.fftfreq(f.size)*2*np.pi/dr
        # in order to get a discretisation of the continuous
        # Fourier transform we need to multiply g by a phase factor
        g = g * dr * np.exp(1j*k[:,None]*rmax) / (2*np.pi)**0.5
        return k, g

    # def twohalo(self, cosmo, bias, func, R, logm=np.logspace(12,16,41),
    #             bias_norm=1, **kwargs):
    def _twohalo(self, cosmo, offset_func, R, logm_2h=np.logspace(12,16,41),
                 z_2h=np.linspace(0,2,21), **kwargs):
        """Calculate the two-halo term associated with the profile

        Parameters
        ----------
        cosmo : `pyccl.Cosmology` object
        offset_func : callable or str
            if callable, it must be the offset version of the
            function in question. For instance, if one is
            modeling the convergence, the function supplied
            must be the offset convergence.
        R : np.ndarray
        logm : np.ndarray, optional
            masses over which to calculate the 2h term. If not
            specified, the masses used when defining the profile will be used.

        Notes
        -----
        kwargs are passed to the function to be called. If the function
        to be calculated is the convergence, the source redshift must
        be supplied
        """
        if not has_ccl:
            msg = 'Core Cosmology Library (CCL) required for two halo' \
                  ' calculations'
            raise ModuleNotFoundError(msg)
        assert isinstance(cosmo, ccl.Cosmology)
        # assert isinstance(bias, ccl.halos.HaloBias)
        # which function are we using?
        _valid_funcs = ('esd', 'kappa', 'sigma', 'convergence',
                        'projected', 'projected_cumulative',
                        'projected_excess',
                        'enclosed_surface_density', 'excess_surface_density',
                        'surface_density')
        if isinstance(offset_func, str):
            assert offset_func in _valid_funcs, \
                f'offset_func must be one of {_valid_funcs}'
            if offset_func in ('sigma', 'projected', 'surface_density'):
                offset_func = self.offset_projected
            elif offset_func in ('projected_cumulative',
                                 'enclosed_surface_density'):
                offset_func = self.offset_projected_cumulative
            elif offset_func in ('esd', 'projected_excess',
                                 'excess_surface_density'):
                offset_func = self.offset_projected_excess
            elif offset_func in ('kappa', 'convergence'):
                offset_func = self.offset_convergence
        else:
            assert callable(offset_func), \
                'argument offset_func must be a string or callable'
            assert offset_func.__name__.startswith('offset'), \
                'if argument offset_func is a function as opposed to' \
                'a string, it must be the offset version of the' \
                'function of interest.'
        # for each distance Ri, calculate the contribution from
        # all halos that will have some contribution at that distance
        # In order to speed this up a little, consider I don't need
        # to calculate the offset profile for all halos at all distances
        # but only for those who are close enough to that distance
        # So probably what I actually need to do is just calculate
        # for a grid in Roff and then move that to the required distance.
        # For example,
        # for i, Ri in R:
        #     twoh = offset_func(R, R[:i+1], **kwargs)
        return

    ### aliases for backward compatibility

    @deprecated('1.3.0', instead=f'projected')
    def surface_density(self, *args, **kwargs):
        """Alias for ``self.projected``

        .. note::
            Deprecated since version 1.3.0
        """
        return self.projected(*args, **kwargs)

    @deprecated('1.3.0', instead='projected_cumulative')
    def enclosed_surface_density(self, *args, **kwargs):
        """Alias for ``self.projected_cumulative``

        .. note::
            Deprecated since version 1.3.0
        """
        return self.projected_cumulative(*args, **kwargs)

    @deprecated('1.3.0', instead='projected_excess')
    def excess_surface_density(self, *args, **kwargs):
        """Alias for ``self.projected_excess``

        .. note::
            Deprecated since version 1.3.0
        """
        return self.projected_excess(*args, **kwargs)

    @deprecated('1.3.0', instead='offset_projected')
    def offset_surface_density(self, *args, **kwargs):
        """Alias for ``self.offset_projected``

        .. note::
            Deprecated since version 1.3.0
        """
        return self.offset_projected(*args, **kwargs)

    @deprecated('1.3.0', instead='offset_projected_cumulative')
    def offset_enclosed_surface_density(self, *args, **kwargs):
        """Alias for ``self.offset_projected_cumulative``

        .. note::
            Deprecated since version 1.3.0
        """
        return self.offset_projected_cumulative(*args, **kwargs)

    @deprecated('1.3.0', instead='offset_projected_excess')
    def offset_excess_surface_density(self, *args, **kwargs):
        """Alias for ``self.offset_projected_excess``

        .. note::
            Deprecated since version 1.3.0
        """
        return self.offset_projected_excess(*args, **kwargs)

    ### auxiliary methods to test integration performance ###

    @inMpc
    @array
    def _quad_projected(self, R):
        """Not yet implemented"""
        integrand = lambda r, Ro: self.profile((r**2+Ro**2)**0.5)
        return np.array([[quad(integrand, 0, np.inf, args=(Rij,))
                          for Rij in Ri] for Ri in R])

    @inMpc
    @array
    def _test_integration(self, R, output=None):
        """Test the fast-integration methods against the slow
        but accurate quad function

        Not yet implemented
        """
        qsd = self.quad_projected(R)
        sd = self.projected(R)
        return

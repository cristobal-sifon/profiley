from astropy import units as u
from astropy.cosmology import Planck15
import numpy as np
from scipy.integrate import cumtrapz, quad, simps

try:
    import pyccl as ccl
    has_ccl = True
except ImportError:
    has_ccl = False

from .helpers.cosmology import BaseCosmo
from .helpers.decorators import array, inMpc
from .helpers.lensing import BaseLensing


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
                super().__init__(**kwargs)
                self.norm = norm
                self.slope = slope
                self._set_shape(norm*slope)

            @array
            def profile(self, r):
                return self.norm * r**self.slope

    That's it! The ``__init__()`` method needs only two lines of code
    (in addition to attribute definitions). The last line is necessary
    to allow ``profiley`` to automatically handle arbitrary shapes,
    through the definition of a ``_shape`` attribute. Note that
    ``set_shape`` takes only one argument (besides ``self``) - the
    *product* of the class arguments. That is, if the arguments are
    arrays, their dimensions must be such that a product can be carried
    out without any manipulation.

    Profile projections
    -------------------

    If the projection of this profile is analytical, any or all of the
    following methods can also be specified: ::

        surface_density(self, R)
        enclosed_surface_density(self, R)
        excess_surface_density(self, R)
        offset_profile3d(self, R, Roff)
        offset_surface_density(self, R, Roff)
        offset_enclosed_surface_density(self, R, Roff)
        offset_excess_surface_density(self, R, Roff)

    If it does not have analytical expressions, these methods will also
    exist, but they will be calculated numerically, so they may be
    somewhat slower depending on the precision required.

    Cosmology
    ---------

    All ``Profile`` objects contain all cosmological information with
    they have been initialized through the ``self.cosmo`` attribute,
    which can be any ``astropy.cosmology.FLRW`` object.

    """

    def __init__(self, z=0, los_loglimit=6, Rlos=200, resampling=20,
                 logleft=-10, left_samples=100, **kwargs):
        """Initialize a profile object

        Optional parameters for numerical integration
        for the (enclosed) surface density (see notes below)
        ----------------------------------------------------
        los_loglimit : int
            log10-limit for the line-of-sight integration, in units
            of the radius of the cluster (e.g., r200, r500, etc,
            as defined when the object was initialized)
        Rlos : int
            number of samples for the line-of-sight integrals
        resampling : int
            number of samples into which each R-interval in the
            data will be re-sampled. For instance, if two adjacent
            data points are at Rbin=0.1,0.2 then for the integration
            they will be replaced by
                newRbin = np.logspace(np.log10(0.1), np.log10(0.2),
                                      resampling, endpoint=False)
            (the endpoint will be added when sampling the following bin)
        logleft : int
            log10-starting point for the integration of the enclosed
            surface density, in units of the cluster radius. The closer
            to zero this number the better
        left_samples : int
            number of samples to use between `logleft` and `R[0]`,
            with a logarithmic sampling

        Notes on numerical integration:
        -------------------------------
        The values for the integration parameters above have been
        chosen as a compromise between speed and accuracy (<0.5%
        for 0.01<R/Mpc<10) for a gNFW with alpha=1 (i.e., numerical
        integration of an NFW profile). For different profiles you
        may want to tune these differently.

        NOTE: for the benchmarking we need a way to do the integration
        that we know is accurate: should implement a function that uses
        `quad` (which is more accurate but a lot slower than the
        currently used `simps` integration).
        """
        super().__init__(z, **kwargs)
        # for numerical integration -- perhaps these could be passed
        # in a single dictionary
        self.los_loglimit = los_loglimit
        self.Rlos = Rlos
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

    def _define_array(self, x):
        if not np.iterable(x):
            return x * np.ones(self._shape)
        return x

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
    def surface_density(self, R, single_R=False):
        """Surface density at radius R, calculated by numerical integration

        `single_R` refers to whether we use a single R array
        to integrate all profiles (assuming more than one were defined).
        If True, it seems both this function and `enclosed_surface_density`
        work to about 10 Mpc within ~0.1% (compared to the analytical NFW)
        but they go wild beyond that, reaching 10% at 100 Mpc for the
        surface density and 1% for the enclosed surface density.

        NOTE: the above seems to be true also for multiple R (which I'm
        not sure I'd expect). More strangely, `single_R=True` now doesn't
        seem to work
        """
        # Need to test single_R=True again. Perhaps it can speed
        # things up a little without giving up much.
        if single_R:
            Rlos = np.logspace(-10, self.los_loglimit, self.Rlos) \
                * self.radius.max()
            R = np.hypot(*np.meshgrid(Rlos, R[0]))
        else:
            Rlos = np.logspace(-10, self.los_loglimit, self.Rlos)[:,None] \
                * self.radius
            R = np.transpose(
                [np.hypot(*np.meshgrid(Rlos[:,i], R[:,0]))
                 for i in range(Rlos.shape[1])],
                axes=(1,2,0))
        return 2 * simps(self.profile(R), Rlos[None], axis=1)

    @inMpc
    @array
    def enclosed_surface_density(self, R):
        """Surface density enclosed within R, calculated numerically"""
        logR = np.log10(R)
        # resample R
        Ro = np.vstack([
            np.zeros(R.shape[1]),
            np.logspace(-10, logR[0], self.left_samples,
                        endpoint=False)[:,None],
            np.concatenate(
                [np.logspace(logR[i-1], logR[i], self.resampling,
                             endpoint=False)
                 for i in range(1, R.shape[0])])[:,None],
            R.max()*np.ones(R.shape[1])
            ])
        j = np.arange(1+self.left_samples, Ro.shape[0], self.resampling)
        integ = cumtrapz(Ro*self.surface_density(Ro),
                         Ro, initial=0, axis=0)
        return 2 * integ[j] / R**2

    def excess_surface_density(self, R):
        """Excess surface density at projected distance(s) R

        The excess surface density or ESD is the galaxy weak lensing
        observable in physical units, and is calculated as:
            ESD(R) = Sigma(<R) - Sigma(R)
        where Sigma(<R) is the average surface density within R and
        Sigma(R) is the surface density at distance R

        Parameters
        ----------
        R : float or array of float
            projected distance(s)
        kwargs : dict, optional
            passed to both `self.enclosed_surface_density` and
            `self.surface_density`
        """
        return self.enclosed_surface_density(R) \
            - self.surface_density(R)

    def offset(self, func, R, Roff, **kwargs):
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

        Returns
        -------
        offset : np.ndarray
            offset profile

        Notes
        -----
        kwargs are passed to ``func``
        """
        if not np.iterable(Roff):
            Roff = np.array([Roff])
        assert len(Roff.shape) == 1, 'argument Roff must be 1d'
        # can't get this to work using the @array decorator
        R = R.reshape((R.size,*self._dimensions))
        Roff = Roff.reshape((Roff.size,*self._dimensions,1,1))
        theta = np.linspace(0, 2*np.pi, 500)
        theta1 = theta.reshape((500,*self._dimensions,1))
        x = (Roff**2 + R**2 + 2*R*Roff*np.cos(theta1))**0.5
        print('x =', x.shape)
        print('func =', func(x[0], **kwargs).shape)
        # looping slower but avoids memory issues
        off = np.array([simps(func(xi, **kwargs), theta, axis=0)
                        for xi in x])
        return off / (2*np.pi)

    def offset_density(self, R, Roff):
        return self.offset(self.profile, R, Roff)

    def offset_surface_density(self, R, Roff, **kwargs):
        """Surface density measured around a position offset from the
        profile center

        Parameters
        ----------
        R : np.ndarray, shape (N,)
            radii at which to calculate the offset surface density
        Roff : np.ndarray, shape (M,)
            offsets with respect to the profile center

        Returns
        -------
        offset_surface_density : np.ndarray
            offset surface density
        """
        return self.offset(self.surface_density, R, Roff, **kwargs)

    def offset_enclosed_surface_density(self, R, Roff, **kwargs):
        return self.offset(self.enclosed_surface_density, R, Roff, **kwargs)

    def offset_excess_surface_density(self, R, Roff, **kwargs):
        return self.offset(self.excess_surface_density, R, Roff, **kwargs)

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
    def twohalo(self, cosmo, offset_func, R, logm_2h=np.logspace(12,16,41),
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
                        'excess_surface_density', 'surface_density')
        if isinstance(offset_func, str):
            assert offset_func in _valid_funcs, \
                f'offset_func must be one of f{_valid_funcs}'
            if offset_func in ('sigma', 'surface_density'):
                offset_func = self.offset_surface_density
            elif offset_func in ('esd', 'excess_surface_density'):
                offset_func = self.offset_excess_surface_density
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

    ### auxiliary methods to test integration performance ###

    @inMpc
    @array
    def _quad_surface_density(self, R):
        integrand = lambda r, Ro: self.profile((r**2+Ro**2)**0.5)
        return np.array([[quad(integrand, 0, np.inf, args=(Rij,))
                          for Rij in Ri] for Ri in R])

    @inMpc
    @array
    def _test_integration(self, R, output=None):
        """Test the fast-integration methods against the slow
        but accurate quad function
        """
        qsd = self.quad_surface_density(R)
        sd = self.surface_density(R)

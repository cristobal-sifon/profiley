from astropy import units as u
from astropy.cosmology import Planck15
import numpy as np
from scipy.integrate import cumtrapz, quad, simps

try:
    import pixell.enmap
    import pixell.utils
    has_pixell = True
except ImporError:
    has_pixell = False

from .helpers.cosmology import BaseCosmo
from .helpers.decorators import array, inMpc
from .helpers.lensing import BaseLensing


#class Profile(BaseCosmo):
class Profile:

    def __init__(self, los_loglimit=6, Rlos=200, resampling=20,
                 logleft=-10, left_samples=100):
        """Initialize a profile object

        Optional parameters for numerical integration
        for the (enclosed) surface density (see notes below)
        ----------------------------------------------------
        los_loglimit : int
            log10-limit for the line-of-sight integration, in units
            of `rvir`
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
            surface density, in units of `rvir`. The closer to zero this
            number the better
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
        #super().__init__(cosmo=cosmo)
        # for numerical integration -- perhaps these could be passed
        # in a single dictionary
        self.los_loglimit = los_loglimit
        self.Rlos = Rlos
        self.resampling = resampling
        self.logleft = logleft
        self.left_samples = left_samples
        # aliases
        self.barsigma = self.enclosed_surface_density
        self.esd = self.excess_surface_density
        self.sigma = self.surface_density

    @property
    def _one(self):
        if self.__one is None:
            self.__one = u.dimensionless_unscaled
        return self.__one

    ### private methods ###

    def _define_array(self, x):
        if not np.iterable(x):
            return x * np.ones_like(self._shape)
        return x

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
                * self.rvir.max()
            R = np.hypot(*np.meshgrid(Rlos, R[0]))
        else:
            Rlos = np.logspace(-10, self.los_loglimit, self.Rlos)[:,None] \
                * self.rvir
            R = np.transpose(
                [np.hypot(*np.meshgrid(Rlos[:,i], R[:,0]))
                 for i in range(Rlos.shape[1])],
                axes=(1,2,0))
        return 2 * simps(self.density(R), Rlos[None], axis=1)

    @inMpc
    @array
    def enclosed_surface_density(self, R):
        """Surface density enclosed within R, calculated numerically"""
        logR = np.log10(R)
        # resample R
        Ro = np.vstack([
            np.zeros(R.shape[1]),
            np.logspace(-10, logR[0], self.left_samples, endpoint=False)[:,None],
            np.concatenate(
                [np.logspace(logR[i-1], logR[i], self.resampling, endpoint=False)
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

    def fourier(self, rmax=10, dr=0.1):
        """This is not working yet! Might just need to fall back to quad"""
        r = np.arange(dr, rmax, dr)
        f = self.density(r)
        # compute Fourier transform by numpy's FFT function
        g = np.fft.fft(f)
        print('g =', g.shape)
        # frequency normalization factor is 2*np.pi/dt
        k = np.fft.fftfreq(f.size)*2*np.pi/dr
        # in order to get a discretisation of the continuous
        # Fourier transform we need to multiply g by a phase factor
        g = g * dr * np.exp(1j*k[:,None]*rmax) / (2*np.pi)**0.5
        return k, g

    #def kfilter(self, R, filter_file):

    ### auxiliary methods to test integration performance ###

    @inMpc
    @array
    def quad_surface_density(self, R):
        integrand = lambda r, Ro: self.density((r**2+Ro**2)**0.5)
        return np.array([[quad(integrand, 0, np.inf, args=(Rij,))
                          for Rij in Ri] for Ri in R])

    @inMpc
    @array
    def test_integration(self, R, output=None):
        """Test the fast-integration methods against the slow
        but accurate quad function
        """
        qsd = self.quad_surface_density(R)
        sd = self.surface_density(R)


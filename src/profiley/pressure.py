from astropy import units as u
from astropy.cosmology import Planck15

from .nfw import BaseNFW
from .helpers.decorators import array


class PressureGNFW(BaseNFW):
    """Generalized NFW profile, with the parameterization commonly used
    to define pressure profiles in galaxy clusters (Nagai et al.
    2007)

    *New in version 1.2.0*

    Parameters
    ----------
    mass : float or np.ndarray
        cluster mass. If defin
    P0, c, alpha, beta, gamma : float or np.ndarray
        parameters of the dimensionless universal pressure profile, as
        defined in Eq. (11) of Arnaud et al. (2010)

    Optional parameters
    -------------------
    z : float or np.ndarray
        redshift
    normalization_name : {'M500','P500'}
        specifies to what parameter ``normalization`` corresponds
    P500_M500_params : list-like, length=4
        the four parameters that determine the scaling relation between
        M500 and P500. See ``P500_from_M500``
    cosmo : astropy.cosmology.FLRW
        cosmology object
    """
    # in case I implement it later:
    #normalization : float or np.ndarray
        #either P500 or M500. If the latter, P500 is calculated assuming
        #a power-law scaling relation following Eq. (3) of Nagain et al.
        #(2007).

    def __init__(self, mass, P0, c, alpha, beta, gamma, z=0,
                 #normalization_name='M500',
                 P500_M500_params=(1.45e-11*u.erg/u.cm**3,1e15,2/3,8/3),
                 **kwargs):
        #assert normalization_name in ('M500','P500'), \
            #"argument `normalization_name` must be one of ('M500','P500')," \
            #f" received {normalization_name} instead"
        #self.normalization = normalization
        #self.normalization_name = normalization_name
        super().__init__(mass, c, z, **kwargs)
        self.P0 = P0
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self._set_shape(mass*P0*c*alpha*beta*gamma*z)
        #self._set_shape(normalization*P0*c*alpha*beta*gamma*z)
        #if self.normalization_name == 'P500':
            #self.P500 = self.normalization
        #else:
        self.P500 = self.P500_from_M500(*P500_M500_params)

    @array
    def upp(self, x):
        """Dimensionless universal pressure profile

        .. math::

            p(x) = \\frac{P_0}
                {(cx)^\\gamma\\left[1+(cx)^\\alpha\\right]^(\\beta-\\gamma)/\\alpha

        where :math:`x=r/r_{500}`
        """
        exp = (self.beta-self.gamma) / self.alpha
        return self.P0 / (self.c*x)**self.gamma \
            / (1+(self.c*x)**self.alpha)**exp

    def P500_from_M500(self, a=1.45e-11*u.erg/u.cm**3, b=1e15, c=2/3, d=8/3):
        """Calculate P500 given M500 using a redshift-corrected power-law:

        .. math::

            P_{500} = a \\left(\\frac{M_{500}}{b}\\right)^c\\,E(z)^d

        where :math:`b` is in :math:`\mathrm{M}_\odot`. Default parameters
        correspond to those derived by Nagain et al. (2007).

        Both P500 and M500 may actually refer to any overdensity; the function
        name simply reflects the usual parameterization
        """
        return a * (self.mass/b)**c \
            * (self.cosmo.H(self.z)/self.cosmo.H0)**d

    @array
    def profile(self, r):
        return self.P500 * self.upp(r/self.rs)


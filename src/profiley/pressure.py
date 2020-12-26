from astropy import units as u
from astropy.cosmology import Planck15

from .nfw import BaseNFW
from .helpers.decorators import array, inMpc


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
    pm_params : list-like, length=4
        the four parameters that determine the scaling relation between
        M500 and P500 (or any other overdensity). See ``P500_from_M500``
    cosmo : astropy.cosmology.FLRW
        cosmology object
    """
    def __init__(self, mass, P0, c, alpha, beta, gamma, z=0,
                 pm_params=(1.45e-11*u.erg/u.cm**3,1e15,2/3,8/3),
                 cosmo=Planck15, **kwargs):
        assert len(pm_params) == 4, \
            'argument ``pm_params`` must contain 4 elements, received' \
            ' {len(pm_params)} instead'
        super().__init__(mass, c, z, **kwargs)
        self.P0 = P0
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.pm_params = pm_params
        self._set_shape(mass*P0*c*alpha*beta*gamma*z)
        self.P500 = self.P500_from_M500(*self.pm_params)

    @array
    def upp(self, x):
        """Dimensionless universal pressure profile

        .. math::

            p(x) = \\frac{P_0}
                {(cx)^\\gamma\\left[1+(cx)^\\alpha\\right]^(\\beta-\\gamma)/\\alpha}

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
        return a * (self.mass/b)**c * self.cosmo.efunc(self.z)**d

    @array
    @inMpc
    def profile(self, r):
        return self.P500 * self.upp(r/self.rs)


class Arnaud10(PressureGNFW):
    """GNFW profile with the best-fit parameters from Arnaud et al.
    (2010)

    All parameters can be modified as desired, but default parameters
    correspond to equations 12 and 13 from Arnaud et al. (2010), which
    makes it convenient for modifying only one or a few parameters at a
    time
    """
    def __init__(self, mass, P0=8.403, c=1.177, alpha=1.0510, beta=5.4905,
                 gamma=0.3081, z=0,
                 pm_params=(1.65e-3*u.keV/u.cm**3,3e14,2/3+0.12,8/3),
                 cosmo=Planck15, **kwargs):
        super().__init__(
            mass, P0, c, alpha, beta, gamma, z=z, pm_params=pm_params,
            cosmo=cosmo, **kwargs)


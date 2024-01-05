"""Test cosmological helpers. Equations based on the CCL v1 paper,
`https://ui.adsabs.harvard.edu/abs/2019ApJS..242....2C/abstract`_
"""
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM, cosmology_equal
import numpy as np

from profiley.einasto import Einasto
from profiley.nfw import GNFW, NFW, WebskyNFW


def test_cosmo():
    """Test whether the cosmology attribute is set correctly"""
    # any cosmology different from what we might have as default!
    cosmo = FlatLambdaCDM(Om0=0.33, Ob0=0.049, H0=64, Tcmb0=2.725 * u.K)
    od = 200
    bg = "m"
    frame = "physical"
    p = NFW(1e14, 5, 0.3, overdensity=od, background=bg, frame=frame, cosmo=cosmo)
    assert cosmology_equal(cosmo, p.cosmo)


def test_density_physical():
    """Equations 3, 4, and 5 from the CCL paper"""
    cosmo = FlatLambdaCDM(Om0=0.30, Ob0=0.049, H0=71, Tcmb0=2.725 * u.K)
    od = 200
    p = WebskyNFW(
        7e14, 3, 0.3, overdensity=od, background="c", frame="physical", cosmo=cosmo
    )
    rho_bg = cosmo.critical_density0 * (cosmo.H(p.z) / cosmo.H0) ** 2
    assert np.allclose(p.rho_bg, rho_bg.to(u.Msun / u.Mpc**3).value)
    p = GNFW(
        2e15, 2, 0.8, overdensity=od, background="m", frame="physical", cosmo=cosmo
    )
    rho_bg = (
        cosmo.critical_density0
        * (cosmo.H(p.z) / cosmo.H0) ** 2
        * cosmo.Om0
        * (1 + p.z) ** 3
    )
    assert np.allclose(p.rho_bg, rho_bg.to(u.Msun / u.Mpc**3).value)


def test_density_comoving():
    cosmo = FlatLambdaCDM(Om0=0.27, Ob0=0.049, H0=75, Tcmb0=2.725 * u.K)
    od = 200
    p = WebskyNFW(
        3e14, 8, 0.4, overdensity=od, background="c", frame="comoving", cosmo=cosmo
    )
    rho_bg = cosmo.critical_density0 * (cosmo.H(p.z) / cosmo.H0) ** 2 / (1 + p.z) ** 3
    assert np.allclose(p.rho_bg, rho_bg.to(u.Msun / u.Mpc**3).value)
    p = NFW(3e14, 8, 0.4, overdensity=od, background="m", frame="comoving", cosmo=cosmo)
    rho_bg = cosmo.critical_density0 * (cosmo.H(p.z) / cosmo.H0) ** 2 * cosmo.Om0
    assert np.allclose(p.rho_bg, rho_bg.to(u.Msun / u.Mpc**3).value)

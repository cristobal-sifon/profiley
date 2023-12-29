from astropy.cosmology import Planck15, cosmology_equal
import numpy as np
from profiley.nfw import NFW


def _test_einasto():
    return


def _test_gnfw():
    return


def _test_hernquist():
    return


def test_nfw():
    """Test NFW profile by calculating from scratch"""
    mass = 2e14
    c = 5
    z = 0.5
    od = 200
    bg = "m"
    frame = "comoving"
    nfw = NFW(mass, c, z, overdensity=od, background=bg, cosmo=Planck15, frame=frame)
    assert nfw.mass == mass
    assert nfw.c == c
    assert nfw.z == z
    assert nfw.overdensity == od
    assert nfw.background == bg
    assert cosmology_equal(nfw.cosmo, Planck15)
    assert nfw.frame == frame
    R = np.logspace(-2, 1, 10)
    # implement from scratch
    delta_c = (od / 3) * c**3 / (np.log(1 + c) - c / (1 + c))
    assert np.allclose(nfw.delta_c, delta_c)
    rdelta = (3 * mass / (4 * np.pi * od * nfw.rho_bg)) ** (1 / 3)
    rs = rdelta / c
    assert np.allclose(nfw.rs, rs)
    rho = delta_c * nfw.rho_bg / ((R / rs) * (1 + R / rs) ** 2)
    assert np.allclose(nfw.profile(R)[:, 0], rho)


def _test_tnfw():
    return

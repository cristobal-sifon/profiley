from astropy.cosmology import Planck15
import numpy as np
from profiley.nfw import GNFW, NFW


def test_projected_1d():
    """Test numerical projection for 1d profile objects comparing the
    numerical GNFW to the analytical NFW"""
    m = np.logspace(13, 15, 5)
    z = 0.5
    c = 5
    od = 200
    bg = "c"
    nfw = NFW(m, c, z, overdensity=od, background=bg, cosmo=Planck15)
    gnfw = GNFW(m, c, z, overdensity=od, background=bg, cosmo=Planck15)
    R = np.logspace(-2, 1, 25)
    assert np.allclose(nfw.profile(R), gnfw.profile(R))
    rtol = 1e-3
    atol = 1e12
    assert np.allclose(nfw.projected(R), gnfw.projected(R), rtol=rtol, atol=atol)
    assert np.allclose(
        nfw.projected_cumulative(R), gnfw.projected_cumulative(R), rtol=rtol, atol=atol
    )


def test_projected_2d():
    """Test numerical projection for 2d profile objects comparing the
    numerical GNFW to the analytical NFW"""
    m = np.logspace(13, 15, 5)
    z = np.linspace(0, 1, 3)[:, None]
    c = 5
    od = 200
    bg = "c"
    nfw = NFW(m, c, z, overdensity=od, background=bg, cosmo=Planck15)
    gnfw = GNFW(m, c, z, overdensity=od, background=bg, cosmo=Planck15)
    R = np.logspace(-2, 1, 25)
    assert np.allclose(nfw.profile(R), gnfw.profile(R))
    rtol = 1e-3
    atol = 1e12
    assert np.allclose(nfw.projected(R), gnfw.projected(R), rtol=rtol, atol=atol)
    assert np.allclose(
        nfw.projected_cumulative(R), gnfw.projected_cumulative(R), rtol=rtol, atol=atol
    )


def test_projected_3d():
    """Test numerical projection for 3d profile objects comparing the
    numerical GNFW to the analytical NFW"""
    m = np.logspace(13, 15, 5)
    z = np.linspace(0, 1, 3)[:, None]
    c = np.linspace(3, 7, 2)[:, None, None]
    od = 200
    bg = "c"
    nfw = NFW(m, c, z, overdensity=od, background=bg, cosmo=Planck15)
    gnfw = GNFW(m, c, z, overdensity=od, background=bg, cosmo=Planck15)
    R = np.logspace(-2, 1, 25)
    assert np.allclose(nfw.profile(R), gnfw.profile(R))
    rtol = 1e-3
    atol = 1e12
    assert np.allclose(nfw.projected(R), gnfw.projected(R), rtol=rtol, atol=atol)
    assert np.allclose(
        nfw.projected_cumulative(R), gnfw.projected_cumulative(R), rtol=rtol, atol=atol
    )


def test_mdelta():
    m = np.logspace(13, 15, 5)
    z = np.linspace(0, 1, 3)[:, None]
    c = np.linspace(3, 7, 2)[:, None, None]
    # default args should be sufficient to reach 1%
    rtol = 0.01
    nfw_500c = NFW(m, c, z, overdensity=500, background="c", cosmo=Planck15)
    m200m, r200m, density_errors = nfw_500c.mdelta(
        200, "m", rtol=rtol, return_errors=True
    )
    assert np.all(density_errors < rtol)
    nfw_200m = NFW(
        m200m, r200m / nfw_500c.rs, z, overdensity=200, background="m", cosmo=Planck15
    )
    r = np.logspace(-2, 1, 100)
    # we don't care about atol so setting to a large value
    assert np.allclose(nfw_200m.profile(r), nfw_500c.profile(r), rtol=rtol, atol=1e20)

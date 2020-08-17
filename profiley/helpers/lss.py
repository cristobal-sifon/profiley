"""Helpers for Large-Scale Structure calculations"""
from itertools import count
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline

from . import hankel


def power2xi(lnPk_lnk, R):
    """Calculate the correlation function using the Hankel
    transform of the power spectrum

    Taken from the KiDS-GGL pipeline, credit Andrej Dvornik.

    Reference:
        H. Ogata,
        A Numerical Integration Formula Based on the Bessel Functions,
        Publ. Res. Inst. Math. Sci. 41 (2005), 949-970.
        doi: 10.2977/prims/1145474602

    Parameters
    ----------
    lnPk_lnk : `scipy.interpolate.interp1d` object (or equivalent)
        function to calculate the natural log of the power spectrum
        given the natural log of the wavenumber
    """
    assert len(np.squeeze(R).shape) == 1, 'R must be 1-d'
    result = np.zeros(R.shape)
    h = hankel.SphericalHankelTransform(0,10000,0.00001)
    for i in range(result.size):
        integ = lambda x: \
            np.exp(lnPk_lnk(np.log(x/R[i]))) * (x**2) / (2*np.pi**2)
        result[i] = h.transform(integ)[0]
    return result / R**3


def xi2sigma(R, r_xi, xi, rho_m, threads=1):
    """Calculate the surface density from the correlation function

    Parameters
    ----------
    R : np.ndarray, shape (N,)
        projected distances at which to calculate the surface density,
        in Mpc. Must be 1d.
    r_xi : np.ndarray, shape ([M,P,]Q)
        radii at which the correlation function has been calculated.
        Can have up to two additional dimensions (say, mass and redshift)
    xi : np.ndarray, shape ([M,P,]Q)
        correlation function, calculated at radii `r_xi`. Can have up
        to two additional dimensions (say, mass and redshift)
    threads : int, optional
        number of threads to calculate the surface densities in
        parallel

    Returns
    -------
    sigma : np.ndarray, shape ([M,P,]N)
        surface density calculated at projected radii R

    Notes
    -----
    * This function can take a *long* time to run, depending on
      the number of surface densities to be calculated. Be sure
      to store the results in a file.
    * In order to make it run faster, everything will be passed to
      a single pool (if `threads>1`); if you would like to keep
      track of progress it is advised to call this function for
      pieces of your data and do the progress printing when calling
      this function.
    * <Note on the shape of r_xi>
    """
    assert len(R.shape) == 1, f'R must be 1d; has shape {R.shape}'
    # reshape r_xi and xi to 2-dimensional if necessary
    r_xi_shape = r_xi.shape
    xi_shape = xi.shape
    output_shape = (*xi_shape[:-1],R.size)
    assert len(r_xi_shape) <= len(xi_shape), \
        "r_xi has more dimensions than xi; I don't know how to" \
        " interpret that!"
    assert r_xi_shape == xi_shape[-len(r_xi_shape):], \
        'inconsistent shapes for arguments r_xi and xi: received' \
        f'{r_xi_shape} and {xi_shape}'
    if len(xi_shape) == 1:
        return np.array(xi2sigma_single(R, ln_rxi, ln_1plusxi))
    # if xi is 3d we need to do a little bit of massaging to
    # turn it into a 2d array, so that one loop will suffice
    # to go through it. We reshape as appropriate at the end.
    if len(xi_shape) == 3:
        xi = np.vstack(xi)
        if len(r_xi_shape) == 1:
            r_xi = np.array(
                [[r_xi for j in range(xi_shape[1])]
                 for i in range(xi_shape[0])])
        elif len(r_xi_shape) == 2:
            r_xi = np.array([r_xi for i in range(xi_shape[0])])
        r_xi = np.vstack(r_xi)
    elif len(xi_shape) == 2:
        if len(r_xi_shape) == 1:
            r_xi = np.array([r_xi for i in range(xi_shape[0])])
    # OK done with the massaging
    ln_rxi = np.log(r_xi)
    ln_1plusxi = np.log(1+xi)
    if threads == 1:
        sigma = np.array(
            [_xi2sig_single(R, ln_rxi_i, ln_1plusxi_i)
             for ln_rxi_i, ln_1plusxi_i in zip(ln_rxi, ln_1plusxi)])
        if len(xi_shape) == 3:
            # probably need to transpose this to get it right!
            sigma = sigma.reshape(output_shape)
    # run in parallel
    else:
        pool = mp.Pool(threads)
        out = [pool.apply_async(
                   _xi2sig_single,
                   args=(R,ln_rxi_i, ln_1plusxi_i), kwds={'idx':i})
                   for i, ln_rxi_i,ln_1plusxi_i
                   in zip(count(), ln_rxi, ln_1plusxi)]
        pool.close()
        pool.join()
        # extract results
        sigma = np.zeros((xi.shape[0],R.size))
        for i in out:
            sigma_i, i = i.get()
            sigma[i] = sigma_i
        sigma = sigma.reshape(output_shape)
    return 2 * rho_m * R * sigma


def _xi2sig_single(R, ln_rxi, ln_1plusxi, idx=None):
    """Auxiliary function to calculate the surface density from
    the correlation function for a single profile

    The primary use of this function is as a helper for
    `xi2sigma`, which allows the calculation of multiple
    surface density profiles in parallel.

    Parameters
    ----------
    R : np.ndarray, shape (N,)
        distances at which to calculate the surface density
    ln_rxi : np.ndarray, shape (M,)
        natural log of the distances at which the correlation
        function has been calculated
    ln_1plusxi : np.ndarray, shape (M,)
        natural log of (1+xi) where xi is the correlation function,
        calculated at distances Rxi
    idx : type free, optional
        pointers to the original array to map the results of the
        multiprocessing Pool used to loop through calls to this
        function

    Returns
    -------
    sig_single : list, len N
        integral required for surface density calculated at
        locations `R`. Note that for convenience this is a
        list and not a np.ndarray, and more importantly note
        that this is NOT the surface density, which corresponds
        instead to sigma=2*rho_m*R*sig_single.
    idx : the argument `idx` is also returned only if specified
        when calling this function
    """
    ln_xi = UnivariateSpline(ln_rxi, ln_1plusxi, s=0, ext=0)
    integrand = lambda x, R: \
        (np.exp(ln_xi(np.log(R/x)))-1) / (x**2*(1-x**2)**0.5)
    sig_single = [quad(integrand, 0, 1, args=(Ri,))[0] for Ri in R]
    if idx:
        return sig_single, idx
    return sig_single

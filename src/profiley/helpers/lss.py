"""Helpers for Large-Scale Structure calculations"""
from itertools import count
import multiprocessing as mp
import numpy as np
import os
import pyccl as ccl
from scipy.integrate import quad, IntegrationWarning
from scipy.interpolate import UnivariateSpline
from time import time
import warnings

from . import hankel


warnings.filterwarnings('ignore', category=IntegrationWarning)

"""
class ProfilesFile:

    def __init__(self, filename):
        self.filename = filename

    def
"""


def load_profiles(filename, x=None, precision=2, force_match_all=True):
    """Load profiles created by ``save_profiles``

    Parameters
    ----------
    filename : str
        name of the file to be read. To ensure consistency, this file
        should have been created with ``save_profiles``; this is
        assumed but not checked
    x : iterable, optional
        quantities for which the profiles are needed. Each dimension
        must correspond to the columns storing the variables in the
        file (e.g., z and logm).
        Each column in ``x`` must be a unique array.
    precision : int or list of 2 ints, optional
        number of decimal places with which to match ``x`` to the
        columns read from the file.
    force_match_all : bool, optional
        whether if ``x is defined``, all values of x *must* be present
        in the file, otherwise a ``ValueError`` is raised.

    Returns
    -------
    profiles : ndarray, shape (M,N,P)
        grid of profiles
    R : ndarray, shape (P,)
        radii at which the profiles were calculated
    x1, x2 : ndarray, shapes (M,) and (N,), respectively
        coordinates over which the grid of profiles was calculated
    info : tuple, len 5
        contains the following information:
            -name of the profile
            -radial units
            -name of x1
            -name of x2
            -dictionary of cosmological parameters, or None

    Raises
    ------
    ValueError: if ``x`` is not composed of unique arrays

    Notes
    -----
    - It is assumed but not check that the radial coordinates make sense.
      In particular, if any value in the radial coordinate is equal to
      -1, this function will raise an exception.
    """
    with open(filename) as f:
        # in python 3.8 I can do `while (hdr := f.readline().strip()) == '#':`
        hdr = f.readline().strip()
        while hdr[0] == '#':
            hdr = f.readline().strip()
            continue
        ncols = np.isin(hdr.split(), '-1').sum()
    cols = np.loadtxt(filename, unpack=True)[:ncols,1:]
    values = [np.unique(c) for c in cols]
    # for now
    x1, x2 = values
    # check whether we need to match values
    if x is not None:
        if not np.all([np.unique(xi).size == xi.size]):
            raise ValueError('Not all columns of x are unique')
        isin = [[]] * ncols
        for i in range(ncols):
            if not np.iterable(precision):
                precision = [precision] * len(x)
            isin[i] = np.isin(
                np.round(x[i], precision[i]),
                np.round(values[i], precision[i]))
            if force_match_all and isin.sum() != x[i].size:
                raise ValueError(
                    f'Not all values found in column #{i}:' \
                    f'found {values[i]}, expected {x[i]}')
    # load profiles, excluding already-loaded columns
    profiles = np.loadtxt(filename)[:,ncols:]
    # first line contains radial coordinates
    r = profiles[0]
    profiles = profiles[1:]
    profiles = np.transpose(
        np.reshape(profiles, (x2.size,x1.size,r.size)), axes=(1,0,2))
    if x is not None:
        x1 = x1[isin[0]]
        x2 = x2[isin[1]]
        profiles = profiles[isin[0],isin[1]]
    # load information from the comments
    with open(filename) as file:
        _, xlabel, ylabel, label = file.readline().split()
        runit = file.readline().split()[-1]
        cosmo = file.readline()
        if cosmo == '# No cosmology provided':
            cosmo = None
        else:
            cosmo = np.transpose(
                [param.split('=') for param in cosmo.split()[1:]])
            cosmo = {key: float(val) for key, val in zip(*cosmo)}
    info = (label, runit, xlabel, ylabel, cosmo)
    return profiles, r, x1, x2, info


def power2xi(lnPgm_lnk, R):
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
    lnPgm_lnk : `scipy.interpolate.interp1d` object (or equivalent)
        function to calculate the natural log of the power spectrum
        given the natural log of the wavenumber
    """
    assert len(np.squeeze(R).shape) == 1, 'R must be 1-d'
    h = hankel.SphericalHankelTransform(0,10000,0.00001)
    result = np.array(
        [h.transform(lambda x: np.exp(lnPgm_lnk(np.log(x/Ri))) * x**2)[0]
         for Ri in R])
    return result / (2*np.pi**2 * R**3)


def save_profiles(file, x, y, R, profiles, xlabel='z', ylabel='logm',
                  label='profile', R_units='Mpc',
                  cosmo_params=None, verbose=True):
    """Save 2d grid of profiles into file

    Parameters
    ----------
    file : str
        output file name
    x, y : ndarray, shapes (M,) and (N,)
        variables over which the profiles have been calculated.
        Typically mass and redshift
    R : ndarray, shape (P,)
        radial coordinates
    profiles : ndarray, shape (M,N,P)
        profiles to be saved
    xlabel, ylabel : str, optional
        names of the x and y coordinates
    label : str, optional
        name of the profile being stored. Default 'profile'.
    R_units : str, optional
        units of the radial coordinates, annotated in the comments
    cosmo : dict, optional
        cosmological parameters used to calculate the profiles,
        annotated as a comment
    verbose : bool, optional
        verbosity. If ``True`` will only print the filename
        when finished.

    -The first row contains to `-1` entries in the first two columns,
    which are there only to make the contents of a file rectangular,
    and are ignored. Elements from the third on correspond to the
    radial coordinates.
    -Starting from the second row, the first two columns contain all
    combinations of the arrays `x` and `y`, and columns from the third
    on contain the profile for each combination.
    -The comments contain the units of the radial coordinates as well
    as the cosmological parameters used in the calculations. If
    cosmological parameters are not provided, the line says "# No
    cosmology provided".

    """
    assert len(x.shape) == 1, 'x must be 1-d'
    assert len(y.shape) == 1, 'y must be 1-d'
    assert len(R.shape) == 1, 'R must be 1-d'
    assert profiles.shape == (x.size,y.size,R.size), \
        'shape of profiles inconsistent, must be (x.size,y.size,R.size)' \
        f'=({x.size},{y.size},{R.size}); received instead {profiles.shape}'
    if os.path.isfile(file):
        warnings.warn(f'Saving backup of existing file at {file}.backup')
        os.system(f'cp -p {file} {file}.backup')
    ygrid, xgrid = np.meshgrid(y, x)
    with open(file, 'w') as f:
        print(f'# {xlabel} {ylabel} {label}', file=f)
        print(f'# radial bins in units of {R_units}', file=f)
        if cosmo_params is None:
            print('# No cosmology provided', file=f)
        else:
            line = ' '.join([f'{key}={val:.5e}'
                             for key, val in cosmo_params.items()])
            print(f'# {line}', file=f)
        print('-1  -1  {0}'.format(' '.join([f'{Ri:.5e}' for Ri in R])),
              file=f)
        for j in range(y.size):
            for i in range(x.size):
                line = '{0:.5e}  {1:.5e}  {2}'.format(
                    xgrid[i,j], ygrid[i,j],
                    ' '.join([f'{ij:.5e}' for ij in profiles[i,j]]))
                print(line, file=f)
    print(f'Saved to {file}')
    return


def xi2sigma(R, r_xi, xi, rho_m, threads=1, full_output=False, verbose=2):
    """Calculate the surface density from the correlation function

    Parameters
    ----------
    R : np.ndarray, shape ([M,[P,]]N)
        projected distances at which to calculate the surface density,
        in Mpc. Can have up to two additional dimensions (typically
        mass and redshift)
    r_xi : np.ndarray, shape ([M,P,]Q)
        radii at which the correlation function has been calculated.
        Can have up to two additional dimensions (say, mass and
        redshift)
    xi : np.ndarray, shape ([M,P,]Q)
        correlation function, calculated at radii `r_xi`. Can have up
        to two additional dimensions (say, mass and redshift). Note
        that it is assumed that at least one of these additional
        dimensions is indeed present
    rho_m : float
        average comoving matter density
    threads : int, optional
        number of threads to calculate the surface densities in
        parallel
    full_output : bool, optional
        whether to return the radial coordinates as well as the
        surface density (default False)
    full_output : bool, optional
        if ``True``, return the radii as well as the surface density,
        both with the same shapes
    verbose : {0,1,2}
        verbosity. 0 is quiet, 2 is full verbose

    Returns
    -------
    sigma : np.ndarray, shape ([M,[P,]]N)
        surface density calculated at projected radii R
    Rsigma : np.ndarray, shape ([M,[P,]]N)
        radial coordinate reshaped to have the same shape as ``sigma``.
        Returned only if ``full_output==True``

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

    """
    # we need to know the original shapes as we reshape these
    R_shape = R.shape
    r_xi_shape = r_xi.shape
    xi_shape = xi.shape
    n = np.prod(xi_shape[:-1])
    output_shape = (*xi_shape[:-1],R.shape[-1])
    assert len(r_xi_shape) <= len(xi_shape), \
        "r_xi has more dimensions than xi; I don't know how to" \
        " interpret that!"
    assert len(R.shape) <= len(xi_shape), \
        "R has more dimensions than xi; I don't know how to" \
        " interpret that!"
    assert r_xi_shape == xi_shape[-len(r_xi_shape):], \
        'inconsistent shapes for arguments r_xi and xi: received' \
        f'{r_xi_shape} and {xi_shape}'
    assert verbose in (0,1,2)
    if len(xi_shape) == 1:
        return np.array(xi2sigma_single(R, ln_rxi, ln_1plusxi))
    # if xi is 3d we need to do a little bit of massaging to
    # turn it into a 2d array, so that one loop will suffice
    # to go through it. We reshape as appropriate at the end.
    if verbose == 2:
        print('Internally rearranging input data...')
    # it's a little more complicated if they have 2 dimensions
    if len(R.shape) == 1:
        R = np.array([R for i in range(n)])
    if len(r_xi_shape) == 1:
        r_xi = np.array([r_xi for i in range(n)])
    # do those cases in here
    if len(xi_shape) == 3:
        xi = np.vstack(xi)
        # arrange r_xi to match xi's shape
        if len(r_xi_shape) == 2:
            r_xi = np.array([r_xi for i in range(xi_shape[0])])
            r_xi = np.vstack(r_xi)
        # arrange R
        if len(R_shape) == 2:
            if R.shape[0] == xi_shape[0]:
                R = np.vstack([[Ri for i in range(xi_shape[1])] for Ri in R])
            else:
                R = np.vstack([[Ri for Ri in R] for i in range(xi_shape[0])])
    # OK done with the massaging
    ln_rxi = np.log(r_xi)
    ln_1plusxi = np.log(1+xi)
    if verbose:
        print(f'Calculating {n} surface densities using {threads} threads.' \
              ' This may take a while...')
    to = time()
    if threads == 1:
        sigma = np.array(
            [_xi2sig_single(*args) for args in zip(R, ln_rxi, ln_1plusxi)])
    # run in parallel
    else:
        pool = mp.Pool(threads)
        out = [[]] * n
        for i, args in enumerate(zip(R, ln_rxi, ln_1plusxi)):
            out[i] = pool.apply_async(
                _xi2sig_single, args=(args), kwds={'idx':i})
        pool.close()
        pool.join()
        # extract results
        sigma = np.zeros((xi.shape[0],R.shape[-1]))
        for i, x in enumerate(out):
            sigma_j, j = x.get()
            sigma[j] = sigma_j
    if verbose:
        t = time() - to
        print(f'Calculated {n} surface densities in {t/60:.2f} min,' \
              f' for an average time of {t/n:.2f} sec per call.')
    if len(output_shape) == 3:
        sigma = sigma.reshape(output_shape)
        R = R.reshape(output_shape)
    sigma = 2 * rho_m * R * sigma
    if full_output:
        return sigma, R
    return sigma


def _xi2sig_single(R, ln_rxi, ln_1plusxi, idx=None, verbose=True):
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
    if idx is not None:
        return sig_single, idx
    return sig_single

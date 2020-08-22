"""Helpers for Large-Scale Structure calculations"""
from itertools import count
import multiprocessing as mp
import numpy as np
import os
import pyccl as ccl
from scipy.integrate import quad, IntegrationWarning
from scipy.interpolate import UnivariateSpline
from time import time
from tqdm import tqdm
import warnings

from . import hankel


warnings.filterwarnings('ignore', category=IntegrationWarning)

"""
class ProfilesFile:

    def __init__(self, filename):
        self.filename = filename

    def 
"""


def load_profiles(filename, output_R_unit='Mpc'):
    """Load profiles created by ``save_profiles``

    Parameters
    ----------
    filename : str
        name of the file to be read. To ensure consistency, this file
        should have been created with ``save_profiles``; this is
        assumed but not checked

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
    """
    x1, x2 = np.loadtxt(filename, unpack=True)[:2,1:]
    x1 = np.unique(x1)
    x2 = np.unique(x2)
    profiles = np.loadtxt(filename)[:,2:]
    # first line contains radial coordinates
    r = profiles[0]
    profiles = profiles[1:]
    profiles = np.transpose(
        np.reshape(profiles, (x2.size,x1.size,r.size)), axes=(1,0,2))
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
            print(cosmo)
            cosmo = {key: float(val) for key, val in zip(*cosmo)}
    info = (label, runit, xlabel, ylabel, cosmo)
    return profiles, r, x1, x2, info


#def power2xi(k, Pgm, R):
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
    #assert len(k.shape) == 1, 'k must be 1-d'
    assert len(np.squeeze(R).shape) == 1, 'R must be 1-d'
    result = np.zeros(R.shape)
    h = hankel.SphericalHankelTransform(0,10000,0.00001)
    for i in range(result.size):
        integ = lambda x: \
            np.exp(lnPgm_lnk(np.log(x/R[i]))) * (x**2) / (2*np.pi**2)
        result[i] = h.transform(integ)[0]
    return result / R**3
    #xi = np.zeros((z.size,m.size,Rxi.size))
    #for i in range(z.size):
        #for j in range(m.size):
            #lnPgm_lnk = interp1d(lnk, lnPgm[i,j])
            #xi[i,j] = lss.power2xi(lnPgm_lnk, Rxi)


def save_profiles(file, x, y, R, profiles, xlabel='z', ylabel='m',
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


def xi2sigma(R, r_xi, xi, rho_m, threads=1, verbose=2):
    """Calculate the surface density from the correlation function

    Parameters
    ----------
    R : np.ndarray, shape ([M,[P,]]N
        projected distances at which to calculate the surface density,
        in Mpc. Can have up to two additional dimensions (typically
        mass and redshift)
    r_xi : np.ndarray, shape ([M,P,]Q)
        radii at which the correlation function has been calculated.
        Can have up to two additional dimensions (say, mass and redshift)
    xi : np.ndarray, shape ([M,P,]Q)
        correlation function, calculated at radii `r_xi`. Can have up
        to two additional dimensions (say, mass and redshift)
    threads : int, optional
        number of threads to calculate the surface densities in
        parallel
    verbose : {0,1,2}
        verbosity. 0 is quiet, 2 is full verbose

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
    # reshape r_xi and xi to 2-dimensional if necessary
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
    if len(xi_shape) == 3:
        xi = np.vstack(xi)
        # arrange r_xi to match xi.shape
        if len(r_xi_shape) == 1:
            r_xi = np.array(
                [[r_xi for j in range(xi_shape[1])]
                 for i in range(xi_shape[0])])
        elif len(r_xi_shape) == 2:
            r_xi = np.array([r_xi for i in range(xi_shape[0])])
        r_xi = np.vstack(r_xi)
        # arrange R
        if len(R.shape) == 1:
            R = np.array([R for i in range(n)])
        elif R.shape[0] == xi_shape[0]:
            R = np.vstack([[Ri for i in range(xi_shape[1])] for Ri in R])
        else:
            R = np.vstack([[Ri for Ri in R] for i in range(xi_shape[0])])
    elif len(xi_shape) == 2:
        if len(r_xi_shape) == 1:
            r_xi = np.array([r_xi for i in range(xi_shape[0])])
        if len(R.shape) == 1:
            R = np.array([Ri for i in range(xi_shape[0])])
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
        #if len(xi_shape) == 3:
            # probably need to transpose this to get it right!
            #sigma = sigma.reshape(output_shape)
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
        """
            if i % (n//20) == 0:
                print('{0:.2f}% done after {1:.2f} min ...\r'.format(
                    100*i/n, (time()-ti)/60), end='')
        print()
        """
    if verbose:
        t = time() - to
        print(f'Calculated {n} surface densities in {t/60:.2f},' \
              f' for an average time of {t/n:.2f} sec per call.')
    if len(output_shape) == 3:
        sigma = sigma.reshape(output_shape)
        R = R.reshape(output_shape)
    return 2 * rho_m * R * sigma, R


def _xi2sig_single(R, ln_rxi, ln_1plusxi, idx=None, verbose=0):
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

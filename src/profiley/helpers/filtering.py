"""k-space filtering utilities"""
from astropy import units as u
from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from pixell import enmap
import pixell.fft

from .binning import Bin2D

class Filter:
    """Class used to filter a real-space profile in Fourier space"""

    def __init__(self, filename):
        """
        Parameters
        ----------
        filename : str
            name of FITS file containing k-space mask. Must comply
            with `enmap` requirements
        """
        self.filename = filename
        self.kmask = enmap.read_map(self.filename)
        self.modrmap = self.kmask.modrmap()
        self.wcs = self.kmask.wcs

    def __repr__(self):
        return f"Filter('{self.filename}')"

    def __str__(self):
        return f'Filter: {self.filename}\n' \
               f'WCS: {self.wcs}'

    ### methods ###

    def filter(self, x, profile, bins, units='rad'):
        """Filter a given profile

        Parameters
        ----------
        x : array of floats, shape (N,)
            locations where the profile was calculated. If not provided,
            the mid points of the bin edges will be used.
        profile : array of floats, shape (N,)
            profile in real space
        bins : array of floats, shape (M,)
            bin edges where the profile is calculated. Min and max of
            `bin_edges` must be within the min and max of `x`
        units : {'arcmin', 'arcsec', 'deg', 'rad'}, optional
            bin units. Default 'rad'.

        Returns
        -------
        x_filtered : array of floats, shape (M-1,)
            mid points of bin_edges, where the filtered profile is
            calculated, in units given by `units`
        filtered : array of floats, shape (M-1,)
            filtered profile, in real space
        """
        _valid_units = ('arcmin', 'arcsec', 'deg', 'rad')
        assert units in ('arcmin', 'arcsec', 'deg', 'rad'), \
            f'arguments units must be one of {_valid_units}'
        assert x.size == profile.size
        # convert bins to rad
        if units != 'rad':
            conversion = \
                (1+(59*(units[:3] == 'arc'))) * (1+(59*(units == 'arcsec')))
            conversion = np.pi / 180 / conversion
            bins = conversion * bins
            x = conversion * x
        f = self.interp1d(x, profile)
        profile2d = enmap.enmap(f(self.modrmap), self.wcs)
        rmap_filtered = self.filter_map(profile2d)
        binner = Bin2D(self.modrmap, bins)
        xout, filtered = binner.bin(rmap_filtered)
        # convert back to input units
        if units != 'rad':
            xout = xout / conversion
        return xout, filtered

    def filter_map(self, xmap):
        kmap = pixell.fft.fft(xmap, axes=[-2,-1])
        # self.kmask loses its wcs somewhere somehow when I call it
        # in particular applications, so this helps bypass that
        _kmask = enmap.enmap(self.kmask.data, self.wcs)
        rmap = np.real(pixell.fft.ifft(
            kmap*_kmask, axes=[-2,-1], normalize=True))
        return enmap.enmap(rmap, xmap.wcs)

    def interp1d(self, x, y, bounds_error=False, fill_value=0, **kwargs):
        return interp1d(x, y, bounds_error=bounds_error,
                        fill_value=fill_value, **kwargs)


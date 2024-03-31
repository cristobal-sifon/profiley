"""Binning"""
import numpy as np


class Bin2D:

    def __init__(self, modrmap, bin_edges):
        self.centers = (bin_edges[1:]+bin_edges[:-1]) / 2
        self.digitized = np.digitize(modrmap.reshape(-1), bin_edges,right=True)
        self.bin_edges = bin_edges
        self.modrmap = modrmap

    def bin(self, data2d, weights=None, err=False, get_count=False, mask_nan=False):
        """
        At most one of `err` or `get_count` can be provided
        """
        assert not (err and get_count)
        if weights is None:
            if mask_nan:
                keep = ~np.isnan(data2d.reshape(-1))
            else:
                keep = np.ones(data2d.size, dtype=bool)
            count = np.bincount(self.digitized[keep])[1:-1]
            res = np.bincount(
                    self.digitized[keep], data2d.reshape(-1)[keep])[1:-1] \
                / count
            if err:
                meanmap = np.zeros(self.modrmap.size)
                for i in range(self.centers.size):
                    meanmap[self.digitized==i] = res[i]
                diff = ((data2d-meanmap.reshape(self.modrmap.shape))**2).reshape(-1)
                std = (np.bincount(self.digitized[keep], diff[keep])[1:-1] \
                    / (count*(count-1)))**0.5
                assert np.allclose(std1, std)
        else:
            count = np.bincount(self.digitized, weights.reshape(-1))[1:-1]
            res = np.bincount(
                    self.digitized, (data2d*weights).reshape(-1))[1:-1] \
                /count
        if get_count:
            return self.centers, res, count
        if err:
            return self.centers, res, std
        return self.centers, res

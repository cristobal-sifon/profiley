import numpy as np

from .nfw import BaseNFW


class tNFW5(BaseNFW):
    """INCOMPLETE (but you get the idea)"""

    def __init__(self, mass, c, rt, z, **kwargs):
        super().__init__(mass, c, z, **kwargs)
        self.rt = rt

    ### methods ###

    @inMpc
    def density(self, R):
        return

    ### auxiliary methods ###

    @inMpc
    def F(self, R):
        x = R / self.rs
        f = np.ones_like(x)
        f[x < 1] = np.log(1/x[x<1] + (1/x[x<1]**2 - 1)**0.5) \
            / (1 - x[x<1]**2)**0.5
        f[x > 1] = np.arccos(1/x[x>1]) / (x[x>1]**2 - 1)**0.5
        return f

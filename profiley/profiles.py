"""Profile types definition

Attributes
----------
DensityProfile

Instructions
------------
There are three possibilities for defining a new type of profile. The
one with the most functionality is

.. code-block ::

    class NewProfileType(BaseLensing, Profile):
        def __init__(self, *args, cosmo=cosmo, numeric_kwargs={}, **kwargs):
            super().__init__(z, cosmo=cosmo, **numeric_kwargs)
            # ...

inherits all lensing functionality. As indicated above, this profile
type must define a redshift. A profile type that does not include
lensing utilities can be defined as

.. code-block ::

    class NewProfileType(BaseCosmo, Profile):
        def __init__(self, *args, cosmo=cosmo, numeric_kwargs={}, **kwargs):
            super().__init__(z, cosmo=cosmo, **numeric_kwargs)
            # ...

inherits a few cosmology-related quantities, such as the background
density, and also requires a redshift. Finally, a minimal type can
be defined through

.. code-block ::

    class NewProfileType(Profile):
        def __init__(self, *args, numeric_kwargs={}, **kwargs):
            super().__init__(**numeric_kwargs)
            # ...

which only inherits basic profile functionality, related to projection
along the line of sight.

As is evident, the keyword argument ``numeric_kwargs`` belongs to the
``Profile`` class, and relates to the accuracy required for numerical
integration of the profile along the line of sight
"""

from astropy import units as u
from astropy.cosmology import Planck15
import numpy as np
from scipy.integrate import cumtrapz, quad, simps

try:
    import pixell.enmap
    import pixell.utils
    has_pixell = True
except ImporError:
    has_pixell = False

from .core import Profile
from .helpers.decorators import array, inMpc
from .helpers.lensing import BaseLensing


#class DensityProfile(Profile, BaseLensing):
class DensityProfile(BaseLensing, Profile):

    def __init__(self, mass, z, cosmo=Planck15, numeric_kwargs={}):
        """Density profile object

        Parameters
        ----------
        mass : float, astropy.units.Quantity or np.ndarray
            total mass(es) (definition arbitrary)
        z : float or np.ndarray
            redshift
        cosmo : ``astropy.cosmology.FLRW`` object, optional
            cosmology object
        numeric_kwargs : dict, optional
            see ``Profile``
        """
        if isinstance(mass, u.Quantity):
            mass = mass.to(u.Msun).value
        if not np.iterable(mass):
            mass = np.array([mass])
        self.mass = mass
        self._shape = self.mass.shape
        self.z = self._define_array(z)
        #Profile().__init__(**numeric_kwargs)
        #BaseLensing().__init__(self.z, cosmo=cosmo)
        #Profile(**numeric_kwargs).__init__()
        #BaseLensing(self.z, cosmo=cosmo).__init__()
        super().__init__(self.z, cosmo=cosmo, **numeric_kwargs)


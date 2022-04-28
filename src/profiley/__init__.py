"""Profile objects for galaxy cluster modeling

Instructions
------------
There are three possibilities for defining a new type of profile. The
one with the most functionality is

.. code-block ::

    class NewProfile(BaseLensing, Profile):
        def __init__(self, *args, cosmo=cosmo, numeric_kwargs={}, **kwargs):
            super().__init__(z, cosmo=cosmo, **numeric_kwargs)
            # ...

inherits all lensing functionality. As indicated above, this profile
type must define a redshift. A profile type that does not include
lensing utilities can be defined as

.. code-block ::

    class NewProfile(BaseCosmo, Profile):
        def __init__(self, *args, cosmo=cosmo, numeric_kwargs={}, **kwargs):
            super().__init__(z, cosmo=cosmo, **numeric_kwargs)
            # ...

inherits a few cosmology-related quantities, such as the background
density, and also requires a redshift. Finally, a minimal type can
be defined through

.. code-block ::

    class NewProfile(Profile):
        def __init__(self, *args, numeric_kwargs={}, **kwargs):
            super().__init__(**numeric_kwargs)
            # ...

which only inherits basic profile functionality, related to projection
along the line of sight.

As is evident, the keyword argument ``numeric_kwargs`` belongs to the
``Profile`` class, and relates to the accuracy required for numerical
integration of the profile along the line of sight.

the parent classes can be imported with

from .core import Profile
from .helpers.cosmology import BaseCosmo
from .helpers.lensing import BaseLensing
"""

__version__ = '1.4.0'

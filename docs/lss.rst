Two-halo terms
===================================

It is possible to calculate the contribution
from large-scale structure, usually known as the "two-halo term". Doing
this requires calculation of the matter power spectrum, which is not
implemented in ``profiley``, but given a power spectrum ``profiley`` provides
convenience functions to calculate the surface density.

There are several steps involved in this calculation; for a full working
example see `this notebook
<https://github.com/cristobal-sifon/profiley/blob/master/examples/twohalo.ipynb>`_,
where we use the `Core Cosmology Library
<https://ccl.readthedocs.io/en/latest>`_ to obtain the matter power spectrum.

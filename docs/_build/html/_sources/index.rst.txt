.. profiley documentation master file, created by
   sphinx-quickstart on Fri Dec 25 20:04:05 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

profiley
========

``profiley`` is an object-oriented implementation of some of the most common astrophysical profiles
used to describe galaxies and clusters of galaxies.

Installation
++++++++++++

``profiley`` is available through the Python Package Index (`PyPI 
<https://pypi.org/>`_), which means installation is as easy as

.. code-block::

    pip install profiley

Alternatively, you may download the development version by forking or cloning 
the `github repository <https://github.com/cristobal-sifon/profiley>`_.


Basic Usage
+++++++++++

For example, to obtain the surface density profile of an NFW profile we simply need the following:

.. code-block:: python

    import numpy as np
    from profiley.nfw import NFW

    mass = 1e14
    concentration = 5
    z = 0.5
    nfw = NFW(mass, concentration, z)

    R = np.logspace(-1, 2, 10)
    rho = nfw.surface_density(R)

``profiley`` can handle ``np.ndarray`` objects of any shape, provided all the 
arguments can be multiplied without manipulation. For instance,

.. code-block:: python

    mass = np.logspace(14, 15, 11)
    concentration = np.linspace(4, 5, 3)[:,None]
    z = np.linspace(0.2, 1, 5)[:,None,None]

    nfw = NFW(mass, concentration, z)

will produce an array of profiles with shape ``(R.size,5,3,11)``.



License
+++++++

``profiley`` is free software and is distributed with an MIT License. See `License`_.


.. toctree::
    :maxdepth: 2
    :caption: Contents:

    profiles/profile
    profiles/index
    profiles/custom
    LICENSE


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. _License: LICENSE.html

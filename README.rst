========
profiley
========

Profile calculations for galaxies and clusters

.. image:: https://readthedocs.org/projects/profiley/badge/?version=latest
    :target: https://profiley.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

To install, simply run ::

    pip install profiley

Some examples of the functionality available can be found in the `examples 
<examples/>`_. For more information see the `documentation 
<https://profiley.readthedocs.io/en/latest/index.html>`_.


References
==========

``profiley`` has been used in `Madhavacheril et al. (2020)`__ and `Shirasaki et al. (2024)`__.


Change Log
==========

Version 2.0.0 (ongoing)
-----------------------

* Renamed all relevant profiles from ``enclosed`` to ``cumulative``
* Fix critical bugs on critical surface density and background density
* Added API documentation

Version 1.5.0
-------------
* Added ``WebskyNFW`` profile
* Fixed bug in ``mdelta`` method, plus it is now a ``Profile`` method inherited by all profiles.

Version 1.4.4
-------------
* Fixed bug in the normalization of a 1-d surface density in ``xi2sigma``
* Fixed bug whereby only one beta was returned

Version 1.4.3
-------------
* Fixed bug calculating a single surface density in ``xi2sigma``

Version 1.4.0
-------------
* Added stand-alone offset calculation for arbitrary profiles

Version 1.3.2
-------------
* Fixed bug in installation (!)

Version 1.3.0
-------------
* Updated names of ``Profile`` methods
* Bug fixes

Version 1.2.2
-------------

* Moved ``background`` argument from ``BaseNFW`` to ``BaseCosmo`` so it is visible to all ``Profile`` objects
* Moved ``overdensity`` argument from ``BaseNFW`` to ``Profile``



.. _Madhavacheril: https://ui.adsabs.harvard.edu/abs/2020ApJ...903L..13M/abstract
.. _Shirasaki: https://ui.adsabs.harvard.edu/abs/2024arXiv240708201S/abstract

__ Madhavacheril_
__ Shirasaki_



========
profiley
========

Profile calculations for galaxies and clusters

To install, simply run ::

    pip install profiley

Some examples of the functionality available can be found in the `examples 
<examples/>`_. For more information see the `documentation 
<https://profiley.readthedocs.io/en/latest/index.html>`_.


References
==========

``profiley`` has been used in `Madhavacheril et al. (2020)`__.


Change Log
==========

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

__ Madhavacheril_



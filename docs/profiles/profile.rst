The ``Profile`` base class
==========================

All profiles in ``profiley`` inherit from the ``Profile`` base class, which 
implements numerical calculation of all methods, starting from a ``profile`` 
method in which the three-dimensional profile is calculated. The ``Profile``
class allows for all profiles to have a unique API with the following methods:

* ``profile(R)``: three-dimensional profile
* ``surface_density(R)``: two-dimensional projected profile
* ``enclosed_surface_density(R)``: cumulative two-dimensional projected profile
* ``excess_surface_density(R)``: difference between the two above (most commonly used as the weak gravitational lensing observable)

Where available, these methods are implemented using the analytical expressions;
otherwise they are calculated numerically. All projected profiles also have a
variant where the profile is offset from the reference position:

* ``offset_surface_density(R, R_off)``
* ``offset_enclosed_surface_density(R, R_off)``
* ``offset_excess_surface_density(R, R_off)``

There are a few optional arguments that can be passed to ``Profile`` upon
initialization, which are used by parent classes that extend the functionality
of ``Profile`` objects:

* ``z_s``: Source redshift. If not specified, it can be passed separately to methods requiring it
* ``cosmo``: Astropy cosmology ``FLRW`` object
* ``frame``: whether calculations proceed in comoving or proper coordinates


Cosmology
---------

The cosmology in which a ``Profile`` object is embedded is specified through the
``cosmo`` optional argument, which must be any ``astropy.cosmology.FLRW`` object.
This allows for the definition of the background density as well as calculations
of distances detailed below.


Gravitational lensing functionality
-----------------------------------

The ``Profile`` class inherits from the ``BaseLensing`` helper class,
which implements quantities relevant for gravitational lensing analysis.

Attributes inherited from this class:

* ``z_s``: source redshift
* ``Dl``: angular diameter distance from observer to lens object
* ``Dls``: angular diameter distance between lens and lensed source
* ``Ds``: angular diameter distance from observer to lensed source

Methods inherited from this class:

* ``beta``: :math:`\max(0, D_\mathrm{ls}/D_\mathrm{s})`
* ``convergence``: lensing convergence
* ``offset_convergence``
* ``sigma_crit``: critical surface density


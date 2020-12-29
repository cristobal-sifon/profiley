The ``Profile`` base class
==========================

All profiles in ``profiley`` inherit from the ``Profile`` base class, which 
implements numerical calculation of all methods, starting from a ``profile`` 
method in which the three-dimensional profile is calculated. The ``Profile``
class allows for all profiles to have a unique API:

+--------------------------------------------------------------------------------------------------------------------------------------+
| Methods defined in this class                                                                                                        |
+=================================+====================================================================================================+
| ``enclosed_surface_density(R)`` | cumulative two-dimensional projected profile                                                       |
+---------------------------------+----------------------------------------------------------------------------------------------------+
| ``excess_surface_density(R)``   | difference between the two above (most commonly used as the weak gravitational lensing observable) |
+---------------------------------+----------------------------------------------------------------------------------------------------+
| ``profile(R)``                  | three-dimensional profile                                                                          |
+---------------------------------+----------------------------------------------------------------------------------------------------+
| ``surface_density(R)``          | two-dimensional projected profile                                                                  |
+---------------------------------+----------------------------------------------------------------------------------------------------+

Where available, these methods are implemented using the analytical expressions;
otherwise they are calculated numerically. All projected profiles also have a
variant where the profile is offset from the reference position:

+-----------------------------------------------+
| Additional methods                            |
+===============================================+
| ``offset_enclosed_surface_density(R, R_off)`` |
+-----------------------------------------------+
| ``offset_excess_surface_density(R, R_off)``   |
+-----------------------------------------------+
| ``offset_surface_density(R, R_off)``          |
+-----------------------------------------------+

In the above, ``R_off`` should be either a ``float`` or a ``np.ndarray``.


What follows are the descriptions of helper classes from which ``Profile`` inherits. These classes
are not to be instantiated directly, but the description is separated for clarity.


----


``BaseCosmo``: Cosmology
------------------------

The cosmology in which a ``Profile`` object is embedded is specified through the
``cosmo`` optional argument, which must be any ``astropy.cosmology.FLRW`` object.
This allows for the definition of the background density as well as calculations
of distances detailed below.

+------------------------------------------------------------------------------------------------------------------------+
| Optional arguments inherited from this class                                                                           |
+================+===============================+=======================================================================+
| ``background`` |         ``{'c','m'}``         | Whether overdensities are defined w.r.t. the critical or mean density |
+----------------+-------------------------------+-----------------------------------------------------------------------+
|   ``cosmo``    |   ``astropy.cosmology.FLRW``  | Cosmology (default: ``Planck15``)                                     |
+----------------+-------------------------------+-----------------------------------------------------------------------+
|   ``frame``    |  ``{'comoving','physical'}``  | Whether to work in comoving or physical coordinates                   |
+----------------+-------------------------------+-----------------------------------------------------------------------+

+---------------------------------------------------------------------------------------------------------------------------------------------+
| Attributes inherited from this class                                                                                                        |
+======================+================+=====================================================================================================+
| ``critical_density`` | ``np.ndarray`` | critical density of the universe at all supplied redshifts                                          |
+----------------------+----------------+-----------------------------------------------------------------------------------------------------+
| ``mean_density``     | ``np.ndarray`` | mean density of the universe at all supplied redshifts                                              |
+----------------------+----------------+-----------------------------------------------------------------------------------------------------+
| ``rho_bg``           | ``np.ndarray`` | alias for either ``critical_density`` or ``mean_density`` depending on the ``background`` attribute |
+----------------------+----------------+-----------------------------------------------------------------------------------------------------+


----


``BaseLensing``: Gravitational lensing functionality
----------------------------------------------------

The ``Profile`` class inherits from the ``BaseLensing`` helper class,
which implements quantities relevant for gravitational lensing analysis.

+----------------------------------------------+
| Optional arguments inherited from this class |
+=========+===========+========================+
| ``z_s`` | ``float`` | source redshift        |
+---------+-----------+------------------------+

+--------------------------------------------------------------------------------+
| Attributes inherited from this class                                           |
+=========+===========+==========================================================+
| ``Dl``  | ``float`` | angular diameter distance from observer to lens object   |
+---------+-----------+----------------------------------------------------------+
| ``Dls`` | ``float`` | angular diameter distance between lens and lensed source |
+---------+-----------+----------------------------------------------------------+
| ``Ds``  | ``float`` | angular diameter distance from observer to lensed source |
+---------+-----------+----------------------------------------------------------+

+---------------------------------------------------------------------------------------+
| Methods inherited from this class                                                     |
+=========================================+=============================================+
| ``beta([z_s])``                         | :math:`\max(0, D_\mathrm{ls}/D_\mathrm{s})` |
+-----------------------------------------+---------------------------------------------+
| ``convergence(R[, z_s])``               | lensing convergence                         |
+-----------------------------------------+---------------------------------------------+
| ``offset_convergence(R, R_off[, z_s])`` | offset lensing convergence                  |
+-----------------------------------------+---------------------------------------------+
| ``sigma_crit([z_s])``                   | critical surface density                    |
+-----------------------------------------+---------------------------------------------+

In all the methods above, the source redshift, ``z_s``, may be specified as a 
keyword argument, in which case it will override the ``self.z_s`` attribute *for 
that particular call of the method only*.

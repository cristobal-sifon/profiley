The ``Profile`` base class
==========================

All profiles in ``profiley`` inherit from the ``Profile`` base class, which 
implements numerical calculation of all methods, starting from a ``profile`` 
method in which the three-dimensional profile is calculated. The ``Profile`` 
class allows for all profiles to have a unique API, described in the following.

+-------------------------------------------------------------------------------------------------------------+
| Methods defined in this class                                                                               |
+=======================================+=====================================================================+
| ``profile(R)``                        | three-dimensional profile                                           |
+---------------------------------------+---------------------------------------------------------------------+
| ``projected(R, **kwargs)``            | Line-of-sight projected profile                                     |
+---------------------------------------+---------------------------------------------------------------------+
| ``projected_cumulative(R, **kwargs)`` | Line-of-sight projected cumulative profile                          |
+---------------------------------------+---------------------------------------------------------------------+
| ``projected_excess(R, **kwargs)``     | difference between ``projected_cumulative(R)`` and ``projected(R)`` |
+---------------------------------------+---------------------------------------------------------------------+

where ``R`` are the radii at which to return the profiles. Where available, 
these methods are implemented using the analytical expressions; otherwise they 
are calculated numerically. Additional arguments absorbed in ``kwargs`` relate 
to the precision (and speed) of numerical integration. See below.


Numerical profile projections
+++++++++++++++++++++++++++++

If the projections of a given profile do not have analytical forms, they are 
calculated by numerical integration, using ``scipy.integrate.simps``.

Projected profile
-----------------

The projection of the profile along the 
line of sight, :math:`\Sigma(R)`, can be calculated as follows:

.. math::

    \Sigma(R) = 2\int_0^{+\infty} dr \rho(\sqrt{r^2+R^2})




Cumulative projected profile
----------------------------

The enclosed (or cumulative) projected profile is then

.. math::

    \Sigma(<R) = \frac2{R^2}\int_0^R dr\,r\Sigma(r)


Excess projected profile
------------------------

The excess projected profile is defined as

.. math::

    \Delta\Sigma(R) = \Sigma(<R) - \Sigma(R)

This quantity is particularly useful in weak gravitational lensing studies, 
where :math:`\Delta\Sigma(R)` is the excess surface density (ESD), which is 
directly related to the weak lensing shape distortion, called *shear*, 
:math:`\gamma`, through :math:`\gamma=\Delta\Sigma/\Sigma_\mathrm{c}`, where
:math:`\Sigma_\mathrm{c}` is the critical surface density (see `Lensing functionality <#lensing>`_).


Offset profiles
---------------

It is also possible to calculate a given profile projection when the reference 
point is not the center of the profile itself. If the projected distance between 
the reference point and the center of the profile is :math:`R_\mathrm{off}`, 
then the measured projected profile is

.. math::

    \Sigma_\mathrm{off}(R,R_\mathrm{off}) = \frac1{2\pi}
        \int_0^{2\pi}d\theta\,
            \Sigma\left(
                \sqrt{R_\mathrm{off}^2 + R^2 + 2RR_\mathrm{off}\cos\theta}
            \right)

and analogously for other projections. These offset profiles are calculated 
numerically in the following methods:

+-----------------------------------------------------+
| ``offset_profile(R, R_off, **kwargs)``              |
+-----------------------------------------------------+
| ``offset_projected(R, R_off, **kwargs)``            |
+-----------------------------------------------------+
| ``offset_projected_cumulative(R, R_off, **kwargs)`` |
+-----------------------------------------------------+
| ``offset_projected_excess(R, R_off, **kwargs)``     |
+-----------------------------------------------------+

In the above, ``R_off`` should be either a ``float`` or a 1-d ``np.ndarray``.


----

.. inheritance:

Inheritance
+++++++++++

What follows are the descriptions of helper classes from which ``Profile`` inherits. These classes
are not to be instantiated directly, but the description is separated for clarity.


----

.. cosmology:

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

.. _lensing:

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

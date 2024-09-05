The ``Profile`` base class
==========================

All profiles in ``profiley`` inherit from the ``Profile`` base class, which
implements numerical calculations of all methods, starting from a ``profile``
method in which the three-dimensional profile is defined. The ``Profile`` base
class allows for all profiles to have a common, simple API. See all implemented
profiles `here <../index.html>`_.

+-------------------------------------------------------------------------------------------------------------+
| Methods defined in this class                                                                               |
+=======================================+=====================================================================+
| ``profile(R)``                        | three-dimensional profile                                           |
+---------------------------------------+---------------------------------------------------------------------+
| ``cumulative(R)``                     | mean value within ``R``                                             |
+---------------------------------------+---------------------------------------------------------------------+
| ``mass_cumulative(R)``                | spherical integral within ``R``                                     |
+---------------------------------------+---------------------------------------------------------------------+
| ``projected(R)``                      | Line-of-sight projected profile                                     |
+---------------------------------------+---------------------------------------------------------------------+
| ``projected_cumulative(R)``           | Line-of-sight projected cumulative profile                          |
+---------------------------------------+---------------------------------------------------------------------+
| ``projected_excess(R)``               | difference between ``projected_cumulative(R)`` and ``projected(R)`` |
+---------------------------------------+---------------------------------------------------------------------+
| ``potential(R)``                      | gravitational potential                                             |
+---------------------------------------+---------------------------------------------------------------------+
| ``escape_velocity(R)``                | escape velocity                                                     |
+---------------------------------------+---------------------------------------------------------------------+
| ``mdelta(overdensity, background)``   | mass within a specified overdensity                                 |
+---------------------------------------+---------------------------------------------------------------------+
| ``rdelta(overdensity, background)``   | radius enclosing a specified overdensity                            |
+---------------------------------------+---------------------------------------------------------------------+

where ``R`` are the radii at which to return the profiles. Where available,
these methods are implemented using the analytical expressions; otherwise they
are calculated numerically. Additional arguments absorbed in ``kwargs`` relate
to the precision (and speed) of numerical integration. See below.

.. numerical:

Numerical calculations
++++++++++++++++++++++

If any of the above methods are not defined analytically, they are
calculated by numerical integration, using ``scipy.integrate.simps``.


Cumulative profile
------------------

Mean value of the profile within the specified radius:

.. math::

    \bar\rho(R) = \frac{\int_{V(<R)} d^3r\,\rho(r)}{\int_{V(<R)} d^3r}


Cumulative mass
---------------

Spherical integral within a radius ``R``. For a density profile this corresponds
to the enclosed mass:

.. math::

    M(<R) = \int_{V(<R)}d^3r\,\rho(r) = 4\pi\int_0^R dr\,r^2\rho(r)


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
:math:`\Sigma_\mathrm{c}` is the critical surface density (see `Lensing`_).

Gravitational potential
-----------------------

The Poisson equation implies that the gravitational potential can be calculated as

.. math::

    \phi(r) = 4\pi G\left[\frac1r\int_0^r \rho(y)y^2\,dy + \int_r^\infty \rho(y)y\,dy\right]

The gravitational potential is returned in units of :math:`\mathrm{Mpc^2\,s^{-2}}`. 

Escape velocity
---------------

The escape velocity is related to the gravitational potential through

.. math::

    v_\mathrm{esc}(r) = \sqrt{2\phi(r)}

and is returned in :math:`\mathrm{km\,s^{-1}}`.


Precision of numerical integration
----------------------------------

Several optional arguments allow the user to find their own sweet-spot in the
trade-off between precision and execution time. Using the default parameters the
precision of the numerical calculations for an NFW profile is better than 0.1%
at all radii, as demonstrated in the |ex-numeric|_.


.. offset:

Offset profiles
+++++++++++++++

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
numerically using the ``offset`` wrapper:

.. code-block::

    offset(func, R, R_off, theta_samples=360, weights=None, **kwargs)

where ``func`` is any of the methods above, ``R_off`` is either a ``float`` or a
1-d ``np.ndarray``, ``theta_samples`` sets the precision for the angular
integral, and ``weights`` is a weight array applied to average the profiles
evaluated at different :math:`R_\mathrm{off}`. Each of the projected profiles
is implemented in the following convenience functions:

+-----------------------------------------------------+
| ``offset_profile(R, R_off, **kwargs)``              |
+-----------------------------------------------------+
| ``offset_projected(R, R_off, **kwargs)``            |
+-----------------------------------------------------+
| ``offset_projected_cumulative(R, R_off, **kwargs)`` |
+-----------------------------------------------------+
| ``offset_projected_excess(R, R_off, **kwargs)``     |
+-----------------------------------------------------+

Stand-alone offset functionality
--------------------------------

Since ``v1.4.0`` it is also possible to offset an arbitrary profile given as an
array, using the ``offset`` function within the ``numeric`` module:

.. code-block::

    from profiley.nfw import NFW
    from profiley.numeric import offset

    mass = 1e14
    concentration = 5
    z = 0.2
    nfw = NFW(mass, concentration, z)

    R = np.logspace(-1, 1, 20)
    sigma = nfw.projected(R)

    Roff = np.arange(0, 1, 10)
    weights = np.normal(0.2, 0.1, Roff.size)
    sigma_off = offset(sigma, R, Roff, weights=weights)

For more details, see the |ex-offset|_.

In fact, the latter implementation is about an order of magnitude faster
than the ``Profile`` method described above, and should be preferred for
the time being. The current methods will be replaced by this implementation in the future.


A note on the radial coordinate
+++++++++++++++++++++++++++++++

All examples in these docs employ one-dimensional radial arrays, ``R``, to calculate profiles. In fact, ``profiley`` can manage ``R`` of any shape. The resulting profiles will depend on the shape of ``R``: dimensions will be added to ``R`` to the extent that they are needed to be able to multiply ``R`` with an array of shape ``p.shape``. Below are a few examples, assuming ``p`` is a ``Profile`` object with ``shape=(12, 7, 5)``. For instance,

.. code-block:: python

    mass = np.logspace(14, 15, 5)
    concentration = np.linspace(2, 9, 7)
    z = np.linspace(0, 1, 12)
    p = NFW(mass, concentration[:,None], z[:,None,None])

The shape of the result of any of ``p``'s `profile methods <profiles/Profile/index.html>`_ will be as follows:

+-----------------+----------------------+
| ``R.shape``     | profile shape        |
+=================+======================+
| ``(25,)``       | ``(25,12,7,5)``      |
+-----------------+----------------------+
| ``(25,100)``    | ``(25,100,12,7,5)``  |
+-----------------+----------------------+
| ``(25,12)``     | ``(25,12,7,5)``      |
+-----------------+----------------------+
| ``(25,12,7)``   | ``(25,12,7,5)``      |
+-----------------+----------------------+
| ``(25,12,7,5)`` | ``(25,12,7,5)``      |
+-----------------+----------------------+
| ``(25,1,7)``    | ``(25,12,7,5)``      |
+-----------------+----------------------+
| ``(25,12,5)``   | ``(25,12,5,12,7,5)`` |
+-----------------+----------------------+

Etc. That is, if the last N dimensions of ``R`` match the first N dimensions of ``p`` (considering empty dimensions appropriately), they will be assumed to correspond to each set of profiles. Exceptional situations, e.g., when the number of radial elements (or the last dimension) is equal to the first element of ``p.shape`` but it is not meant to represent one radial vector per profile, will not behave as expected. Such fringe cases must be appropriately handled by the user, but should generally be avoided.


.. inheritance:

Inheritance
+++++++++++

What follows are the descriptions of helper classes from which ``Profile``
inherits. These classes are not to be instantiated directly, but the description
of attributes and methods defined within these classes is separated for clarity.



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


.. _lensing:

``Lens``: Gravitational lensing functionality
---------------------------------------------

The ``Profile`` class inherits from the ``Lens`` helper class,
which implements quantities relevant for gravitational lensing analysis.

+--------------------------------------------------------------------------------------+
| Attributes inherited from this class                                                 |
+=========+===========+================================================================+
| ``Dl``  | ``float`` | angular diameter distance from observer to lens object, in Mpc |
+---------+-----------+----------------------------------------------------------------+

+-----------------------------------------------------------------------------------------------+
| Methods inherited from this class                                                             |
+=======================================+=======================================================+
| ``Dls(z_s)``                          | angular diameter distance from lens to source, in Mpc |
+---------------------------------------+-------------------------------------------------------+
| ``Dls_over_Ds(z_s)``                  | :math:`\max(0, D_\mathrm{ls}/D_\mathrm{s})`           |
+---------------------------------------+-------------------------------------------------------+
| ``convergence(R, z_s)``               | lensing convergence                                   |
+---------------------------------------+-------------------------------------------------------+
| ``offset_convergence(R, R_off, z_s)`` | offset lensing convergence                            |
+---------------------------------------+-------------------------------------------------------+
| ``sigma_crit(z_s)``                   | critical surface density                              |
+---------------------------------------+-------------------------------------------------------+

In all these methods, ``z_s`` is the source redshift.

.. include:: ../../github-links.rst
NFW-like profiles
=================

``profiley`` implements the most common NFW-like profiles, all of which are 
under the ``nfw`` module.

.. nfw:

NFW profile
+++++++++++

The ``NFW`` class implements the original NFW profile from Navarro et al. (2015),

.. math::

    \rho(r) = \frac{\delta_\mathrm{c}\rho_\mathrm{bg}}{x(1+x)^2}

where :math:`x=r/r_\mathrm{s}`. 
The signature is

.. code-block::

    from profiley.nfw import NFW
    NFW(mass, concentration, z, **kwargs)

where ``kwargs`` are passed to ``BaseLensing`` and ``BaseCosmo``. See
`Inheritance <../profile.html#inheritance>`_.


.. gnfw:

Generalized NFW profile
+++++++++++++++++++++++

The ``GNFW`` class implements a generalized NFW profile,

.. math::

    \rho(r) = \frac{\delta_\mathrm{c}\rho_\mathrm{bg}}
                   {x^\gamma\left(1+x^\alpha\right)^{(\beta-\gamma)/\alpha}}

The signature is

.. code-block::

    from profiley import GNFW
    GNFW(mass, concentration, z, alpha=alpha, beta=beta, gamma=gamma, **kwargs)

Using default values for ``alpha``, ``beta``, and ``gamma`` results in the 
regular NFW profile.


.. tnfw:

Truncated NFW profile
+++++++++++++++++++++

The ``TNFW`` class implements a truncated NFW profile,

.. math::

    \rho(r) = \frac{\delta_\mathrm{c}\rho_\mathrm{bg}}{x(1+x)^2}
              \left(\frac{\tau^2}{\tau^2+x^2}\right)^\eta

where :math:`\tau=r_\mathrm{t}/r_\mathrm{s}`, with
:math:`r_\mathrm{s}` the scale radius and :math:`r_\mathrm{t}` the truncation radius.
Analytical expressions have been derived for the cases :math:`\eta=\{1,2\}` by `Baltz,
Marshall & Oguri (2009) <http://adsabs.harvard.edu/abs/2009JCAP...01..015B>`_, but
they have not yet been implemented in ``profiley``, which means the projections of
TNFW profiles are calculated numerically.

The signature is

.. code-block::

    from profiley.nfw import TNFW
    TNFW(mass, concentration, z, tau, eta, **kwargs)


.. hernquist:

Hernquist profile
+++++++++++++++++

The ``Hernquist`` class implements the `Hernquist (1990)
<https://ui.adsabs.harvard.edu/abs/1990ApJ...356..359H/abstract>`_ profile,
which is a special case of the GNFW profile with :math:`\alpha=1`,
:math:`\beta=4`, and :math:`\gamma=1`.

The signature is

.. code-block::

    from profiley.nfw import Hernquist
    Hernquist(mass, concentration, z, **kwargs)

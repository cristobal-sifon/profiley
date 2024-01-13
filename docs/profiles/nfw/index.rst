NFW-like profiles
=================

``profiley`` implements the most common NFW-like profiles, all of which are 
under the ``nfw`` module.

.. _nfw:

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

where ``kwargs`` are passed to ``Lens`` and ``BaseCosmo``. See
`Inheritance <../Profile/index.html#inheritance>`_.


.. _gnfw:

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


.. _tnfw:

Truncated NFW profile
+++++++++++++++++++++

The ``TNFW`` class implements a truncated NFW profile,

.. math::

    \rho(r) = \frac{\delta_\mathrm{c}\rho_\mathrm{bg}}{x(1+x)^2}
              \left(\frac{\tau^2}{\tau^2+x^2}\right)^\eta

where :math:`\tau=r_\mathrm{t}/r_\mathrm{s}`, with
:math:`r_\mathrm{s}` the scale radius and :math:`r_\mathrm{t}` the truncation radius.
Analytical expressions have been derived for the cases :math:`\eta=\{1,2\}` by |Baltz2009|_, but
they have not yet been implemented in ``profiley``, which means the projections of
TNFW profiles are calculated numerically.

The signature is

.. code-block::

    from profiley.nfw import TNFW
    TNFW(mass, concentration, z, tau, eta, **kwargs)


.. _hernquist:

Hernquist profile
+++++++++++++++++

The ``Hernquist`` class implements the |Hernquist1990|_ profile,
which is a special case of the GNFW profile with :math:`\alpha=1`,
:math:`\beta=4`, and :math:`\gamma=1`.

The signature is

.. code-block::

    from profiley.nfw import Hernquist
    Hernquist(mass, concentration, z, **kwargs)


.. _webskynfw:

Websky Modified NFW
+++++++++++++++++++

The ``WebskyNFW`` class implements the modified NFW profile adopted for the Websky simulations by |Websky|_,

.. math::

    \rho(r) = \begin{cases}
        \rho_\mathrm{NFW}(r)                                            & r < r_\mathrm{200m} \\
        \rho_\mathrm{NFW}(r)\left(\frac{r}{r_\mathrm{200m}}\right)^{-\alpha}  & r_\mathrm{200m} < r < 2r_\mathrm{200m} \\
        0                                                               & r > r_\mathrm{200m}
    \end{cases}

where :math:`r_\mathrm{200m}` is the radius enclosing 200 times the mean matter density at the specified redshift.
The Websky simulations adopt :math:`\alpha=2` but it can be modified here for additional freedom.

The signature is

.. code-block::

    from profiley.nfw import WebskyNFW
    WebskyNFW(mass, concentration, z, **kwargs)


.. include:: ../../reference-links.rst

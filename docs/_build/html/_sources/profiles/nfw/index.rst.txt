NFW-like profiles
=================

``profiley`` implements the most common NFW-like profiles. Specifically,

* The ``NFW`` class implements the original NFW profile from Navarro et al. (2015),

.. math::

    \rho(r) = \frac{\delta_\mathrm{c}\rho_\mathrm{bg}}{x(1+x)^2}

* The ``GNFW`` class implements a generalized NFW profile,

.. math::

    \rho(r) = \frac{\delta_\mathrm{c}\rho_\mathrm{bg}}
                   {x^\gamma\left(1+x^\alpha\right)^{(\beta-\gamma)/\alpha}}

* The ``TNFW`` class implements a truncated NFW profile,

.. math::

    \rho(r) = \frac{\delta_\mathrm{c}\rho_\mathrm{bg}}{x(1+x)^2}
              \left(\frac{\tau^2}{\tau^2+x^2}\right)^\eta

where :math:`x=r/r_\mathrm{s}` and :math:`\tau=r_\mathrm{t}/r_\mathrm{s}`, with
:math:`r_\mathrm{s}` the scale radius and :math:`r_\mathrm{t}` the truncation radius.

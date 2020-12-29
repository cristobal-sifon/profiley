Einasto profile
===============

The ``einasto`` module implements the Einasto (1965) density profile,

.. math::

    \rho(r) = \rho_\mathrm{s}
        \exp\left(-\frac2\alpha
                \left[\left(\frac{r}{r_\mathrm{s}}\right)^\alpha - 1\right]
            \right)

where :math:`\alpha>0`. The calling signature is

.. code-block::

    from profiley.einasto import Einasto
    Einasto(rho_s, r_s, alpha, **kwargs)

Unlike the NFW profile, the mass profile derived from this density profile
converges to a total mass

.. math::

    M_\mathrm{tot} = \frac{4\pi\rho_\mathrm{s}r_\mathrm{s}^3}{\alpha}
        \left(\frac\alpha2\right)^{3/\alpha}\Gamma(3/\alpha)\exp(2/\alpha)

and the mass profile is described by

.. math::

    M(x) = M_\mathrm{tot}
        \frac{\Gamma(3/\alpha) - \Gamma(3/\alpha,2x^\alpha/\alpha)}
             {\Gamma(3/\alpha)}

where as usual :math:`x=r/r_\mathrm{s}` (`Cardone, Piedipalumbo and Tortora, 2005 
<https://ui.adsabs.harvard.edu/abs/2005MNRAS.358.1325C/abstract>`_).

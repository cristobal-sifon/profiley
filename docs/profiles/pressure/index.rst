Pressure profiles
=================

This module does not contain new functional forms, but implements GNFW profiles in a particular parameterization, commonly used to describe the pressure profiles of galaxy clusters. The most general pressure profile object is the ``PressureGNFW``, which defines the following profile:

.. math::

    P(r) = \frac{P_0}{x^\gamma(1+x^\alpha)^{(\beta-\gamma)/\alpha}} P_\Delta

where :math:`\Delta` is a spherical overdensity of choice (typically 500), and

.. math::

    P_\Delta = a\left(\frac{M_\Delta}{b}\right)^c E(z)^d

The parameters :math:`(\alpha,\beta,\gamma)` are specified as optional arguments ``(alpha,beta,gamma)`` in the definition of the ``PressureGNFW`` object, while the parameters :math:`(a,b,c,d)` are specified in the ``pm_scaling`` optional argument, which takes a list of four values. By default, all these parameters are set to those found by |Nagai2007|_.

Specific cases
++++++++++++++

A specific case of a ``PressureGNFW`` is the ``Arnaud10`` class, which is a ``PressureGNFW`` object but with all default parameters set to those obtained by |Arnaud2010|_.

.. include:: ../../reference-links.rst
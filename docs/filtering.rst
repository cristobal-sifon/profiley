Fourier-space filtering
=======================

``profiley`` includes a module that allows the application of a given 
Fourier-space filter to any profile. This module relies on the `pixell 
<https://github.com/simonsobs/pixell>`_ library and is adapted from code written 
by `Mathew Madhavacheril <https://github.com/msyriac>`_. Because ``pixell`` is 
only required by ``profiley`` for this particular application, it is not a basic 
requirement of ``profiley``, so it should be installed separately:

.. code-block::

    pip install pixell

Head over to the ``pixell`` repository for more details. The Fourier-space 
filtering can be applied by initializing a ``Filter`` object by passing the name 
of the FITS file containing the Fourier-space two-dimensional filter, which 
should have been created with pixell:

.. code-block::

    from profiley.filtering import Filter

    filt = Filter(kfilter_fits_filename)


The filtering functionality is contained in the ``filter`` method. For a working 
example, see `this Jupyter notebook 
<https://github.com/cristobal-sifon/profiley/blob/master/examples/filtering.ipynb>`_.

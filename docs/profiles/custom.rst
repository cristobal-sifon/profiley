Custom profiles
===============

Creating a custom profile is easy. As an example, let's
define a simple power-law profile with two free parameters, the
normalization and the slope:

.. math::

    f(r|a,b) = a\,r^b

The code should look as follows: ::

    from profiley import Profile
    from profiley.helpers.decorators import array, inMpc

    class PowerLaw(Profile):

        def __init__(self, norm, slope, **kwargs):
            self._set_shape(norm*slope)
            super().__init__(**kwargs)
            self.norm = norm
            self.slope = slope

        @array
        @inMpc
        def profile(self, r):
            return self.norm * r**self.slope

That's it! The ``__init__()`` method needs only two lines of code (in addition 
to attribute definitions). There are three things to pay attention to:

* The first line is necessary to allow ``profiley`` to automatically handle arbitrary shapes, through the definition of a ``_shape`` attribute, and must be called before ``super``. Note that ``set_shape`` takes only one argument (besides ``self``) - the *product* of the arguments in ``__init__``. That is, if  the arguments are arrays, their dimensions must be such that a product can be carried out without any manipulation.
* The ``array`` decorator allows manipulation of arrays of arbitrary shape, as specified above.
* The ``inMpc`` decorator is optional and allows the method to receive an astropy ``Quantity`` object, which will be converted to Mpc before calculating the profile.


Profile projections
+++++++++++++++++++

If the projection of this profile is analytical, any or all of the
following methods can also be specified: ::

    projected(self, R, **kwargs)
    projected_cumulative(self, R, **kwargs)
    projected_excess(self, R, **kwargs)
    offset_profile(self, R, Roff)
    offset_projected(self, R, Roff, **kwargs)
    offset_projected_cumulative(self, R, Roff, **kwargs)
    offset_projected_excess(self, R, Roff, **kwargs)

Those projections that do not have analytical expressions will also exist in the 
newly-built profile object, but they will be calculated numerically, so they may 
be somewhat slower depending on the precision required.

**Important:** For methods defined explicitly, the ``**kwargs`` constructor *is 
mandatory*, to allow for easy integration with the generic numerical 
implementations in the ``Profile`` base class.


from astropy import units as u
from functools import wraps
import numpy as np
import warnings


def array(f):
    """Add dimensions to the first argument, assumed to be the radial coordinate at
    which a given profile will be calculated, to match the ``Profile`` object shape
    """

    @wraps(f)
    def decorated(*args, **kwargs):
        args = list(args)
        # args[0] is self
        shape = getattr(args[0], "shape")
        ndim = len(shape)
        # if R is 1d it's easy, just add all dimensions
        if args[1].ndim == 1:
            args[1] = np.expand_dims(args[1], tuple(range(-ndim, 0)))
        else:
            ones = np.ones(shape)
            while True:
                try:
                    args[1] * ones
                except ValueError:
                    args[1] = args[1][..., None]
                else:
                    break
        return f(*args, **kwargs)

    return decorated


def deprecated(since="", instead="", category=DeprecationWarning):
    def inner(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            msg = f"``{f.__name__}`` is deprecated"
            if since:
                msg = f"{msg} since {since}"
            msg = f"{msg} and will be removed in a future version."
            if instead:
                msg = f"{msg} Use ``{instead}`` instead"
            warnings.warn(msg, category)
            return f(*args, **kwargs)

        return decorated

    return inner


def float_args(f):
    """Test whether all arguments are float (or None)"""

    @wraps(f)
    def decorated(*args, **kwargs):
        has_type_error = False
        for i, arg in enumerate(args, 1):
            try:
                arg / 1
            except TypeError as e:
                msg = (
                    f"argument #{i} of {f.__name__} must be float" " or arrays of float"
                )
                raise TypeError(msg) from e
        for name, arg in kwargs.items():
            if name in ("cosmo", "units"):
                continue
            if arg is None:
                continue
            try:
                arg / 1
            except TypeError as e:
                msg = f"argument {name} must be a float or array of float"
                raise TypeError(msg) from e
        return f(*args, **kwargs)

    return decorated


def inMpc(f):
    """Change units of a Quantity to Mpc and return a float."""

    @wraps(f)
    def decorated(*args, **kwargs):
        if isinstance(args[1], u.Quantity):
            args = list(args)
            args[1] = args[1].to(u.Mpc).value
        return f(*args, **kwargs)

    return decorated

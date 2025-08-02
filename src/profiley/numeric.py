"""Stand-alone numerical implementations"""

import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import UnivariateSpline


def offset(profile, R, Roff, weights=None, *, theta_samples=360):
    """Calcuate any profile with a reference point different
    from its center

    This is useful, for instance, when large-scale structure profiles
    beyond the one-halo profiles implemented directly in profiley,
    need to be calculated around offset positions.

    This implementation is significantly faster than ``Profile.offset``
    so it should be preferred.

    Parameters
    ----------
    profile : array, shape ([M,...,]N)
        the numerical value of the profile(s) to offset. Note that
        ``profile`` is expected to be transposed with respect to the
        output of ``profiley``. The returned profile, therefore, should
        be transposed back if working with ``profiley``
    R : np.ndarray, shape (N,)
        radii at which ``profile`` has been calculated. The same radii will be used to calculate the offset profile
    Roff : np.ndarray, shape (P,)
        offsets with respect to the profile center

    Optional parameters
    -------------------
        weights to apply to each profile corresponding to every
        value of ``Roff``. See ``Returns`` below
    theta_samples : int
        number of samples for the angular integral from 0 to :math:`2\\pi`
    weights : array of floats, shape (P,)

    Returns
    -------
    offset : np.ndarray,
        offset profile. The shape of the array depends on whether
        the ``weights`` argument is specified: if *not* specified
        (default), then ::

            shape: ([M,...,]P,N)

        if ``weights`` is provided, then the first axis will be
        weight-averaged over so that ::

            shape: ([M,...,]N)

    """
    if not isinstance(theta_samples, (int, np.integer)):
        raise TypeError(
            "argument theta_samples must be int, received"
            f" {theta_samples} ({type(theta_samples)})"
        )
    if not np.iterable(Roff):
        Roff = np.array([Roff])
    assert Roff.ndim == 1, "argument Roff must be 1d"
    if weights is not None:
        if weights.size != Roff.size:
            msg = (
                "weights must have the same size as Roff,"
                f" received {weights.size}, {Roff.size},"
                " respectively."
            )
            raise ValueError(msg)

    theta = np.linspace(0, 2 * np.pi, theta_samples)
    Roff = Roff[:, None]
    x = (Roff**2 + R**2 + 2 * R * Roff * np.cos(theta[:, None, None])) ** 0.5
    if profile.ndim == 1:
        func = UnivariateSpline(R, profile, k=1, s=0)
        off = simpson(func(x), theta, axis=0)
    else:
        shape = profile.shape
        ndim = profile.ndim
        profile = profile.reshape((*shape[:-1], 1, 1, shape[-1]))
        if ndim == 2:
            # we have to iterate over additional dimensions
            funcs = (UnivariateSpline(R, pi, k=1, s=0) for pi in profile)
            off = np.array([simpson(f(x), theta, axis=0) for f in funcs])
        if ndim == 3:
            funcs = (
                (UnivariateSpline(R, pij, k=1, s=0) for pij in pi) for pi in profile
            )
            off = np.array(
                [[simpson(fij(x), theta, axis=0) for fij in fi] for fi in funcs]
            )
    if weights is not None:
        off = simpson(weights[:, None] * off, Roff, axis=-2) / simpson(weights, Roff[:, 0])
    return off / (2 * np.pi)


def project(
    func, R, log_rmin: float = -10, log_rmax: float = 5, integral_samples: int = 1005
):
    R_los = np.logspace(log_rmin, log_rmax, integral_samples)
    # we must add dimensions to R_los rather than R so that @array
    # works as expected. It can be quite slow though
    print("R =", R, R.shape, R_los.shape)
    R = (np.expand_dims(R_los, tuple(range(1, R.ndim + 1))) ** 2 + R**2) ** 0.5
    print("R =", R, R.shape)
    return 2 * simpson(func(R), R_los, axis=0)

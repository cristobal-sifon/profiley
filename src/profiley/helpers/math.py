from scipy import special as sc


def gamma(a, x=0):
    """Incomplete Gamma function as defined in Mathematica,

    .. math::

        \Gamma(a,x) = \int_0^\infty dt\,t^{a-1}e^{-t}

    such that :math:`\Gamma(a,0)=\Gamma(a)` is the Gamma function
    """
    if x == 0:
        return sc.gamma(a)
    return sc.gamma(a) * sc.gammaincc(a, x)




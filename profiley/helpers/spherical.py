import numpy as np


def mass_from_radius(r, overdensity, rho_bg):
    return overdensity*rho_bg * 4*np.pi*r**3/3


def radius_from_mass(m, overdensity, rho_bg):
    return (m / (4*np.pi/3) / (overdensity*rho_bg))**(1/3)

from pyweierstrass.weierstrass import wp, omega_from_g
from math import gamma, pi, sqrt

def test_equianharmonic():
    z = gamma(1/3)**3 / 4 / pi * (1 + 1j/sqrt(3))
    omega = omega_from_g(0, 1)
    expected = 0
    obtained = wp(z, omega)
    assert abs(obtained - expected) < 1e-14
    
def test_sum_ei_is_zero():
    omega1 = 1.4 - 1j
    omega2 = 1.6 + 0.5j
    omega = (omega1, omega2)
    e1 = wp(omega1, omega)
    e2 = wp(omega2, omega)
    e3 = wp(-omega1 - omega2, omega)
    expected = 0
    obtained = e1 + e2 + e3
    assert abs(obtained - expected) < 1e-14


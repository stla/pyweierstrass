from pyweierstrass.weierstrass import wsigma, omega_from_g
from cmath import sin
from math import sqrt, exp

def test_wsigma_value1():
    omega = omega_from_g(12, -8)
    expected = sin(1j*sqrt(3))/(1j*sqrt(3)*sqrt(exp(1)))
    obtained = wsigma(1, omega)
    assert abs(expected - obtained) < 1e-14
    
def test_wsigma_value2():
    omega = omega_from_g(1, 2j)
    expected = 1.8646253716 - 0.3066001355j
    obtained = wsigma(2, omega)
    assert abs(expected - obtained) < 1e-9

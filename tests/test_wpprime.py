from pyweierstrass.weierstrass import wp, wpprime, omega_from_g

def test_differential_equation():
    z = 1 + 1j
    g2 = 5 + 3j
    g3 = 2 + 7j
    omega = omega_from_g(g2, g3)
    p = wp(z, omega)
    pdash2 = wpprime(z, omega)**2
    assert abs((4*p**3 - g2*p - g3) - pdash2) < 1e-14
    

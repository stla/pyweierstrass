from pyweierstrass.weierstrass import wp, invwp

def test_differential_equation():
    omega = (1.4 - 1j, 1.6 + 0.5j)
    w = 1 + 1j
    z = invwp(w, omega)
    expected = w
    obtained = wp(z, omega)
    assert abs(expected - obtained) < 1e-14
    

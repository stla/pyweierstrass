from pyweierstrass.weierstrass import wzeta, omega_from_g
    
def test_wzeta_value1():
    omega = omega_from_g(5+3j, 5+3j)
    expected = 0.802084165492408 - 0.381791358666872j
    obtained = wzeta(1+1j, omega)
    assert abs(expected - obtained) < 1e-14

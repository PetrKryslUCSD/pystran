from sympy import *
import matplotlib

var('x, L, q, s, EI')

N1 = 1 - 3*(x/L)**2 + 2*(x/L)**3
N2 = x * (1 - x/L)**2
N3 = 3*(x/L)**2 - 2*(x/L)**3
N4 = x**2/L*(-1 + x/L)

q = (1 - x/L) * 45000
Fi = integrate(N1 * q, (x, 0, L))
Mi = integrate(N2 * q, (x, 0, L))
Fj = integrate(N3 * q, (x, 0, L))
Mj = integrate(N4 * q, (x, 0, L))

print(Fi, Mi, Fj, Mj)

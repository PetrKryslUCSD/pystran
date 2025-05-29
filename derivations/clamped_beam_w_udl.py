from sympy import *
import matplotlib

var("x, L, q, s, EI")

N1 = 1 - 3 * (x / L) ** 2 + 2 * (x / L) ** 3
N2 = x * (1 - x / L) ** 2
N3 = 3 * (x / L) ** 2 - 2 * (x / L) ** 3
N4 = x**2 / L * (-1 + x / L)

Fi = integrate(N1 * q, (x, 0, L))
Mi = integrate(N2 * q, (x, 0, L))
Fj = integrate(N3 * q, (x, 0, L))
Mj = integrate(N4 * q, (x, 0, L))

M = Mi - Fi * x + q * x**2 / 2
# plot(M.subs(((q, 1), (L, 1), )), (x, 0, 1))

th = integrate(M.subs(x, s) / EI, (s, 0, x))
v = integrate(th.subs(x, s), (s, 0, x))
plot(v.subs(((q, 1), (L, 1), (EI, 1))), (x, 0, 1))

print("Deflection at the midpoint = ", v.subs(x, L / 2))

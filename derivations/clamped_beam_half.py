# Solution for the bending moment in a clamped-clamped beam
# with uniform distributed load
# modeling only HALF (with a symmetry condition).
# The beam is clamped at x=-h/2 and has a sliding clamp at x=0.

from sympy import *

# Coefficients of the assumed polynomial for the deflection curve
a0, a1, a2, a3, a4 = symbols('a0, a1, a2, a3, a4')
# Other variables. h  = length of beam element, q = uniform distributed loading
EI, h, x, q = symbols('EI, h, x, q')

# We assumed the deflection curve to be this polynomial
w = a0 + a1*x + a2*x**2 + a3*x**3 + a4*x**4

# From the equation of motion we know that the fourth order derivative should match q.
wpppp = diff(w, x, 4)

# This is the equation of motion:
eq = Eq(EI * wpppp - q, 0)

# We will solve it for a4...
sol = solve(eq, a4)
# ...which we will now substitute.
w = simplify(w.subs(a4, sol[0]))

print(simplify(w))

# Now we will try to satisfy the boundary conditions (zero slope
# and zero displacement at either end). Those will be four equations
# for the four unknown coefficients.
wp = diff(w, x)
wpp = diff(w, x, 3)

sol = solve([Eq(w.subs(x, -h/2), 0), Eq(wp.subs(x, -h/2), 0),
            Eq(wpp.subs(x, 0), 0), Eq(wp.subs(x, 0), 0)], [a0, a1, a2, a3])
print(sol)

# So now we can construct the true curve of deflection where 
# all coefficients are at this point known.

w = w.subs(sol)
print('w = ', w)

print('w(middle) = ', w.subs(x, 0))

# Clearly, the deflection curve is a fourth order polynomial.

# Now we are ready to compute the moment expression by computing 
# the second derivative of the deflection curve, and multiplying 
# by the bending stiffness coefficient.
M = EI * diff(w, x, 2)
print('M = ', M)

print('M(middle) = ', M.subs(x, 0))
print('M(clamp) = ', M.subs(x, -h/2))

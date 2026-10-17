# Solution for the bending moment in a clamped-clamped beam
# with a central force equivalent to a uniform distributed load
# modeling only HALF of the beam (with a symmetry condition).
# The beam is clamped at x=-h/2 and has a sliding clamp at x=0.

from sympy import *

# Coefficients of the assumed polynomial for the deflection curve
a0, a1, a2, a3 = symbols('a0, a1, a2, a3')
# Other variables. h  = length of beam element, q = distrib load
EI, h, x, q = symbols('EI, h, x, q')

# We assumed the deflection curve to be this polynomial
w = a0 + a1*x + a2*x**2 + a3*x**3 

# Now we will try to satisfy the boundary conditions (zero slope
# and zero displacement at either end). Those will be four equations
# for the four unknown coefficients.
wp = diff(w, x)
wpp = diff(w, x, 3)

sol = solve([Eq(w.subs(x, -h/2), 0), Eq(wp.subs(x, -h/2), 0),
            Eq(-EI * wpp.subs(x, 0), q*h/4), Eq(wp.subs(x, 0), 0)], [a0, a1, a2, a3])
print(sol)

# So now we can construct the true curve of deflection where 
# all coefficients are at this point known.

w = w.subs(sol)
print('w = ', w)

print('w(middle) = ', w.subs(x, 0))


# Now we are ready to compute the moment expression by computing 
# the second derivative of the deflection curve, and multiplying 
# by the bending stiffness coefficient.
M = EI * diff(w, x, 2)
print('M = ', M)

print('M(middle) = ', M.subs(x, 0))
print('M(clamp) = ', M.subs(x, -h/2))

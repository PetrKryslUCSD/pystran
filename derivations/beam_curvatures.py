# %% [markdown]
# # Beam curvatures 

# %% [markdown]
# In this notebook we aim to derive the expression for the curvature of the beam, when its deflections are described with the basis functions derived earlier.
# 
# Recall that the deflection $w$ of the beam in the $x-z$ plane is assumed to be described by the expression 
# $$
# w(\xi)=N_1(\xi)W_1 + (h/2)N_2(\xi)W_2 + N_3(\xi)W_3 + (h/2)N_4(\xi)W_4=N_1(\xi)w_1 + (h/2)N_2(\xi)\theta_1 + N_3(\xi)w_2 + (h/2)N_4(\xi)\theta_2
# $$.
# 
# Here the degrees of freedom $W_1=w_1$ (deflection at the left hand side, i.e. at $\xi=-1$), $W_2=\theta_1$ (rotation at the left hand side), 
# $W_3=w_2$ (deflection at the right hand side, i.e. at $\xi=+1$), $W_4=\theta_2$ (rotation at the right hand side).
# 
# Further, the bending moment in the beam is expressed as $M(\xi) = - EI \partial^2 w(\xi)/\partial x^2$.
# The expression $\partial^2 w(\xi)/\partial x^2$ is referred to as curvature of the curve $w(\xi)$. 
# 
# Note that the curvature is expressed in terms of the nondimensional parametric coordinate $\xi$, but we need to take the derivatives with respect to $x$, the physical coordinates along the length of the beam.

# %%
from sympy import *
import matplotlib.pyplot as plt

# %%
xi = symbols('xi')
a, b, c, d = symbols('a b c d')

# %% [markdown]
# The coefficient matrix `C` that defines the four conditions (deflection and slope at either end of the beam), and the vector of their unknown coefficients $a, b, c$, and $d$ are defined first. The first row corresponds to deflection at $\xi=-1$, the second to the slope at $\xi=-1$, and the third and fourth to the deflection and slope at $\xi=+1$.

# %%
C = Matrix([
    [1, -1, 1, -1], 
    [0, 1, -2, 3],
    [1, 1, 1, 1],
    [0, 1, 2, 3]
])
abcd = Matrix([[a], [b], [c], [d]])

# %% [markdown]
# Previously we have shown that the first basis function results from this equality:

# %%
sol = solve(Eq(C * abcd, Matrix([[1], [0], [0], [0]])))
N1 = sum(coeff*xi**j for j, coeff in enumerate(sol.values()))
print(N1)

# %% [markdown]
# The second basis function results from

# %%
sol = solve(Eq(C * abcd, Matrix([[0], [-1], [0], [0]])))
N2 = sum(coeff*xi**j for j, coeff in enumerate(sol.values()))
print(N2)


# %% [markdown]
# The third basis function follows from the deflection at $\xi=+1$

# %%
sol = solve(Eq(C * abcd, Matrix([[0], [0], [+1], [0]])))
N3 = sum(coeff*xi**j for j, coeff in enumerate(sol.values()))
print(N3)

# %% [markdown]
# and, finally, the fourth basis function is

# %%
sol = solve(Eq(C * abcd, Matrix([[0], [0], [0], [-1]])))
N4 = sum(coeff*xi**j for j, coeff in enumerate(sol.values()))
print(N4)

# %% [markdown]
# The mapping between the physical coordinate $x$ (the beam extends between $0$ and $h$, where $h$ is the length of the beam) and the parametric coordinate $-1\le\xi\le+1$ can be set up as $x = (\xi+1)/2\times h$. The jacobian $J=\partial x/\partial \xi=h/2$ can be used to link derivatives of the  shape functions with respect to the physical coordinate to the derivatives with respect to the parametric coordinate
# 
# $$
# \partial N_k(\xi)/ \partial x=\partial N_k(\xi) / \partial \xi \times (J)^{-1} = (2/h) \partial N_k(\xi) / \partial \xi
# $$
# 
# This may be continued with a second derivative
# 
# $$
# \partial N^2_k(\xi)/ \partial x^2=\partial N^2_k(\xi) / \partial \xi^2 \times (J)^{-2} = (2/h)^2 \partial N^2_k(\xi) / \partial \xi^2
# $$
# 

# %% [markdown]
# So now we can take what we have learned above and symbolically compute the second derivatives of all four basis functions.
# The symbol $h$ will be needed. 

# %%
h = symbols('h')

# %% [markdown]
# So for instance the second derivative of the first basis function with respect to $x$ reads

# %%
d2N1x2 = diff(N1, xi, 2) * (2/h)**2
print('d2N1x2 = ', d2N1x2)

# %% [markdown]
# and therefore

# %%
d2N2x2 = diff(N2, xi, 2) * (2/h)**2
print('d2N2x2 = ', d2N2x2)
d2N3x2 = diff(N3, xi, 2) * (2/h)**2
print('d2N3x2 = ', d2N3x2)
d2N4x2 = diff(N4, xi, 2) * (2/h)**2
print('d2N4x2 = ', d2N4x2)

# %% [markdown]
# Note that all the second derivatives (curvatures) are linear functions of the position along the length of the beam. This is consistent with our expectation that the moments will vary linearly from one end to the other.

# %% [markdown]
# The curvature of the beam is a fundamental quantity. It depends linearly on the degrees of freedom $W_k$. Such a relationship can be expressed with a matrix expression. The crucial matrix is the curvature-displacement matrix, $B$:

# %%
B = Matrix([d2N1x2, (h/2) * d2N2x2, d2N3x2, (h/2) * d2N4x2]).reshape(1, 4)
print(B)

# %% [markdown]
# The curvature can then be expressed as
# $$
# \partial^2 w(\xi)/\partial x^2 = B(\xi) \cdot W
# $$
# where $W=[W_1, W_2, W_3, W_4]$.



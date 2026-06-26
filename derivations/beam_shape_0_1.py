# %% [markdown]
# # Derivation of beam shape functions: Alternative coordinates

# %% [markdown]
# In this notebook we aim to derive the cubic  functions that describe the shape of a Bernoulli-Euler beam. The python package `sympy` is used for the symbolic manipulations.

# %% [markdown]
# The deflection of the beam is here considered in axes $x-y$, with the deflection $v=N_1 V_1+h N_2 V_2+N_3 V_3+h N_4 V_4$ positive upwards, along the $y$ axis.

# %%
from sympy import *

# %% [markdown]
# Each of the four shape functions of a beam will be expressed in the parametric coordinate $\gamma$, as a cubic function $N_k=a_k+b_k \gamma+c_k \gamma^2+d_k\gamma^3$, where each of the coefficients $a$, ..., $d$ is different for each shape function. Instead of using subscripts or superscripts, we keep the names the same, but each shape function has a set of different values.

# %%
gamma = symbols('gamma')
a, b, c, d = symbols('a b c d')
N = a + b * gamma + c * gamma**2 + d * gamma**3

# %% [markdown]
# The derivatives of these functions may be expressed as

# %%
dNdg = diff(N, gamma)
d2Ndg2 = diff(N, gamma, 2)
d3Ndg3 = diff(N, gamma, 3)
print(dNdg)
print(d2Ndg2)
print(d3Ndg3)

# %% [markdown]
# The four coefficients for each of the four shape functions may be computed from four conditions: these conditions are the value and slope of the shape function at either end of the interval $0\le \gamma \le +1$. For instance, the first shape function, $N_1$, should have a value of +1 at $\gamma=0$. So the polynomial expression proposed above evaluated at  $\gamma=0$ reads

# %%
N.subs(gamma, 0)

# %% [markdown]
# and should be equal to 1. Further, the slope at $\gamma=0$ and the value and the slope at $\gamma=+1$ should all be zero.

# %%
dNdg.subs(gamma, 0)

# %%
N.subs(gamma, +1)

# %%
dNdg.subs(gamma, +1)

# %% [markdown]
# Thus we have these four equations
# $$
# \begin{array}{l}
# a=1 \\
# b=0 \\
# a+b+c+d=0 \\
# b+2c+3d=0
# \end{array}
# $$
# and we can solve them for the coefficients $a, b, c, d$.
# 
# The system of equations can be written and solved as:

# %%
sol = solve([Eq(N.subs(gamma, 0), +1), 
             Eq(dNdg.subs(gamma, 0), 0), 
             Eq(N.subs(gamma, 1), 0), 
             Eq(dNdg.subs(gamma, 1), 0)])
print(sol)
N1 = sum(coeff*gamma**j for j, coeff in enumerate(sol.values()))
print('N1 = ', N1)

# %% [markdown]
# Hence, the first shape function is written
# as $N_1 = 2\gamma^3 - 3\gamma^2 + 1$.

# %% [markdown]
# The second shape function should correspond to a positive unit rotation at $\gamma=0$, and therefore to a positive slope. 

# %%
sol = solve([Eq(N.subs(gamma, 0), 0), 
             Eq(dNdg.subs(gamma, 0), +1), 
             Eq(N.subs(gamma, 1), 0), 
             Eq(dNdg.subs(gamma, 1), 0)])
print(sol)
N2 = sum(coeff*gamma**j for j, coeff in enumerate(sol.values()))
print('N2 = ', N2)

# %% [markdown]
# The second shape function is therefore $N_2=\gamma^3 - 2\gamma^2 + \gamma$.

# %% [markdown]
# The third shape function should have a value of +1 at $\gamma=+1$.

# %%
sol = solve([Eq(N.subs(gamma, 0), 0), 
             Eq(dNdg.subs(gamma, 0), 0), 
             Eq(N.subs(gamma, 1), +1), 
             Eq(dNdg.subs(gamma, 1), 0)])
print(sol)
N3 = sum(coeff*gamma**j for j, coeff in enumerate(sol.values()))
print('N3 = ', N3)

# %% [markdown]
# Which gives $N_3=-2\gamma^3 + 3\gamma^2$

# %% [markdown]
# Finally, the fourth shape function should represent rotation at $\gamma=+1$ (and again, the slope is positive).

# %%
sol = solve([Eq(N.subs(gamma, 0), 0), 
             Eq(dNdg.subs(gamma, 0), 0), 
             Eq(N.subs(gamma, 1), 0), 
             Eq(dNdg.subs(gamma, 1), +1)])
print(sol)
N4 = sum(coeff*gamma**j for j, coeff in enumerate(sol.values()))
print('N4 = ', N4)

# %% [markdown]
# The fourth shape function is therefore $N_4=\gamma^3 - \gamma^2$.

# %% [markdown]
# Here is a graphical representation of the four shape functions.

# %%
plot(N1, N2, N3, N4, xlim=(0, 1), ylim=(-0.25, 1), xlabel='$\\gamma$', ylabel='$N_k$')
None



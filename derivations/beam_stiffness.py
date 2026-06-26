# %% [markdown]
# # Beam stiffness matrix

# %% [markdown]
# In this notebook we derive the stiffness matrix of the basic 2D beam. We assume that the cross section is uniform: the beam is prismatic.

# %%
from sympy import *
import matplotlib.pyplot as plt

# %% [markdown]
# The fundamental  building blocks will be expressions for the cubic (Hermite) basis functions. The independent variable is $\xi$, on the interval $-1 \le \xi \le +1$.

# %%
xi = symbols('xi')

# %% [markdown]
# The basis functions are:

# %%
N1 = xi**3/4 - 3*xi/4 + 1/2
N2 = -xi**3/4 + xi**2/4 + xi/4 - 1/4
N3 = -xi**3/4 + 3*xi/4 + 1/2
N4 = -xi**3/4 - xi**2/4 + xi/4 + 1/4

# %% [markdown]
# The bending moment is given as $M(\xi) = - EI \partial^2 w(\xi)/\partial x^2$. With the curvature-displacement matrix $B$ and a vector of the degrees of freedom, $W$, where $W_1=w_1$ (deflection at the left hand side, i.e. at $\xi=-1$), $W_2=\theta_1$ (rotation at the left hand side), 
# $W_3=w_2$ (deflection at the right hand side, i.e. at $\xi=+1$), $W_4=\theta_2$ (rotation at the right hand side), we can write
# 
# $$
# M(\xi) = - EI B(\xi) W
# $$
# 
# Using `sympy` we write

# %%
W_1, W_2, W_3, W_4 = symbols('W_1, W_2, W_3, W_4')
W = Matrix([[W_1], [W_2], [W_3], [W_4]])
E, I, h = symbols('E, I, h')

# %% [markdown]
# The curvature-displacement matrix $B$ is constructed from the second derivatives of the basis functions with respect to $x$, using the chain rule

# %%
d2N1x2 = diff(N1, xi, 2) * (2/h)**2
d2N2x2 = diff(N2, xi, 2) * (2/h)**2
d2N3x2 = diff(N3, xi, 2) * (2/h)**2
d2N4x2 = diff(N4, xi, 2) * (2/h)**2

# %% [markdown]
# as

# %%
B = Matrix([d2N1x2, (h/2) * d2N2x2, d2N3x2, (h/2) * d2N4x2]).reshape(1, 4)
print(B)

# %% [markdown]
# So now we can define symbolically an expression for the bending moment. Note that `(B * W)` is a $1\times1$ matrix, and by subscripting with `[0]` we make it into a scalar (number):

# %%
M = - E * I * (B * W)[0]
print('M = ', M)


# %% [markdown]
# Note that the bending moment along the beam varies linearly as a function of $\xi$, and depends linearly on $W_1, ..., W_4$. 

# %% [markdown]
# We are looking for the stiffness matrix of the beam, namely  the matrix $K$ that gives the forces acting on the endpoints of the beam $F$ in terms of the displacements $W$ (well, generalized displacements, really, since they include translations and rotations). So
# 
# $$
# F = K \cdot W
# $$

# %% [markdown]
# We can obtain the entries of the stiffness matrix using Castigliano's theorem. First we express the strain energy stored in the beam as
# 
# $$
# U  = (1/2) \int_0^h M^2/(EI) dx = 1/(2EI) \int_{-1}^{+1} M^2 d\xi (h/2) = h/(4EI)\int_{-1}^{+1} M^2 d\xi
# $$
# 
# where we can symbolically evaluate the necessary integral as

# %%
U = h/(4*E*I) * integrate(M**2, (xi, -1, +1))
print(simplify(U))

# %% [markdown]
# Now the partial derivative of the strain energy  $U$ with respect to the first degree of freedom $W_1$ will reveal the work-conjugate generalized force,  namely the shear force that works on the first degree of freedom (vertical displacement at the left hand side end)

# %%
print(simplify(diff(U, W_1)))

# %% [markdown]
# This is actually the first row of the $4\times4$ stiffness matrix multiplied by values of the degrees of freedom. The first row of the stiffness matrix can therefore be written as
# 
# $$
# [12 EI/h^3, -6EI/h^2, -12 EI/h^3, -6EI/h^2]
# $$
# 
# And similarly the second row follows as

# %%
print(simplify(diff(U, W_2)))

# %% [markdown]
# The remaining two rows follow from

# %%
print(simplify(diff(U, W_3)))
print(simplify(diff(U, W_4)))

# %% [markdown]
# The weighted residual method (the method of choice in the book) states that the stiffness matrix can be obtained as
# 
# $$
# K = EI \int_{-1}^{+1} B^T B \; d\xi\; (h/2)
# $$

# %% [markdown]
# The symbolic code below does precisely this.

# %%
K = E * I * integrate(B.T * B, (xi, -1, 1)) * (h/2) 
print(K)

# %% [markdown]
# Note that the matrix `K` is symmetric

# %%
K - K.T

# %% [markdown]
# We can finally show that the same result for the stiffness matrix can be obtained with numerical integration. Here we use a two-point Gauss rule, which should be "exact", since the integrand is only a quadratic function of $\xi$. `B1` is a numerical matrix obtained by substituting $\xi$ into the symbolic expression for $B$.

# %%
xiG = [-1/sqrt(3), 1/sqrt(3)]
WG = [1, 1]
K = zeros(4, 4)
for q in range(2):
    B1 = B.subs(xi, xiG[q])
    K += E * I * B1.T * B1 * WG[q] * (h/2)
print(simplify(K))
    

# %% [markdown]
# It is also possible to go backwards at this point: express the strain energy $U$ using the matrix quantities as $(1/2)W^TKW$. We should get the same expression for $U$ as above.

# %%
simplify(U - (1/2) * (W.T * K * W)[0])

# %% [markdown]
# Note that we had to write `(W.T * K * W)[0]`, because `(W.T * K * W)` is a $1\times1$ matrix, and therefore we cannot subtract it from the scalar $U$.

# %% [markdown]
# We also note here that the coefficients of the stiffness matrix can be obtained as second derivatives of the strain function. For instance
# 
# $$
# \frac{\partial^2 U}{\partial W_1 \partial W_2}
# $$
# 
# yields
# 

# %%
diff(diff(U, W_1), W_2)

# %% [markdown]
# We should have gotten $K_{12}$:

# %%
simplify(K[0, 1])



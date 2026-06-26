# %% [markdown]
# # Derivation of beam stiffness matrix via flexibility
# 
# We show how to derive the stiffness matrix of the basic beam element from complementary strain energy, proceeding via the flexibility matrix.

# %% [markdown]
# ![Beam](beam2d.png)
# 
# 

# %% [markdown]
# The figure above shows the beam with two degrees of freedom, deflection $v_i$ and  $v_j$, and rotation $\phi_i$ and $\phi_j$, at the joints.
# The length of the beam is taken to be $h$, and we take the coordinate along the axis of the beam $x$ to be such that $x=0$ is the location of joint $i$, and $x=h$ is the location of joint $j$.
# 
# This structure is unsupported, which means its stiffness matrix is singular. Therefore, that matrix cannot be obtained by inverting a flexibility matrix for the unsupported structure, since that would also be singular.

# %% [markdown]
# We need to construct the flexibility matrix for a _supported_ structure. 
# There need to be sufficient supports to remove rigid body displacements.
# Since the structure cannot translate horizontally, there are two rigid body displacement modes: 
# vertical translation and rotation. To suppress the rigid body displacements we need to constrain
# two degrees of freedom  to zero.
# 
# We will first employ the example of a simply-supported beam. Consider the situation shown in the figure below:

# %% [markdown]
# ![Beam](beam2d-ss.png)

# %% [markdown]
# The beam is supported on rollers at the joints, and loaded with end moments $M_i$ and $M_j$. Under these loads the joints rotate by
# $\theta_i$ and $\theta_j$. The moment along the axis of the beam is 
# $$
# M(x)=M_i + (x/h)(M_j-M_i)
# $$. 
# 
# The complementary strain energy stored in the beam (which for this linearly elastic structure is the same as strain energy) can be calculated as $U=(1/2/EI)\int_0^h (M(x))^2\;dx$. Using `sympy`: 

# %%
from sympy import *
Mi, Mj, x, h, EI = symbols('Mi, Mj, x, h, EI')
M = Mi - (x / h) * (Mi + Mj)
U = (1/2) * integrate(M**2 / EI, (x, 0, h))
U = simplify(U)
print('U = ', U)


# %% [markdown]
# Please keep in mind that the numbers with many digits need to be interpreted as fractions (0.166666666666667 means $1/6$).

# %% [markdown]
# This expression can then be differentiated with respect to
# the applied moments to obtain the rotations at the joints, as follows from Castiliagno's second theorem.

# %%
thi = simplify(diff(U, Mi))
print('thi = ', thi)
thj = simplify(diff(U, Mj))
print('thj = ', thj)

# %% [markdown]
# The relationships can be written in matrix form as
# $$
# \left[\begin{array}{c}\theta_i\\\theta_j\end{array}\right]
# = \frac{h}{EI}\left[\begin{array}{cc}1/3, & -1/6\\ -1/6,& 1/3 \end{array}\right]
# \left[\begin{array}{c}M_i\\ M_j\end{array}\right],
# $$
# where we introduce the flexibility matrix, i.e.

# %%
F = (h/EI) * Matrix([[1/3, -1/6], [-1/6, 1/3]])
display(F)

# %% [markdown]
# The stiffness matrix of this system is obtained by inverting the flexibility matrix:

# %%
S = F**(-1)
display(S)

# %% [markdown]
# Next we imagine that the structure is freed of the constraint of the rollers and can move vertically
# in its deformed state. In other words, to the deformed shape of the structure we add a rigid body motion.
# The resulting picture illustrates  the total rotation of joint $i$ obtained as $\phi_i=\theta_i+(v_j-v_i)/h$,
# and the total rotation of joint $j$ obtained as $\phi_j=\theta_j+(v_j-v_i)/h$.

# %% [markdown]
# ![Beam](beam2d-ss-rb.png)

# %% [markdown]
# Thus we can write
# $$
# \left[\begin{array}{c}\theta_i\\\theta_j\end{array}\right]
# = \left[\begin{array}{cccc} 1/h, & 1, & -1/h, & 0 \\ 
#                             1/h, & 0, & -1/h, & 1\end{array}\right]
# \left[\begin{array}{c}v_i\\ \phi_i \\ v_j \\ \phi_j \end{array}\right].
# $$
# where we can define 
# $$
# G
# = \left[\begin{array}{cccc} 1/h, & 1, & -1/h, & 0 \\ 
#                             1/h, & 0, & -1/h, & 1\end{array}\right].
# $$

# %% [markdown]
# The next step is to take advantage of the fact that the strain energy $U$ can also be written in terms of the stiffness matrix,
# $U = (1/2)\theta^T \cdot S  \cdot \theta $ (where $\theta$ is a vector collecting $\theta_i, \theta_j$; refer to the equation above)

# %%
th = Matrix([thi, thj])
simplify(U - 1/2*(th.T * S * v)[0])

# %% [markdown]
# So, nearly zero (not exactly zero, due to the floating-point arithmetic round-off errors).

# %% [markdown]
# When we take this formula, and substitute the transformation $\theta=G v$ (where $v=[v_i, \theta_i, v_j, \theta_j]^T$), we obtain
# $$U=(1/2)\theta^T \cdot S  \cdot \theta =(1/2)v^T \cdot G^T \cdot S  \cdot G \cdot v = (1/2)v^T   \cdot K \cdot v$$
# where the $4\times 4 $ stiffness matrix $K = G^T \cdot S  \cdot G$ of the free-free beam is introduced. That is

# %%
G = Matrix([[1/h, 1, -1/h, 0], 
            [1/h, 0, -1/h, 1]])
K = G.T * S * G
display(K)

# %% [markdown]
# The second approach outlined next supports the beam by a clamped condition at the left hand 
# side end. 

# %% [markdown]
# ![Beam](beam2d-cl.png)

# %% [markdown]
# The right hand side 
# is assumed to be loaded by a vertical force $V_j$  and  a bending moment
# $M_j$. Therefore, at any location of the beam the bending moment is $M(x)=M_j+V_j(h-x)$.
# Consequently, we can express the strain energy stored in the beam as a function of the applied loads:

# %%
from sympy import *
Vj, Mj, x, h, EI = symbols('Vj, Mj, x, h, EI')
M = Mj + Vj * (h - x)
U = (1/2) * integrate(M**2 / EI, (x, 0, h))
U = simplify(U)
print('U = ', U)

# %% [markdown]
# And, consequently, we can calculate the displacement of the tip $d_j$ and its slope $\theta_j$ as derivatives with respect to the applied loads

# %%
dj = simplify(diff(U, Vj))
print('Tip deflection dj = ', dj)
thj = simplify(diff(U, Mj))
print('Tip slope thj =', thj)

# %% [markdown]
# Now the flexibility matrix can be written down:

# %%
F = Matrix([[diff(dj, Vj), diff(dj, Mj)], [diff(thj, Vj), diff(thj, Mj)]])
print('F = ', )
display(F)

# %% [markdown]
# This flexibility matrix yields the tip deflection and the slope of the beam at the tip as a function of the applied load.
# That relationship can be inverted to give the stiffness matrix: the forces necessary for equilibrium that are produced
# when we know the deflection and the slope.

# %%
S = F**(-1)
print('S = ')
display(S)

# %% [markdown]
# Now we again consider that the support may move, in this case the clamped joint $i$ is assumed to translate by $v_i$ 
# and rotate by the angle $\phi_i$. 

# %% [markdown]
# ![Beam](beam2d-cl-rb.png)

# %% [markdown]
# Therefore we have the relationships
# $$
# d_j = v_j - v_i - h\phi_i
# $$
#  and 
#  $$
#  \theta_j = \phi_j - \phi_i
#  $$
# In matrix form,
# $$
# \left[\begin{array}{c} d_j \\ \theta_j\end{array}\right]
# = \left[\begin{array}{cccc} -1, & -h, & 1,& 0 \\ 
#                             0, & -1, & 0, &1\end{array}\right]
# \left[\begin{array}{c}v_i\\ \phi_i \\ v_j \\ \phi_j \end{array}\right].
# $$
# Now the $G$ matrix reads:

# %%
G = Matrix([[-1, -h, 1, 0], [0, -1, 0, 1]])

# %% [markdown]
# The stiffness matrix of the free-free beam follows as

# %%
K = G.T * S * G
display(K)

# %% [markdown]
# ## Conclusions
# 
# For both systems of supports of the beam, simply-supported and clamped,
# we derived a flexibility matrix from the strain energy, employing the second derivatives
# of the (complementary) strain energy (Castiliagno's second theorem). These flexibility matrices could be inverted, to yield $2\times2$
# stiffness matrices. Then the original degrees of freedom were expanded in both cases by adding in rigid body motion.
# The resulting transformation gave in both cases the well-known beam stiffness matrix.



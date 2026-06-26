# %% [markdown]
# # Check of the 3D beam stiffness matrix
# 
# In this example it is demonstrated that the stiffness matrix computed by the
# `pystran` package results in the same matrix that is assembled from a textbook formula
# that produces the stiffness matrix of in the local beam coordinate system, and
# then transforms it using a  $12\times12$ transformation matrix into the global
# coordinates.
# 
# We consider a beam in general orientation, meaning that the joint locations are "random" (in the sense of not being special in any way).

# %%
from numpy import zeros, dot
import context
import pystran
from numpy.linalg import norm
from pystran import model
from pystran import section

# %% [markdown]
# These are the parameters that characterize the three dimensional beam.

# %%
E = 2.0e6
G = E / (2 * (1 + 0.3))
H = 0.13
B = 0.5
A = H * B
Iy = H * B**3 / 12
Iz = H**3 * B / 12
Ix = Iy + Iz
J = Ix
xz_vector = [0, 0, 1]

# %% [markdown]
# In `pystran` we set up a "structure" consisting of a single beam member and two joints.

# %%
m = model.create(3)

# General orientation. Pick some nearly random locations. Just make sure that the beam is not parallel to xz_vector. 
model.add_joint(m, 1, [-1.199, 2.45, 3.01])
model.add_joint(m, 2, [-10.06, 7.70, -8.23])
# Default orientation: Try this alternative, in which case the individual submatrices may be easily identified.
# model.add_joint(m, 1, [0.0, 0.0, 0.0])
# model.add_joint(m, 2, [10.0, 0.0, 0.0])

s1 = section.beam_3d_section(
    "sect_1", E=E, G=G, A=A, Ix=Ix, Iy=Iy, Iz=Iz, J=J, xz_vector=xz_vector
)
model.add_beam_member(m, 1, [1, 2], s1)

# %% [markdown]
# It is possible to proceed in at least two ways: If we support the beam sufficiently, we can solve a static problem. Otherwise, we have to assemble the stiffness matrix ourselves, because the static solution cannot be run successfully with a "floppy", unsupported,  structure. Here we go with the second option.
# The degrees of freedom are numbered, and we can check that the degrees of freedom are numbered such that joint `i` gets the first six, and joint `j` the second six (12 degrees of freedom overall). 

# %%
model.number_dofs(m)
print(m['joints'])

# %%
nt, nf = m["ntotaldof"], m["nfreedof"]
print(nt, nf)

# %% [markdown]
# The total number of degrees of freedom is therefore 12, which equals the number of free degrees of freedom (and hence the stiffness matrix is singular).
# 
# Now we use the functionality implemented in the `beam` module to compute and assemble the stiffness matrix of this single member.

# %%
# Check that we only have this single member:
print(m["beam_members"].values())
# Allocate the matrix, and assemble the member stiffness.
K1 = zeros((nt, nt))
member = m["beam_members"][1]
connectivity = member["connectivity"]
i, j = m["joints"][connectivity[0]], m["joints"][connectivity[1]]
pystran.beam.assemble_stiffness(K1, member, i, j)

e_x, e_y, e_z, h = pystran.geometry.member_3d_geometry(i, j, xz_vector)

# %% [markdown]
# At this stage we compute the stiffness matrix of the three dimensional beam the way it is usually done in structural analysis courses. In other words, the $12\times12$ matrix is essentially precomputed analytically for a special orientation of the beam (the beam is oriented such that its local coordinate system agrees with the global coordinate system).

# %%
K = zeros((nt, nt))
# Axial force
K[0, 0] = E * A / h
K[6, 6] = E * A / h
K[0, 6] = -E * A / h
K[6, 0] = -E * A / h
# Torsion
K[3, 3] = G * J / h
K[9, 9] = G * J / h
K[3, 9] = -G * J / h
K[9, 3] = -G * J / h
# Bending in xy plane
K[1, 1] = 12 * E * Iz / h**3
K[7, 7] = 12 * E * Iz / h**3
K[1, 7] = -12 * E * Iz / h**3
K[7, 1] = -12 * E * Iz / h**3
K[1, 5] = 6 * E * Iz / h**2
K[5, 1] = 6 * E * Iz / h**2
K[1, 11] = 6 * E * Iz / h**2
K[11, 1] = 6 * E * Iz / h**2
K[5, 5] = 4 * E * Iz / h
K[11, 11] = 4 * E * Iz / h
K[5, 11] = 2 * E * Iz / h
K[11, 5] = 2 * E * Iz / h
K[5, 7] = -6 * E * Iz / h**2
K[7, 5] = -6 * E * Iz / h**2
K[11, 7] = -6 * E * Iz / h**2
K[7, 11] = -6 * E * Iz / h**2
# Bending in xz plane
K[2, 2] = 12 * E * Iy / h**3
K[8, 8] = 12 * E * Iy / h**3
K[2, 8] = -12 * E * Iy / h**3
K[8, 2] = -12 * E * Iy / h**3
K[2, 4] = -6 * E * Iy / h**2
K[4, 2] = -6 * E * Iy / h**2
K[2, 10] = -6 * E * Iy / h**2
K[10, 2] = -6 * E * Iy / h**2
K[4, 4] = 4 * E * Iy / h
K[10, 10] = 4 * E * Iy / h
K[4, 10] = 2 * E * Iy / h
K[10, 4] = 2 * E * Iy / h
K[4, 8] = 6 * E * Iy / h**2
K[8, 4] = 6 * E * Iy / h**2
K[10, 8] = 6 * E * Iy / h**2
K[8, 10] = 6 * E * Iy / h**2

# %% [markdown]
# At this point the traditional approach constructs the so-called transformation matrix $T$, a $12\times12$ matrix.
# The matrix for the beam in the general orientation is given by 
# 
# $$
# K^\prime = T \cdot K \cdot     T^T
# $$
# 
# The $T$ consists of $3\times3$ blocks, which we will construct from the basis vectors, $e_x$, $e_y$, and $e_z$, of the local coordinate system of the beam.

# %%
i, j = m["joints"][connectivity[0]], m["joints"][connectivity[1]]
e_x, e_y, e_z, h = pystran.geometry.member_3d_geometry(i, j, xz_vector=xz_vector)

# %% [markdown]
# The transformation matrix consists of four blocks $[e_x,e_y,e_z]$ on the diagonal.

# %%
# Transformation matrix
T = zeros(K.shape)
T[0:3, 0] = e_x
T[0:3, 1] = e_y
T[0:3, 2] = e_z
T[3:6, 3:6] = T[0:3, 0:3]
T[6:9, 6:9] = T[0:3, 0:3]
T[9:12, 9:12] = T[0:3, 0:3]

# %% [markdown]
# Now we have the transformation matrix, and we can calculate the final form of the beam stiffness matrix transformed into the general orientation.

# %%
Kprim = dot(T, dot(K, T.T))

# %% [markdown]
# Finally, we measure the difference between the two matrices. `K1`, calculated using pystran, and `Kprim`, calculated using the formula for the stiffness matrix in a special orientation + the transformation from special to general orientation.

# %%
for r in range(12):
    for c in range(12):
        if abs(Kprim[r, c] - K1[r, c]) > 1e-12 * (abs(K1[r, c]) + abs(Kprim[r, c])):
            print(r, c, Kprim[r, c], K1[r, c])
            raise ValueError("Stiffness matrix is not correct")
print("Stiffness matrix is correct")

# %% [markdown]
# The amount of work to get the stiffness matrix can be approximated as follows.
# 
# For the `pystran` implementation, we need one outer product of $6\times1$ strain-displacement matrices for the axial deformation (36 operations),
# one outer product of $12\times1$ strain-displacement matrices for each of two numerical quadrature points  (144 operations)
# for a bending in the x-z plane, one outer product of $12\times1$ strain-displacement matrices for each of two numerical quadrature points  (144 operations)
# for a bending in the x-y plane, and  one outer product of $6\times1$ strain-displacement matrices for torsion (36 operations). Total: 

# %%
print(36 + 2*144 + 2*144 + 36)

# %% [markdown]
# For the classical implementation, we need the product $K^\prime = T \cdot K \cdot     T^T$, which means that 
# we need to compute first  $temp = T \cdot K$ and then $temp \cdot T^T$. This represents two products of $12\times12$
# matrices. For each of these products we need to calculate  $12\times12$ coefficients of the result, for which we need a dot product 
# of $12\times1$ vectors. Hence

# %%
print(2 * 12**2 * 12)

# %% [markdown]
# or approximately five times as many operations. This number can be reduced by taking advantage of the blocked nature of the $T$  matrix (it consists of $3\times3$ submatrices on the diagonal, otherwise it consists of zeros)

# %%
print(16 * 3**2 * 3 * 2)


# %% [markdown]
# Conclusion: The beam matrix can be calculated either way. The `pystran` way is not only a little bit cheaper, but also shows clearly how the pie is made.



# %% [markdown]
# [pystran](https://github.com/PetrKryslUCSD/pystran) - Python package for structural analysis with trusses and beams 
# 
# (C) 2025-2026, Petr Krysl, pkrysl@ucsd.edu
# 
# # A gentle introduction to matrices in Python
# 
# This tutorial should to be executed line by line.
# 

# %% [markdown]
# We begin with the standard imports:

# %%
import math
import numpy

# %% [markdown]
# Here is a Python list:

# %%
al = [1, 2, 3]

# %% [markdown]
# We can make and array from it:
# 

# %%
am = numpy.array(al)
print(am)

# %% [markdown]
# A two dimensional array (matrix) can be made from a list of lists:
# 

# %%
A = 1000.0 * numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("A = ", A)

# %% [markdown]
# We can access the type of the elements of the matrix and its shape:
# 

# %%
print(f"Type ({A.dtype}) and shape ({A.shape}) of A")

# %% [markdown]
# Individual elements (entries) of the matrix can be accessed by their indices:
# 

# %%
print("Top left - A[0, 0] = ", A[0, 0])
print("Bottom left - A[2, 0] = ", A[2, 0])
print("Top right - A[0, 2] = ", A[0, 2])
print("Bottom right - A[2, 2] = ", A[2, 2])

# %% [markdown]
# Indices are really the offsets of the rows and columns from the top left corner.
# Therefore to access row r=2 and column c=3, we use `A[r-1, c-1] = A[1, 2]`.

# %% [markdown]
# So called slicing can be used to extract submatrices: 

# %%
print("Top row - A[0, :] = ", A[0, :])
print("Rightmost column - A[:, -1] = ", A[:, -1])

# %% [markdown]
# The colon means "all elements in the row/column". The minus sign means
# "counting from the end".

# %% [markdown]
# A list like this
# 

# %%
B = numpy.array([6, -7])

# %% [markdown]
# produces a vector.
# 

# %%
print(B.shape)

# %% [markdown]
# The length of this vector can be computed using the dot product of two vectors.
# 

# %%
print("Length of B = ", math.sqrt(numpy.dot(B, B)))

# %% [markdown]
# Two vectors can be added or subtracted.

# %%
C = numpy.array([3, 2])
print("B + C = ", B + C)

# %% [markdown]
# The diagonal of the square matrix A

# %%
print("A = ", A)

# %% [markdown]
# can be extracted as follows:
# 

# %%
print("Diagonal of A = ", numpy.diag(A))

# %% [markdown]
# Square matrices are often transposed:
# 

# %%
print("Transpose of A = ", A.T)

# %% [markdown]
# Matrix is symmetric if it is equal to its transpose, i.e., A - A.T is the zero matrix:

# %%
print("A - A.T = ", A - A.T)
# This matrix is not symmetric.

# %% [markdown]
# A matrix can be decomposed into its symmetric and antisymmetric part.
# The symmetric part is `(A + A.T)/2`, the antisymmetric part is `(A - A.T)/2`.
# 

# %%
As = (A + A.T) / 2
Aa = (A - A.T) / 2
print("Symmetric part of A = ", As)
print("Antisymmetric part of A = ", Aa)

# %% [markdown]
# Here is an identity matrix:
# 

# %%
I = numpy.eye(4)

# %% [markdown]
# And a zero matrix:
# 

# %%
Z = numpy.zeros((3, 3))
print(Z)

# %% [markdown]
# Matrices can be added and subtracted if they have the same shape.
# 

# %%
D = numpy.random.rand(3, 4)
E = numpy.random.rand(3, 4)

# %%
print("D = ", D)
print("E = ", E)
print("E+D = ", E + D)

# %% [markdown]
# If we are not interested in as many decimal places, we can use the printoptions.  
# 

# %%
with numpy.printoptions(precision=3):
    print("E+D = ", E + D)

# %% [markdown]
# Matrices can be multiplied by scalars.

# %%
print("2000 * D = ", 2000 * D)

# %% [markdown]
# Matrices can be multiplied together if they have compatible shapes (the number
# of columns in the first matrix matches the number of rows of the second
# matrix). There is a function for matrix multiplication called `dot`.
# 

# %%
F = numpy.random.rand(4, 3)
G = numpy.random.rand(4, 3)
print("F = ", F)
print("G = ", G)


# %% [markdown]
# We cannot multiply F and G because they do not have compatible shapes.
# 

# %%
try:
    print("F*G = ", numpy.dot(F, G))
except ValueError:
    print('Incompatible!')       


# %% [markdown]
# But we can multiply `F` and `G.T` because they have compatible shapes.
# 

# %%
print("F*G.T = ", numpy.dot(F, G.T))

# %% [markdown]
# Vectors can be used in the so called outer product to produce matrices.
# 

# %%
B = numpy.array([1, 2, 3, 4])
print("B = ", B)
print("Outer product of B with itself = \n ", numpy.outer(B, B))

# %% [markdown]
# The inverse of a matrix can be computed using the function `inv`.
# 

# %%
K = numpy.array([[1, 2], [3, 4]])
print("K inverse = ", numpy.linalg.inv(K))
print("K*K inverse = ", numpy.dot(K, numpy.linalg.inv(K)))


# %% [markdown]
# We should get identity matrix (within some numerical accuracy margins).

# %% [markdown]
# Some matrices are known to be singular, i.e., their inverse does not exist.
# 

# %%
K = 0.1 * numpy.outer(B, B)
try:
    numpy.linalg.inv(K)
except ValueError:
    print('Yep, singular!')

# %% [markdown]
# That should not surprise us: all the rows in the matrix are linearly
# dependent on the first one.
# 

# %%
with numpy.printoptions(precision=3):
    print("K = ", K)

# %% [markdown]
# We will work with partitioned matrices in `pystran`. Slicing can be used to
# extract the submatrices.

# %% [markdown]
# This particular matrix

# %%
K = numpy.array(
    [
        [266.78, 69.68, -197.1, 0.0, -69.68, -69.68, 0.0, 0.0],
        [69.68, 168.23, 0.0, 0.0, -69.68, -69.68, 0.0, -98.55],
        [-197.1, 0.0, 197.1, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-69.68, -69.68, 0.0, 0.0, 69.68, 69.68, 0.0, 0.0],
        [-69.68, -69.68, 0.0, 0.0, 69.68, 69.68, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, -98.55, 0.0, 0.0, 0.0, 0.0, 0.0, 98.55],
    ]
)

# %% [markdown]
# is for a truss structure with
# four joints (i.e. 8 total degrees of freedom), out of which one joint is free
# to move (2 free degrees of freedom) and the others are pinned.

# %% [markdown]
# This matrix therefore has dimension
# 

# %%
print("K shape = ", K.shape)

# %% [markdown]
# and the first 2 rows and first 2 columns are for the free degrees of freedom.
# 

# %%
with numpy.printoptions(precision=2):
    print("K = \n", K)

# %% [markdown]
# We will pull out four submatrices, such that
# K = [[Kff, Kfd]
#      [Kdf, Kdd]]
# where Kff is 2x2, Kfd is 2x6, Kdf is 6x2, and Kdd is 6x6.

# %%
Kff = K[0:2, 0:2]
Kfd = K[0:2, 2:]
Kdf = K[2:, 0:2]
Kdd = K[2:, 2:]
with numpy.printoptions(precision=3):
    print("Kff = ", Kff)
with numpy.printoptions(precision=3):
    print("Kfd = ", Kfd)
with numpy.printoptions(precision=3):
    print("Kdf = ", Kdf)
with numpy.printoptions(precision=3):
    print("Kdd = ", Kdd)

# %% [markdown]
# The eigenvalues of the matrix can be instructive.
# 

# %%
Kinfo = numpy.linalg.eig(K)
print(Kinfo.eigenvalues)

# %% [markdown]
# The matrix K is singular: look for those (practically or actually) zero eigenvalues.



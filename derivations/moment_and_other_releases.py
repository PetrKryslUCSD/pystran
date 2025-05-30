"""
Derivation of a "reduced" matrix for the case when
a member has a moment hinge at the first joint.
The computation is done in the "local" coordinates.
"""

from sympy import *

var('E, A, L, I')

K = E*I/L**3 * Matrix([
    [A*L**2/I, 0, 0, -A*L**2/I, 0, 0],
    [0, 12, 6*L, 0, -12, 6*L],
    [0, 6*L, 4*L**2, 0, -6*L, 2*L**2],
    [-A*L**2/I, 0, 0, A*L**2/I, 0, 0],
    [0, -12, -6*L, 0, 12, -6*L],
    [0, 6*L, 2*L**2, 0, -6*L, 4*L**2]
            ])

# MT=1 (moment release at joint i)
Ks = K[[0,1,3,4,5,2], :][:, [0,1,3,4,5,2]]

A = Ks[0:5, :][:, 0:5]
B = Ks[0:5, :][:, 5:6]
C = Ks[5:6, :][:, 0:5]
D = Ks[5:6, :][:, 5:6]

Kr = A - B * D**(-1) * C

display(Kr)
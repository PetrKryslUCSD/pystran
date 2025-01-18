# -*- coding: utf-8 -*-
"""
Created on 01/12/2025

Example 5.8 from Matrix Structural Analysis: Second Edition 2nd Edition by
William McGuire, Richard H. Gallagher, Ronald D. Ziemian 

The section properties are not completely defined in the book.  They are
taken from example 4.8, which does not provide both second moments of area.
They are taken here as both the same.
"""
from math import sqrt
from numpy.linalg import norm
from context import pystran
from pystran import model
from pystran import section
from pystran import plots

# SI units
L = 3.0
E = 200e9
G = 80e9
A = 0.01
Iz = 1e-3
Iy = 1e-3
Ix = 2e-3
J = Ix  # Torsional constant
P = 60000

m = model.create(3)

model.add_joint(m, 1, [0.0, 0.0, 0.0])
model.add_joint(m, 2, [0.0, L, 0.0])
model.add_joint(m, 3, [0.0, L, L])
model.add_joint(m, 4, [0.0, L, -L])

model.add_support(m["joints"][1], model.CLAMPED)

xz_vector = [1, 0, 0]
s1 = section.beam_3d_section("property_1", E, G, A, Ix, Iy, Iz, J, xz_vector)
model.add_beam_member(m, 1, [1, 2], s1)
model.add_beam_member(m, 2, [2, 3], s1)
model.add_beam_member(m, 3, [2, 4], s1)

model.add_load(m["joints"][3], model.U1, -P)

model.number_dofs(m)

print("Number of free degrees of freedom = ", m["nfreedof"])
print("Number of all degrees of freedom = ", m["ntotaldof"])

# print([j['dof'] for j in m['joints'].values()])

model.solve(m)

for id in [2, 3, 4]:
    j = m["joints"][id]
    print(id, j["displacements"])

# print(m['K'][0:m['nfreedof'], 0:m['nfreedof']])

# print(m['U'][0:m['nfreedof']])


# if norm(m['joints'][1]['displacements'] - [-8.59409726e-04, 5.77635277e-05, 5.00764459e-03, 2.39333188e-03, -1.62316861e-03, 6.81331291e-04]) > 1.e-5:
#     raise ValueError('Displacement calculation error')
# else:
#     print('Displacement calculation OK')

# if norm(m['joints'][2]['displacements'] - [-0.00117605, 0.00325316, 0.00525552, 0.00128843, 0.00172094, -0.00077147]) > 1.e-5:
#     raise ValueError('Displacement calculation error')
# else:
#     print('Displacement calculation OK')

plots.plot_setup(m)
plots.plot_members(m)
# plots.plot_member_numbers(m)
plots.plot_deformations(m, 100.0)
plots.plot_moments(m, 0.00001, "y")
plots.plot_moments(m, 0.00001, "z")
# ax = plots.plot_shear_forces(m, scale=0.50e-3)
# ax.set_title('Shear forces')
plots.show(m)

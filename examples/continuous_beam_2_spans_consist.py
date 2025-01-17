# -*- coding: utf-8 -*-
"""
Created on 01/12/2025

Introductory example 7.1 from Structural Mechanics. Analytical and Numerical Approaches for
Structural Analysis by Lingyi Lu, Junbo Jia, Zhuo Tang.
"""
from context import pystran
from pystran import model
from pystran import property
from pystran import geometry
from pystran import plots
from numpy import dot
from numpy.linalg import norm

m = model.create(2)

model.add_joint(m, 1, [0.0, 0.0])
model.add_joint(m, 2, [5.0, 0.0])
model.add_joint(m, 3, [12.0, 0.0])
model.add_support(m['joints'][1], model.U1)
model.add_support(m['joints'][1], model.U2)
model.add_support(m['joints'][2], model.U1)
model.add_support(m['joints'][2], model.U2)
model.add_support(m['joints'][3], model.U1)
model.add_support(m['joints'][3], model.U2)

E = 3e10
A = 0.001
I = 1.44e-5
p1 = property.beam_2d_property('material_1', E, A, I)
model.add_beam_member(m, 1, [1, 2], p1)
E = 2.06e11
A = 0.001
I = 1.152e-5
p2 = property.beam_2d_property('material_2', E, A, I)
model.add_beam_member(m, 2, [2, 3], p2)

model.add_load(m['joints'][1], 2, -15e3)
model.add_load(m['joints'][2], 2, -25e3)
model.add_load(m['joints'][3], 2, +35e3)

model.number_dofs(m)

nf = m['nfreedof']
nt = m['ntotaldof']

print('Number of free degrees of freedom = ', nf)
print('Number of all degrees of freedom = ', nt)

print([j['dof'] for j in m['joints'].values()])

model.solve(m)

print(m['K'][0:3, 0:3])

Kdf = m['K'][nf:nt, 0:nf]    
Kdd = m['K'][nf:nt, nf:nt]    
Uf = m['U'][0:nf]
Ud = m['U'][nf:nt]
R = dot(Kdf, Uf) + dot(Kdd, Ud)
print('Reactions = ', R)

print(m['U'][0:3])

if norm(m['U'][0:3] - [-0.02969075, -0.02742406, 0.03952194]) > 1.e-3:
    raise ValueError('Displacement calculation error')
else:
    print('Displacement calculation OK')

    
plots.plot_setup(m)
plots.plot_members(m)
plots.plot_beam_orientation(m, 1.0)
plots.show(m)

plots.plot_setup(m)
plots.plot_members(m)
plots.plot_deformations(m, 10.0)
plots.show(m)

plots.plot_setup(m)
plots.plot_members(m)
ax = plots.plot_moments(m, 0.0001)
ax.set_title('Moments')
plots.show(m)

plots.plot_setup(m)
plots.plot_members(m)
ax = plots.plot_shear_forces(m, 0.0001)
ax.set_title('Shear forces')
plots.show(m)

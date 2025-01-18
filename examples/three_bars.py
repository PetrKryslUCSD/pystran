# -*- coding: utf-8 -*-
"""
Analysis of Geometrically 
Nonlinear Structures 
Second Edition 

by 
Robert Levy 
Technion-Israel Institute of Technology, 
Haifa, Israel 

and 
William R. Spillers 
New Jersey Institute of Technology, 

Three bar example on page 32
"""
from numpy import array, dot, outer, concatenate
from numpy.linalg import norm
from context import pystran
from pystran import model
from pystran import section
from pystran import truss
from pystran import plots

m = model.create()

model.add_joint(m, 1, [10.0, 20.0])
model.add_joint(m, 2, [0.0, 20.0])
model.add_joint(m, 3, [0.0, 10.0])
model.add_joint(m, 4, [+10.0, 0.0])

E = 30000000.0
A = 0.65700000
s1 = section.truss_section("steel", E, A)
model.add_truss_member(m, 1, [1, 2], s1)
model.add_truss_member(m, 2, [1, 3], s1)
model.add_truss_member(m, 3, [1, 4], s1)

for i in [2, 3, 4]:
    for d in range(2):
        model.add_support(m["joints"][i], d)

model.add_load(m["joints"][1], 0, -10000.0)
model.add_load(m["joints"][1], 1, -10000.0 / 2.0)

model.number_dofs(m)
print("Total Degrees of Freedom = ", m["ntotaldof"])
print("Free Degrees of Freedom = ", m["nfreedof"])

model.solve(m)

for j in m["joints"].values():
    print(j["displacements"])


if norm(m["joints"][1]["displacements"] - [-0.0033325938, -0.001591621]) > 1.0e-3:
    raise ValueError("Displacement calculation error")
else:
    print("Displacement calculation OK")

for b in m["truss_members"].values():
    connectivity = b["connectivity"]
    i, j = m["joints"][connectivity[0]], m["joints"][connectivity[1]]
    e_x, L = truss.truss_member_geometry(i, j)
    B = truss.strain_displacement(e_x, L)
    u = concatenate((i["displacements"], j["displacements"]))
    eps = dot(B, u)
    print("Bar " + str(connectivity) + " force = ", E * A * eps[0])

print("Reference forces: ", -0.656854250e4, -0.48528137e4, -0.15685425e4)

plots.plot_setup(m)
plots.plot_members(m)
plots.plot_deformations(m, 1000.0)
plots.plot_member_numbers(m)
plots.show(m)

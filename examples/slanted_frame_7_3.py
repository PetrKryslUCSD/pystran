"""
Created on 01/12/2025

Example 7.3 from Structural Mechanics. Analytical and Numerical Approaches for
Structural Analysis by Lingyi Lu, Junbo Jia, Zhuo Tang.

This reference does not appear to be correct.
The calculation here verified with SA_tools.
"""

from numpy import array, dot, outer
from numpy.linalg import norm
from context import pystran
from pystran import model
from pystran import section
from pystran import freedoms

m = model.create(2)

model.add_joint(m, 1, [0.0, 0.0])
model.add_joint(m, 2, [-5.65, -5.65])
model.add_joint(m, 3, [4.0, 0.0])
model.add_joint(m, 4, [8.0, 0.0])
model.add_support(m["joints"][2], freedoms.U1)
model.add_support(m["joints"][2], freedoms.U2)
model.add_support(m["joints"][2], freedoms.UR3)
model.add_support(m["joints"][4], freedoms.U1)
model.add_support(m["joints"][4], freedoms.U2)
model.add_support(m["joints"][4], freedoms.UR3)

E = 2.06e11
A = 1.5e-2
I = 2.813e-5
s2 = section.beam_2d_section("material_2", E, A, I)
model.add_beam_member(m, 1, [1, 2], s2)
model.add_beam_member(m, 2, [3, 1], s2)
model.add_beam_member(m, 3, [3, 4], s2)

model.add_load(m["joints"][1], freedoms.UR3, -50e3)
model.add_load(m["joints"][3], freedoms.U2, -60e3)

model.number_dofs(m)

print("Number of free degrees of freedom = ", m["nfreedof"])
print("Number of all degrees of freedom = ", m["ntotaldof"])

print([j["dof"] for j in m["joints"].values()])

model.solve_statics(m)

print([j["displacements"] for j in m["joints"].values()])

print(m["K"][0 : m["nfreedof"], 0 : m["nfreedof"]])

print(m["U"][0 : m["nfreedof"]])

print("Reference:     8.8622e-05  -2.2798e-04  -1.8971e-02")
print("               4.4311e-05  -4.6696e-02   4.7854e-03")


if norm(
    m["joints"][1]["displacements"] - [8.8622e-05, -2.2798e-04, -1.8971e-02]
) > 1.0e-3 * norm(m["joints"][1]["displacements"]):
    raise ValueError("Displacement calculation error")

print("Displacement calculation OK")

if norm(
    m["joints"][3]["displacements"] - [4.4311e-05, -4.6696e-02, 4.7854e-03]
) > 1.0e-3 * norm(m["joints"][3]["displacements"]):
    raise ValueError("Displacement calculation error")

print("Displacement calculation OK")

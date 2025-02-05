"""
Created on 01/12/2025

Example 7.3 from Structural Mechanics. Analytical and Numerical Approaches for
Structural Analysis by Lingyi Lu, Junbo Jia, Zhuo Tang.
"""

from numpy.linalg import norm
from context import pystran
from pystran import model
from pystran import section
from pystran import freedoms
from pystran import plots

m = model.create(2)

model.add_joint(m, 1, [0.0, 0.0])
model.add_joint(m, 2, [-5.65, -5.65])
model.add_joint(m, 4, [8.0, 0.0])
model.add_support(m["joints"][2], freedoms.U1)
model.add_support(m["joints"][2], freedoms.U2)
model.add_support(m["joints"][2], freedoms.UR3)
model.add_support(m["joints"][4], freedoms.U1)
model.add_support(m["joints"][4], freedoms.UR3)

E = 2.06e11
A = 9.6e-3
I = 1.152e-5
s2 = section.beam_2d_section("material_2", E, A, I)
model.add_beam_member(m, 1, [1, 2], s2)
model.add_beam_member(m, 2, [4, 1], s2)

model.add_load(m["joints"][1], freedoms.UR3, -40e3)
model.add_load(m["joints"][1], freedoms.U2, -50e3)

model.number_dofs(m)

print("Number of free degrees of freedom = ", m["nfreedof"])
print("Number of all degrees of freedom = ", m["ntotaldof"])

print([j["dof"] for j in m["joints"].values()])

model.solve_statics(m)

print([j["displacements"] for j in m["joints"].values()])

print(m["K"][0 : m["nfreedof"], 0 : m["nfreedof"]])

print(m["F"][0 : m["nfreedof"]])

print(m["U"][0 : m["nfreedof"]])

print(m["joints"][1]["displacements"])
print("Reference: ", [0.000236558620403, -0.000674850902417, -0.027039378483428])


if (
    norm(
        m["joints"][1]["displacements"]
        - [0.000236558620403, -0.000674850902417, -0.027039378483428]
    )
    > 1.0e-3
):
    raise ValueError("Displacement calculation error")

print("Displacement calculation OK")

plots.plot_setup(m)
plots.plot_members(m)
plots.plot_deformations(m, 10.0)
plots.show(m)

plots.plot_setup(m)
plots.plot_members(m)
ax = plots.plot_bending_moments(m, scale=1.0e-4)
ax.set_title("Bending moments")
plots.show(m)

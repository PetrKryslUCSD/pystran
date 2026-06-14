"""
pystran - Python package for structural analysis with trusses and beams

(C) 2025-2026, Petr Krysl, pkrysl@ucsd.edu

# Bending of a two incompressible cantilevers rigidly linked at the free end

## Problem description:

Two parallel cantilevers are rigidly linked at the free end.
They bend in sync due to the rigid link.

Displacements and bending moments are provided in the reference.

## References

This is the AFNOR SSLL05/89 test case.

Original source: "Guide de validation des progiciels de calcul de structures"
publié par l'AFNOR 1990 (ISBN 2-12-486611-7).

Data taken from: ICAB Force Exemples Exemples de calculs de statique pour ICAB
Force. www.icab.fr
"""

import context
from pystran import model
from pystran import section
from pystran import plots
import numpy

E = 200e9
A = 1.0
I = 4 / 3 * 10**-8
sb = section.beam_2d_section("sb", E=E, A=A, I=I)

h = 2
F = 1.0e3

# The rigid link formulation requires a penalty for each degree of freedom.
# This penalty needs to be of appropriate magnitude, commensurate with the
# other stiffnesses in the system. This particular formulation does not require
# the penalty to be many times larger, just in the ballpark. Here we take a
# fraction of the `E*A`  product.
sr = section.rigid_link_section("sr", Gamma=1e8 * numpy.diagflat([1.0, 1.0, 1.0]))

m = model.create(2)
freedoms = m["freedoms"]

model.add_joint(m, 1, (0.0, 0.0))
model.add_joint(m, 2, (h, 0.0))
model.add_joint(m, 3, (0.0, -0.2))
model.add_joint(m, 4, (h, -0.2))

# Here are the two beam members.
model.add_beam_member(m, 1, (1, 2), sb)
model.add_beam_member(m, 2, (3, 4), sb)
# The free ends are linked together rigidly.
model.add_rigid_link_member(m, 1, (2, 4), sr)

# The rigid link is shown with a slightly thicker line.
plots.setup(m)
plots.plot_members(m)
plots.show(m)

model.add_support(m["joints"][1], freedoms.ALL_DOFS)
model.add_support(m["joints"][3], freedoms.ALL_DOFS)
model.add_load(m["joints"][4], freedoms.U2, -F)

# The transverse force is applied to only one of the bars.
plots.setup(m)
plots.plot_members(m)
plots.plot_applied_forces(m)
plots.show(m)

# Solve the system. 
model.number_dofs(m)
model.solve_statics(m)

# Using the stiffness coefficients of a beam in the basic configuration, we
# can calculate the deflection from the relationship between the shear force
# in the beams (there are two!) and the applied force, assuming 
# the bars are nearly incompressible in the axial direction.

w2 = F / (12 * E * I / h**3) / 2
print("Analytical deflection in the direction of the force: ", w2)
print("Expected tip deflection: ", [0.0, -1.25250329e-01, 0])
for j in m["joints"].values():
    print(j["displacements"])

plots.setup(m)
plots.plot_members(m)
plots.plot_deformations(m)
plots.show(m)

# Again using the basic stiffness, we deduce the correct magnitude of the bending moment:
M = 6 * h * E * I / h**3 * w2
print("Analytical bending moment magnitude: ", M)
plots.setup(m)
plots.plot_members(m)
plots.plot_bending_moments(m)
plots.show(m)

"""
pystran - Python package for structural analysis with trusses and beams 

(C) 2025, Petr Krysl, pkrysl@ucsd.edu

# Continuous beam with two spans

Introductory example 7.1 from Structural Mechanics. Analytical and Numerical
Approaches for Structural Analysis by Lingyi Lu, Junbo Jia, Zhuo Tang.
"""

from numpy import dot
from numpy.linalg import norm
from context import pystran
from pystran import model
from pystran import section
from pystran import plots

# Create a two dimensional (planar) model.
m = model.create(2)

# There are three joints.
model.add_joint(m, 1, [0.0, 0.0])
model.add_joint(m, 2, [5.0, 0.0])
model.add_joint(m, 3, [12.0, 0.0])

# The supports are added to the model. The pinned supports are added to the
# joint by listing the degree of freedom designations, `model.U1` and
# `model.U2`. These degrees of freedom are suppressed (set to zero). Note that
# each of the joints also has a rotation degree of freedom, `model.UR3`, which
# are free at all joints.
model.add_support(m["joints"][1], model.U1)
model.add_support(m["joints"][1], model.U2)
model.add_support(m["joints"][2], model.U1)
model.add_support(m["joints"][2], model.U2)
model.add_support(m["joints"][3], model.U1)
model.add_support(m["joints"][3], model.U2)

# The beam members have different material properties and cross sectional
# properties. We create two separate sections for the two beam members.
E = 3e10
A = 0.001
I = 1.44e-5
s1 = section.beam_2d_section("section_1", E, A, I)
model.add_beam_member(m, 1, [1, 2], s1)
E = 2.06e11
A = 0.001
I = 1.152e-5
s2 = section.beam_2d_section("section_2", E, A, I)
model.add_beam_member(m, 2, [2, 3], s2)

# The loads are  moments at the joints.
model.add_load(m["joints"][1], model.UR3, -15e3)
model.add_load(m["joints"][2], model.UR3, -25e3)
model.add_load(m["joints"][3], model.UR3, +35e3)

# The model is solved.
model.number_dofs(m)
nt = m["ntotaldof"]
nf = m["nfreedof"]
print("Total Degrees of Freedom = ", nt)
print("Free Degrees of Freedom = ", nf)

model.solve_statics(m)

# The stiffness matrix for the free degrees of freedom can be printed.
print(m["K"][0:3, 0:3])

# Here are the calculated free degrees of freedom:
print(m["U"][0:3])

# These displacements can be compared with the reference values from literature.
if norm(m["U"][0:3] - [-0.02969075, -0.02742406, 0.03952194]) > 1.0e-3:
    raise ValueError("Displacement calculation error")
else:
    print("Displacement calculation OK")


# We can calculate the reactions at the supports. This is the manual approach to
# that using the partitioning of the stiffness matrix and the displacement
# vector.
Kdf = m["K"][nf:nt, 0:nf]
Kdd = m["K"][nf:nt, nf:nt]
Uf = m["U"][0:nf]
Ud = m["U"][nf:nt]

# The reactions follow:
R = dot(Kdf, Uf) + dot(Kdd, Ud)
print("Reactions = ", R)

# In order to understand moment and shear diagrams, we start by plotting the
# geometry with the orientation of the local coordinate system on each beam.
plots.plot_setup(m)
plots.plot_members(m)
ax = plots.plot_beam_orientation(m, 1.0)
ax.set_title("Local coordinate systems (red -- local x, blue -- local z)")
plots.show(m)

# The deformed shape shows the curvatures of the beam.
plots.plot_setup(m)
plots.plot_members(m)
ax = plots.plot_deformations(m, 10.0)
ax.set_title("Deformed shape (magnified 10 times)")
plots.show(m)

# The bending moment can be compared with the curvature of the beam.
plots.plot_setup(m)
plots.plot_members(m)
ax = plots.plot_bending_moments(m, 0.0001)
ax.set_title("Moments")
plots.show(m)

# The shear forces are the slopes of the moment diagram.
plots.plot_setup(m)
plots.plot_members(m)
ax = plots.plot_shear_forces(m, 0.0001)
ax.set_title("Shear forces")
plots.show(m)

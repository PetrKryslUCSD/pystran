"""
pystran - Python package for structural analysis with trusses and beams 

(C) 2025, Petr Krysl, pkrysl@ucsd.edu

# Spatial frame with a hinge and spring supports

Spatial frame with spring supports and a spherical ball hinge. This is AFNOR
SSLL04/89 test case. 

Original source: "Guide de validation des progiciels de calcul de structures"
publié par l'AFNOR 1990 (ISBN 2-12-486611-7). Ces tests ont été élaborés par la
Société Française des Mécaniciens (SFM) avec pour objectif de "contribuer à
l'amélioration de la qualité et de la fiabilité des progiciels de calcul de
structures, outils indispensables à la conception et au dimensionnement des
équipements mécaniques". 

Data taken from: ICAB Force Exemples Exemples de calculs de statique pour ICAB
Force. www.icab.fr

Displacements and internal forces are provided in the verification manual.
"""

# We begin with the standard imports:

from numpy import array
from numpy.linalg import norm
from context import pystran
from pystran import model
from pystran import section
from pystran import freedoms
from pystran import plots
from pystran import beam

# Define a few constants:
E = 2.1e11  # N/m^2
nu = 1 / 3
G = E / (2 * (1 + nu))
A = 1e-3  # m^2
Iz = 1e-6  # m^4
Iy = 1e-6  # m^4
Ix = 2e-6  # m^4
J = Ix  # Torsional constant
K = 52500  # N/m (translation spring) or N/m*m (rotation spring)
F = 10000

# The model is created as three dimensional.
m = model.create(3)

# Joints are added at their locations. The original source refers to the nodes
# (joints) as N_A, N_B, etc. N_A and N_B are supported, N_H1 and N_H2 form the
# two joints that are linked in a hinge.
model.add_joint(m, 1, [0.0, 2.0, 1.0])  # N_A
model.add_joint(m, 2, [2.0, 0.0, -1.0])  # N_B
model.add_joint(m, 3, [0.0, 0.0, -1.0])  # N_C
model.add_joint(m, 4, [0.0, 0.0, +1.0])  # N_D
model.add_joint(m, 5, [0.0, 0.0, 0.0])  # N_H1
model.add_joint(m, 6, [0.0, 0.0, 0.0])  # N_H2

# There are four beams. The cross sectional properties are the same, but the
# beam orientations need to be different.

# Beams along the Z-axis.
xz_vector = [1, 0, 0]
sect_1 = section.beam_3d_section(
    "sect_1", E=E, G=G, A=A, Ix=Ix, Iy=Iy, Iz=Iz, J=J, xz_vector=xz_vector
)
# Beams along the X- and Y-axis.
xz_vector = [0, 0, 1]
sect_2 = section.beam_3d_section(
    "sect_2", E=E, G=G, A=A, Ix=Ix, Iy=Iy, Iz=Iz, J=J, xz_vector=xz_vector
)


# With the above definitions of the sections at hand, we define the four beam
# members.

model.add_beam_member(m, 1, [1, 4], sect_2)
model.add_beam_member(m, 2, [3, 2], sect_2)
model.add_beam_member(m, 3, [4, 5], sect_1)
model.add_beam_member(m, 4, [6, 3], sect_1)

# Now we can plot the geometry of the structure. We show the members, the member
# numbers, and the orientations of the local coordinate systems.

ax = plots.plot_setup(m)
plots.plot_members(m)
plots.plot_member_numbers(m)
plots.plot_joint_numbers(m)
plots.plot_beam_orientation(m, 0.20)
ax.set_title("Frame geometry")
plots.show(m)

# Next we add the link between the two joints that form the hinge. The hinge is
# ball joint, meaning all three translations are the same for joints at the
# hinge.
model.add_links(m, [6, 5], freedoms.U1)
model.add_links(m, [6, 5], freedoms.U2)
model.add_links(m, [6, 5], freedoms.U3)

# The supports are next. The original source refers to the supports at N_A and
# N_B.
model.add_support(m["joints"][1], freedoms.U1)
model.add_support(m["joints"][1], freedoms.U3)
model.add_support(m["joints"][1], freedoms.UR2)

model.add_support(m["joints"][2], freedoms.U2)
model.add_support(m["joints"][2], freedoms.U3)
model.add_support(m["joints"][2], freedoms.UR1)

# Next we add the springs to the ground. The translation and rotation spring
# constants have the same numerical value. First at joint 1.
model.add_extension_spring_to_ground(m["joints"][1], 1, [0, 1, 0], K)
model.add_moment_spring_to_ground(m["joints"][1], 1, [1, 0, 0], K)
model.add_moment_spring_to_ground(m["joints"][1], 2, [0, 0, 1], K)
# Then at joint 3.
model.add_extension_spring_to_ground(m["joints"][2], 1, [1, 0, 0], K)
model.add_moment_spring_to_ground(m["joints"][2], 1, [0, 1, 0], K)
model.add_moment_spring_to_ground(m["joints"][2], 2, [0, 0, 1], K)


# Let us look at the translation and rotation supports:
ax = plots.plot_setup(m)
plots.plot_members(m)
plots.plot_joint_numbers(m)
plots.plot_translation_supports(m)
ax.set_title("Translation supports")
plots.show(m)

ax = plots.plot_setup(m)
plots.plot_members(m)
plots.plot_joint_numbers(m)
plots.plot_rotation_supports(m)
ax.set_title("Rotation supports")
plots.show(m)

# Next we add the forces and moments applied at the joint N_D.
model.add_load(m["joints"][4], freedoms.U3, -F)


# Now we can solve the static equilibrium of the frame.
model.number_dofs(m)
model.solve_statics(m)

# The solution to the problem can be visualized with a number of plots. We start
# with the deformed shape of the frame.
ax = plots.plot_setup(m)
plots.plot_members(m)
plots.plot_joint_numbers(m)
ax = plots.plot_deformations(m, 2.0)
ax.set_title("Deformed shape (magnified 2 times)")
plots.show(m)

# These are the displacements at all the joints.
for j in m["joints"].values():
    print(j["jid"], ": ", j["displacements"])

# The displacements of the joints can be compared to the reference values.
# These are the displacements of joint 1:
ref1 = array([0.0, -0.02976196, 0.0, 0.16072649, 0.0, -0.05951626])
for i in range(6):
    if abs(m["joints"][1]["displacements"][i] - ref1[i]) > 0.001 * abs(ref1[i]):
        raise ValueError("Displacement calculation error")

# These are the displacements of joint 3:
ref3 = array([0.02977301, 0.13888914, -0.37002052])
for i in range(3):
    if abs(m["joints"][3]["displacements"][i] - ref3[i]) > 0.001 * abs(ref3[i]):
        raise ValueError("Displacement calculation error")

print("Displacement calculation OK")

# Moment diagrams can be produced for the torsional and bending moments in the
# members.
ax = plots.plot_setup(m)
plots.plot_members(m)
plots.plot_joint_numbers(m)
plots.plot_beam_orientation(m, 0.2)
ax = plots.plot_torsion_moments(m, scale=0.0001)
ax.set_title("Torsion moments")
plots.show(m)

ax = plots.plot_setup(m)
plots.plot_members(m)
plots.plot_joint_numbers(m)
plots.plot_beam_orientation(m, 0.2)
ax = plots.plot_bending_moments(m, scale=0.0001, axis="y")
ax.set_title("Moments M_y")
plots.show(m)

ax = plots.plot_setup(m)
plots.plot_members(m)
plots.plot_joint_numbers(m)
plots.plot_beam_orientation(m, 0.2)
ax = plots.plot_bending_moments(m, scale=0.0001, axis="z")
ax.set_title("Moments M_z")
plots.show(m)


# The internal "end forces", i.e., the forces and moments at the ends of the
# members that act on the joints, are reported.
for k in m["beam_members"].keys():
    member = m["beam_members"][k]
    connectivity = member["connectivity"]
    i, j = m["joints"][connectivity[0]], m["joints"][connectivity[1]]
    f = beam.beam_3d_end_forces(member, i, j)
    print(f"Member {k}: ")
    print(
        f"   Joint {connectivity[0]}: N={f['Ni']:.5}, Qy={f['Qyi']:.5}, Qz={f['Qzi']:.5}, T={f['Ti']:.5}, My={f['Myi']:.5}, Mz={f['Mzi']:.5}: "
    )
    print(
        f"   Joint {connectivity[1]}: N={f['Nj']:.5}, Qy={f['Qyj']:.5}, Qz={f['Qzj']:.5}, T={f['Tj']:.5}, My={f['Myj']:.5}, Mz={f['Mzj']:.5}: "
    )

# The reference provides values of moments at joint 1 (N_A) in member 1:
member = m["beam_members"][1]
connectivity = member["connectivity"]
i, j = m["joints"][connectivity[0]], m["joints"][connectivity[1]]
f = beam.beam_3d_end_forces(member, i, j)
if abs(f["Ti"] - -1562.3) > 1e-1:
    raise ValueError("Member 1, joint i, internal moment error")
if abs(f["Myi"] - 8438.1) > 1e-1:
    raise ValueError("Member 1, joint i, internal moment error")
if abs(f["Mzi"] - -3124.6) > 1e-1:
    raise ValueError("Member 1, joint i, internal moment error")

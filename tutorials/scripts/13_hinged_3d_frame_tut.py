# %% [markdown]
# [pystran](https://github.com/PetrKryslUCSD/pystran) - Python package for structural analysis with trusses and beams
# 
# (C) 2025-2026, Petr Krysl, pkrysl@ucsd.edu
# 
# # Spatial frame with a hinge and spring supports
# 
# Last updated: 07/12/26
# 
# ## Problem description
# 
# Structure consisting of two L-shaped members connected by a spherical ball joint
# (hinge), and supported partially by extension and torsion springs at the two
# extremities. A static solution is sought for loading at one of the corner joints.
# 
# Displacements and internal forces are provided in the verification manual.
# 
# ## References
# 
# This is the AFNOR SSLL04/89 test case.
# 
# Original source: "Guide de validation des progiciels de calcul de structures"
# publié par l'AFNOR 1990 (ISBN 2-12-486611-7).
# 
# Data taken from: ICAB Force Exemples Exemples de calculs de statique pour ICAB
# Force. www.icab.fr
# 
# ## Documentation
# 
# [pystran docs](https://petrkryslucsd.github.io/pystran)
# 

# %% [markdown]
# We begin with the standard imports:

# %%
import scipy
from numpy import array
import context
from pystran import model
from pystran import section
from pystran import plots
from pystran import beam

# %%
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

# %% [markdown]
# The model is created as three dimensional.
# 

# %%
m = model.create(3)
freedoms = m["freedoms"]

# %% [markdown]
# Joints are added at their locations. The original source refers to the nodes
# (joints) as `N_A`, `N_B`, etc. `N_A` and `N_B` are supported, `N_H1` and `N_H2` are 
# two joints at the same location that are linked to form a hinge.
# 

# %%
model.add_joint(m, "N_A", [0.0, 2.0, 1.0])
jA = m['joints']["N_A"]
model.add_joint(m, "N_B", [2.0, 0.0, -1.0])
jB = m['joints']["N_B"]
model.add_joint(m, "N_C", [0.0, 0.0, -1.0])
model.add_joint(m, "N_D", [0.0, 0.0, +1.0])
model.add_joint(m, "N_H1", [0.0, 0.0, 0.0]) # the same location
model.add_joint(m, "N_H2", [0.0, 0.0, 0.0]) # the same location


# %% [markdown]
# The ground is represented as joint, which we will make immovable below.
# 

# %%
model.add_joint(m, "ground", [2.0, 2.0, 0.0])
ground = m['joints']['ground']

# %% [markdown]
# There are four beams. The cross sectional properties are the same, but the
# beam orientations need to be different.

# %%
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

# %% [markdown]
# With the above definitions of the sections at hand, we define the four beam
# members.

# %%
model.add_beam_member(m, 1, ["N_A", "N_D"], sect_2)
model.add_beam_member(m, 2, ["N_C", "N_B"], sect_2)
model.add_beam_member(m, 3, ["N_D", "N_H1"], sect_1)
model.add_beam_member(m, 4, ["N_H2", "N_C"], sect_1)

# %% [markdown]
# Now we can plot the geometry of the structure. We show the members,
# the member numbers, and the orientations of the local coordinate systems.

# %%
ax = plots.setup(m)
plots.plot_members(m)
plots.plot_member_ids(m)
plots.plot_joint_ids(m)
plots.plot_member_orientation(m, 0.20)
ax.set_title("Frame geometry")
plots.show(m)

# %% [markdown]
# Next we add the link between the two joints that form the hinge. The hinge is
# a ball joint, meaning all three translations are the same for the two joints at the
# hinge.
# 

# %%
model.add_dof_links(m, ["N_H1", "N_H2"], freedoms.U1)
model.add_dof_links(m, ["N_H1", "N_H2"], freedoms.U2)
model.add_dof_links(m, ["N_H1", "N_H2"], freedoms.U3)

# %% [markdown]
# The supports are next. The original source refers to the supports at `N_A` and
# `N_B`.
# 

# %%
model.add_support(jA, freedoms.U1)
model.add_support(jA, freedoms.U3)
model.add_support(jA, freedoms.UR2)

# %%
model.add_support(jB, freedoms.U2)
model.add_support(jB, freedoms.U3)
model.add_support(jB, freedoms.UR1)

# %% [markdown]
# The ground "joint" is fully supported.
# 

# %%
model.add_support(ground, freedoms.ALL_DOFS)

# %% [markdown]
# Next we add the springs to the ground. The translation and rotation spring
# constants have the same numerical value. First at joint `N_A`.
# 

# %%
model.add_spring_member(
    m,
    1,
    ["N_A", "ground"],
    section.spring_section("EXT_A", "extension", [0, 1, 0], K),
)
model.add_spring_member(
    m,
    2,
    ["N_A", "ground"],
    section.spring_section("TOR_A_X", "torsion", [1, 0, 0], K),
)
model.add_spring_member(
    m,
    3,
    ["N_A", "ground"],
    section.spring_section("TOR_A_Z", "torsion", [0, 0, 1], K),
)


# %% [markdown]
# Then at joint `N_B`.
# 

# %%
model.add_spring_member(
    m,
    4,
    ["N_B", "ground"],
    section.spring_section("EXT_B", "extension", [1, 0, 0.0], K),
)
model.add_spring_member(
    m,
    5,
    ["N_B", "ground"],
    section.spring_section("TOR_B_Y", "torsion", [0, 1, 0], K),
)
model.add_spring_member(
    m,
    6,
    ["N_B", "ground"],
    section.spring_section("TOR_B_Z", "torsion", [0, 0, 1], K),
)

# %% [markdown]
# Let us look at the translation and rotation supports. Note  the support of the ground "joint".
# 
# 

# %%
ax = plots.setup(m)
plots.plot_members(m)
# plots.plot_joint_ids(m)
plots.plot_translation_supports(m)
ax.set_title("Translation supports")
plots.show(m)

# %%
ax = plots.setup(m)
plots.plot_members(m)
# plots.plot_joint_ids(m)
plots.plot_rotation_supports(m)
ax.set_title("Rotation supports")
plots.show(m)

# %% [markdown]
# Next we add the forces and moments applied at the joint N_D.
# 

# %%
model.add_load(m["joints"]["N_D"], freedoms.U3, -F)

# %% [markdown]
# Now we can solve the static equilibrium of the frame.
# 

# %%
model.number_dofs(m)
model.solve_statics(m)

# %% [markdown]
# The solution to the problem can be visualized with a number of plots. We start
# with the deformed shape of the frame.
# 

# %%
ax = plots.setup(m)
plots.plot_members(m)
plots.plot_joint_ids(m)
ax = plots.plot_deformations(m, 2.0)
ax.set_title("Deformed shape (magnified 2 times)")
plots.show(m)

# %% [markdown]
# These are the displacements at all the joints.
# 

# %%
for j in m["joints"].values():
    print(j["jid"], ": ", j["displacements"])

# %% [markdown]
# The displacements of the joints can be compared to the reference values.
# These are the displacements of joint 1:
# 

# %%
ref1 = array([0.0, -0.02976196, 0.0, 0.16072649, 0.0, -0.05951626])
for i in range(6):
    if abs(jA["displacements"][i] - ref1[i]) > 0.001 * abs(ref1[i]):
        raise ValueError("Displacement calculation error")

# %% [markdown]
# These are the displacements of joint 3:
# 

# %%
ref3 = array([0.02977301, 0.13888914, -0.37002052])
for i in range(3):
    if abs(m["joints"]["N_C"]["displacements"][i] - ref3[i]) > 0.001 * abs(ref3[i]):
        raise ValueError("Displacement calculation error")

# %%
print("Displacement calculation OK")

# %% [markdown]
# Moment diagrams can be produced for the torsional and bending moments in the
# members.
# 

# %%
ax = plots.setup(m)
plots.plot_members(m)
# plots.plot_joint_ids(m)
plots.plot_member_orientation(m, 0.2)
ax = plots.plot_torsion_moments(m, scale=0.0001)
ax.set_title("Torsion moments")
plots.show(m)

# %%
ax = plots.setup(m)
plots.plot_members(m)
# plots.plot_joint_ids(m)
plots.plot_member_orientation(m, 0.2)
ax = plots.plot_bending_moments(m, scale=0.0001, axis="y")
ax.set_title("Moments M_y")
plots.show(m)

# %%
ax = plots.setup(m)
plots.plot_members(m)
# plots.plot_joint_ids(m)
plots.plot_member_orientation(m, 0.2)
ax = plots.plot_bending_moments(m, scale=0.0001, axis="z")
ax.set_title("Moments M_z")
plots.show(m)

# %% [markdown]
# The internal "end forces", i.e., the forces and moments at the ends of the
# members that act on the joints, are reported.
# 

# %%
for k in m["beam_members"].keys():
    member = m["beam_members"][k]
    connectivity = member["connectivity"]
    i, j = m["joints"][connectivity[0]], m["joints"][connectivity[1]]
    f = beam.beam_3d_end_forces(member, i, j)
    print(f"Member {k}: ")
    print(f"   Joint {connectivity[0]}: ")
    print(f"   N={f['Ni']:.5}, Qy={f['Qyi']:.5}, Qz={f['Qzi']:.5}")
    print(f"   T={f['Ti']:.5}, My={f['Myi']:.5}, Mz={f['Mzi']:.5}")
    print(f"   Joint {connectivity[1]}: ")
    print(f"   N={f['Nj']:.5}, Qy={f['Qyj']:.5}, Qz={f['Qzj']:.5}")
    print(f"   T={f['Tj']:.5}, My={f['Myj']:.5}, Mz={f['Mzj']:.5}")

# %% [markdown]
# The reference provides values of moments at joint N_A in member 1:
# 

# %%
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

# %% [markdown]
# ## Conclusions
# 
# This frame may look peculiar, but it is carefully constructed so that an analytical solution can be found. Our numerical solution can then be compared with the analytical one.



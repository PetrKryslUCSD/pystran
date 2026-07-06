# %% [markdown]
# pystran - Python package for structural analysis with trusses and beams
# 
# (C) 2025, Petr Krysl, pkrysl@ucsd.edu
# 
# # Weaver 3D frame benchmark
# 
# ## Problem description:
# 
# This example is a 3D frame for which multiple independent solutions are
# available in the open literature (hence, it is a benchmark).
# 
# Displacements, reactions, and internal forces are provided in the verification
# manual.
# 
# A number of quantities can be checked against independent solutions. 
# 
# These reactions are consistent with STAAD.Pro 2023.00.03 User Manual, assuming
# Poisson ratio 0.25:
# ```
# Joint 3:
#    Rx=-1.1041, Ry=-0.43222, Rz=0.21731, Mx=48.785, My=-17.973, Mz=96.122:
# Joint 4:
#    Rx=-0.89588, Ry=1.4322, Rz=-0.21731, Mx=123.08, My=47.246, Mz=-11.72:
# ```
# The internal forces can also be checked against the manual (but the number of
# available digits in the manual is limited).
# 
# ## References
# 
# The original source is Weaver Jr., W., Computer Programs for Structural
# Analysis, page 146, problem 8.
# 
# It is accessible online via the STAAD.Pro 2023.00.03 User Manual.
# 

# %% [markdown]
# We begin with the standard imports:

# %%
from numpy import array
from numpy.linalg import norm
import context
from pystran import model
from pystran import section
from pystran import plots
from pystran import rotation
from pystran import beam

# %% [markdown]
# from pystran import plots
# from pystran import beam
# from pystran import rotation

# %% [markdown]
# Define a few constants:
# US customary units, inches, pounds, seconds are assumed.
# 

# %%
L = 120.0
E = 30000
nu = 0.25  # Poisson's ratio assumed in STAAD documentation
G = E / (2 * (1 + nu))
A = 11
Iz = 56
Iy = 56
Ix = 83
J = Ix  # Torsional constant
F = 2
P = 1
M = 120

# %% [markdown]
# The model is created as three dimensional (3 as argument!).
# 

# %%
m = model.create(3)
freedoms = m["freedoms"]

# %% [markdown]
# Joints are added at their locations, and the supports of the clamped joints
# are specified.
# 

# %%
model.add_joint(m, 3, [0.0, 0.0, 0.0])
model.add_joint(m, 1, [0.0, L, 0.0])
model.add_joint(m, 2, [2 * L, L, 0.0])
model.add_joint(m, 4, [3 * L, 0.0, L])

# %% [markdown]
# All degrees of freedom suppressed -- fixed (clamped) condition.

# %%
model.add_support(m["joints"][3], freedoms.ALL_DOFS)
model.add_support(m["joints"][4], freedoms.ALL_DOFS)

# %% [markdown]
# At this point we can visualize the supports. The translation supports are
# shown with arrow heads.
# 

# %%
ax = plots.setup(m)
plots.plot_joint_ids(m)
plots.plot_translation_supports(m)
ax.view_init(elev=137, azim=-67, roll=30)
ax.set_title("Translation supports")
plots.show(m)

# %% [markdown]
# The rotation supports are shown next.
# 

# %%
ax = plots.setup(m)
plots.plot_joint_ids(m)
plots.plot_rotation_supports(m)
ax.view_init(elev=137, azim=-67, roll=30)
ax.set_title("Rotation supports")
plots.show(m)

# %% [markdown]
# There are three beams. The cross sectional properties are the same, but they
# have different orientation of their local coordinate systems in space.
# Therefore, we create two sections. The first section orients the beam local
# coordinate system such that the x-z planes contains the vector [0, 0, 1].

# %%
xz_vector = [0, 0, 1]
sect_1 = section.beam_3d_section(
    "sect_1", E=E, G=G, A=A, Ix=Ix, Iy=Iy, Iz=Iz, J=J, xz_vector=xz_vector
)

# %% [markdown]
# The second section is for the inclined beam, whose local coordinate system
# needs to be oriented so that one of the axes is parallel to the global plane
# `X-Z`. We accomplish that here by rotating the vector `[0, 1, 0]` by ninety
# degrees about the axis of the beam.

# %%
xz_vector = rotation.rotate(m["joints"][2], m["joints"][4], [0, 1, 0], 90)
sect_2 = section.beam_3d_section(
    "sect_3", E=E, G=G, A=A, Ix=Ix, Iy=Iy, Iz=Iz, J=J, xz_vector=xz_vector
)

# %% [markdown]
# With the above definitions of the sections at hand, we define the three
# members.

# %%
model.add_beam_member(m, 1, [1, 2], sect_1)
model.add_beam_member(m, 2, [3, 1], sect_1)
model.add_beam_member(m, 3, [2, 4], sect_2)

# %% [markdown]
# Now we can plot the geometry of the structure. We show the members, the
# member numbers, and the ids of the joints.

# %%
ax = plots.setup(m)
plots.plot_members(m)
plots.plot_member_ids(m)
plots.plot_joint_ids(m)
ax.view_init(elev=137, azim=-67, roll=30)
ax.set_title("Frame geometry")
plots.show(m)

# %% [markdown]
# We can also show the local coordinate systems for each member (member x is red, y is green, z is blue -- rgb).

# %%
ax = plots.setup(m)
plots.plot_members(m)
plots.plot_joint_ids(m)
plots.plot_member_orientation(m)
ax.view_init(elev=137, azim=-67, roll=30)
ax.set_title("Frame geometry: Member orientation")
plots.show(m)

# %% [markdown]
# Next we add the forces and moments applied at the joints.

# %%
model.add_load(m["joints"][1], freedoms.U1, F)
model.add_load(m["joints"][2], freedoms.U2, -P)
model.add_load(m["joints"][2], freedoms.UR3, -M)

# %% [markdown]
# And the applied forces can now be rendered.

# %%
ax = plots.setup(m)
plots.plot_members(m)
ax = plots.plot_applied_forces(m, 100.0)
ax.view_init(elev=130, azim=-70, roll=25)
ax.set_title("Applied forces")
plots.show(m)

# %% [markdown]
# Similarly, we can render the applied moments:

# %%
ax = plots.setup(m)
plots.plot_members(m)
ax = plots.plot_applied_moments(m, 1.0)
ax.view_init(elev=130, azim=-70, roll=25)
ax.set_title("Applied moments")
plots.show(m)

# %% [markdown]
# Now we can solve the static equilibrium of the frame. First we number the
# degrees of freedom, and then we call the solver that will construct the
# stiffness matrix, the right-hand side vector of applied generalized forces
# (forces plus moments), and solve a system of equations. The solution is then
# distributed to the joints.
# 

# %%
model.number_dofs(m)
model.solve_statics(m)

# %% [markdown]
# The displacements of the joints can be printed out. Recall that joints 3 and
# 4 are clamped, and their displacements are therefore zero.
# 

# %%
for jid in [1, 2]:
    j = m["joints"][jid]
    print(jid, j["displacements"])

# %% [markdown]
# The displacements of the joints can be compared to the reference values.
# These are the displacements of joint 1:
# 

# %%
ref1 = [0.22267, 0.00016, -0.17182, -0.00255, 0.00217, -0.00213]
if norm(m["joints"][1]["displacements"] - ref1) > 1.0e-1 * norm(ref1):
    raise ValueError("Displacement calculation error")
print("Displacement calculation OK")
# These are the displacements of joint 2:
ref2 = [0.22202, -0.48119, -0.70161, -0.00802, 0.00101, -0.00435]
if norm(m["joints"][2]["displacements"] - ref2) > 1.0e-1 * norm(ref2):
    raise ValueError("Displacement calculation error")
print("Displacement calculation OK")

# %% [markdown]
# The internal forces in the members can be calculated. Here we report the
# "end forces", i.e., the forces and moments at the ends of the members that
# act on the joints.
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
# The reactions at the supports can be calculated. The reactions at joints 3
# and 4 are compared to the reference values.
# 

# %%
model.statics_reactions(m)

# %%
for jid in [3, 4]:
    j = m["joints"][jid]
    print(f"Joint {jid}:")
    print(f"   Rx={j['reactions'][0]:.5}")
    print(f"   Ry={j['reactions'][1]:.5}")
    print(f"   Rz={j['reactions'][2]:.5}")
    print(f"   Mx={j['reactions'][3]:.5}")
    print(f"   My={j['reactions'][4]:.5}")
    print(f"   Mz={j['reactions'][5]:.5}")

# %%
# For joint 3:
j = m["joints"][3]
react3 = array(
    [
        j["reactions"][0],
        j["reactions"][1],
        j["reactions"][2],
        j["reactions"][3],
        j["reactions"][4],
        j["reactions"][5],
    ]
)
ref = array([-1.1041, -0.43222, 0.21731, 48.785, -17.973, 96.122])
if norm((react3 - ref) / ref) > 1e-3:
    raise ValueError("Reaction error")
print("Reaction calculation OK")

# %% [markdown]
# The frame can be considered, with the supports removed and replaced by
# reactions, a free body. Therefore we can formulate a condition that all
# forces and all moments need to add up to zero. The function below calculates
# those resultants:

# %%
allforces = model.free_body_check(m)
print("Sum of forces and moments: ", allforces)
if norm(allforces) > 1.0e-10:
    raise ValueError("Sum of forces and moments not zero")

# %% [markdown]
# We can see that the overall balance as a free body is satisfied.

# %% [markdown]
# The solution to the problem can be visualized with a number of plots. We
# start with the deformed shape of the frame.
# 

# %%
ax = plots.setup(m)
plots.plot_members(m)
plots.plot_member_orientation(m)
ax = plots.plot_deformations(m)
ax.set_title("Deformed shape (magnified)")
ax.view_init(elev=137, azim=-67, roll=30)
plots.show(m)

# %% [markdown]
# The shear forces in the members can be visualized with diagrams.
# For instance, the shear forces along the z-axis are shown.
# 

# %%
ax = plots.setup(m)
plots.plot_members(m)
plots.plot_member_orientation(m)
ax = plots.plot_shear_forces(m)
ax.set_title("Shear forces along z-axis")
ax.view_init(elev=137, azim=-67, roll=30)
plots.show(m)

# %% [markdown]
# Analogous diagrams can be produced for the torsional moments in the members.
# 

# %%
ax = plots.setup(m)
plots.plot_members(m)
plots.plot_member_orientation(m)
ax = plots.plot_torsion_moments(m)
ax.set_title("Torsional moments")
ax.view_init(elev=137, azim=-67, roll=30)
plots.show(m)

# %% [markdown]
# Finally, the axial forces in the members can be visualized.
# 

# %%
ax = plots.setup(m)
plots.plot_members(m)
plots.plot_member_orientation(m)
ax = plots.plot_axial_forces(m)
ax.set_title("Axial forces")
ax.view_init(elev=137, azim=-67, roll=30)
plots.show(m)

# %% [markdown]
# We have computed the reactions above. Now we can visualize them.
# 

# %%
ax = plots.setup(m)
plots.plot_members(m)
ax = plots.plot_reaction_forces(m)
ax.set_title("Reaction forces")
ax.view_init(elev=137, azim=-67, roll=30)
plots.show(m)

# %%
ax = plots.setup(m)
plots.plot_members(m)
ax = plots.plot_reaction_moments(m)
ax.set_title("Reaction moments")
ax.view_init(elev=137, azim=-67, roll=30)
plots.show(m)



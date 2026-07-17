# %% [markdown]
# [pystran](https://github.com/PetrKryslUCSD/pystran) - Python package for structural analysis with trusses and beams
# 
# (C) 2025, Petr Krysl, pkrysl@ucsd.edu
# 
# # Example of a support-settlement problem
# 
# Last updated: 07/11/26
# 
# ## Problem description:
# 
# Statically indeterminate beam with two spans. The beam is supported at a pin
# that settles by a given amount.
# 
# Displacements and internal forces are provided in the book, and we can check our
# solution against these reference values.
# 
# ## References
# 
# This example is completely solved in the book Matrix Analysis of Structures by
# Robert E. Sennett, ISBN 978-1577661436 (Section 3.8).
# 
# ## Documentation
# 
# [pystran docs](https://petrkryslucsd.github.io/pystran)

# %%
import scipy
import context
from pystran import model
from pystran import section
from pystran import beam
from pystran import plots

# %% [markdown]
# US customary units, inches, pounds, seconds are assumed.

# %% [markdown]
# The book gives the product of the modulus of elasticity and the moment
# of inertia as 2.9e6. The cross-sectional area does not influence the results.
# 

# %%
E = 2.9e6
I = 1.0
A = 1.0  # cross-sectional area does not influence the results
L = 10 * 12  # span in inches

# %% [markdown]
# The model is created as two dimensional.
# 

# %%
m = model.create(2)
freedoms = m["freedoms"]

# %%
model.add_joint(m, 1, [0.0, 0.0])
model.add_joint(m, 2, [L, 0.0])
model.add_joint(m, 3, [2 * L, 0.0])

# %% [markdown]
# The left hand side is clamped (all degrees of freedom set to zero), the other
# joints are simply supported.
# 

# %%
model.add_support(m["joints"][1], freedoms.ALL_DOFS)


# %% [markdown]
# The middle support moves down by 0.25 inches (notice the non zero value of the
# enforced displacement).
# 

# %%
model.add_support(m["joints"][2], freedoms.U2, -0.25)
model.add_support(m["joints"][3], freedoms.U2)

# %% [markdown]
# Define the beam members.
# 

# %%
s1 = section.beam_2d_section("s1", E, A, I)
model.add_beam_member(m, 1, [1, 2], s1)
model.add_beam_member(m, 2, [2, 3], s1)

# %% [markdown]
# Next we display the displacements. We have to magnify the prescribed
# displacement 50x, otherwise it would not be visible properly (the default
# scale doesn't work well in this case).
# 

# %%
ax = plots.setup(m, set_limits=True)
plots.plot_members(m)
plots.plot_translation_supports(m, 50)
ax.set_title("Supports (magnified 50x)")
plots.show(m)

# %% [markdown]
# Solve the discrete model.
# 

# %%
model.number_dofs(m)
model.solve_statics(m)

# %% [markdown]
# Now we print the displacements at the joints.

# %%
for j in m["joints"].values():
    print("Joint", j["jid"], "displacements", j["displacements"])

# %% [markdown]
# The first sanity check is the plot of the deformation.
# 

# %%
plots.setup(m)
plots.plot_members(m)
ax = plots.plot_deformations(m, 50.0)
ax.set_title("Deformations (x50)")
plots.show(m)

# %% [markdown]
# The reference text provides the following values for the internal forces. We
# check the end forces at the starting joint in member 1:
# 

# %%
member = m["beam_members"][1]
connectivity = member["connectivity"]
i, j = m["joints"][connectivity[0]], m["joints"][connectivity[1]]
f = beam.beam_2d_end_forces(member, i, j)
print("Member 1 end forces: ", f)
if abs(f["Ni"]) > 1e-3:
    raise ValueError("Incorrect force")
if abs(f["Qzi"] / 3.9558 - 1) > 1e-3:
    raise ValueError("Incorrect force")
if abs(f["Myi"] / -258.92857 - 1) > 1e-3:
    raise ValueError("Incorrect force")

# %% [markdown]
#  Next, we check the end forces at the starting joint in member 2:
# 

# %%
member = m["beam_members"][2]
connectivity = member["connectivity"]
i, j = m["joints"][connectivity[0]], m["joints"][connectivity[1]]
f = beam.beam_2d_end_forces(member, i, j)
print("Member 2 end forces: ", f)
if abs(f["Ni"]) > 1e-3:
    raise ValueError("Incorrect force")
if abs(f["Qzi"] / -1.7981 - 1) > 1e-3:
    raise ValueError("Incorrect force")
if abs(f["Myi"] / 215.7738 - 1) > 1e-3:
    raise ValueError("Incorrect force")

# %% [markdown]
# Here we show the local coordinate systems, in which the internal resultants
# are displayed in the next two graphs.
# 

# %%
plots.setup(m, set_limits=True)
plots.plot_members(m)
plots.plot_member_ids(m)
plots.plot_joint_ids(m)
ax = plots.plot_member_orientation(m, 20.0)
ax.set_title("Continuous beam geometry")
plots.show(m)

# %% [markdown]
# The internal forces are shown in the local coordinate system of the beams.
# These are the bending moments.
# 

# %%
plots.setup(m)
plots.plot_members(m)
ax = plots.plot_bending_moments(m)
ax.set_title("Moments")
plots.show(m)

# %% [markdown]
# And these are the shear forces.
# 

# %%
plots.setup(m)
plots.plot_members(m)
ax = plots.plot_shear_forces(m)
ax.set_title("Shear forces")
plots.show(m)



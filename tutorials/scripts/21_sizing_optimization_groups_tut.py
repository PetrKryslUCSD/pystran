# %% [markdown]
# [pystran](https://github.com/PetrKryslUCSD/pystran) - Python package for structural analysis with trusses and beams
# 
# (C) 2025-2026, Petr Krysl, pkrysl@ucsd.edu
# 
# # A 2-D truss sizing optimization:  Groups of bars.
# 
# Last updated: 06/27/26
# 
# ## Description
# 
# Optimize the cross sectional areas of a truss structure to achieve its 
# minimum weight. The bars are grouped into four groups: bottom chord, top chord, verticals, and diagonals.
# Each of the four groups can have a different cross sectional area.
# 
# Objective function: mass of the structure. The design variables are the relative cross sectional areas of the four groups of bars.
# 
# Constraints: (1) limit on maximum deflection, and (2) the design variables (i.e. the cross sectional areas) are bounded from below (so that they are greater than zero).
# 
# Mathematically
# $$
#        x^* = \arg\min f(x)
# $$
# subject to the constraint
# $$
#     c_j(x) \ge 0 \; , j=1,2
# $$
# Note: $f(x)$ is the objective function, whose argument is the vector of the design variables, $x$.
# 
# 
# ## Documentation
# 
# [pystran docs](https://petrkryslucsd.github.io/pystran)
# 

# %% [markdown]
# First we bring in the modules and functions that we will need.

# %%
import scipy
import context
from numpy import max, ones
from pystran import model
from pystran import section
from pystran import plots
from scipy.optimize import minimize

# %% [markdown]
# We are working in SI(mm) units. Next, we define some useful constants.

# %% [markdown]
# Initially, all members in all groups have cross sections of these dimensions (in millimeters square).

# %%
INITIAL_AREA = 150.0

# %% [markdown]
# The minimum area of any group of bars is 1/100 of the initial area. Therefore, the smallest admissible value of a design variable is 0.01.  This constraint will be enforced by the "lower bound" constraint defined below.

# %%
MINIMUM_DV = 0.01

# %% [markdown]
# No joint is allowed to move than this maximum allowed deflection (in millimeters):
# 

# %%
MAXIMUM_ALLOWED_DEFLECTION = 20.0

# %% [markdown]
# 
# The material properties correspond roughly to steel.
# 

# %%
E = 200000
RHO = 7.8e-9

# %% [markdown]
# 
# The magnitude of the vertical (downward) forces in Newton.
# 

# %%
W = 6000

# %% [markdown]
# The following lists define the bars for all four groups: 

# %%
BOTTOM_CHORD_BAR_CONNECTIVITIES = [
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
    ]
TOP_CHORD_BAR_CONNECTIVITIES = [
    [6, 7],
    [7, 8],
    [8, 9],
    [9, 10],
]
VERTICAL_BAR_CONNECTIVITIES = [
    [1, 6],
    [2, 7],
    [3, 8],
    [4, 9],
    [5, 10],
]
DIAGONAL_BAR_CONNECTIVITIES = [
    [1, 7],
    [3, 9],
    [3, 7],
    [5, 9],
]

# %% [markdown]
# The design variables are nondimensional multipliers of the initial cross sectional area of each group.
# The cross sectional area of each group of bars is `dvs[i] * INITIAL_AREA`. The design variables start 
# at the value of 1.0.

# %%
dvs0 = ones(4)

# %% [markdown]
# This function defines the `pystran` model of the structure, based on the values of the
# design variables, `dvs`. 
# 

# %%
def truss_model(dvs):
    m = model.create(2)
    freedoms = m['freedoms']
    model.add_joint(m, 1, [-6000, -2500])
    model.add_joint(m, 2, [-3000, -2500])
    model.add_joint(m, 3, [0, -2500])
    model.add_joint(m, 4, [3000, -2500])
    model.add_joint(m, 5, [6000, -2500])
    model.add_joint(m, 6, [-6000, 0])
    model.add_joint(m, 7, [-3000, 0])
    model.add_joint(m, 8, [0, 0])
    model.add_joint(m, 9, [3000, 0])
    model.add_joint(m, 10, [6000, 0])
    model.add_support(m["joints"][1], freedoms.U2)
    model.add_support(m["joints"][5], freedoms.U2)
    model.add_support(m["joints"][8], freedoms.U1)
    model.add_load(m["joints"][2], freedoms.U2, -W)
    model.add_load(m["joints"][3], freedoms.U2, -W)
    model.add_load(m["joints"][4], freedoms.U2, -W)
    # We start with the bottom chord bars.
    s = section.truss_section(f"s_bottom_chord", E=E, A=INITIAL_AREA * dvs[0], rho=RHO)
    for k, c in enumerate(BOTTOM_CHORD_BAR_CONNECTIVITIES):
        model.add_truss_member(m, f"bottom_chord_{k}", c, s)
    # Next we have the top chord.
    s = section.truss_section(f"s_top_chord", E=E, A=INITIAL_AREA * dvs[1], rho=RHO)
    for k, c in enumerate(TOP_CHORD_BAR_CONNECTIVITIES):
        model.add_truss_member(m, f"top_chord_{k}", c, s)
    # Next the verticals.
    s = section.truss_section(f"s_vertical", E=E, A=INITIAL_AREA * dvs[2], rho=RHO)
    for k, c in enumerate(VERTICAL_BAR_CONNECTIVITIES):
        model.add_truss_member(m, f"vertical_{k}", c, s)
    # Finally the diagonals.
    s = section.truss_section(f"s_diagonal", E=E, A=INITIAL_AREA * dvs[3], rho=RHO)
    for k, c in enumerate(DIAGONAL_BAR_CONNECTIVITIES):
        model.add_truss_member(m, f"diagonal_{k}", c, s)
    return m

# %% [markdown]
# At this point we can display the initial structure: members, joints, applied forces, and supports.
# 

# %%
m = truss_model(dvs0)
plots.setup(m)
plots.plot_members(m)
plots.plot_joints(m)
plots.plot_applied_forces(m)
plots.plot_translation_supports(m)
plots.show(m)

# %% [markdown]
# The following calculates the total volume of all 
# the members of the structure. We can use it to evaluate 
# the total mass of the structure. 
# 

# %%
mass = RHO * model.volume(m)
print('Initial mass = ', 1000 * mass, ' [kg]')

# %% [markdown]
# This helper function is defined to compute the design 
# responses (`drs`). Static response of the structure is computed. The design 
# responses are the mass of the structure and  
# the maximum displacement magnitude. They are returned as a tuple.
# 

# %%
def solve(dvs):
    m = truss_model(dvs)
    model.number_dofs(m)
    model.solve_statics(m)
    drs = (
        RHO * model.volume(m),
        max(abs(m["U"])),
    )
    return drs

# %% [markdown]
# Now we can report on the performance of the structure 
# as originally designed.
# 

# %%
drs = solve(dvs0)
initial_mass = drs[0]
initial_max_deflection = drs[1]
print("\nInitial structure")
print("-----------------")
print("Initial Design Variables: ", dvs0)
print("Mass: ", 1000 * initial_mass, ' [kg]')
print("Initial largest deflection: ", initial_max_deflection, '[mm]')


# %% [markdown]
# Note that as designed, the structure has a smaller deflection than allowed. Therefore we may expect the mass of the structure to go down.

# %% [markdown]
# At this point we start on the optimization. The objective 
# function and the constraints need to be defined. 

# %%
# Objective function is the normalized mass.
def objective(dvs):
    drs = solve(dvs)
    return drs[0] / initial_mass

# %% [markdown]
# Define a constraint on the maximum deflection.  The constraint here is on the maximum deflection, $(u_{max}-\max u)/u_{max}\ge 0$: Here $u_{max}$=`MAXIMUM_ALLOWED_DEFLECTION`.

# %%
def constrain_deflection(dvs):
    drs = solve(dvs)
    max_deflection = drs[1]
    return (MAXIMUM_ALLOWED_DEFLECTION - max_deflection) / MAXIMUM_ALLOWED_DEFLECTION

cons = [
    {"type": "ineq", "fun": constrain_deflection},
]

# %% [markdown]
# The function `constrain_deflection` is used to define an inequality constraint. All such constraints are collected in the list `cons`.
# 
# 

# %% [markdown]
# 
# Define lower bounds for the design variables. There are no upper bounds (the `None`).
# 

# %%
bounds = [(MINIMUM_DV, None) for _ in dvs0]

# %% [markdown]
# Invoke the optimization function. 
# 

# %%
solution = minimize(
    objective,
    dvs0,
    method="SLSQP",
    bounds=bounds,
    constraints=cons,
    options={"ftol": 1e-7, "maxiter": 1000, "disp": True},
)

# %% [markdown]
# Retrieve the values of the design variables from the solution, and compute the design responses
# for the optimal design variables.

# %%
dvs = solution.x
drs = solve(dvs)

# %% [markdown]
#  Now report the characteristics of the optimized structure. The largest deflection is equal to the maximum allowed, and that constraint is then the only active constraint.

# %%
mass = drs[0]
max_deflection = drs[1]
print("\nOptimized structure")
print("-----------------")
print("Solution success: ", solution.success, f" ({solution.nit} iterations)")
print("Design Variables: ", dvs)
print("Mass: ", 1000 * mass, ' [kg]')
print("Largest deflection: ", max_deflection, '[mm]')

# %% [markdown]
# Print out the cross sectional areas of the four groups (in millimeters square).
# 

# %%
print("Areas of groups of bars: ")
m = truss_model(dvs)
for (j, v) in enumerate(dvs):
    print(f"Group {j}: {INITIAL_AREA * v}")

# %% [markdown]
# The following visualization provides a graphical assessment of the optimized structure: bars with large cross sectional area are shown with thick lines, and conversely bars whose cross sectional area is small are very thin.

# %%
plots.setup(m)
plots.plot_members(m, min_linewidth=1, max_linewidth=8)
plots.show(m)

# %% [markdown]
# Notice the corners: there are no more null bars.
# 

# %% [markdown]
# ## Conclusions
# 
# Optimizing the cross sectional areas of groups of bars facilitates construction: for instance the chords can be made continuous, from a single profile.



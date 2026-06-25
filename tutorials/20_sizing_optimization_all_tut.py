"""
pystran - Python package for structural analysis with trusses and beams

(C) 2025-2026, Petr Krysl, pkrysl@ucsd.edu

# A 2-D truss sizing optimization.

Last updated: 06/24/26

## Description

Optimize the cross sectional areas of a truss structure to achieve its 
minimum weight. Each bar can have a different cross section area.

Objective Function: mass of the structure

Constraints: 
    (i) limit on maximum deflection
    (ii) limit on minimum cross-sectional area

## References

- 

"""

import context
from numpy import max, ones
from pystran import model
from pystran import section
from pystran import plots
from scipy.optimize import minimize

# Initially, all members have cross sections of these dimensions (in millimeters square).
INITIAL_AREA = 150.0

# The design variables are nondimensional multipliers of the initial cross sectional area.
# Each cross sectional area is `dvs[i] * INITIAL_AREA`. The design variables start 
# at the value of 1.0.
dvs0 = ones(17)

# Maximum allowed deflection (in millimeters)
MAXIMUM_ALLOWED_DEFLECTION = 20.0

# The material properties correspond roughly to steel.
E = 200000
RHO = 7.8e-9

# The magnitude of the vertical (downward) forces.
W = 6000

# This function defines the model of the structure, based on the values of the
# design variables, `dvs`. 
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
    a = 0
    bar_connectivities = [
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
    ]
    for k, c in enumerate(bar_connectivities):
        s = section.truss_section(f"s{k}", E=E, A=INITIAL_AREA * dvs[a], rho=RHO)
        model.add_truss_member(m, f"bottom_chord_{k}", c, s)
        a += 1
    bar_connectivities = [
        [6, 7],
        [7, 8],
        [8, 9],
        [9, 10],
    ]
    for k, c in enumerate(bar_connectivities):
        s = section.truss_section(f"s{k}", E=E, A=INITIAL_AREA * dvs[a], rho=RHO)
        model.add_truss_member(m, f"top_chord_{k}", c, s)
        a += 1
    bar_connectivities = [
        [1, 6],
        [2, 7],
        [3, 8],
        [4, 9],
        [5, 10],
    ]
    for k, c in enumerate(bar_connectivities):
        s = section.truss_section(f"s{k}", E=E, A=INITIAL_AREA * dvs[a], rho=RHO)
        model.add_truss_member(m, f"vertical_{k}", c, s)
        a += 1
    bar_connectivities = [
        [1, 7],
        [3, 9],
        [3, 7],
        [5, 9],
    ]
    for k, c in enumerate(bar_connectivities):
        s = section.truss_section(f"s{k}", E=E, A=INITIAL_AREA * dvs[a], rho=RHO)
        model.add_truss_member(m, f"diagonal_{k}", c, s)
        a += 1
    return m

# At this point we can display the initial structure.
m = truss_model(dvs0)
plots.setup(m)
plots.plot_members(m)
plots.plot_joints(m)
plots.show(m)

# The following function calculates the total volume of all 
# the members of the structure. We can use it till evaluate 
# the total mass of the structure. 

mass = RHO * model.volume(m)
print('Initial mass = ', mass, ' [metric tones]')


# This helper function is defined to compute the design 
# responses (drs). Static response is obtained. The design 
# responses are the mass of the structure and  
# the maximum displacement magnitude.
def solve(dvs):
    m = truss_model(dvs)
    model.number_dofs(m)
    model.solve_statics(m)
    drs = (
        RHO * model.volume(m),
        max(abs(m["U"])),
    )
    return drs

# Now we can report on the performance of the structure 
# as originally designed.
drs = solve(dvs0)
initial_mass = drs[0]
initial_max_deflection = drs[1]
print("\nInitial structure")
print("-----------------")
print("Initial Design Variables: ", dvs0)
print("Mass: ", initial_mass, ' [metric tones]')
print("Initial deflection: ", initial_max_deflection, '[mm]')

# At this point we start on the optimization. The objective 
# function and the constraints need to be defined.

# Objective function is the normalized mass.
def objective(dvs):
    drs = solve(dvs)
    return drs[0] / initial_mass


# Define a constraint on the maximum deflection. 
def constrain_deflection(dvs):
    drs = solve(dvs)
    max_deflection = drs[1]
    return (MAXIMUM_ALLOWED_DEFLECTION - max_deflection) / MAXIMUM_ALLOWED_DEFLECTION


# Define constraints. The the only constraint here is on the maximum deflection.
cons = [
    {"type": "ineq", "fun": constrain_deflection},
]

# Define lower bounds for the design variables. There are no upper bounds.
bounds = [(0.001 * INITIAL_AREA, None) for _ in dvs0]

# Invoke the optimization function.
solution = minimize(
    objective,
    dvs0,
    method="SLSQP",
    bounds=bounds,
    constraints=cons,
    options={"ftol": 1e-7, "maxiter": 1000, "disp": True},
)

# Retrieve the values of the design variables, and compute the design responses
# for the optimal design variables.
dvs = solution.x
drs = solve(dvs)

mass = drs[0]
max_deflection = drs[1]
print("\nOptimized structure")
print("-----------------")
print("Solution success: ", solution.success, f" ({solution.nit} iterations)")
print("Design Variables: ", dvs)
print("Mass: ", mass, ' [metric tones]')
print("Initial deflection: ", max_deflection, '[mm]')

print("Areas of individual bars: ")
m = truss_model(dvs)
a = 0
for _m in m['truss_members'].values():
    print(f"{_m['mid']}: {INITIAL_AREA * dvs[a]}")
    a += 1


plots.setup(m)
plots.plot_members(m, max_linewidth=9)
plots.show(m)

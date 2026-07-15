"""
Optimize the cross sectional areas of a truss structure to achieve its 
minimum weight. Here each bar can have different cross section area.
The bars are assumed to be made from hollow square tubes.
"""

from math import pi
import numpy
from scipy.optimize import minimize, NonlinearConstraint
import context 
from pystran import model
from pystran import section
from pystran import geometry
from pystran import plots


# The design variables are the dimension and the thickness of the square tubes.

# Initially, all members have cross sections of these dimensions.
INITIAL_INNER_DIM = 50.0
INITIAL_THICKNESS = 3.0
def area(d, t):
    return (d+2*t)**2 - d**2
groups = ['bottom_chord', 'top_chord', 'vertical', 'diagonal']
bar_connectivities = [
    [
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],
    ], [
    [6, 7],
    [7, 8],
    [8, 9],
    [9, 10],
    ], [
    [1, 6],
    [2, 7],
    [3, 8],
    [4, 9],
    [5, 10],
    ],  [
    [1, 7],
    [3, 9],
    [3, 7],
    [5, 9],
    ]
    ]
NBARS = sum(len(c) for c in bar_connectivities)
dvs0 = numpy.concatenate((INITIAL_INNER_DIM * numpy.ones(NBARS), INITIAL_THICKNESS * numpy.ones(NBARS)))

# Maximum allowed deflection (in millimeters)
MAXIMUM_ALLOWED_DEFLECTION = 20.0

# This function defines the model of the structure, based on the values of the
# design variables. 
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
    W = 6000
    model.add_load(m["joints"][2], freedoms.U2, -W)
    model.add_load(m["joints"][3], freedoms.U2, -W)
    model.add_load(m["joints"][4], freedoms.U2, -W)
    E = 200000
    rho = 2.7e-9
    v = 0
    for g in range(len(groups)):
        for k, c in enumerate(bar_connectivities[g]):
            dim = dvs[v]
            t = dvs[v + NBARS]
            A = area(dim, t)
            s = section.truss_section(f"s{k}", E=E, A=A, rho=rho)
            model.add_truss_member(m, f"{groups[g]}_{k}", c, s)
            v += 1
    return m



# m = truss_model(dvs0)
# plots.setup(m)
# plots.plot_members(m)
# plots.show(m)

# Helper function to compute the current mass of the structure.
def current_mass(m, dvs):
    mass = 0.0
    v = 0
    for g in range(len(groups)):
        for k, c in enumerate(bar_connectivities[g]):
            dim = dvs[v]
            t = dvs[v + NBARS]
            A = area(dim, t)
            member = m["truss_members"][f"{groups[g]}_{k}"]
            c = member["connectivity"]
            i, j = m["joints"][c[0]], m["joints"][c[1]]
            sect = member["section"]
            e_x, _, h = geometry.member_2d_geometry(i, j)
            mass += A * h * sect["rho"]
            v += 1
    return mass


# Helper function to compute the design responses. Static response is obtained.
def solve(dvs):
    m = truss_model(dvs)
    model.number_dofs(m)
    model.solve_statics(m)
    drs = (
        current_mass(m, dvs),
        numpy.max(abs(m["U"])),
    )
    return drs



# Report on the initial performance of the structure.
drs = solve(dvs0)
initial_mass = drs[0]
initial_max_deflection = drs[1]
print("Initial structure")
print("-----------------")
print("Areas: ", dvs0)
print("Mass: ", initial_mass)
print("Initial deflection: ", initial_max_deflection)


# Objective function is the normalized mass.
def objective(dvs):
    drs = solve(dvs)
    return drs[0] / initial_mass


# Define a constraint on the maximum deflection. 
def constrain_deflection(dvs):
    drs = solve(dvs)
    max_deflection = drs[1]
    return (MAXIMUM_ALLOWED_DEFLECTION - max_deflection) / MAXIMUM_ALLOWED_DEFLECTION


# Define constraints. They are both inequalities.
cons = [
    {"type": "ineq", "fun": constrain_deflection},
]

# Define lower bounds for the design variables. There are no upper bounds.
bounds = list([(0.01, numpy.inf) for _ in range(NBARS)]) +  \
            list([(0.01, numpy.inf) for _ in range(NBARS)])
    

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
print("Optimal structure")
print("-----------------")
print("Design variables: ", dvs)
print("Mass: ", mass)
print("Maximum deflection: ", max_deflection)


m = truss_model(dvs)
plots.setup(m)
plots.plot_members(m, max_linewidth=6)
plots.show(m)

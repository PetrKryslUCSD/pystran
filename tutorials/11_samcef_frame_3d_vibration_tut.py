"""
pystran - Python package for structural analysis with trusses and beams 

(C) 2025, Petr Krysl, pkrysl@ucsd.edu

# Two story 3D Frame vibration

Example 5.3 from the book: MECHANICAL VIBRATIONS THEORY AND APPLICATION TO
STRUCTURAL DYNAMICS, ThirdEdition, Michel Géradin, Daniel J. Rixen

The agreement is not very close, but reasonable; the source of the discrepancy
is at this point unknown.
    
Original citation: Samcef. 1992 Samcef manual: Asef–Stabi–Dynam–Repdyn (M4)
Samtech SA Liège, Belgium.
"""

from context import pystran
from pystran import model
from pystran import section
from pystran import plots

# The material is steel, SI units (m).
E = 2.1e11
G = E / (2 * (1 + 0.3))
rho = 7.8e3

# The cross section properties are given in SI units (m).

# Columns.
A = 5.14e-3
Iy = 6.9e-6
Iz = 8.49e-5
Ix = Iy + Iz
J = 1.73e-7
xz_vector = [0, 1, 0]
sverti = section.beam_3d_section(
    "sverti", E=E, rho=rho, G=G, A=A, Ix=Ix, Iy=Iy, Iz=Iz, J=J, xz_vector=xz_vector
)

# Horizontal beams.
A = 5.68e-3
Iy = 1.2e-4
Iz = 7.3e-6
Ix = Iy + Iz
J = 1.76e-7
xz_vector = [0, 0, 1]
shoriz = section.beam_3d_section(
    "shoriz", E=E, rho=rho, G=G, A=A, Ix=Ix, Iy=Iy, Iz=Iz, J=J, xz_vector=xz_vector
)

# Grid spacing and height of the floors.
a = 5.49
b = 3.66

m = model.create(3)

# Bottoms of columns.
model.add_joint(m, 1, [0.0, 0.0, 0.0])
model.add_joint(m, 2, [a, 0.0, 0.0])
model.add_joint(m, 3, [a, a, 0.0])
model.add_joint(m, 4, [0.0, a, 0.0])

# First floor.
model.add_joint(m, 11, [0.0, 0.0, b])
model.add_joint(m, 12, [a, 0.0, b])
model.add_joint(m, 13, [a, a, b])
model.add_joint(m, 14, [0.0, a, b])

# Second floor.
model.add_joint(m, 21, [0.0, 0.0, 2 * b])
model.add_joint(m, 22, [a, 0.0, 2 * b])
model.add_joint(m, 23, [a, a, 2 * b])
model.add_joint(m, 24, [0.0, a, 2 * b])

# Fix the bottoms of the columns.
for jid in range(1, 5):
    model.add_support(m["joints"][jid], model.ALL_DOFS)

# Add the members.
model.add_beam_member(m, 1, [1, 11], sverti)
model.add_beam_member(m, 2, [2, 12], sverti)
model.add_beam_member(m, 3, [3, 13], sverti)
model.add_beam_member(m, 4, [4, 14], sverti)

model.add_beam_member(m, 5, [11, 21], sverti)
model.add_beam_member(m, 6, [12, 22], sverti)
model.add_beam_member(m, 7, [13, 23], sverti)
model.add_beam_member(m, 8, [14, 24], sverti)

model.add_beam_member(m, 9, [11, 12], shoriz)
model.add_beam_member(m, 10, [12, 13], shoriz)
model.add_beam_member(m, 11, [13, 14], shoriz)
model.add_beam_member(m, 12, [14, 11], shoriz)

model.add_beam_member(m, 13, [21, 22], shoriz)
model.add_beam_member(m, 14, [22, 23], shoriz)
model.add_beam_member(m, 15, [23, 24], shoriz)
model.add_beam_member(m, 16, [24, 21], shoriz)

# Optionally we can refine the members into multiple finite elements. In the
# book, the claim is that the results are for a single element per member. The
# accuracy of the frequencies will be driven by the number of elements, since a
# single element cannot represent the deflections in the interior of the
# members.
nref = 3
for i in range(16):
    model.refine_member(m, i + 1, nref)

model.number_dofs(m)
model.solve_free_vibration(m)

reffs = [3.08, 4.65, 7.87, 8.23]
for mode in range(0, 4):
    print(f"Mode {mode}: {m["frequencies"][mode]:.3f} Hz")
    print(f"  Reference: ", reffs[mode])
    ax = plots.plot_setup(m)
    plots.plot_members(m)
    model.copy_mode(m, mode)
    plots.plot_deformations(m, 100.0)
    ax.set_title(f"Mode {mode}, frequency = {m['frequencies'][mode]:.2f} Hz")
    plots.show(m)

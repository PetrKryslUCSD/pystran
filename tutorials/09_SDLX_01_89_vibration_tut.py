"""
pystran - Python package for structural analysis with trusses and beams

(C) 2025-2026, Petr Krysl, pkrysl@ucsd.edu

# Two story planar frame vibration

## Problem description:

Planar frame of two stories, vibrating only in the plane.
The frame members are automatically refined into multiple elements.

Analytical solutions are available for the first few modes of vibration.

## References

SCIA Engineer 24.0.1020 test case SDLX 01/89
"""

import context
from pystran import model
from pystran import section

from pystran import plots

# The material is steel, SI units (m).
E = 2.1e11
rho = 7.85e3

H = 29.0e-3  # millimeters to meters
B = 4.8e-3
A = H * B
I = H * B**3 / 12

# Create the model in usual fashion.
m = model.create(2)
freedoms = m["freedoms"]

model.add_joint(m, 1, [-0.3, 0.0])
model.add_joint(m, 2, [-0.3, 0.810])
model.add_joint(m, 3, [0.3, 0.0])
model.add_joint(m, 4, [0.3, 0.810])
model.add_joint(m, 5, [-0.3, 0.360])
model.add_joint(m, 6, [0.3, 0.360])

for jid in [1, 3]:
    model.add_support(m["joints"][jid], freedoms.ALL_DOFS)

s1 = section.beam_2d_section("section_1", E, A, I, rho)

model.add_beam_member(m, 1, [5, 1], s1)
model.add_beam_member(m, 2, [2, 5], s1)
model.add_beam_member(m, 3, [2, 4], s1)
model.add_beam_member(m, 4, [6, 4], s1)
model.add_beam_member(m, 5, [6, 3], s1)
model.add_beam_member(m, 6, [6, 5], s1)

plots.setup(m, set_limits=True)
plots.plot_members(m)
plots.plot_member_ids(m)
plots.plot_member_orientation(m, 0.05)
ax = plots.plot_joint_ids(m)
ax.set_title("Structure before refinement")
plots.show(m)

# All members will now be refined into eight finite elements. Without the
# refinement, the reference solutions cannot be reproduced: there simply
# wouldn't be enough degrees of freedom. Unfortunately the reference publication
# does not mention with how many finite elements used per member 
# the reference solution was calculated.
nref = 8
for i in range(6):
    model.refine_member(m, i + 1, nref)

plots.setup(m, set_limits=True)
plots.plot_members(m)
ax = plots.plot_joint_ids(m)
ax.set_title("Structure after refinement")
plots.show(m)

# Solve a free vibration analysis problem.
model.number_dofs(m)
model.solve_free_vibration(m)

# Compare with the reference frequencies.
reffs = [8.75, 29.34, 43.71, 56.12, 95.86, 102.37, 146.64, 174.39, 178.36]
for mode, reff in enumerate(reffs):
    print(f'Mode {mode}: {m["frequencies"][mode]} vs {reff} Hz')
    if abs((m["frequencies"][mode] - reff) / reff) > 1e-2:
        raise ValueError("Incorrect frequency")

# Show the first four modes.
for mode in range(0, 4):
    print(f"Mode {mode}: ", m["frequencies"][mode])
    ax = plots.setup(m)
    plots.plot_members(m)
    model.set_solution(m, m["eigvecs"][:, mode])
    plots.plot_deformations(m, 0.2)
    ax.set_title(f"Mode {mode}, frequency = {m['frequencies'][mode]:.2f} Hz")
    plots.show(m)

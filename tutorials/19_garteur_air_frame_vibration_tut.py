"""
pystran - Python package for structural analysis with trusses and beams

(C) 2025-2026, Petr Krysl, pkrysl@ucsd.edu

# GARTEUR SM-AG19 Testbed: Vibration of idealized airframe

## Last updated: 05/23/26

Description

This virtual test application is based on the test article  
used by the GARTEUR Structures & Materials Action Group 19  
which organized a Round Robin exercise where 12 European laboratories  
tested a single structure between 1995 and 1997. The benchmark structure   
was a laboratory structure built to simulate the dynamic behaviour  
of an aeroplane. The structure was initially built for a benchmark  
study on experimental modal analysis conducted by the  
Structures and Materials Action Group (SM-AG19) of the Group  
for Aeronautical Research and Technology in EURope (GARTEUR).  
The test-bed was designed and manufactured by ONERA, France.

![](IMAC97photo.png)

## References

- [GARTEUR] Ground Vibration Test Techniques, compiled by A Gravelle, GARTEUR
  Structures & Materials Action Group 19 Technical report TP-115, 1999.
- [BW] Etienne Balmes, Jan R. Wright, GARTEUR GROUP ON GROUND VIBRATION
  TESTING | RESULTS FROM THE TEST OF A SINGLE STRUCTURE BY 12 LABORATORIES IN
  EUROPE, Proceedings of DETC'97, 1997 ASME Design Engineering Technical
  Conferences, September 14-17, 1997, Sacramento, California.

"""

import context
from numpy import eye
from pystran import model
from pystran import section
from pystran import plots

# The material is aluminum, SI units (mm).
# [GARTEUR] has an indication which material properties were considered
# in the original finite element simulations.
E = 72000.0
G = E / (2 * (1 + 0.3))
rho = 2.7e-9

# The cross section properties are given in SI units (mm).

# All cross sections are oriented so that the local z axis is aligned with 
# the global z (except for the tail).
xz_vector = [0, 0, 1]
xz_vector_tail = [1, 0, 0]

# Body.
A, Ix, Iy, Iz, J = section.rectangle(150.0, 50.0)
sbody = section.beam_3d_section(
    "sbody", E=E, rho=rho, G=G, A=A, Ix=Ix, Iy=Iy, Iz=Iz, J=J, xz_vector=xz_vector
)

# Wings. Note we could approximately compensate 
# for the absence of the damping layer by adding ~1mm
# thickness to the wing thickness.
A, Ix, Iy, Iz, J = section.rectangle(10.0, 100.0)
swing = section.beam_3d_section(
    "swing", E=E, rho=rho, G=G, A=A, Ix=Ix, Iy=Iy, Iz=Iz, J=J, xz_vector=xz_vector
)


# Drums.
A, Ix, Iy, Iz, J = section.rectangle(10.0, 100.0)
sdrum = section.beam_3d_section(
    "sdrum", E=E, rho=rho, G=G, A=A, Ix=Ix, Iy=Iy, Iz=Iz, J=J, xz_vector=xz_vector
)

# Tail.
A, Ix, Iy, Iz, J = section.rectangle(100.0, 10.0)
stail = section.beam_3d_section(
    "stail", E=E, rho=rho, G=G, A=A, Ix=Ix, Iy=Iy, Iz=Iz, J=J, xz_vector=xz_vector_tail
)

# Tailplane.
A, Ix, Iy, Iz, J = section.rectangle(10.0, 100.0)
stplane = section.beam_3d_section(
    "stplane", E=E, rho=rho, G=G, A=A, Ix=Ix, Iy=Iy, Iz=Iz, J=J, xz_vector=xz_vector
)

# The offset beams are connected through rigid links.
srig = section.rigid_link_section("sr", Gamma=E * 100**3 * 10 / 50 * 10 * eye(6))

# Create the model.
m = model.create(3)
freedoms = m["freedoms"]

# On the body
model.add_joint(m, 1, [600.0, 0.0, 0.0])
model.add_joint(m, 2, [0.0, 0.0, 0.0])
model.add_joint(m, 3, [-900.0, 0.0, 0.0])

# The midpoint on the wing.
model.add_joint(m, 4, [0.0, 0.0, 150.0 / 2 + 16 + 10.0 / 2])
# Wing tips.
model.add_joint(m, 5, [0.0, -1000.0, 150.0 / 2 + 16 + 10.0 / 2])
model.add_joint(m, 6, [0.0, 1000.0, 150.0 / 2 + 16 + 10.0 / 2])

# The midpoint on the left drum.
model.add_joint(m, 16, [0.0, 1000.0 - 50, 150.0 / 2 + 16 + 10.0 / 2 + 10.0 / 2])
# Left drum tips.
model.add_joint(m, 9, [200.0, 1000.0 - 50, 150.0 / 2 + 16 + 10.0 / 2 + 10.0 / 2])
model.add_joint(m, 10, [-200.0, 1000.0 - 50, 150.0 / 2 + 16 + 10.0 / 2 + 10.0 / 2])

# The midpoint on the right drum.
model.add_joint(m, 15, [0.0, -1000.0 + 50, 150.0 / 2 + 16 + 10.0 / 2 + 10.0 / 2])
# Right drum tips.
model.add_joint(m, 7, [200.0, -1000.0 + 50, 150.0 / 2 + 16 + 10.0 / 2 + 10.0 / 2])
model.add_joint(m, 8, [-200.0, -1000.0 + 50, 150.0 / 2 + 16 + 10.0 / 2 + 10.0 / 2])

# The tail.
model.add_joint(m, 11, [-850.0, 0.0, 50.0])
model.add_joint(m, 12, [-850.0, 0.0, 300 + 50.0])

# The tailplane midpoint.
model.add_joint(m, 17, [-850.0, 0.0, 300 + 50.0 + 10 / 2])
# Tailplane tips.
model.add_joint(m, 14, [-850.0, 200.0, 300 + 50.0 + 10 / 2])
model.add_joint(m, 13, [-850.0, -200.0, 300 + 50.0 + 10 / 2])

# Add attachment point for the concentrated mass on the left drum.
model.add_joint(m, 20, [200.0 - 20.0, 1000.0 - 50 - 20, 150.0 / 2 + 16 + 10.0 / 2 + 10.0])

# Add attachment point for the concentrated mass on the left drum.
model.add_joint(m, 19, [200.0 - 20.0, -1000.0 + 50 + 20, 150.0 / 2 + 16 + 10.0 / 2 + 10.0])

# Add the body members.
model.add_beam_member(m, 'b1', [1, 2], sbody)
model.add_beam_member(m, 'b2', [2, 3], sbody)

# Add the wing members.
model.add_beam_member(m, 'w1', [4, 5], swing)
model.add_beam_member(m, 'w2', [4, 6], swing)

# Add the left drum members.
model.add_beam_member(m, 'ld1', [16, 9], sdrum)
model.add_beam_member(m, 'ld2', [16, 10], sdrum)

# Add the right drum members.
model.add_beam_member(m, 'rd1', [15, 7], sdrum)
model.add_beam_member(m, 'rd2', [15, 8], sdrum)

# Add the tail
model.add_beam_member(m, 't', [11, 12], stail)

# Add the tailplane members.
model.add_beam_member(m, 'tp1', [17, 13], stplane)
model.add_beam_member(m, 'tp2', [17, 14], stplane)

# Connect the body and the wing
model.add_rigid_link_member(m, 'bwlink', (2, 4), srig)

# Connect the left drum and the wing
model.add_rigid_link_member(m, 'dwllink', (5, 15), srig)

# Connect the right drum and the wing
model.add_rigid_link_member(m, 'dwrlink', (6, 16), srig)

# Connect the body and the tail
model.add_rigid_link_member(m, 'btlink', (3, 11), srig)

# Connect the tail and the tailplane
model.add_rigid_link_member(m, 'tplink', (12, 17), srig)

# Connect the left drum and the added mass
model.add_rigid_link_member(m, 'mllink', (9, 20), srig)

# Connect the right drum and the added mass
model.add_rigid_link_member(m, 'mrlink', (7, 19), srig)

# Add the masses on the drums.
drum_added_mass = 0.2 / 1000 # convert to metric tons
model.add_mass(m['joints'][19], freedoms.TRANSLATION_DOFS, drum_added_mass)
model.add_mass(m['joints'][19], freedoms.ROTATION_DOFS, drum_added_mass / 100)
model.add_mass(m['joints'][20], freedoms.TRANSLATION_DOFS, drum_added_mass)
model.add_mass(m['joints'][20], freedoms.ROTATION_DOFS, drum_added_mass / 100)

# Add the mass of the attachment between the body and the wing.
body_wing_attachment_mass = 16 * 70 * 130 * 7.85e-9 # steel
model.add_mass(m['joints'][4], freedoms.TRANSLATION_DOFS, body_wing_attachment_mass)

# Add the mass of the auxiliary metal on the tail
fin_tail_plan_attachment_mass = 2 * 15 * 15 * 100 * 7.85e-9
model.add_mass(m['joints'][12], freedoms.TRANSLATION_DOFS, fin_tail_plan_attachment_mass)

vol = model.volume(m)
mass = (rho * vol) + drum_added_mass + body_wing_attachment_mass + fin_tail_plan_attachment_mass
print('Model mass = ', mass * 1000, '[kg]')

# plots.setup(m)
# # plots.plot_joint_ids(m)
# plots.plot_members(m)
# plots.plot_member_orientation(m)
# plots.show(m)

# # Solve the problem.
model.number_dofs(m)
# # The number of free degrees of freedom is
print(f"Number of degrees of freedom: {m['nfreedof']}")
model.solve_free_vibration(m, 5.0)

# Plot the modes and compare the frequencies. The mode shapes correlate with
# those published in the reference.
for mode in range(6, 14):
    print(f"Mode {mode+1-6}: {m['frequencies'][mode]:.3f} Hz")
    ax = plots.setup(m)
    plots.plot_members(m)
    model.set_solution(m, m["eigvecs"][:, mode])
    plots.plot_deformations(m, 5.0)
    ax.set_title(f"Mode {mode+1-6}, frequency = {m['frequencies'][mode]:.2f} Hz")
    plots.show(m)

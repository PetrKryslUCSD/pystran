"""
pystran unit tests
"""

import unittest

import context
from math import sqrt, pi, cos, sin
from numpy import array, dot, outer, concatenate
from numpy.linalg import norm
from pystran import model
from pystran import section
from pystran import geometry
from pystran import beam
from pystran import truss
from pystran import rotation


class UnitTestsPlanarFrames(unittest.TestCase):

    def test_cant_w_masses(self):
        """
        Created on 01/19/2025

        Structural Analysis: A Unified Classical and Matrix, Ghali, Amin; Neville, Adam
        -- Edition 7, 2017, Taylor and Francis

        Example 24.2 - Cantilever with added masses

        The mass density of the beam itself is artificially reduced so that there are
        only the added masses.
        """

        E = 2.0e11
        G = E / (2 * (1 + 0.3))
        rho = 7.85e3 / 10000  # artificially reduce the mass density of the beam

        h = 0.12
        b = 0.03
        A = b * h
        Iy = b * h**3 / 12
        sbar = section.beam_2d_section("sbar", E=E, rho=rho, A=A, I=Iy)
        L = 2.0
        W = 3.0 * 9.81
        g = 9.81

        m = model.create(2)
        freedoms = m["freedoms"]

        model.add_joint(m, 1, [0.0, 3 * L])
        model.add_joint(m, 2, [0.0, 2 * L])
        model.add_joint(m, 3, [0.0, 1 * L])
        model.add_joint(m, 4, [0.0, 0.0])

        model.add_support(m["joints"][4], freedoms.ALL_DOFS)

        model.add_beam_member(m, 1, [1, 2], sbar)
        model.add_beam_member(m, 2, [2, 3], sbar)
        model.add_beam_member(m, 3, [3, 4], sbar)

        model.add_mass(m["joints"][1], freedoms.U1, 4 * W / g)
        model.add_mass(m["joints"][1], freedoms.U2, 4 * W / g)
        model.add_mass(m["joints"][2], freedoms.U1, W / g)
        model.add_mass(m["joints"][2], freedoms.U2, W / g)
        model.add_mass(m["joints"][3], freedoms.U1, W / g)
        model.add_mass(m["joints"][3], freedoms.U2, W / g)

        model.number_dofs(m)

        model.solve_free_vibration(m)

        expected = (
            array([0.1609, 1.7604, 5.0886]) * sqrt(g * E * Iy / W / L**3) / 2 / pi
        )
        # print("Expected frequencies (zero mass of beam): ", expected)
        # print("Computed frequencies: ", m["frequencies"][0:3])
        self.assertAlmostEqual(m["frequencies"][0], expected[0], places=2)
        self.assertAlmostEqual(m["frequencies"][1], expected[1], places=1)
        self.assertAlmostEqual(m["frequencies"][2], expected[2], places=0)

        # for mode in range(3):
        #     plots.setup(m)
        #     plots.plot_members(m)
        #     model.set_solution(m, mode)
        #     ax = plots.plot_deformations(m, 50.0)
        #     ax.set_title(f"Mode {mode}: f = {sqrt(m['eigvals'][mode])/2/pi:.3f} Hz")
        #     plots.show(m)

    def test_supp_settle(self):
        """
        # Example of a support-settlement problem (Section 3.8)

        This example is completely solved in the book Matrix Analysis of Structures by
        Robert E. Sennett, ISBN 978-1577661436.

        Displacements and internal forces are provided in the book, and we can check our
        solution against these reference values.


        Important note: Our orientation of the local coordinate system is such that web
        of the H-beams is parallel to z axis! This is different from the orientation in
        the book, where the web is parallel to the y axis.
        """

        # US customary units, inches, pounds, seconds are assumed.

        # The book gives the product of the modulus of elasticity and the moment of inertia as 2.9e6.
        E = 2.9e6
        I = 1.0
        A = 1.0  # cross-sectional area does not influence the results
        L = 10 * 12  # span in inchesc

        m = model.create(2)
        freedoms = m["freedoms"]

        model.add_joint(m, 1, [0.0, 0.0])
        model.add_joint(m, 2, [L, 0.0])
        model.add_joint(m, 3, [2 * L, 0.0])

        # The left hand side is clamped, the other joints are simply supported.
        model.add_support(m["joints"][1], freedoms.ALL_DOFS)
        # The middle support moves down by 0.25 inches.
        model.add_support(m["joints"][2], freedoms.U2, -0.25)
        model.add_support(m["joints"][3], freedoms.U2)

        # Define the beam members.
        s1 = section.beam_2d_section("s1", E, A, I)
        model.add_beam_member(m, 1, [1, 2], s1)
        model.add_beam_member(m, 2, [2, 3], s1)

        model.number_dofs(m)

        model.solve_statics(m)

        member = m["beam_members"][1]
        connectivity = member["connectivity"]
        i, j = m["joints"][connectivity[0]], m["joints"][connectivity[1]]
        f = beam.beam_2d_end_forces(member, i, j)
        # print("Member 1 end forces: ", f)
        if abs(f["Ni"]) > 1e-3:
            raise ValueError("Incorrect force")
        if abs(f["Qzi"] / 3.9558 - 1) > 1e-3:
            raise ValueError("Incorrect force")
        if abs(f["Myi"] / -258.92857 - 1) > 1e-3:
            raise ValueError("Incorrect force")

        member = m["beam_members"][2]
        connectivity = member["connectivity"]
        i, j = m["joints"][connectivity[0]], m["joints"][connectivity[1]]
        f = beam.beam_2d_end_forces(member, i, j)
        # print("Member 2 end forces: ", f)
        if abs(f["Ni"]) > 1e-3:
            raise ValueError("Incorrect force")
        if abs(f["Qzi"] / -1.7981 - 1) > 1e-3:
            raise ValueError("Incorrect force")
        if abs(f["Myi"] / 215.7738 - 1) > 1e-3:
            raise ValueError("Incorrect force")

        # plots.setup(m, set_limits=True)
        # plots.plot_members(m)
        # plots.plot_member_ids(m)
        # plots.plot_joint_ids(m)
        # plots.plot_member_orientation(m, 10.0)
        # plots.show(m)

        # plots.setup(m)
        # plots.plot_members(m)
        # plots.plot_deformations(m, 100.0)
        # plots.show(m)

        # plots.setup(m)
        # plots.plot_members(m)
        # ax = plots.plot_bending_moments(m, 0.5)
        # ax.set_title("Moments")
        # plots.show(m)

        # plots.setup(m)
        # plots.plot_members(m)
        # ax = plots.plot_shear_forces(m, 5.5)
        # ax.set_title("Shear forces")
        # plots.show(m)

    def test_hinge_frame(self):
        """
        # Example of a support-settlement problem (Section 7.4)

        This example is completely solved in the book Matrix Analysis of Structures by
        Robert E. Sennett, ISBN 978-1577661436.

        Displacements and internal forces are provided in the book, and we can check our
        solution against these reference values.
        """

        # US customary units, inches, pounds, seconds are assumed.

        # The book gives the product of the modulus of elasticity and the moment of inertia as 2.9e6.
        E = 29e6
        I = 100.0
        A = 10.0  # cross-sectional area does not influence the results
        L = 10 * 12  # span in inches

        m = model.create(2)
        freedoms = m["freedoms"]

        model.add_joint(m, 1, [0.0, 0.0])
        model.add_joint(m, 2, [0, L])
        model.add_joint(m, 5, [0, L])
        model.add_joint(m, 3, [L, L])
        model.add_joint(m, 4, [L, 0.0])

        # The left hand side is clamped, the other joints are simply supported.
        model.add_support(m["joints"][1], freedoms.TRANSLATION_DOFS)
        model.add_support(m["joints"][4], freedoms.TRANSLATION_DOFS)

        # Define the beam members.
        s1 = section.beam_2d_section("s1", E, A, I)
        model.add_beam_member(m, 1, [1, 2], s1)
        model.add_beam_member(m, 2, [5, 3], s1)
        model.add_beam_member(m, 3, [4, 3], s1)

        model.add_dof_links(m, [2, 5], freedoms.U1)
        model.add_dof_links(m, [2, 5], freedoms.U2)

        model.add_load(m["joints"][2], freedoms.U1, 1000.0)

        model.number_dofs(m)

        model.solve_statics(m)

        # for jid in [2, 3]:
        #     j = m["joints"][jid]
        #     print(jid, j["displacements"])

        d2 = m["joints"][2]["displacements"]
        d5 = m["joints"][5]["displacements"]
        if norm(d2[0:2] - d5[0:2]) > 1e-3:
            raise ValueError("Incorrect displacement")

        if abs(d2[0] - 0.3985) / 0.3985 > 1e-3:
            raise ValueError("Incorrect displacement")
        if abs(d2[1] - 0.00041) / 0.00041 > 1e-2:
            raise ValueError("Incorrect displacement")

        # member = m["beam_members"][1]
        # connectivity = member["connectivity"]
        # i, j = m["joints"][connectivity[0]], m["joints"][connectivity[1]]
        # f = beam.beam_2d_end_forces(member, i, j)
        # print("Member 1 end forces: ", f)
        # if abs(f["Ni"]) > 1e-3:
        #     raise ValueError("Incorrect force")
        # if abs(f["Qzi"] / 3.9558 - 1) > 1e-3:
        #     raise ValueError("Incorrect force")
        # if abs(f["Myi"] / -258.92857 - 1) > 1e-3:
        #     raise ValueError("Incorrect force")

        # member = m["beam_members"][2]
        # connectivity = member["connectivity"]
        # i, j = m["joints"][connectivity[0]], m["joints"][connectivity[1]]
        # f = beam.beam_2d_end_forces(member, i, j)
        # print("Member 2 end forces: ", f)
        # if abs(f["Ni"]) > 1e-3:
        #     raise ValueError("Incorrect force")
        # if abs(f["Qzi"] / -1.7981 - 1) > 1e-3:
        #     raise ValueError("Incorrect force")
        # if abs(f["Myi"] / 215.7738 - 1) > 1e-3:
        #     raise ValueError("Incorrect force")

        # plots.setup(m, set_limits=True)
        # plots.plot_members(m)
        # plots.plot_member_ids(m)
        # plots.plot_joint_ids(m)
        # plots.plot_member_orientation(m, 10.0)
        # plots.show(m)

        # plots.setup(m)
        # plots.plot_members(m)
        # plots.plot_deformations(m, 100.0)
        # plots.show(m)

        # plots.setup(m)
        # plots.plot_members(m)
        # ax = plots.plot_bending_moments(m, 0.0005)
        # ax.set_title("Moments")
        # plots.show(m)

        # plots.setup(m)
        # plots.plot_members(m)
        # ax = plots.plot_shear_forces(m, 0.01)
        # ax.set_title("Shear forces")
        # plots.show(m)

    def test_SDLX_01_89_vibration_tut(self):
        """
        pystran - Python package for structural analysis with trusses and beams

        (C) 2025-2026, Petr Krysl, pkrysl@ucsd.edu

        # Two story planar frame vibration

        SCIA Engineer 24.0.1020 test case SDLX 01/89
        """

        #         # from pystran import model
        # from pystran import section
        # from pystran import plots

        # The material is steel, SI units (m).
        E = 2.1e11
        rho = 7.85e3

        H = 29.0e-3  # millimeters to meters
        B = 4.8e-3
        A = H * B
        I = H * B**3 / 12

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

        # plots.setup(m, set_limits=True)
        # plots.plot_members(m)
        # plots.plot_member_ids(m)
        # plots.plot_member_orientation(m, 0.05)
        # ax = plots.plot_joint_ids(m)
        # ax.set_title("Structure before refinement")
        # plots.show(m)

        # All members will now be refined into eight finite elements. Without the
        # refinement, the reference solutions cannot be reproduced: there simply
        # wouldn't be enough degrees of freedom. Unfortunately the reference publication
        # does not mention the numbers of finite elements used per member.
        nref = 8
        for i in range(6):
            model.refine_member(m, i + 1, nref)

        # plots.setup(m, set_limits=True)
        # plots.plot_members(m)
        # ax = plots.plot_joint_ids(m)
        # ax.set_title("Structure after refinement")
        # plots.show(m)

        # Solve a free vibration analysis problem.
        model.number_dofs(m)
        model.solve_free_vibration(m)

        # Compare with the reference frequencies.
        reffs = [8.75, 29.34, 43.71, 56.12, 95.86, 102.37, 146.64, 174.39, 178.36]
        for mode, reff in enumerate(reffs):
            # print(f'Mode {mode}: {m["frequencies"][mode]} vs {reff} Hz')
            if abs((m["frequencies"][mode] - reff) / reff) > 1e-2:
                raise ValueError("Incorrect frequency")

        # Show the first four modes.
        # for mode in range(0, 4):
        #     print(f"Mode {mode}: ", m["frequencies"][mode])
        #     ax = plots.setup(m)
        #     plots.plot_members(m)
        #     model.set_solution(m, mode)
        #     plots.plot_deformations(m, 0.2)
        #     ax.set_title(f"Mode {mode}, frequency = {m['frequencies'][mode]:.2f} Hz")
        #     plots.show(m)

    def test_12_timo_beam_on_springs_tut(self):
        """
        pystran - Python package for structural analysis with trusses and beams

        (C) 2025-2026, Petr Krysl, pkrysl@ucsd.edu

        # Natural Frequency of Mass supported by a Beam on Springs

        Reference: Timoshenko, S., Young, D., and Weaver, W., Vibration Problems in
        Engineering, John Wiley & Sons, 4th edition, 1974. page 11, problem 1.1-3.

        Problem: A simple beam is supported by two spring at the endpoints. Neglecting
        the distributed mass of the beam, calculate the period of free vibration of the
        beam given a concentrated mass of weight W.

        The answer in the book is: T = 0.533 sec., corresponding to the frequency =
        1.876 CPS.
        """

        # from math import sqrt, pi
        #         # from pystran import model
        # from pystran import section
        # from pystran import plots

        # US customary units, converted to inches, lbf, and lbm
        da = 12 * 7  # 7 ft converted to inches
        db = 12 * 3  # 3 ft converted to inches
        K = 300.0  # 300 lbf/in
        W = 1000.0  # lbf
        M = W / (12 * 32.174)  # mass in lbm
        E = 3e7  # 30e6 psi
        A = 1.0
        I = 1.0
        rho = 1e-12  # artificially reduce the mass density of the beam

        m = model.create(2)
        freedoms = m["freedoms"]

        model.add_joint(m, 1, [0.0, 0.0])
        model.add_joint(m, 2, [da, 0.0])
        model.add_joint(m, 3, [da + db, 0.0])
        model.add_joint(m, "ground", [0.0, 0.0])
        model.add_support(m["joints"][1], freedoms.U1)
        model.add_support(m["joints"][3], freedoms.U1)
        model.add_support(m["joints"]["ground"], freedoms.ALL_DOFS)

        s2 = section.beam_2d_section("s2", E, A, I, rho)
        model.add_beam_member(m, 1, [1, 2], s2)
        model.add_beam_member(m, 2, [2, 3], s2)

        model.add_mass(m["joints"][2], freedoms.U1, M)
        model.add_mass(m["joints"][2], freedoms.U2, M)

        # In the vertical direction, the spring stiffness is K. A spring is added at
        # either end of the beam.
        ss = section.spring_section("ss", "extension", [0, 1], K)
        model.add_spring_member(m, 1, [1, "ground"], ss)
        model.add_spring_member(m, 2, [3, "ground"], ss)

        model.number_dofs(m)
        model.solve_free_vibration(m)

        # The expected frequency is 1.876 CPS, hence the vibration period is 0.533 sec.
        self.assertAlmostEqual(m["frequencies"][0], 1.876, places=3)
        # for mode in range(1):
        #     plots.setup(m)
        #     plots.plot_members(m)
        #     model.set_solution(m, m["eigvecs"][:, mode])
        #     ax = plots.plot_deformations(m, 50.0)
        #     print(
        #         f"Mode {mode}: f = {sqrt(m['eigvals'][mode])/2/pi:.6f} Hz, T = {2*pi/sqrt(m['eigvals'][mode]):.6f} sec"
        #     )
        #     ax.set_title(
        #         f"Mode {mode}: f = {sqrt(m['eigvals'][mode])/2/pi:.3f} Hz, T = {2*pi/sqrt(m['eigvals'][mode]):.3f} sec"
        #     )
        #     plots.show(m)


def main():
    unittest.main()


if __name__ == "__main__":
    main()

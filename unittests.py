"""
pystran unit tests
"""

import unittest

from numpy import array, dot, outer, concatenate
from numpy.linalg import norm
from pystran import model
from pystran import section
from pystran import freedoms
from pystran import beam
from pystran import rotation


class UnitTestsPlaneTrusses(unittest.TestCase):

    def test_truss_dome(self):
        """
        Analysis of Geometrically
        Nonlinear Structures
        Second Edition

        by
        Robert Levy
        Technion-Israel Institute of Technology,
        Haifa, Israel

        and
        William R. Spillers
        New Jersey Institute of Technology,


        Space truss dome in section 2.4.2

        Vertical deflection at the crown: -.20641184e+00 in (linear analysis)
        """

        m = model.create(3)

        model.add_joint(m, 1, [0.0, 0.0, 0.32346000e1])
        model.add_joint(m, 2, [0.49212500e1, 0.85239000e1, 0.24472000e1])
        model.add_joint(m, 3, [-0.49212500e1, 0.85239000e1, 0.24472000e1])
        model.add_joint(m, 4, [-0.98425000e1, 0.0, 0.24472000e1])
        model.add_joint(m, 5, [-0.49212500e1, -0.85239000e1, 0.24472000e1])
        model.add_joint(m, 6, [0.49212500e1, -0.85239000e1, 0.24472000e1])
        model.add_joint(m, 7, [0.98425000e1, 0.0, 0.24472000e1])
        model.add_joint(m, 8, [0.0, 0.19685000e02, 0.0])
        model.add_joint(m, 9, [-0.17047200e02, 0.98425000e1, 0.0])
        model.add_joint(m, 10, [-0.17047200e02, -0.98425000e1, 0.0])
        model.add_joint(m, 11, [0.0, -0.19685000e02, 0.0])
        model.add_joint(m, 12, [0.17047200e02, -0.98425000e1, 0.0])
        model.add_joint(m, 13, [0.17047200e02, 0.98425000e1, 0.0])

        E = 30000000.0
        A = 0.0155
        s1 = section.truss_section("steel", E, A)

        for id, j in enumerate(
            [
                [1, 2],
                [1, 3],
                [1, 4],
                [1, 5],
                [1, 6],
                [1, 7],
                [6, 12],
                [7, 12],
                [7, 13],
                [2, 13],
                [2, 8],
                [3, 8],
                [3, 9],
                [4, 9],
                [4, 10],
                [5, 10],
                [5, 11],
                [6, 11],
                [2, 3],
                [3, 4],
                [4, 5],
                [5, 6],
                [6, 7],
                [7, 2],
            ]
        ):
            model.add_truss_member(m, id, j, s1)

        for i in [8, 9, 10, 11, 12, 13]:
            for d in range(m["dim"]):
                model.add_support(m["joints"][i], d)

        model.add_load(m["joints"][1], 2, -220.46)

        model.number_dofs(m)
        # print("Total Degrees of Freedom = ", m["ntotaldof"])
        # print("Free Degrees of Freedom = ", m["nfreedof"])

        model.solve_statics(m)

        # for j in m["joints"].values():
        #     print(j["displacements"])

        if norm(m["joints"][1]["displacements"][2] - (-0.20641184e00)) > 1.0e-3 * abs(
            -0.20641184e00
        ):
            raise ValueError("Displacement calculation error")
        # else:
        #     print("Displacement calculation OK")


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

        from math import sqrt, pi
        from numpy import array
        from numpy.linalg import norm
        from pystran import model
        from pystran import section
        from pystran import plots

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
        #     plots.plot_setup(m)
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

        # plots.plot_setup(m, set_limits=True)
        # plots.plot_members(m)
        # plots.plot_member_numbers(m)
        # plots.plot_joint_numbers(m)
        # plots.plot_beam_orientation(m, 10.0)
        # plots.show(m)

        # plots.plot_setup(m)
        # plots.plot_members(m)
        # plots.plot_deformations(m, 100.0)
        # plots.show(m)

        # plots.plot_setup(m)
        # plots.plot_members(m)
        # ax = plots.plot_bending_moments(m, 0.5)
        # ax.set_title("Moments")
        # plots.show(m)

        # plots.plot_setup(m)
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

        model.add_links(m, [2, 5], freedoms.U1)
        model.add_links(m, [2, 5], freedoms.U2)

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

        # plots.plot_setup(m, set_limits=True)
        # plots.plot_members(m)
        # plots.plot_member_numbers(m)
        # plots.plot_joint_numbers(m)
        # plots.plot_beam_orientation(m, 10.0)
        # plots.show(m)

        # plots.plot_setup(m)
        # plots.plot_members(m)
        # plots.plot_deformations(m, 100.0)
        # plots.show(m)

        # plots.plot_setup(m)
        # plots.plot_members(m)
        # ax = plots.plot_bending_moments(m, 0.0005)
        # ax.set_title("Moments")
        # plots.show(m)

        # plots.plot_setup(m)
        # plots.plot_members(m)
        # ax = plots.plot_shear_forces(m, 0.01)
        # ax.set_title("Shear forces")
        # plots.show(m)

    def test_SDLX_01_89_vibration_tut(self):
        """
        pystran - Python package for structural analysis with trusses and beams

        (C) 2025, Petr Krysl, pkrysl@ucsd.edu

        # Two story planar frame vibration

        SCIA Engineer 24.0.1020 test case SDLX 01/89
        """

        # from context import pystran
        # from pystran import model
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

        # plots.plot_setup(m, set_limits=True)
        # plots.plot_members(m)
        # plots.plot_member_numbers(m)
        # plots.plot_beam_orientation(m, 0.05)
        # ax = plots.plot_joint_numbers(m)
        # ax.set_title("Structure before refinement")
        # plots.show(m)

        # All members will now be refined into eight finite elements. Without the
        # refinement, the reference solutions cannot be reproduced: there simply
        # wouldn't be enough degrees of freedom. Unfortunately the reference publication
        # does not mention the numbers of finite elements used per member.
        nref = 8
        for i in range(6):
            model.refine_member(m, i + 1, nref)

        # plots.plot_setup(m, set_limits=True)
        # plots.plot_members(m)
        # ax = plots.plot_joint_numbers(m)
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
        #     ax = plots.plot_setup(m)
        #     plots.plot_members(m)
        #     model.set_solution(m, mode)
        #     plots.plot_deformations(m, 0.2)
        #     ax.set_title(f"Mode {mode}, frequency = {m['frequencies'][mode]:.2f} Hz")
        #     plots.show(m)

    def test_12_timo_beam_on_springs_tut(self):
        """
        pystran - Python package for structural analysis with trusses and beams

        (C) 2025, Petr Krysl, pkrysl@ucsd.edu

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
        # from context import pystran
        # from pystran import model
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

        model.add_joint(m, 1, [0.0, 0.0])
        model.add_joint(m, 2, [da, 0.0])
        model.add_joint(m, 3, [da + db, 0.0])
        model.add_support(m["joints"][1], freedoms.U1)
        model.add_support(m["joints"][3], freedoms.U1)

        s2 = section.beam_2d_section("s2", E, A, I, rho)
        model.add_beam_member(m, 1, [1, 2], s2)
        model.add_beam_member(m, 2, [2, 3], s2)

        model.add_mass(m["joints"][2], freedoms.U1, M)
        model.add_mass(m["joints"][2], freedoms.U2, M)

        # In the vertical direction, the spring stiffness is K. A spring is added at
        # either end of the beam.
        model.add_extension_spring_to_ground(m["joints"][1], 1, [0, 1, 0], K)
        model.add_extension_spring_to_ground(m["joints"][3], 2, [0, 1, 0], K)

        model.number_dofs(m)
        model.solve_free_vibration(m)

        # The expected frequency is 1.876 CPS, hence the vibration period is 0.533 sec.
        self.assertAlmostEqual(m["frequencies"][0], 1.876, places=3)
        # for mode in range(1):
        #     plots.plot_setup(m)
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


class UnitTestsSpaceFrames(unittest.TestCase):

    def test_spfr_weaver_gere_1(self):
        """
        Created on 01/12/2025

        Example 5.8 from Matrix Structural Analysis: Second Edition 2nd Edition by
        William McGuire, Richard H. Gallagher, Ronald D. Ziemian

        The section properties are not completely defined in the book.  They are
        taken from example 4.8, which does not provide both second moments of area.
        They are taken here as both the same.
        """

        # SI units
        L = 3.0
        E = 200e9
        G = 80e9
        A = 0.01
        Iz = 1e-3
        Iy = 1e-3
        Ix = 2e-3
        J = Ix  # Torsional constant
        P = 60000

        m = model.create(3)

        jA, jB, jC, jD, jE = 3, 1, 2, 4, 5
        model.add_joint(m, jA, [0.0, 0.0, 0.0])
        model.add_joint(m, jB, [0.0, L, 0.0])
        model.add_joint(m, jC, [2 * L, L, 0.0])
        model.add_joint(m, jD, [3 * L, 0.0, L])
        model.add_joint(m, jE, [L, L, 0.0])

        model.add_support(m["joints"][jA], freedoms.ALL_DOFS)
        model.add_support(m["joints"][jD], freedoms.ALL_DOFS)

        xz_vector = [1, 0, 0]
        s1 = section.beam_3d_section(
            "sect_1", E=E, G=G, A=A, Ix=Ix, Iy=Iy, Iz=Iz, J=J, xz_vector=xz_vector
        )
        model.add_beam_member(m, 1, [jA, jB], s1)
        xz_vector = [0, 1, 0]
        s2 = section.beam_3d_section(
            "sect_2", E=E, G=G, A=A, Ix=Ix, Iy=Iy, Iz=Iz, J=J, xz_vector=xz_vector
        )
        model.add_beam_member(m, 2, [jE, jB], s2)
        model.add_beam_member(m, 3, [jE, jC], s2)
        model.add_beam_member(m, 4, [jC, jD], s1)

        model.add_load(m["joints"][jB], freedoms.U1, 2 * P)
        model.add_load(m["joints"][jE], freedoms.U3, 4 * P)
        model.add_load(m["joints"][jC], freedoms.U2, -P)
        model.add_load(m["joints"][jC], freedoms.UR3, -P * L)

        model.number_dofs(m)

        # print("Number of free degrees of freedom = ", m["nfreedof"])
        # print("Number of all degrees of freedom = ", m["ntotaldof"])

        # print([j['dof'] for j in m['joints'].values()])

        model.solve_statics(m)

        # for jid in [jB, jC, jE]:
        #     j = m["joints"][jid]
        #     print(jid, j["displacements"])

        # print(m['K'][0:m['nfreedof'], 0:m['nfreedof']])

        # print(m['U'][0:m['nfreedof']])

        if (
            norm(
                m["joints"][1]["displacements"]
                - [
                    -8.59409726e-04,
                    5.77635277e-05,
                    5.00764459e-03,
                    2.39333188e-03,
                    -1.62316861e-03,
                    6.81331291e-04,
                ]
            )
            > 1.0e-5
        ):
            raise ValueError("Displacement calculation error")
        # else:
        #     print("Displacement calculation OK")

        if (
            norm(
                m["joints"][2]["displacements"]
                - [
                    -0.00117605,
                    0.00325316,
                    0.00525552,
                    0.00128843,
                    0.00172094,
                    -0.00077147,
                ]
            )
            > 1.0e-5
        ):
            raise ValueError("Displacement calculation error")
        # else:
        #     print("Displacement calculation OK")

        # print('Reference: ', [-0.02238452,  0.00419677,  0.00593197])

        # plots.plot_setup(m)
        # plots.plot_members(m)
        # # plots.plot_member_numbers(m)
        # plots.plot_deformations(m, 100.0)
        # # ax = plots.plot_shear_forces(m, scale=0.50e-3)
        # # ax.set_title('Shear forces')
        # plots.show(m)

    def test_weaver_1(self):
        """
        Created on 01/20/2025

        Weaver Jr., W., Computer Programs for Structural Analysis, page 146,
        problem 8. From: STAAD.Pro 2023.00.03
        """

        # US customary units, inches, pounds, seconds
        L = 120.0
        E = 30000
        G = E / (2 * (1 + 0.3))
        A = 11
        Iz = 56
        Iy = 56
        Ix = 83
        J = Ix  # Torsional constant
        F = 2
        P = 1
        M = 120

        m = model.create(3)

        model.add_joint(m, 3, [0.0, 0.0, 0.0])
        model.add_joint(m, 1, [0.0, L, 0.0])
        model.add_joint(m, 2, [2 * L, L, 0.0])
        model.add_joint(m, 4, [3 * L, 0.0, L])

        model.add_support(m["joints"][3], freedoms.ALL_DOFS)
        model.add_support(m["joints"][4], freedoms.ALL_DOFS)

        xz_vector = [1, 0, 0]
        sect_1 = section.beam_3d_section(
            "sect_1", E=E, G=G, A=A, Ix=Ix, Iy=Iy, Iz=Iz, J=J, xz_vector=xz_vector
        )
        model.add_beam_member(m, 1, [3, 1], sect_1)
        xz_vector = [0, 1, 0]
        sect_2 = section.beam_3d_section(
            "sect_2", E=E, G=G, A=A, Ix=Ix, Iy=Iy, Iz=Iz, J=J, xz_vector=xz_vector
        )
        model.add_beam_member(m, 2, [1, 2], sect_2)
        model.add_beam_member(m, 3, [2, 4], sect_2)

        model.add_load(m["joints"][1], freedoms.U1, F)
        model.add_load(m["joints"][2], freedoms.U2, -P)
        model.add_load(m["joints"][2], freedoms.UR3, -M)

        model.number_dofs(m)

        # print("Number of free degrees of freedom = ", m["nfreedof"])
        # print("Number of all degrees of freedom = ", m["ntotaldof"])

        # print([j['dof'] for j in m['joints'].values()])

        model.solve_statics(m)

        # for jid in [1, 2]:
        #     j = m["joints"][jid]
        #     print(jid, j["displacements"])

        # print(m['K'][0:m['nfreedof'], 0:m['nfreedof']])

        # print(m['U'][0:m['nfreedof']])

        # 0.22267   0.00016  -0.17182  -0.00255   0.00217  -0.00213
        # 0.22202  -0.48119  -0.70161  -0.00802   0.00101  -0.00435
        ref1 = [0.22267, 0.00016, -0.17182, -0.00255, 0.00217, -0.00213]
        if norm(m["joints"][1]["displacements"] - ref1) > 1.0e-1 * norm(ref1):
            raise ValueError("Displacement calculation error")
        # else:
        #     print("Displacement calculation OK")
        ref2 = [0.22202, -0.48119, -0.70161, -0.00802, 0.00101, -0.00435]
        if norm(m["joints"][2]["displacements"] - ref2) > 1.0e-1 * norm(ref2):
            raise ValueError("Displacement calculation error")
        # else:
        #     print("Displacement calculation OK")

        # plots.plot_setup(m)
        # plots.plot_members(m)
        # # plots.plot_member_numbers(m)
        # plots.plot_deformations(m, 30.0)
        # # ax = plots.plot_shear_forces(m, scale=0.50e-3)
        # # ax.set_title('Shear forces')
        # plots.show(m)

    def test_original_weaver_1(self):
        """
        Created on 01/12/2025

        Example 5.8 from Matrix Structural Analysis: Second Edition 2nd Edition by
        William McGuire, Richard H. Gallagher, Ronald D. Ziemian

        The section properties are not completely defined in the book.  They are
        taken from example 4.8, which does not provide both second moments of area.
        They are taken here as both the same.
        """

        # US customary units, inches, pounds, seconds
        L = 120.0
        E = 30000
        G = E / (2 * (1 + 0.3))
        A = 11
        Iz = 56
        Iy = 56
        Ix = 83
        J = Ix  # Torsional constant
        F = 2
        P = 1
        M = 120

        m = model.create(3)

        model.add_joint(m, 3, [0.0, 0.0, 0.0])
        model.add_joint(m, 1, [0.0, L, 0.0])
        model.add_joint(m, 2, [2 * L, L, 0.0])
        model.add_joint(m, 4, [3 * L, 0.0, L])

        model.add_support(m["joints"][3], freedoms.ALL_DOFS)
        model.add_support(m["joints"][4], freedoms.ALL_DOFS)

        xz_vector = [0, 0, 1]
        sect_1 = section.beam_3d_section(
            "sect_1", E=E, G=G, A=A, Ix=Ix, Iy=Iy, Iz=Iz, J=J, xz_vector=xz_vector
        )
        xz_vector = [0, 0, 1]
        sect_2 = section.beam_3d_section(
            "sect_2", E=E, G=G, A=A, Ix=Ix, Iy=Iy, Iz=Iz, J=J, xz_vector=xz_vector
        )
        xz_vector = rotation.rotate(m["joints"][2], m["joints"][4], [0, 1, 0], 90)
        sect_3 = section.beam_3d_section(
            "sect_3", E=E, G=G, A=A, Ix=Ix, Iy=Iy, Iz=Iz, J=J, xz_vector=xz_vector
        )

        model.add_beam_member(m, 1, [1, 2], sect_1)
        model.add_beam_member(m, 2, [3, 1], sect_2)
        model.add_beam_member(m, 3, [2, 4], sect_3)

        model.add_load(m["joints"][1], freedoms.U1, F)
        model.add_load(m["joints"][2], freedoms.U2, -P)
        model.add_load(m["joints"][2], freedoms.UR3, -M)

        model.number_dofs(m)

        # print("Number of free degrees of freedom = ", m["nfreedof"])
        # print("Number of all degrees of freedom = ", m["ntotaldof"])

        # print([j['dof'] for j in m['joints'].values()])

        model.solve_statics(m)

        # for jid in [1, 2]:
        #     j = m["joints"][jid]
        #     print(jid, j["displacements"])

        # print(m['K'][0:m['nfreedof'], 0:m['nfreedof']])

        # print(m['U'][0:m['nfreedof']])

        # 0.22267   0.00016  -0.17182  -0.00255   0.00217  -0.00213
        # 0.22202  -0.48119  -0.70161  -0.00802   0.00101  -0.00435
        ref1 = [0.22267, 0.00016, -0.17182, -0.00255, 0.00217, -0.00213]
        if norm(m["joints"][1]["displacements"] - ref1) > 1.0e-1 * norm(ref1):
            raise ValueError("Displacement calculation error")
        # else:
        #     print("Displacement calculation OK")
        ref2 = [0.22202, -0.48119, -0.70161, -0.00802, 0.00101, -0.00435]
        if norm(m["joints"][2]["displacements"] - ref2) > 1.0e-1 * norm(ref2):
            raise ValueError("Displacement calculation error")
        # else:
        #     print("Displacement calculation OK")

        # for k in m["beam_members"].keys():
        #     member = m["beam_members"][k]
        #     connectivity = member["connectivity"]
        #     i, j = m["joints"][connectivity[0]], m["joints"][connectivity[1]]
        #     f = beam.beam_3d_end_forces(member, i, j)
        #     print(f"Member {k}: ")
        #     print(
        #         f"   Joint {connectivity[0]}: N={f['Ni']:.5}, Qy={f['Qyi']:.5}, Qz={f['Qzi']:.5}, T={f['Ti']:.5}, My={f['Myi']:.5}, Mz={f['Mzi']:.5}: "
        #     )
        #     print(
        #         f"   Joint {connectivity[1]}: N={f['Nj']:.5}, Qy={f['Qyj']:.5}, Qz={f['Qzj']:.5}, T={f['Tj']:.5}, My={f['Myj']:.5}, Mz={f['Mzj']:.5}: "
        #     )
        member = m["beam_members"][1]
        connectivity = member["connectivity"]
        i, j = m["joints"][connectivity[0]], m["joints"][connectivity[1]]
        f = beam.beam_3d_end_forces(member, i, j)
        # Member 1:
        self.assertAlmostEqual(f["Ni"], -0.89607, places=3)
        self.assertAlmostEqual(f["Qyi"], 0.43381, places=3)
        self.assertAlmostEqual(f["Qzi"], -0.21393, places=3)
        self.assertAlmostEqual(f["Ti"], -22.661, places=3)
        self.assertAlmostEqual(f["Myi"], 17.656, places=3)
        self.assertAlmostEqual(f["Mzi"], 36.188, places=3)
        # Nj=0.89607, Qy=-0.43381, Qz=0.21393, T=22.661, My=33.689, Mz=67.927
        # Member 2:
        member = m["beam_members"][2]
        connectivity = member["connectivity"]
        i, j = m["joints"][connectivity[0]], m["joints"][connectivity[1]]
        f = beam.beam_3d_end_forces(member, i, j)
        self.assertAlmostEqual(f["Ni"], 0.43381, places=3)
        self.assertAlmostEqual(f["Qyi"], -1.1039, places=3)
        self.assertAlmostEqual(f["Qzi"], -0.21393, places=3)
        self.assertAlmostEqual(f["Ti"], 17.656, places=3)
        self.assertAlmostEqual(f["Myi"], 48.333, places=3)
        self.assertAlmostEqual(f["Mzi"], -96.284, places=3)
        # Nj=-0.43381, Qy=1.1039, Qz=0.21393, T=-17.656, My=-22.661, Mz=-36.188
        # Member 3:
        member = m["beam_members"][3]
        connectivity = member["connectivity"]
        i, j = m["joints"][connectivity[0]], m["joints"][connectivity[1]]
        f = beam.beam_3d_end_forces(member, i, j)
        self.assertAlmostEqual(f["Ni"], -1.4687, places=3)
        self.assertAlmostEqual(f["Qyi"], 0.71755, places=3)
        self.assertAlmostEqual(f["Qzi"], 0.48234, places=3)
        self.assertAlmostEqual(f["Ti"], 36.432, places=3)
        self.assertAlmostEqual(f["Myi"], -15.499, places=3)
        self.assertAlmostEqual(f["Mzi"], 52.845, places=3)
        # Nj=1.4687, Qy=-0.71755, Qz=-0.48234, T=-36.432, My=-84.753, Mz=96.294

        model.statics_reactions(m)

        # for jid in [3, 4]:
        #     j = m["joints"][jid]
        #     print(f"Joint {jid}:")
        #     print(
        #         f"   Rx={j['reactions'][0]:.5}, Ry={j['reactions'][1]:.5}, Rz={j['reactions'][2]:.5}, Mx={j['reactions'][3]:.5}, My={j['reactions'][4]:.5}, Mz={j['reactions'][5]:.5}: "
        #     )
        j = m["joints"][3]
        self.assertAlmostEqual(j["reactions"][0], -1.1039, places=3)
        self.assertAlmostEqual(j["reactions"][1], -0.43381, places=3)
        self.assertAlmostEqual(j["reactions"][2], 0.21393, places=3)
        self.assertAlmostEqual(j["reactions"][3], 48.333, places=3)
        self.assertAlmostEqual(j["reactions"][4], -17.656, places=3)
        self.assertAlmostEqual(j["reactions"][5], 96.284, places=3)

        # plots.plot_setup(m)
        # plots.plot_members(m)
        # plots.plot_beam_orientation(m, 20)
        # plots.plot_deformations(m, 80.0)
        # # ax = plots.plot_shear_forces(m, scale=0.50e-3)
        # # ax.set_title('Shear forces')
        # plots.show(m)

    def test_linked_cantilevers_free(self):
        """
        Created on 01/22/2025

        Linked cantilevers through their tips, with a prescribed displacement given at
        the linked joints.
        """
        # SI units
        L = 3.0
        H = 0.3
        B = 0.2
        E = 200e9
        G = 80e9
        P = 60000

        A, Ix, Iy, Iz, J = section.rectangle(H, B)
        xz_vector = [0, 0, 1]
        sect_1 = section.beam_3d_section(
            "sect_1", E=E, G=G, A=A, Ix=Ix, Iy=Iy, Iz=Iz, J=J, xz_vector=xz_vector
        )
        A, Ix, Iy, Iz, J = section.rectangle(H, B / 2)
        xz_vector = [0, 0, 1]
        sect_2 = section.beam_3d_section(
            "sect_2", E=E, G=G, A=A, Ix=Ix, Iy=Iy, Iz=Iz, J=J, xz_vector=xz_vector
        )

        m = model.create(3)

        model.add_joint(m, 1, [0.0, 0.0, 0.0])
        model.add_joint(m, 2, [L, 0, 0.0])
        model.add_joint(m, 3, [2 * L, 0, 0])
        model.add_joint(m, 4, [L, 0, 0.0])

        model.add_support(m["joints"][1], freedoms.ALL_DOFS)
        model.add_support(m["joints"][3], freedoms.ALL_DOFS)

        model.add_beam_member(m, 1, [1, 2], sect_1)
        model.add_beam_member(m, 2, [3, 4], sect_2)

        model.add_load(m["joints"][4], freedoms.U3, -P)

        model.add_links(m, [2, 4], freedoms.U1)
        model.add_links(m, [2, 4], freedoms.U2)
        model.add_links(m, [2, 4], freedoms.U3)

        model.number_dofs(m)

        # print("Number of free degrees of freedom = ", m["nfreedof"])
        # print("Number of all degrees of freedom = ", m["ntotaldof"])

        # print([j['dof'] for j in m['joints'].values()])

        model.solve_statics(m)

        # for id in [2, 4]:
        #     j = m["joints"][id]
        #     print(id, j["displacements"])

        for k in m["beam_members"].keys():
            member = m["beam_members"][k]
            connectivity = member["connectivity"]
            i, j = m["joints"][connectivity[0]], m["joints"][connectivity[1]]
            f = beam.beam_3d_end_forces(member, i, j)
            # print(f"Member {k}: ")
            # print(
            #     f" Joint {connectivity[0]}: N={f['Ni']:.5}, Qy={f['Qyi']:.5}, Qz={f['Qzi']:.5}, T={f['Ti']:.5}, My={f['Myi']:.5}, Mz={f['Mzi']:.5}: "
            # )
            # print(
            #     f" Joint {connectivity[1]}: N={f['Nj']:.5}, Qy={f['Qyj']:.5}, Qz={f['Qzj']:.5}, T={f['Tj']:.5}, My={f['Myj']:.5}, Mz={f['Mzj']:.5}: "
            # )

        model.statics_reactions(m)

        # for jid in [1, 3]:
        #     j = m["joints"][jid]
        #     print(f"Joint {jid}:")
        #     print(
        #         f"   Rx={j['reactions'][0]:.5}, Ry={j['reactions'][1]:.5}, Rz={j['reactions'][2]:.5}, Mx={j['reactions'][3]:.5}, My={j['reactions'][4]:.5}, Mz={j['reactions'][5]:.5}: "
        #     )
        j1 = m["joints"][1]
        j3 = m["joints"][3]
        self.assertAlmostEqual(j1["reactions"][2] + j3["reactions"][2], P)

        # plots.plot_setup(m)
        # plots.plot_members(m)
        # plots.plot_member_numbers(m)
        # plots.plot_deformations(m, 100.0)
        # plots.plot_beam_orientation(m, 0.5)
        # # plots.plot_bending_moments(m, 0.00001, "y")
        # # plots.plot_bending_moments(m, 0.00001, "z")
        # # ax = plots.plot_shear_forces(m, scale=0.50e-3)
        # # ax.set_title('Shear forces')
        # plots.show(m)

    def test_linked_cantilevers_prescribed(self):
        """
        Created on 01/22/2025

        Linked cantilevers through their tips, with a force given at the linked joints.
        """
        # SI units
        L = 3.0
        H = 0.3
        B = 0.2
        E = 200e9
        G = 80e9
        P = 60000

        A, Ix, Iy, Iz, J = section.rectangle(H, B)
        xz_vector = [0, 0, 1]
        sect_1 = section.beam_3d_section(
            "sect_1", E=E, G=G, A=A, Ix=Ix, Iy=Iy, Iz=Iz, J=J, xz_vector=xz_vector
        )
        A, Ix, Iy, Iz, J = section.rectangle(H, B / 2)
        xz_vector = [0, 0, 1]
        sect_2 = section.beam_3d_section(
            "sect_2", E=E, G=G, A=A, Ix=Ix, Iy=Iy, Iz=Iz, J=J, xz_vector=xz_vector
        )

        m = model.create(3)

        model.add_joint(m, 1, [0.0, 0.0, 0.0])
        model.add_joint(m, 2, [L, 0, 0.0])
        model.add_joint(m, 3, [2 * L, 0, 0])
        model.add_joint(m, 4, [L, 0, 0.0])

        model.add_links(m, [2, 4], freedoms.TRANSLATION_DOFS)

        model.add_support(m["joints"][1], freedoms.ALL_DOFS)
        model.add_support(m["joints"][3], freedoms.ALL_DOFS)
        model.add_support(m["joints"][4], freedoms.U2, 0.003)

        model.add_beam_member(m, 1, [1, 2], sect_1)
        model.add_beam_member(m, 2, [3, 4], sect_2)

        model.number_dofs(m)

        # print(m["joints"])

        # print("Number of free degrees of freedom = ", m["nfreedof"])
        # print("Number of all degrees of freedom = ", m["ntotaldof"])

        # print([j['dof'] for j in m['joints'].values()])

        model.solve_statics(m)

        # for id in [2, 3, 4]:
        #     j = m["joints"][id]
        #     print(id, j["displacements"])

        for k in m["beam_members"].keys():
            member = m["beam_members"][k]
            connectivity = member["connectivity"]
            i, j = m["joints"][connectivity[0]], m["joints"][connectivity[1]]
            f = beam.beam_3d_end_forces(member, i, j)
            # print(f"Member {k}: ")
            # print(
            #     f" Joint {connectivity[0]}: N={f['Ni']:.5}, Qy={f['Qyi']:.5}, Qz={f['Qzi']:.5}, T={f['Ti']:.5}, My={f['Myi']:.5}, Mz={f['Mzi']:.5}: "
            # )
            # print(
            #     f" Joint {connectivity[1]}: N={f['Nj']:.5}, Qy={f['Qyj']:.5}, Qz={f['Qzj']:.5}, T={f['Tj']:.5}, My={f['Myj']:.5}, Mz={f['Mzj']:.5}: "
            # )

        model.statics_reactions(m)

        # for jid in [1]:
        #     j = m["joints"][jid]
        #     print(f"Joint {jid}:")
        #     print(
        #         f"   Rx={j['reactions'][0]:.5}, Ry={j['reactions'][1]:.5}, Rz={j['reactions'][2]:.5}, Mx={j['reactions'][3]:.5}, My={j['reactions'][4]:.5}, Mz={j['reactions'][5]:.5}: "
        #     )

        j2 = m["joints"][2]
        j4 = m["joints"][4]

        self.assertAlmostEqual(
            j2["displacements"][freedoms.U2], j4["displacements"][freedoms.U2]
        )

        # plots.plot_setup(m)
        # plots.plot_members(m)
        # plots.plot_member_numbers(m)
        # plots.plot_deformations(m, 100.0)
        # plots.plot_beam_orientation(m, 0.5)
        # # plots.plot_bending_moments(m, 0.00001, "y")
        # # plots.plot_bending_moments(m, 0.00001, "z")
        # # ax = plots.plot_shear_forces(m, scale=0.50e-3)
        # # ax.set_title('Shear forces')
        # plots.show(m)

    def test_linked_four_bars_free(self):
        """
        Created on 01/23/2025

        Linked cantilevers through their tips, with a force acting at the linked joints.
        """
        # SI units
        L = 3.0
        H = 0.3
        B = 0.2
        E = 200e9
        G = 80e9
        P = 60000

        A, Ix, Iy, Iz, J = section.rectangle(H, B / 2)
        xz_vector = [0, 0, 1]
        sect_2 = section.beam_3d_section(
            "sect_2", E=E, G=G, A=A, Ix=Ix, Iy=Iy, Iz=Iz, J=J, xz_vector=xz_vector
        )

        m = model.create(3)

        model.add_joint(m, 1, [0.0, 0.0, 0.0])
        model.add_joint(m, 2, [0.0, 0.0, 0.0])
        model.add_joint(m, 3, [0.0, 0.0, 0.0])
        model.add_joint(m, 4, [0.0, 0.0, 0.0])
        model.add_joint(m, 5, [-L, 0, 0.0])
        model.add_joint(m, 6, [L, 0, 0.0])
        model.add_joint(m, 7, [0, -L, 0.0])
        model.add_joint(m, 8, [0, L, 0.0])

        model.add_support(m["joints"][5], freedoms.ALL_DOFS)
        model.add_support(m["joints"][6], freedoms.ALL_DOFS)
        model.add_support(m["joints"][7], freedoms.ALL_DOFS)
        model.add_support(m["joints"][8], freedoms.ALL_DOFS)

        model.add_beam_member(m, 1, [1, 5], sect_2)
        model.add_beam_member(m, 2, [2, 6], sect_2)
        model.add_beam_member(m, 3, [3, 7], sect_2)
        model.add_beam_member(m, 4, [4, 8], sect_2)

        model.add_load(m["joints"][4], freedoms.U3, -P)

        model.add_links(m, [1, 2, 3, 4], freedoms.TRANSLATION_DOFS)

        model.number_dofs(m)

        # print("Number of free degrees of freedom = ", m["nfreedof"])
        # print("Number of all degrees of freedom = ", m["ntotaldof"])

        # print([j['dof'] for j in m['joints'].values()])

        model.solve_statics(m)

        # for id in [2, 4]:
        #     j = m["joints"][id]
        #     print(id, j["displacements"])

        for k in m["beam_members"].keys():
            member = m["beam_members"][k]
            connectivity = member["connectivity"]
            i, j = m["joints"][connectivity[0]], m["joints"][connectivity[1]]
            f = beam.beam_3d_end_forces(member, i, j)
            # print(f"Member {k}: ")
            # print(
            #     f" Joint {connectivity[0]}: N={f['Ni']:.5}, Qy={f['Qyi']:.5}, Qz={f['Qzi']:.5}, T={f['Ti']:.5}, My={f['Myi']:.5}, Mz={f['Mzi']:.5}: "
            # )
            # print(
            #     f" Joint {connectivity[1]}: N={f['Nj']:.5}, Qy={f['Qyj']:.5}, Qz={f['Qzj']:.5}, T={f['Tj']:.5}, My={f['Myj']:.5}, Mz={f['Mzj']:.5}: "
            # )

        sum_end_forces = 0.0
        for k in m["beam_members"].keys():
            member = m["beam_members"][k]
            connectivity = member["connectivity"]
            i, j = m["joints"][connectivity[0]], m["joints"][connectivity[1]]
            f = beam.beam_3d_end_forces(member, i, j)
            sum_end_forces += f["Qzi"]

        self.assertAlmostEqual(sum_end_forces, P)

        model.statics_reactions(m)

        # for jid in [5, 6, 7, 8]:
        #     j = m["joints"][jid]
        #     print(f"Joint {jid}:")
        #     print(
        #         f"   Rx={j['reactions'][0]:.5}, Ry={j['reactions'][1]:.5}, Rz={j['reactions'][2]:.5}, Mx={j['reactions'][3]:.5}, My={j['reactions'][4]:.5}, Mz={j['reactions'][5]:.5}: "
        #     )

        # plots.plot_setup(m)
        # plots.plot_members(m)
        # plots.plot_member_numbers(m)
        # plots.plot_deformations(m, 200.0)
        # plots.plot_beam_orientation(m, 0.5)
        # # plots.plot_bending_moments(m, 0.00001, "y")
        # # plots.plot_bending_moments(m, 0.00001, "z")
        # # ax = plots.plot_shear_forces(m, scale=0.50e-3)
        # # ax.set_title('Shear forces')
        # plots.show(m)

    def test_frame_4_12(self):
        """
        Created on 01/12/2025

        Example 4.12 from Matrix Structural Analysis: Second Edition 2nd Edition by
        William McGuire, Richard H. Gallagher, Ronald D. Ziemian

        Not all section properties are provided. Assuming that the second moments of
        area are the same four y and z here.
        """
        E = 200000  # SI units with lengths in millimeters
        G = E / (2 * (1 + 0.3))

        m = model.create(3)

        model.add_joint(m, 1, [0.0, 0.0, 0.0])
        model.add_joint(m, 2, [8000.0, 0.0, 0.0])
        model.add_joint(m, 3, [13000.0, 0.0, 0.0])
        model.add_joint(m, 4, [8000.0, 0.0, 40])

        a = m["joints"][1]
        model.add_support(a, freedoms.U1)
        model.add_support(a, freedoms.U2)
        model.add_support(a, freedoms.U3)
        model.add_support(a, freedoms.UR1)
        model.add_support(a, freedoms.UR2)
        model.add_support(a, freedoms.UR3)
        c = m["joints"][3]
        model.add_support(c, freedoms.U1)
        model.add_support(c, freedoms.U2)
        model.add_support(c, freedoms.U3)
        model.add_support(c, freedoms.UR1)
        model.add_support(c, freedoms.UR2)
        model.add_support(c, freedoms.UR3)

        A = 6000
        Iy = 200e6
        Iz = Iy
        Ix = Iy + Iz
        J = 300e3
        xz_vector = [0, 0, 1]
        s1 = section.beam_3d_section(
            "sect_1", E=E, G=G, A=A, Ix=Ix, Iy=Iy, Iz=Iz, J=J, xz_vector=xz_vector
        )
        model.add_beam_member(m, 1, [1, 2], s1)

        A = 4000
        Iy = 50e6
        Iz = Iy
        Ix = Iy + Iz
        J = 100e3
        xz_vector = [0, 0, 1]
        s2 = section.beam_3d_section(
            "sect_2", E=E, G=G, A=A, Ix=Ix, Iy=Iy, Iz=Iz, J=J, xz_vector=xz_vector
        )
        model.add_beam_member(m, 2, [3, 2], s2)

        # artificially increased cross section properties for the short bracket
        A = 4000
        Iy = 5000e6
        Iz = Iy
        Ix = Iy + Iz
        J = 10000e3
        xz_vector = [0, 1, 0]
        s3 = section.beam_3d_section(
            "sect_3", E=E, G=G, A=A, Ix=Ix, Iy=Iy, Iz=Iz, J=J, xz_vector=xz_vector
        )
        model.add_beam_member(m, 3, [4, 2], s3)

        d = m["joints"][4]
        model.add_load(d, freedoms.U2, -1e3)

        # plots.plot_setup(m)
        # plots.plot_members(m)
        # plots.plot_beam_orientation(m, 1.0)
        # plots.plot_member_numbers(m)
        # plots.show(m)

        model.number_dofs(m)

        # print("Number of free degrees of freedom = ", m["nfreedof"])
        # print("Number of all degrees of freedom = ", m["ntotaldof"])

        # print([j["dof"] for j in m["joints"].values()])

        model.solve_statics(m)

        # print([j["displacements"] for j in m["joints"].values()])

        # print(abs(m["joints"][2]["displacements"][1] - -0.545) / (0.545))
        self.assertAlmostEqual(m["joints"][2]["displacements"][1], -0.545, places=2)
        self.assertAlmostEqual(m["joints"][2]["displacements"][5], -0.263e-4, places=2)

        for k in m["beam_members"].keys():
            member = m["beam_members"][k]
            connectivity = member["connectivity"]
            i, j = m["joints"][connectivity[0]], m["joints"][connectivity[1]]
            f = beam.beam_3d_end_forces(member, i, j)
            # print(f"Member {k}: ")
            # print(
            #     f"   Joint {connectivity[0]}: N={f['Ni']:.5}, Qy={f['Qyi']:.5}, Qz={f['Qzi']:.5}, T={f['Ti']:.5}, My={f['Myi']:.5}, Mz={f['Mzi']:.5}: "
            # )
            # print(
            #     f"   Joint {connectivity[1]}: N={f['Nj']:.5}, Qy={f['Qyj']:.5}, Qz={f['Qzj']:.5}, T={f['Tj']:.5}, My={f['Myj']:.5}, Mz={f['Mzj']:.5}: "
            # )

        model.statics_reactions(m)

        for jid in [1, 3]:
            j = m["joints"][jid]
            # print(f"Joint {jid}:")
            # print(
            #     f"   Rx={j['reactions'][0]:.5}, Ry={j['reactions'][1]:.5}, Rz={j['reactions'][2]:.5}, Mx={j['reactions'][3]:.5}, My={j['reactions'][4]:.5}, Mz={j['reactions'][5]:.5}: "
            # )
        self.assertAlmostEqual(m["joints"][1]["reactions"][5], 1783365, places=0)
        self.assertAlmostEqual(m["joints"][3]["reactions"][5], -1414997.80, places=0)
        self.assertAlmostEqual(m["joints"][1]["reactions"][3], -2.6087e04, places=0)
        self.assertAlmostEqual(m["joints"][3]["reactions"][3], -1.3913e04, places=0)

        # plots.plot_setup(m)
        # plots.plot_members(m)
        # plots.plot_deformations(m, 1000.0)
        # plots.plot_torsion_moments(m, scale=0.04)
        # # ax = plots.plot_shear_forces(m, scale=0.50e-3)
        # # ax.set_title("Shear forces")
        # plots.show(m)

    def test_11_samcef_frame_3d_vibration_tut(self):
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
            "sverti",
            E=E,
            rho=rho,
            G=G,
            A=A,
            Ix=Ix,
            Iy=Iy,
            Iz=Iz,
            J=J,
            xz_vector=xz_vector,
        )

        # Horizontal beams.
        A = 5.68e-3
        Iy = 1.2e-4
        Iz = 7.3e-6
        Ix = Iy + Iz
        J = 1.76e-7
        xz_vector = [0, 0, 1]
        shoriz = section.beam_3d_section(
            "shoriz",
            E=E,
            rho=rho,
            G=G,
            A=A,
            Ix=Ix,
            Iy=Iy,
            Iz=Iz,
            J=J,
            xz_vector=xz_vector,
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
            model.add_support(m["joints"][jid], freedoms.ALL_DOFS)

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
            self.assertAlmostEqual(
                abs(m["frequencies"][mode] / reffs[mode]), 1.0, places=0
            )
            # print(f"Mode {mode}: {m["frequencies"][mode]:.3f} Hz")
            # print(f"  Reference: ", reffs[mode])
            # ax = plots.plot_setup(m)
            # plots.plot_members(m)
            # model.set_solution(m, mode)
            # plots.plot_deformations(m, 100.0)
            # ax.set_title(f"Mode {mode}, frequency = {m['frequencies'][mode]:.2f} Hz")
            # plots.show(m)

    def test_13_hinged_3d_frame_tut(self):
        """
        pystran - Python package for structural analysis with trusses and beams

        (C) 2025, Petr Krysl, pkrysl@ucsd.edu

        # Spatial frame with a hinge and spring supports

        AFNOR SSLL04/89 test case.

        Spatial frame with spring supports and a spherical ball hinge.

        ICAB Force Exemples Exemples de calculs de statique pour ICAB Force www.icab.fr

        Displacements and internal forces are provided in the verification manual.
        """

        # We begin with the standard imports:

        # from numpy import array
        # from numpy.linalg import norm
        # from context import pystran
        # from pystran import model
        # from pystran import section
        # from pystran import plots
        # from pystran import beam
        # from pystran import rotation

        # Define a few constants:
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

        # The model is created as three dimensional.
        m = model.create(3)

        # Joints are added at their locations. The original source refers to the nodes
        # (joints) as N_A, N_B, etc. N_A and N_B are supported, N_H1 and N_H2 form the
        # two joints that are linked in a hinge.
        model.add_joint(m, 1, [0.0, 2.0, 1.0])  # N_A
        model.add_joint(m, 2, [2.0, 0.0, -1.0])  # N_B
        model.add_joint(m, 3, [0.0, 0.0, -1.0])  # N_C
        model.add_joint(m, 4, [0.0, 0.0, +1.0])  # N_D
        model.add_joint(m, 5, [0.0, 0.0, 0.0])  # N_H1
        model.add_joint(m, 6, [0.0, 0.0, 0.0])  # N_H2

        # There are four beams. The cross sectional properties are the same, but the
        # beam orientations need to be different.

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

        # With the above definitions of the sections at hand, we define the four beam
        # members.

        model.add_beam_member(m, 1, [1, 4], sect_2)
        model.add_beam_member(m, 2, [3, 2], sect_2)
        model.add_beam_member(m, 3, [4, 5], sect_1)
        model.add_beam_member(m, 4, [6, 3], sect_1)

        # Now we can plot the geometry of the structure. We show the members, the member
        # numbers, and the orientations of the local coordinate systems.

        # ax = plots.plot_setup(m)
        # plots.plot_members(m)
        # plots.plot_member_numbers(m)
        # plots.plot_joint_numbers(m)
        # plots.plot_beam_orientation(m, 0.20)
        # ax.set_title("Frame geometry")
        # plots.show(m)

        # Next we add the link between the two joints that form the hinge. The hinge is
        # ball joint, meaning all three translations are the same for joints at the
        # hinge.
        model.add_links(m, [6, 5], freedoms.U1)
        model.add_links(m, [6, 5], freedoms.U2)
        model.add_links(m, [6, 5], freedoms.U3)

        # The supports are next. The original source refers to the supports at N_A and
        # N_B.
        model.add_support(m["joints"][1], freedoms.U1)
        model.add_support(m["joints"][1], freedoms.U3)
        model.add_support(m["joints"][1], freedoms.UR2)

        model.add_support(m["joints"][2], freedoms.U2)
        model.add_support(m["joints"][2], freedoms.U3)
        model.add_support(m["joints"][2], freedoms.UR1)

        # Next we add the springs to the ground. The translation and rotation spring
        # constants have the same numerical value. First at joint 1.
        model.add_extension_spring_to_ground(m["joints"][1], 1, [0, 1, 0], K)
        model.add_moment_spring_to_ground(m["joints"][1], 1, [1, 0, 0], K)
        model.add_moment_spring_to_ground(m["joints"][1], 2, [0, 0, 1], K)
        # Then at joint 3.
        model.add_extension_spring_to_ground(m["joints"][2], 1, [1, 0, 0], K)
        model.add_moment_spring_to_ground(m["joints"][2], 1, [0, 1, 0], K)
        model.add_moment_spring_to_ground(m["joints"][2], 2, [0, 0, 1], K)

        # Let us look at the translation and rotation supports:
        # ax = plots.plot_setup(m)
        # plots.plot_members(m)
        # plots.plot_joint_numbers(m)
        # plots.plot_translation_supports(m)
        # ax.set_title("Translation supports")
        # plots.show(m)

        # ax = plots.plot_setup(m)
        # plots.plot_members(m)
        # plots.plot_joint_numbers(m)
        # plots.plot_rotation_supports(m)
        # ax.set_title("Rotation supports")
        # plots.show(m)

        # Next we add the forces and moments applied at the joint N_D.
        model.add_load(m["joints"][4], freedoms.U3, -F)

        # Now we can solve the static equilibrium of the frame.
        model.number_dofs(m)
        model.solve_statics(m)

        # The solution to the problem can be visualized with a number of plots. We start
        # with the deformed shape of the frame.
        # ax = plots.plot_setup(m)
        # plots.plot_members(m)
        # plots.plot_joint_numbers(m)
        # ax = plots.plot_deformations(m, 2.0)
        # ax.set_title("Deformed shape (magnified 2 times)")
        # plots.show(m)

        # The displacements of the joints can be compared to the reference values.
        # These are the displacements of joint 1:
        ref1 = array([0.0, -0.02976196, 0.0, 0.16072649, 0.0, -0.05951626])
        for i in range(6):
            if abs(m["joints"][1]["displacements"][i] - ref1[i]) > 0.001 * abs(ref1[i]):
                raise ValueError("Displacement calculation error")

        # These are the displacements of joint 3:
        ref3 = array([0.02977301, 0.13888914, -0.37002052])
        for i in range(3):
            if abs(m["joints"][3]["displacements"][i] - ref3[i]) > 0.001 * abs(ref3[i]):
                raise ValueError("Displacement calculation error")

        # print("Displacement calculation OK")

        # Moment diagrams can be produced for the torsional and bending moments in the
        # members.
        # ax = plots.plot_setup(m)
        # plots.plot_members(m)
        # plots.plot_joint_numbers(m)
        # plots.plot_beam_orientation(m, 0.2)
        # ax = plots.plot_torsion_moments(m, scale=0.0001)
        # ax.set_title("Torsion moments")
        # plots.show(m)

        # ax = plots.plot_setup(m)
        # plots.plot_members(m)
        # plots.plot_joint_numbers(m)
        # plots.plot_beam_orientation(m, 0.2)
        # ax = plots.plot_bending_moments(m, scale=0.0001, axis="y")
        # ax.set_title("Moments M_y")
        # plots.show(m)

        # ax = plots.plot_setup(m)
        # plots.plot_members(m)
        # plots.plot_joint_numbers(m)
        # plots.plot_beam_orientation(m, 0.2)
        # ax = plots.plot_bending_moments(m, scale=0.0001, axis="z")
        # ax.set_title("Moments M_z")
        # plots.show(m)

        # The internal "end forces", i.e., the forces and moments at the ends of the
        # members that act on the joints, are reported.
        # for k in m["beam_members"].keys():
        #     member = m["beam_members"][k]
        #     connectivity = member["connectivity"]
        #     i, j = m["joints"][connectivity[0]], m["joints"][connectivity[1]]
        #     f = beam.beam_3d_end_forces(member, i, j)
        #     print(f"Member {k}: ")
        #     print(
        #         f"   Joint {connectivity[0]}: N={f['Ni']:.5}, Qy={f['Qyi']:.5}, Qz={f['Qzi']:.5}, T={f['Ti']:.5}, My={f['Myi']:.5}, Mz={f['Mzi']:.5}: "
        #     )
        #     print(
        #         f"   Joint {connectivity[1]}: N={f['Nj']:.5}, Qy={f['Qyj']:.5}, Qz={f['Qzj']:.5}, T={f['Tj']:.5}, My={f['Myj']:.5}, Mz={f['Mzj']:.5}: "
        #     )

        # The reference provides values of moments at joint 1 (N_A) in member 1:
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


def main():
    unittest.main()


if __name__ == "__main__":
    main()

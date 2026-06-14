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


class UnitTestsPlanarTrusses(unittest.TestCase):

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
        freedoms = m["freedoms"]

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

    def test_truss_thermal(self):
        """
        STAAD.Pro test example
        """

        # from math import sqrt, pi, cos, sin
        # from numpy.linalg import norm
        #         # from pystran import model
        # from pystran import section
        # from pystran import truss
        # from pystran import geometry
        # 
        # from pystran import plots

        # US SI(m) units
        E = 2.0e5  # MPa
        CTE = 1.4e-5  # 1/degC
        A = 900  # mm^2
        DeltaT = {1: 20.0, 2: 70.0, 3: 20.0}  # temperatures of the bars

        m = model.create(2)
        freedoms = m["freedoms"]

        def add_thermal_loads(m):
            for member in m["truss_members"].values():
                sect = member["section"]
                E, A = sect["E"], sect["A"]
                connectivity = member["connectivity"]
                i, j = m["joints"][connectivity[0]], m["joints"][connectivity[1]]
                d = geometry.delt(i["coordinates"], j["coordinates"])
                nd = d / norm(d)
                N_T = E * A * CTE * DeltaT[member["mid"]]
                # print("Thermal force: ", N_T)
                model.add_load(i, freedoms.U1, -nd[0] * N_T)
                model.add_load(i, freedoms.U2, -nd[1] * N_T)
                model.add_load(j, freedoms.U1, +nd[0] * N_T)
                model.add_load(j, freedoms.U2, +nd[1] * N_T)

        model.add_joint(m, 1, [-2121.32, 2121.32])
        model.add_joint(m, 2, [0.0, 2121.32])
        model.add_joint(m, 3, [2121.32, 2121.32])
        model.add_joint(m, 4, [0.0, 0.0])

        model.add_support(m["joints"][1], freedoms.TRANSLATION_DOFS)
        model.add_support(m["joints"][2], freedoms.TRANSLATION_DOFS)
        model.add_support(m["joints"][3], freedoms.TRANSLATION_DOFS)

        s1 = section.truss_section("s1", E, A)

        model.add_truss_member(m, 1, [1, 4], s1)
        model.add_truss_member(m, 2, [2, 4], s1)
        model.add_truss_member(m, 3, [3, 4], s1)

        # ax = plots.setup(m)
        # plots.plot_joint_ids(m)
        # plots.plot_members(m)
        # plots.show(m)

        add_thermal_loads(m)

        # plots.setup(m)
        # plots.plot_members(m)
        # ax = plots.plot_applied_forces(m, 0.001)
        # ax.set_title("Forces")
        # plots.show(m)

        model.number_dofs(m)
        model.solve_statics(m)

        # for j in m["joints"].values():
        #     print(j["jid"], ": ", j["displacements"])

        for member, res in zip(
            m["truss_members"].values(), [22142.727, -31314.545, 22142.727]
        ):
            connectivity = member["connectivity"]
            i, j = m["joints"][connectivity[0]], m["joints"][connectivity[1]]
            N = truss.truss_axial_force(member, i, j, 0.0)
            # print("N = ", N)
            N_T = E * A * CTE * DeltaT[member["mid"]]
            # print("N - N_T = ", N - N_T)
            if abs(N - N_T - res) > 1.0e-3 * abs(res):
                raise ValueError("Force calculation error")

        # plots.setup(m)
        # plots.plot_members(m)
        # ax = plots.plot_deformations(m, 50.0)
        # ax.set_title("Deformed shape (magnification factor = 50)")
        # plots.show(m)

        # plots.setup(m)
        # plots.plot_members(m)
        # ax = plots.plot_axial_forces(m, 0.001)
        # ax.set_title("Deformed shape (magnification factor = 50)")
        # plots.show(m)

    def test_14_truss_skew_support_tut(self):
        """
        pystran - Python package for structural analysis with trusses and beams

        (C) 2025-2026, Petr Krysl, pkrysl@ucsd.edu

        # Truss with a combination of loads

        Original source: "Guide de validation des progiciels de calcul de structures"
        publié par l'AFNOR 1990 (ISBN 2-12-486611-7).

        Data taken from: ICAB Force Exemples Exemples de calculs de statique pour ICAB
        Force. www.icab.fr

        Problem: A truss with a combination of loads. The truss is supported at three
        joints, and one of those supports is inclined. This tutorial demonstrates how to
        use a combination of loads.
        """

        # from math import sqrt, pi, cos, sin
        # from numpy.linalg import norm
        #         # from pystran import model
        # from pystran import section
        # from pystran import geometry
        # 
        # from pystran import plots

        # US SI(m) units
        E = 2.1e11  # Pa
        CTE = 1.0e-5  # 1/degC
        A1 = 1.41e-3  # m^2
        A2 = 2 * 1.41e-3  # m^2
        # The rigid support is a bar of unit length, and artificially increased the
        # cross sectional area.
        Ar = 1.0e5  # m^2
        cth = cos(60 / 180 * pi)
        sth = sin(60 / 180 * pi)
        # Increase of temperature above reference. All bars are affected, except the one
        # used as a rigid support.
        DeltaT = 150.0

        # Define the truss sections for two groups of bars.
        s1 = section.truss_section("s1", E=E, A=A1, CTE=CTE)
        s2 = section.truss_section("s2", E=E, A=A2, CTE=CTE)
        # Define the section of the bar used as a rigid support. Note that it does not
        # thermally expand since its coefficient of thermal expansion is set to zero.
        sr = section.truss_section("sr", E=E, A=Ar, CTE=0.0)

        # Create the model by defining joints and truss members.
        m = model.create(2)
        freedoms = m["freedoms"]

        # A helper function to set up the thermal loads.
        def add_thermal_loads(m):
            """Set up thermal loads."""
            for member in m["truss_members"].values():
                if member["mid"] != 20:
                    sect = member["section"]
                    EA = sect["E"] * sect["A"]
                    _CTE = sect["CTE"]
                    connectivity = member["connectivity"]
                    _i, _j = m["joints"][connectivity[0]], m["joints"][connectivity[1]]
                    d = geometry.delt(_i["coordinates"], _j["coordinates"])
                    nd = d / norm(d)
                    N_T = _CTE * DeltaT * EA
                    model.add_load(_i, freedoms.U1, -nd[0] * N_T)
                    model.add_load(_i, freedoms.U2, -nd[1] * N_T)
                    model.add_load(_j, freedoms.U1, +nd[0] * N_T)
                    model.add_load(_j, freedoms.U2, +nd[1] * N_T)

        model.add_joint(m, 1, [0.0, 0.0])  # A
        model.add_joint(m, 2, [0.0, 4.0])
        model.add_joint(m, 3, [5.0, 0.0])
        model.add_joint(m, 4, [2 * 5.0, 4.0])
        model.add_joint(m, 5, [3 * 5.0, 0.0])
        model.add_joint(m, 6, [4 * 5.0, 4.0])
        model.add_joint(m, 7, [5 * 5.0, 0.0])  # D - monitored joint
        model.add_joint(m, 8, [6 * 5.0, 4.0])
        model.add_joint(m, 9, [7 * 5.0, 0.0])  # C
        model.add_joint(m, 10, [4 * 5.0, -4.0])  # B
        # The direction of the inclined support is defined by the angle 30 degrees of
        # the plane on which the joint can slide. C' is the joint that is pinned, and C
        # is the joint whose motion is controlled.
        model.add_joint(m, 11, [7 * 5.0 + 1.0 * cth, 0.0 - 1.0 * sth])  # C'

        # Group 1.
        model.add_truss_member(m, 1, [1, 2], s1)
        model.add_truss_member(m, 2, [2, 3], s1)
        model.add_truss_member(m, 3, [3, 4], s1)
        model.add_truss_member(m, 4, [4, 5], s1)
        model.add_truss_member(m, 5, [5, 6], s1)
        model.add_truss_member(m, 6, [6, 7], s1)
        model.add_truss_member(m, 7, [7, 8], s1)
        model.add_truss_member(m, 8, [8, 9], s1)

        # Group 2.
        model.add_truss_member(m, 11, [2, 4], s2)
        model.add_truss_member(m, 12, [4, 6], s2)
        model.add_truss_member(m, 13, [6, 8], s2)
        model.add_truss_member(m, 14, [1, 3], s2)
        model.add_truss_member(m, 15, [3, 5], s2)
        model.add_truss_member(m, 16, [5, 10], s2)
        model.add_truss_member(m, 17, [10, 7], s2)
        model.add_truss_member(m, 18, [7, 9], s2)
        model.add_truss_member(m, 19, [10, 6], s2)

        model.add_truss_member(m, 20, [9, 11], sr)

        # Inspect be base structure visually.
        # ax = plots.setup(m)
        # plots.plot_joint_ids(m)
        # plots.plot_members(m)
        # ax.set_title("Truss definition")
        # plots.show(m)

        # Add the supports and loads. We will do that in three load cases. The quantity
        # we are interested in is displacement of the joint 7 (D) in the y-direction.

        # Load case 1: concentrated forces, homogeneous (all components set to zero)
        # supports.

        # We begin by clearing all loads and supports.
        model.remove_loads(m)
        model.remove_supports(m)

        # Then we add concentrated forces.
        model.add_load(m["joints"][4], freedoms.U2, -150.0e3)
        model.add_load(m["joints"][8], freedoms.U2, -100.0e3)

        # And we add supports.
        model.add_support(m["joints"][1], freedoms.U1)
        model.add_support(m["joints"][1], freedoms.U2)
        model.add_support(m["joints"][10], freedoms.U2)
        model.add_support(m["joints"][11], freedoms.U1)
        model.add_support(m["joints"][11], freedoms.U2)

        # The solution for the load case is obtained.
        model.number_dofs(m)
        model.solve_statics(m)

        # Store the displacement of the joint 7 (D) in the y-direction.
        j = m["joints"][7]
        UD1 = j["displacements"][freedoms.U2]
        # print("Case 1: joint 7 (D) vertical displacement", ": ", j["displacements"])

        # Load case 2: support settlement. In this case only the supports are changed to prescribed non zero displacements.

        model.remove_loads(m)
        model.remove_supports(m)

        model.add_support(m["joints"][1], freedoms.U1)
        model.add_support(m["joints"][1], freedoms.U2, -0.02)
        model.add_support(m["joints"][10], freedoms.U2, -0.03)
        model.add_support(m["joints"][11], freedoms.U1, +0.015 * cth)
        model.add_support(m["joints"][11], freedoms.U2, -0.015 * sth)

        model.number_dofs(m)
        model.solve_statics(m)

        j = m["joints"][7]
        UD2 = j["displacements"][freedoms.U2]
        # print("Case 2: joint 7 (D) vertical displacement", ": ", j["displacements"])

        # Load case 3: thermal loads, homogeneous supports.

        # All loads and supports are cleared.
        model.remove_loads(m)
        model.remove_supports(m)

        # Homogeneous displacement supports are added.
        model.add_support(m["joints"][1], freedoms.U1)
        model.add_support(m["joints"][1], freedoms.U2)
        model.add_support(m["joints"][10], freedoms.U2)
        model.add_support(m["joints"][11], freedoms.U1)
        model.add_support(m["joints"][11], freedoms.U2)

        # Thermal loads are added.
        add_thermal_loads(m)

        model.number_dofs(m)
        model.solve_statics(m)

        j = m["joints"][7]
        UD3 = j["displacements"][freedoms.U2]
        # print("Case 3: joint 7 (D) vertical displacement", ": ", j["displacements"])

        # print("Displacement of D for the load combination: ", UD1 + UD2 + UD3)
        # print("Displacement of D reference: ", -0.01618)

        if abs(UD1 + UD2 + UD3 + 0.01618) > 1.0e-4:
            raise ValueError("Displacement of D incorrect.")

        # Now we intend to check that the load combination will lead to the same result
        # as all loadings applied at once.

        # Load cases combined into one: all loadings are applied in a single load case.

        model.remove_loads(m)
        model.remove_supports(m)

        # Non zero displacements.
        model.add_support(m["joints"][1], freedoms.U1)
        model.add_support(m["joints"][1], freedoms.U2, -0.02)
        model.add_support(m["joints"][10], freedoms.U2, -0.03)
        model.add_support(m["joints"][11], freedoms.U1, +0.015 * cth)
        model.add_support(m["joints"][11], freedoms.U2, -0.015 * sth)

        # Concentrated forces.
        model.add_load(m["joints"][4], freedoms.U2, -150.0e3)
        model.add_load(m["joints"][8], freedoms.U2, -100.0e3)

        # Thermal loads.
        add_thermal_loads(m)

        model.number_dofs(m)
        model.solve_statics(m)

        j = m["joints"][7]
        UD = j["displacements"][freedoms.U2]
        # print("Combined loading: joint 7 (D) vertical displacement", ": ", j["displacements"])

        # Check that the load combination will gave the same displacement as all the
        # loadings combined.

        if abs(UD - (UD1 + UD2 + UD3)) > 1.0e-9:
            raise ValueError("Displacement of D incorrect.")

        # Finally, display the deformed truss.
        # ax = plots.setup(m)
        # plots.plot_members(m)
        # plots.plot_joint_ids(m)
        # ax = plots.plot_deformations(m, 20.0)
        # ax.set_title("Deformed shape (magnified 20 times)")
        # plots.show(m)


def main():
    unittest.main()


if __name__ == "__main__":
    main()

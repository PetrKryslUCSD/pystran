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


class UnitTestsSpaceTruss(unittest.TestCase):

    def test_truss_dome(self):
        #         """
        # Analysis of Geometrically
        # Nonlinear Structures
        # Second Edition

        # by
        # Robert Levy
        # Technion-Israel Institute of Technology,
        # Haifa, Israel

        # and
        # William R. Spillers
        # New Jersey Institute of Technology,

        # Space truss dome in section 2.4.2

        # Vertical deflection at the crown: -.20641184e+00 in (linear analysis)
        # """

        # from numpy import array, dot, outer, concatenate
        # from numpy.linalg import norm
        #         # from pystran import model
        # from pystran import section
        # from pystran import plots

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

        # for b in m['truss_members'].values():
        #     connectivity = b['connectivity']
        #     i, j = m['joints'][connectivity[0]], m['joints'][connectivity[1]]
        #     e_x, L = truss.truss_member_geometry(i, j)
        #     B = truss.strain_displacement(e_x, L)
        #     u = concatenate((i['displacements'], j['displacements']))
        #     eps = dot(B, u)
        #     print('Bar ' + str(connectivity) + ' force = ', E * A * eps[0])

        # plots.setup(m)
        # plots.plot_members(m)
        # plots.plot_deformations(m, 5.0)
        # plots.show(m)


def main():
    unittest.main()


if __name__ == "__main__":
    main()

"""
Define mechanical quantities of general springs.
"""

from numpy import array, outer, concatenate
from pystran import assemble
10
def _spring_2d_stiffness(kind, direction, stiffness_coefficient):
    if kind == "torsion":
        return array([stiffness_coefficient])
    else:
        k1 = stiffness_coefficient * outer(direction, direction)
        return concatenate(
            [concatenate([k1, -k1], axis=1), concatenate([-k1, k1], axis=1)], axis=0
        )


def _spring_3d_stiffness(direction, stiffness_coefficient):
    k1 = stiffness_coefficient * outer(direction, direction)
    return concatenate(
        [concatenate([k1, -k1], axis=1), concatenate([-k1, k1], axis=1)], axis=0
    )


def assemble_stiffness(Kg, member, i, j):
    """
    Assemble the stiffness matrix of a general spring.

    The details of the calculation depend on whether the spring is in two
    dimensions or in three dimensions.

    Parameters
    ----------
    Kg
        Global structural stiffness matrix.
    member
        Dictionary that defines the data of the member.
    i
        Dictionary that defines the data of the first joint of the member.
    j
        Dictionary that defines the data of the second joint of the member.

    Returns
    -------
    array
        Updated global matrix is returned.

    See Also
    --------
    :func:`pystran.assemble.assemble`
    """
    sect = member["section"]
    stiffness_coefficient = sect["stiffness_coefficient"]
    direction = sect["direction"]
    dim = len(j["coordinates"])
    if dim == 2:
        k = _spring_2d_stiffness(sect['kind'], direction, stiffness_coefficient)
    else:
        k = _spring_3d_stiffness(direction, stiffness_coefficient)
    dof = [i["dof"][d] for d in member["dofkind"]] + [j["dof"][d] for d in member["dofkind"]]
    Kg = assemble.assemble(Kg, dof, k)

"""
Define truss mechanical quantities.
"""

from numpy import reshape, outer, concatenate, zeros, dot, array
from pystran import geometry
from pystran import assemble
from pystran import gauss


def truss_stiffness(e_x, h, E, A):
    r"""
    Compute truss stiffness matrix.

    The axial stiffness matrix is computed as :math:`K = EA B^T B h`. Here
    :math:`B` is the stretch-displacement matrix, computed by
    :func:`truss_strain_displacement`.

    Parameters
    ----------
    e_x
        Unit vector of the local cartesian coordinate system in the direction
        of the axis of the beam.
    h
        Length of the beam.
    E
        Young's modulus of the material.
    A
        Area of the cross section.

    Returns
    -------
    array
        Member stiffness matrix.

    See Also
    --------
    :func:`truss_strain_displacement`
    """
    B = truss_strain_displacement(e_x, h)
    return E * A * outer(B.T, B) * h


def truss_2d_mass(e_x, e_z, h, rho, A):
    r"""
    Compute 2d truss mass matrix.

    The mass matrix is consistent, which means that it is computed from the discrete
    form of the kinetic energy of the element,

    .. math::
        \int \rho A \left(\dot u \cdot \dot u +  \dot w \cdot \dot w\right)dx

    where :math:`\dot u` and :math:`\dot w` are the velocities in the :math:`x` and
    :math:`z` directions.

    Parameters
    ----------
    e_x
        Unit vector of the local cartesian coordinate system in the direction
        of the axis of the beam.
    e_z
        Unit vector of the local cartesian coordinate system (orthogonal to the
        axis of the beam).
    h
        Length of the beam.
    rho
        Mass density of the material.
    A
        Area of the cross section.

    Returns
    -------
    array
        Member mass matrix.
    """
    xiG, WG = gauss.rule(2)
    WG = [1, 1]
    n = len(e_x) * 2
    m = zeros((n, n))
    for q in range(2):
        N = geometry.lin_basis(xiG[q])
        Nu = array([N[0], 0.0, N[1], 0.0])
        m += rho * A * outer(Nu, Nu) * WG[q] * (h / 2)
        Nv = array([0.0, N[0], 0.0, N[1]])
        m += rho * A * outer(Nv, Nv) * WG[q] * (h / 2)
    return m


def truss_3d_mass(e_x, e_y, e_z, h, rho, A):
    r"""
    Compute 3d truss mass matrix.

    The mass matrix is consistent, which means that it is computed from the discrete
    form of the kinetic energy of the element,

    .. math::
        \int \rho A \left(\dot u \cdot \dot u + \dot v \cdot \dot v + \dot w \cdot \dot w\right)dx

    where :math:`\dot u`, :math:`\dot v`, and :math:`\dot w` are the velocities
    in the :math:`x`, :math:`y`, and :math:`z` directions.

    Parameters
    ----------
    e_x
        Unit vector of the local cartesian coordinate system in the direction
        of the axis of the beam.
    e_y
        Unit vector of the local cartesian coordinate system (orthogonal to the
        axis of the beam).
    e_z
        Unit vector of the local cartesian coordinate system (orthogonal to the
        axis of the beam).
    h
        Length of the beam.
    rho
        Mass density of the material.
    A
        Area of the cross section.

    Returns
    -------
    array
        Member mass matrix.
    """
    xiG, WG = gauss.rule(2)
    n = len(e_x) * 2
    m = zeros((n, n))
    for q in range(2):
        N = geometry.lin_basis(xiG[q])
        Nu = array([N[0], 0.0, 0.0, N[1], 0.0, 0.0])
        m += rho * A * outer(Nu, Nu) * WG[q] * (h / 2)
        Nv = array([0.0, N[0], 0.0, 0.0, N[1], 0.0])
        m += rho * A * outer(Nv, Nv) * WG[q] * (h / 2)
        Nw = array([0.0, 0.0, N[0], 0.0, 0.0, N[1]])
        m += rho * A * outer(Nw, Nw) * WG[q] * (h / 2)
    return m


def truss_strain_displacement(e_x, h):
    r"""
    Compute truss strain-displacement matrix.

    The axial strain is computed as :math:`\varepsilon = B u`, using the strain
    displacement matrix :math:`B` and the displacement vector :math:`u`.

    The dimension of the strain-displacement matrix depends on the number of
    space dimensions. The vector :math:`e_x` is the unit vector along the truss
    member, and it could have two or three components.

    Parameters
    ----------
    e_x
        Unit vector along the axis of the member.
    h
        Length of the member.

    Returns
    -------
    array
        Strain-displacement matrix.
    """
    return reshape(concatenate((-e_x / h, e_x / h)), (1, 2 * len(e_x)))


def assemble_stiffness(Kg, member, i, j):
    """
    Assemble truss stiffness matrix.

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
    E, A = sect["E"], sect["A"]
    if E <= 0.0:
        raise ValueError("Elastic modulus must be positive")
    if A <= 0.0:
        raise ValueError("Area must be positive")
    dim = len(i["coordinates"])
    if dim == 2:
        e_x, _, h = geometry.member_2d_geometry(i, j)
    else:
        e_x, _, _, h = geometry.member_3d_geometry(i, j, array([]))
    k = truss_stiffness(e_x, h, E, A)
    dof = concatenate([i["dof"][0:dim], j["dof"][0:dim]])
    return assemble.assemble(Kg, dof, k)


def assemble_mass(Mg, member, i, j):
    """
    Assemble truss mass matrix.

    Parameters
    ----------
    Mg
        Global structural mass matrix.
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
    rho, A = sect["rho"], sect["A"]
    if rho <= 0.0:
        raise ValueError("Mass density must be positive")
    if A <= 0.0:
        raise ValueError("Area must be positive")
    dim = len(i["coordinates"])
    if dim == 2:
        e_x, e_z, h = geometry.member_2d_geometry(i, j)
        m = truss_2d_mass(e_x, e_z, h, rho, A)
    else:
        e_x, e_y, e_z, h = geometry.member_3d_geometry(i, j, array([]))
        m = truss_3d_mass(e_x, e_y, e_z, h, rho, A)
    dim = len(e_x)
    dof = concatenate([i["dof"][0:dim], j["dof"][0:dim]])
    return assemble.assemble(Mg, dof, m)


def truss_axial_force(member, i, j, xi):
    r"""
    Compute truss axial force based on the displacements stored at the joints.

    The force is computed as :math:`N = EA B U`, where :math:`B` is the
    strain-displacement matrix (computed by :func:`truss_strain_displacement`),
    :math:`U` is the displacement vector (so that :math:`\varepsilon  = BU` is
    the axial strain), and :math:`EA` is the axial stiffness.

    Parameters
    ----------
    member
        Dictionary that defines the data of the member.
    i
        Dictionary that defines the data of the first joint of the member.
    j
        Dictionary that defines the data of the second joint of the member.
    xi
        Location along the bar in terms of the parameter coordinate. Unused, as
        the force along the bar is constant.

    Returns
    -------
    float
        Axial force is returned.
    """
    sect = member["section"]
    E, A = sect["E"], sect["A"]
    dim = len(i["coordinates"])
    if dim == 2:
        e_x, _, h = geometry.member_2d_geometry(i, j)
        ui, uj = i["displacements"][0:2], j["displacements"][0:2]
    else:
        e_x, _, _, h = geometry.member_3d_geometry(i, j, None)
        ui, uj = i["displacements"][0:3], j["displacements"][0:3]
    u = concatenate([ui, uj])
    B = truss_strain_displacement(e_x, h)
    N = (E * A * dot(B, u))[0]
    return N


def truss_volume(member, i, j):
    r"""
    Compute truss volume.

    Parameters
    ----------
    member
        Dictionary that defines the data of the member.
    i
        Dictionary that defines the data of the first joint of the member.
    j
        Dictionary that defines the data of the second joint of the member.
    
    Returns
    -------
    float
        The volume.
    """
    truss_is_2d = len(i["coordinates"]) == len(j["coordinates"]) == 2
    A = member["section"]["A"]
    if truss_is_2d:
        _, _, h = geometry.member_2d_geometry(i, j)
        return A * h
    else:
        _, _, _, h = geometry.member_3d_geometry(i, j, None)
        return A * h

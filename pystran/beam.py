"""
Define beam mechanical quantities.
"""

from numpy import dot, outer, concatenate, zeros
from pystran import geometry
from pystran.geometry import herm_basis_xi2, herm_basis_xi3, herm_basis
from pystran import gauss
from pystran import assemble
from pystran import truss


def beam_3d_xz_shape_fun(xi):
    r"""
    Compute the beam shape functions for deflection in the :math:`x-z` plane.

    Parameters
    ----------
    xi
        Parametric coordinate (:math:`-1\le\xi\le+1`): first joint is at
        :math:`\xi =-1`, second is at :math:`\xi =+1`.

    Returns
    -------
    array
        An array of shape function values is returned (i.e. :math:`[N_1(\xi),
        ..., N_4(\xi)]`).

    See Also
    --------
    :func:`pystran.geometry.herm_basis`
    """
    return herm_basis(xi)


def beam_3d_xz_shape_fun_xi2(xi):
    r"""
    Compute the second derivative of the beam shape functions for deflection in
    the :math:`x-z` plane.

    Parameters
    ----------
    xi
        Parametric coordinate (:math:`-1\le\xi\le+1`): first joint is at
        :math:`\xi =-1`, second is at :math:`\xi =+1`.

    Returns
    -------
    array
        An array of second derivatives of shape functions is returned (i.e.
        :math:`[d^2N_1(\xi)/d\xi^2, ..., d^2N_4(\xi)/d\xi^2]`).

    See Also
    --------
    :func:`pystran.geometry.herm_basis`
    """
    return herm_basis_xi2(xi)


def beam_3d_xy_shape_fun(xi):
    r"""
    Compute the beam shape functions for deflection in the :math:`x-y` plane.

    Parameters
    ----------
    xi
        Parametric coordinate (:math:`-1\le\xi\le+1`): first joint is at
        :math:`\xi =-1`, second is at :math:`\xi =+1`.

    Returns
    -------
    array
        The signs of the shape functions that go with the rotations (i.e. the
        second and fourth) need to be reversed relative to the :math:`x-z`
        plane of bending: An array of second derivatives of shape functions is
        returned (i.e. :math:`[N_1(\xi), -N_2(\xi), N_3(\xi), -N_4(\xi)]`).

    See Also
    --------
    :func:`pystran.geometry.herm_basis`
    """
    N = herm_basis(xi)
    N[1] *= -1.0
    N[3] *= -1.0
    return N


def beam_3d_xy_shape_fun_xi2(xi):
    r"""
    Compute the second derivative of the beam shape functions for deflection in
    the :math:`x-y` plane.

    Parameters
    ----------
    xi
        Parametric coordinate (:math:`-1\le\xi\le+1`): first joint is at
        :math:`\xi =-1`, second is at :math:`\xi =+1`.

    Returns
    -------
    array
        The signs of the shape functions that go with the rotations (i.e. the
        second and fourth) need to be reversed: An array of second derivatives
        of shape functions is returned (i.e. :math:`[d^2N_1(\xi)/d\xi^2,
        -d^2N_2(\xi)/d\xi^2, d^2N_3(\xi)/d\xi^2, -d^2N_4(\xi)/d\xi^2]`).

    See Also
    --------
    :func:`pystran.geometry.herm_basis`
    """
    d2Ndxi2 = herm_basis_xi2(xi)
    d2Ndxi2[1] *= -1.0
    d2Ndxi2[3] *= -1.0
    return d2Ndxi2


def beam_3d_xz_shape_fun_xi3(xi):
    r"""
    Compute the third derivative of the beam shape functions with respect to
    ``xi`` for deflection in the :math:`x-z` plane.

    Parameters
    ----------
    xi
        Parametric coordinate (:math:`-1\le\xi\le+1`): first joint is at
        :math:`\xi =-1`, second is at :math:`\xi =+1`.

    Returns
    -------
    array
        An array of third derivatives of shape functions is returned (i.e.
        :math:`[d^3N_1(\xi)/d\xi^3, ..., d^3N_4(\xi)/d\xi^3]`).

    See Also
    --------
    :func:`pystran.geometry.herm_basis`
    """
    return herm_basis_xi3(xi)


def beam_3d_xy_shape_fun_xi3(xi):
    r"""
    Compute the third derivative of the beam shape functions with respect to
    ``xi`` for deflection in the :math:`x-y` plane.

    Parameters
    ----------
    xi
        Parametric coordinate (:math:`-1\le\xi\le+1`): first joint is at
        :math:`\xi =-1`, second is at :math:`\xi =+1`.

    Returns
    -------
    array
        The signs of the shape functions that go with the rotations (i.e. the
        second and fourth) need to be reversed: An array of third derivatives
        of shape functions is returned (i.e. :math:`[d^3N_1(\xi)/d\xi^3,
        -d^3N_2(\xi)/d\xi^3, d^3N_3(\xi)/d\xi^3, -d^3N_4(\xi)/d\xi^3]`).

    See Also
    --------
    :func:`pystran.geometry.herm_basis`
    """
    d3Ndxi3 = herm_basis_xi3(xi)
    d3Ndxi3[1] *= -1.0
    d3Ndxi3[3] *= -1.0
    return d3Ndxi3


def beam_2d_bending_stiffness(e_z, h, E, I):
    r"""
    Compute 2d beam stiffness matrix.

    The formula reads

    .. math::

        K = (h/2) \int_{-1}^{+1} EI B^T B  d\xi,

    where :math:`B` is the curvature-displacement matrix (computed by
    :func:`beam_2d_curv_displ_matrix`), and :math:`h/2` is the Jacobian. :math:`I` is the
    second moment of area about the :math:`y` axis (which is orthogonal to the plane
    of bending).

    Two-point Gauss quadrature is used to compute the stiffness matrix.

    Parameters
    ----------
    e_z
        Unit vector of the local cartesian coordinate system (orthogonal to the
        axis of the beam).
    h
        Length of the beam.
    E
        Young's modulus of elasticity.
    I
        Second moment of area for bending about the :math:`y`-axis (i.e. bending in the
        :math:`x-z` plane).

    Returns
    -------
    array
        Stiffness matrix of the beam.


    See Also
    --------
    :func:`beam_2d_curv_displ_matrix`
    """
    xiG, WG = gauss.rule(2)
    K = zeros((6, 6))
    for xi, W in zip(xiG, WG):
        B = beam_2d_curv_displ_matrix(e_z, h, xi)
        K += E * I * outer(B.T, B) * W * (h / 2)
    return K


def beam_3d_xz_curv_displ_matrix(e_y, e_z, h, xi):
    r"""
    Compute beam curvature-displacement matrix in the local :math:`x-z` plane
    (i.e. bending about the :math:`y` axis).

    The curvature :math:`d^2w/dx^2` is computed in the local coordinate system
    of the beam as :math:`d^2w/dx^2 = B U`. Here :math:`B` is the
    curvature-displacement matrix and :math:`U` is the displacement vector. All
    three displacement components and all three rotation components at each
    joint are assumed, so the matrix :math:`B` has one row and twelve columns.

    Parameters
    ----------
    e_y
        Unit vector of the local cartesian coordinate system (orthogonal to the
        axis of the beam).
    e_z
        Unit vector of the local cartesian coordinate system (orthogonal to the
        axis of the beam).
    h
        Length of the beam.
    xi
        Parametric coordinate (:math:`-1\le\xi\le+1`): first joint is at
        :math:`\xi =-1`, second is at :math:`\xi =+1`.

    Returns
    -------
    array
        The curvature-displacement matrix.
    """
    d2Ndxi2 = beam_3d_xz_shape_fun_xi2(xi)
    B = zeros((1, 12))
    B[0, 0:3] = d2Ndxi2[0] * (2 / h) ** 2 * e_z
    B[0, 3:6] = (h / 2) * d2Ndxi2[1] * (2 / h) ** 2 * e_y
    B[0, 6:9] = d2Ndxi2[2] * (2 / h) ** 2 * e_z
    B[0, 9:12] = (h / 2) * d2Ndxi2[3] * (2 / h) ** 2 * e_y
    return B


def beam_3d_xy_curv_displ_matrix(e_y, e_z, h, xi):
    r"""
    Compute beam curvature-displacement matrix in the local :math:`x-y` plane
    (i.e. bending about the :math:`z` axis).

    The curvature :math:`d^2v/dx^2` is computed in the local coordinate system
    of the beam as :math:`d^2v/dx^2 = B U`. Here :math:`B` is the
    curvature-displacement matrix and :math:`U` is the displacement vector. All
    three displacement components and all three rotation components at each
    joint are assumed, so the matrix :math:`B` has one row and twelve columns.

    Parameters
    ----------
    e_y
        Unit vector of the local cartesian coordinate system (orthogonal to the
        axis of the beam).
    e_z
        Unit vector of the local cartesian coordinate system (orthogonal to the
        axis of the beam).
    h
        Length of the beam.
    xi
        Parametric coordinate (:math:`-1\le\xi\le+1`): first joint is at
        :math:`\xi =-1`, second is at :math:`\xi =+1`.

    Returns
    -------
    array
        The curvature-displacement matrix.
    """
    d2Ndxi2 = beam_3d_xy_shape_fun_xi2(xi)
    B = zeros((1, 12))
    B[0, 0:3] = d2Ndxi2[0] * (2 / h) ** 2 * e_y
    B[0, 3:6] = (h / 2) * d2Ndxi2[1] * (2 / h) ** 2 * e_z
    B[0, 6:9] = d2Ndxi2[2] * (2 / h) ** 2 * e_y
    B[0, 9:12] = (h / 2) * d2Ndxi2[3] * (2 / h) ** 2 * e_z
    return B


def beam_3d_bending_stiffness(e_y, e_z, h, E, Iy, Iz):
    r"""
    Compute 3d beam stiffness matrices for bending in the planes :math:`x-y`
    and :math:`x-z`.

    The formula reads

    .. math::

        K = (h/2) \int_{-1}^{+1} EI_y B_{xz}^T B_{xz}  d\xi,

    for bending in the :math:`x-z` plane, and

    .. math::

        K = (h/2) \int_{-1}^{+1} EI_z B_{xy}^T B_{xy}  d\xi,

    for bending in the :math:`x-y` plane. Here :math:`B_{xz}` is the
    curvature-displacement matrix for bending in the :math:`x-z` plane
    (computed by :func:`beam_3d_xz_curv_displ_matrix`), :math:`B_{xy}` is the
    curvature-displacement matrix for bending in the :math:`x-y` plane
    (computed by :func:`beam_3d_xy_curv_displ_matrix`), and :math:`h/2` is the
    Jacobian. :math:`I_y` is the second moment of area about the :math:`y`
    axis, and :math:`I_z`  is the second moment of area about the :math:`z`
    axis. The overall matrix is the sum of these two contributions.

    Appropriate Gauss quadrature formulas are used to compute the stiffness matrix.

    Parameters
    ----------
    e_y
        Unit vector of the local cartesian coordinate system (orthogonal to the
        axis of the beam).
    e_z
        Unit vector of the local cartesian coordinate system (orthogonal to the
        axis of the beam).
    h
        Length of the beam.
    E
        Young's modulus of elasticity.
    Iy
        Second moment of area for bending about the :math:`y`-axis (i.e. bending in the
        :math:`x-z` plane).
    Iz
        Second moment of area for bending about the :math:`z`-axis (i.e. bending in the
        :math:`x-y` plane).

    Returns
    -------
    array
        Stiffness matrix of the beam.

    See Also
    --------
    :func:`beam_3d_xy_curv_displ_matrix`
    :func:`beam_3d_xz_curv_displ_matrix`
    """
    xiG, WG = gauss.rule(2)
    Kxy = zeros((12, 12))
    for xi, W in zip(xiG, WG):
        B = beam_3d_xy_curv_displ_matrix(e_y, e_z, h, xi)
        Kxy += E * Iz * outer(B.T, B) * W * (h / 2)
    Kxz = zeros((12, 12))
    for xi, W in zip(xiG, WG):
        B = beam_3d_xz_curv_displ_matrix(e_y, e_z, h, xi)
        Kxz += E * Iy * outer(B.T, B) * W * (h / 2)
    return Kxy, Kxz


def beam_2d_curv_displ_matrix(e_z, h, xi):
    r"""
    Compute 2d beam curvature-displacement matrix.

    Here the curvatures is with respect to the physical coordinate measured
    along the member (local :math:`x`).

    The curvature :math:`d^2w/dx^2` is computed in the local coordinate system
    of the beam as :math:`d^2w/dx^2 = B U`. Here :math:`B` is the
    curvature-displacement matrix and :math:`U` is the displacement vector. Two
    displacement components and one rotation component at each joint are
    assumed, so the matrix :math:`B` has one row and six columns.

    Parameters
    ----------
    e_z
        Unit vector of the local cartesian coordinate system (orthogonal to the
        axis of the beam).
    h
        Length of the beam.
    xi
        Parametric coordinate (:math:`-1\le\xi\le+1`): first joint is at
        :math:`\xi =-1`, second is at :math:`\xi =+1`.

    Returns
    -------
    array
        The curvature-displacement matrix.
    """
    d2Ndxi2 = herm_basis_xi2(xi)
    B = zeros((1, 6))
    B[0, 0:2] = d2Ndxi2[0] * (2 / h) ** 2 * e_z
    B[0, 2] = (h / 2) * d2Ndxi2[1] * (2 / h) ** 2
    B[0, 3:5] = d2Ndxi2[2] * (2 / h) ** 2 * e_z
    B[0, 5] = (h / 2) * d2Ndxi2[3] * (2 / h) ** 2
    return B


def beam_2d_3rd_deriv_displ_matrix(e_z, h):
    r"""
    Compute beam third derivative-displacement matrix.

    Here the third derivative is with respect to the physical coordinate
    measured along the member (local :math:`x`).

    The third derivative :math:`d^3w/dx^3` is computed in the local coordinate
    system of the beam as :math:`d^3w/dx^3 = B U`. Here :math:`B` is the
    third-derivative-displacement matrix and :math:`U` is the displacement
    vector. Two displacement components and one rotation component at each
    joint are assumed, so the matrix :math:`B` has one row and six columns.
    Note that this matrix is constant (independent of the location along the
    beam).

    Parameters
    ----------
    e_z
        Unit vector of the local cartesian coordinate system (orthogonal to the
        axis of the beam).
    h
        Length of the beam.

    Returns
    -------
    array
        The third derivative-displacement matrix.
    """
    d2Ndxi3 = herm_basis_xi3(0.0)
    B = zeros((1, 6))
    B[0, 0:2] = d2Ndxi3[0] * (2 / h) ** 3 * e_z
    B[0, 2] = (h / 2) * d2Ndxi3[1] * (2 / h) ** 3
    B[0, 3:5] = d2Ndxi3[2] * (2 / h) ** 3 * e_z
    B[0, 5] = (h / 2) * d2Ndxi3[3] * (2 / h) ** 3
    return B


def beam_3d_xz_3rd_deriv_displ_matrix(e_y, e_z, h):
    r"""
    Compute 3d beam third derivative-displacement matrix for displacements in
    the :math:`x-z` plane.

    Here the third derivative is with respect to the physical coordinate
    measured along the member (local :math:`x`).

    The third derivative :math:`d^3w/dx^3` is computed in the local coordinate
    system of the beam as :math:`d^3w/dx^3 = B U`. Here :math:`B` is the
    third-derivative-displacement matrix and :math:`U` is the displacement
    vector. All three displacement components and all three rotation components
    at each joint are assumed, so the matrix :math:`B` has one row and twelve
    columns. Note that this matrix is constant (independent of the location
    along the beam).

    Parameters
    ----------
    e_y
        Unit vector of the local cartesian coordinate system (orthogonal to the
        axis of the beam).
    e_z
        Unit vector of the local cartesian coordinate system (orthogonal to the
        axis of the beam).
    h
        Length of the beam.

    Returns
    -------
    array
        The third derivative-displacement matrix.
    """
    d2Ndxi3 = beam_3d_xz_shape_fun_xi3(0.0)
    B = zeros((1, 12))
    B[0, 0:3] = d2Ndxi3[0] * (2 / h) ** 3 * e_z
    B[0, 3:6] = (h / 2) * d2Ndxi3[1] * (2 / h) ** 3 * e_y
    B[0, 6:9] = d2Ndxi3[2] * (2 / h) ** 3 * e_z
    B[0, 9:12] = (h / 2) * d2Ndxi3[3] * (2 / h) ** 3 * e_y
    return B


def beam_3d_xy_3rd_deriv_displ_matrix(e_y, e_z, h):
    r"""
    Compute 3d beam third derivative-displacement matrix for displacements in
    the :math:`x-y` plane.

    Here the third derivative is with respect to the physical coordinate
    measured along the member (local :math:`x`).

    The third derivative :math:`d^3v/dx^3` is computed in the local coordinate
    system of the beam as :math:`d^3v/dx^3 = B U`. Here :math:`B` is the
    third-derivative-displacement matrix and :math:`U` is the displacement
    vector. All three displacement components and all three rotation components
    at each joint are assumed, so the matrix :math:`B` has one row and twelve
    columns. Note that this matrix is constant (independent of the location
    along the beam).

    Parameters
    ----------
    e_y
        Unit vector of the local cartesian coordinate system (orthogonal to the
        axis of the beam).
    e_z
        Unit vector of the local cartesian coordinate system (orthogonal to the
        axis of the beam).
    h
        Length of the beam.

    Returns
    -------
    array
        The third derivative-displacement matrix.
    """
    d2Ndxi3 = beam_3d_xy_shape_fun_xi3(0.0)
    B = zeros((1, 12))
    B[0, 0:3] = d2Ndxi3[0] * (2 / h) ** 3 * e_y
    B[0, 3:6] = (h / 2) * d2Ndxi3[1] * (2 / h) ** 3 * e_z
    B[0, 6:9] = d2Ndxi3[2] * (2 / h) ** 3 * e_y
    B[0, 9:12] = (h / 2) * d2Ndxi3[3] * (2 / h) ** 3 * e_z
    return B


def beam_2d_moment(member, i, j, xi):
    r"""
    Compute 2d beam moment based on the displacements stored at the joints. The
    moment is computed at the parametric location ``xi`` along the beam.

    The moment is mathematically defined as :math:`M = -EI d^2w/dx^2`.

    The curvature is computed with the curvature-displacement matrix :math:`B`
    by the function :func:`beam_2d_curv_displ_matrix`.

    Parameters
    ----------
    member
        Dictionary that defines the data of the member.
    i
        Dictionary that defines the data of the first joint of the member.
    j
        Dictionary that defines the data of the second joint of the member.
    xi
        Parametric coordinate (:math:`-1\le\xi\le+1`): first joint is at
        :math:`\xi =-1`, second is at :math:`\xi =+1`.

    Returns
    -------
    float
        The moment resultant.

    See Also
    --------
    :func:`beam_2d_curv_displ_matrix`
    """
    _, e_z, h = geometry.member_2d_geometry(i, j)
    sect = member["section"]
    E, I = sect["E"], sect["I"]
    ui, uj = i["displacements"], j["displacements"]
    u = concatenate([ui, uj])
    B = beam_2d_curv_displ_matrix(e_z, h, xi)
    return -E * I * dot(B, u)


def beam_3d_moment(member, i, j, axis, xi):
    r"""
    Compute 3d beam moment based on the displacements stored at the joints. The
    moment is computed at the parametric location ``xi`` along the beam. The
    moment acts about the axis specified by the string ``axis`` (``'y'`` or
    ``'z'``).

    The moments are mathematically defined as :math:`M_y = -EI_y d^2w/dx^2` for
    bending about the `y` axis, and :math:`M_z = +EI_z d^2v/dx^2` for bending
    about the :math:`z` axis.

    The curvatures are computed with  curvature-displacement matrices :math:`B`
    by the functions :func:`beam_3d_xz_curv_displ_matrix` and
    :func:`beam_3d_xy_curv_displ_matrix`, respectively.

    Parameters
    ----------
    member
        Dictionary that defines the data of the member.
    i
        Dictionary that defines the data of the first joint of the member.
    j
        Dictionary that defines the data of the second joint of the member.
    axis
        Bending about which axis? Specify either `'y'` or `'z'`.
    xi
        Parametric coordinate (:math:`-1\le\xi\le+1`): first joint is at
        :math:`\xi =-1`, second is at :math:`\xi =+1`.

    Returns
    -------
    float
        The moment resultant.
    """
    sect = member["section"]
    _, e_y, e_z, h = geometry.member_3d_geometry(i, j, sect["xz_vector"])
    E, Iy, Iz = sect["E"], sect["Iy"], sect["Iz"]
    ui, uj = i["displacements"], j["displacements"]
    u = concatenate([ui, uj])
    if axis == "y":
        B = beam_3d_xz_curv_displ_matrix(e_y, e_z, h, xi)
        M = -E * Iy * dot(B, u)
    else:
        B = beam_3d_xy_curv_displ_matrix(e_y, e_z, h, xi)
        M = +E * Iz * dot(B, u)
    return M


def beam_3d_torsion_moment(member, i, j, xi):
    r"""
    Compute 3d beam torsion moment based on the displacements stored at the
    joints. The moment is uniform along the beam.

    The moment is mathematically defined as :math:`T = GJ d\theta_x/dx`. The rate of
    change of the axial rotation, :math:`d\theta_x/dx`, is computed with the
    torsion-displacement matrix :math:`B`, obtained by the function
    :func:`beam_3d_torsion_displ_matrix`.

    The torsion moment is uniform along the beam. Hence, ``xi`` does not matter.

    Parameters
    ----------
    member
        Dictionary that defines the data of the member.
    i
        Dictionary that defines the data of the first joint of the member.
    j
        Dictionary that defines the data of the second joint of the member.
    xi
        Parametric coordinate (:math:`-1\le\xi\le+1`): first joint is at
        :math:`\xi =-1`, second is at :math:`\xi =+1`.

    Returns
    -------
    float
        The moment resultant.
    """
    sect = member["section"]
    e_x, _, _, h = geometry.member_3d_geometry(i, j, sect["xz_vector"])
    G, J = sect["G"], sect["J"]
    ui, uj = i["displacements"][3:6], j["displacements"][3:6]
    u = concatenate([ui, uj])
    B = beam_3d_torsion_displ_matrix(e_x, h, 0.0)  # single-point integration
    T = G * J * dot(B, u)
    return T


def beam_2d_axial_force(member, i, j, xi):
    r"""
    Compute 2d beam axial force based on the displacements stored at the
    joints.

    Refer to the function :func:`pystran.truss.truss_strain_displacement` that computes
    the strain-displacement matrix for a truss member.

    The axial force is uniform along the beam. Hence, ``xi`` does not matter.

    Parameters
    ----------
    member
        Dictionary that defines the data of the member.
    i
        Dictionary that defines the data of the first joint of the member.
    j
        Dictionary that defines the data of the second joint of the member.
    xi
        Parametric coordinate (:math:`-1\le\xi\le+1`): first joint is at
        :math:`\xi =-1`, second is at :math:`\xi =+1`.

    Returns
    -------
    float
        The force resultant.

    See Also
    --------
    :func:`pystran.truss.truss_strain_displacement`
    """
    sect = member["section"]
    e_x, _, h = geometry.member_2d_geometry(i, j)
    E, A = sect["E"], sect["A"]
    ui, uj = i["displacements"][0:2], j["displacements"][0:2]
    u = concatenate([ui, uj])
    B = truss.truss_strain_displacement(e_x, h)
    N = E * A * dot(B, u)
    return N


def beam_3d_axial_force(member, i, j, xi):
    r"""
    Compute 3d beam or truss axial force based on the displacements stored at the joints.

    The axial force is uniform along the beam. Hence, ``xi`` does not matter.

    Parameters
    ----------
    member
        Dictionary that defines the data of the member.
    i
        Dictionary that defines the data of the first joint of the member.
    j
        Dictionary that defines the data of the second joint of the member.
    xi
        Parametric coordinate (:math:`-1\le\xi\le+1`): first joint is at
        :math:`\xi =-1`, second is at :math:`\xi =+1`.

    Returns
    -------
    float
        The force resultant.
    """
    sect = member["section"]
    if 'xz_vector' in sect:
        xz_vector = sect["xz_vector"]
    else:
        xz_vector = []
    e_x, _, _, h = geometry.member_3d_geometry(i, j, xz_vector)
    E, A = sect["E"], sect["A"]
    ui, uj = i["displacements"][0:3], j["displacements"][0:3]
    u = concatenate([ui, uj])
    B = beam_3d_stretch_displ_matrix(e_x, h, 0.0)  # single-point integration
    N = E * A * dot(B, u)
    return N


def beam_3d_shear_force(member, i, j, axis, xi):
    r"""
    Compute 3d shear force based on the displacements stored at the joints.

    The shear force in the direction of axis ``axis``  (``'y'`` or ``'z'``) is
    uniform along the beam.  Hence, ``xi`` does not matter.

    Parameters
    ----------
    member
        Dictionary that defines the data of the member.
    i
        Dictionary that defines the data of the first joint of the member.
    j
        Dictionary that defines the data of the second joint of the member.
    axis
        Shearing in the direction of which axis? Specify either `'y'` or `'z'`.
    xi
        Parametric coordinate (:math:`-1\le\xi\le+1`): first joint is at
        :math:`\xi =-1`, second is at :math:`\xi =+1`.

    Returns
    -------
    float
        The force resultant.
    """
    sect = member["section"]
    _, e_y, e_z, h = geometry.member_3d_geometry(i, j, sect["xz_vector"])
    E, Iy, Iz = sect["E"], sect["Iy"], sect["Iz"]
    ui, uj = i["displacements"], j["displacements"]
    u = concatenate([ui, uj])
    if axis == "z":
        B = beam_3d_xz_3rd_deriv_displ_matrix(e_y, e_z, h)
        Q = -E * Iy * dot(B, u)
    else:
        B = beam_3d_xy_3rd_deriv_displ_matrix(e_y, e_z, h)
        Q = -E * Iz * dot(B, u)
    return Q


def beam_2d_shear_force(member, i, j, xi):
    r"""
    Compute 2d beam shear force based on the displacements stored at the
    joints.

    The shear force is uniform along the beam.  Hence, ``xi`` does not matter.

    Parameters
    ----------
    member
        Dictionary that defines the data of the member.
    i
        Dictionary that defines the data of the first joint of the member.
    j
        Dictionary that defines the data of the second joint of the member.
    xi
        Parametric coordinate (:math:`-1\le\xi\le+1`): first joint is at
        :math:`\xi =-1`, second is at :math:`\xi =+1`.

    Returns
    -------
    float
        The force resultant.
    """
    _, e_z, h = geometry.member_2d_geometry(i, j)
    sect = member["section"]
    E, I = sect["E"], sect["I"]
    ui, uj = i["displacements"], j["displacements"]
    u = concatenate([ui, uj])
    B = beam_2d_3rd_deriv_displ_matrix(e_z, h)
    return -E * I * dot(B, u)


def beam_3d_stretch_displ_matrix(e_x, h, xi):
    r"""
    Compute beam stretch-displacement matrix.

    Stretch here means axial strain.

    The job is delegated to the truss module.

    Parameters
    ----------
    e_x
        Unit vector along the axis of the member.
    h
        Length of the member.
    xi
        Parametric coordinate (:math:`-1\le\xi\le+1`): first joint is at
        :math:`\xi =-1`, second is at :math:`\xi =+1`.

    Returns
    -------
    array
        Stretch-displacement matrix.

    See Also
    --------
    :func:`pystran.truss.truss_strain_displacement`
    """
    return truss.truss_strain_displacement(e_x, h)


def beam_2d_3d_axial_stiffness(e_x, h, E, A):
    r"""
    Compute axial stiffness matrix.

    This function works both for 2D and 3D beams.

    The axial stiffness matrix is computed as :math:`K = EA B^T B h`. Here
    :math:`B` is the stretch-displacement matrix, computed by
    :func:`beam_3d_stretch_displ_matrix`.

    Parameters
    ----------
    e_x
        Unit vector of the local cartesian coordinate system in the direction
        of the axis of the beam.
    h
        Length of the beam.
    E
        Young's modulus of elasticity.
    A
        Cross section area.

    Returns
    -------
    array
        Stiffness matrix of the beam.

    See Also
    --------
    :func:`beam_3d_stretch_displ_matrix`
    """
    B = beam_3d_stretch_displ_matrix(e_x, h, 0.0)  # single-point integration
    return E * A * outer(B.T, B) * h


def beam_3d_torsion_stiffness(e_x, h, G, J):
    r"""
    Compute torsion stiffness matrix.

    The torsion stiffness matrix is computed as :math:`K = GJ B^T B h`. Here
    :math:`B` is the torsion-displacement matrix, computed by
    :func:`beam_3d_torsion_displ_matrix`.

    Parameters
    ----------
    e_x
        Unit vector of the local cartesian coordinate system in the direction
        of the axis of the beam.
    h
        Length of the beam.
    G
        Shear modulus of elasticity.
    J
        Torsion constant of the section.

    Returns
    -------
    array
        Stiffness matrix of the beam.

    See Also
    --------
    :func:`beam_3d_torsion_displ_matrix`
    """
    B = beam_3d_torsion_displ_matrix(e_x, h, 0.0)  # single-point integration
    return G * J * outer(B.T, B) * h


def beam_3d_torsion_displ_matrix(e_x, h, xi):
    r"""
    Compute torsion-displacement matrix.

    The torsion-displacement matrix is constant.  Hence, ``xi`` does not
    matter.

    Parameters
    ----------
    e_x
        Unit vector of the local cartesian coordinate system in the direction
        of the axis of the beam.
    h
        Length of the beam.
    xi
        Parametric coordinate (:math:`-1\le\xi\le+1`): first joint is at
        :math:`\xi =-1`, second is at :math:`\xi =+1`.

    Returns
    -------
    array
        The torsion rate-displacement matrix of the bar.
    """
    B = zeros((1, 6))
    B[0, 0:3] = -e_x / h
    B[0, 3:6] = e_x / h
    return B


def assemble_stiffness(Kg, member, i, j):
    """
    Assemble beam stiffness matrix.

    The stiffness matrix depends on whether the beam is considered in two
    dimensions or in three dimensions.

    In two dimensions, the beam stiffness matrix is a superposition of two
    mechanisms: axial bar plus bending in the :math:`x-z` plane.

    In three dimensions, beam stiffness is a superposition of four mechanisms
    -- axial bar, torsion bar, bending in the :math:`x-y` plane, and bending in
    the :math:`x-z` plane.

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
    :func:`beam_2d_3d_axial_stiffness`
    :func:`beam_2d_bending_stiffness`
    :func:`beam_3d_bending_stiffness`
    :func:`beam_3d_torsion_stiffness`
    """
    # Add stiffness in bending.
    beam_is_2d = len(i["coordinates"]) == len(j["coordinates"]) == 2
    dof = concatenate([i["dof"], j["dof"]])
    if beam_is_2d:
        sect = member["section"]
        e_x, e_z, h = geometry.member_2d_geometry(i, j)
        # Add stiffness in bending.
        E, I = sect["E"], sect["I"]
        k = beam_2d_bending_stiffness(e_z, h, E, I)
        Kg = assemble.assemble(Kg, dof, k)
        # Add stiffness in the axial direction.
        E, A = sect["E"], sect["A"]
        k = beam_2d_3d_axial_stiffness(e_x, h, E, A)
        dof = concatenate([i["dof"][0:2], j["dof"][0:2]])
        Kg = assemble.assemble(Kg, dof, k)
    else:
        sect = member["section"]
        e_x, e_y, e_z, h = geometry.member_3d_geometry(i, j, sect["xz_vector"])
        # Add stiffness in bending.
        E, Iy, Iz = sect["E"], sect["Iy"], sect["Iz"]
        kxy, kxz = beam_3d_bending_stiffness(e_y, e_z, h, E, Iy, Iz)
        Kg = assemble.assemble(Kg, dof, kxy)
        Kg = assemble.assemble(Kg, dof, kxz)
        # Add stiffness in the axial direction.
        E, A = sect["E"], sect["A"]
        k = beam_2d_3d_axial_stiffness(e_x, h, E, A)
        dof = concatenate([i["dof"][0:3], j["dof"][0:3]])
        Kg = assemble.assemble(Kg, dof, k)
        # Add stiffness in torsion.
        G, J = sect["G"], sect["J"]
        k = beam_3d_torsion_stiffness(e_x, h, G, J)
        dof = concatenate([i["dof"][3:6], j["dof"][3:6]])
        Kg = assemble.assemble(Kg, dof, k)
    return Kg


def assemble_mass(Mg, member, i, j):
    """
    Assemble beam mass matrix.

    The mass matrix of two-dimensional beam takes into account axial vibration
    and transverse vibration separately.

    In three dimensions, in addition to axial vibration, transverse vibration
    in two principal planes, torsional vibration is also taken into account.

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
    float
        Updated global matrix is returned.

    See Also
    --------
    :func:`pystran.assemble.assemble`
    :func:`beam_2d_mass`
    :func:`beam_3d_mass`
    """
    beam_is_2d = len(i["coordinates"]) == len(j["coordinates"]) == 2
    dof = concatenate([i["dof"], j["dof"]])
    sect = member["section"]
    rho, A = sect["rho"], sect["A"]
    if beam_is_2d:
        e_x, e_z, h = geometry.member_2d_geometry(i, j)
        m = beam_2d_mass(e_x, e_z, h, rho, A)
    else:
        e_x, e_y, e_z, h = geometry.member_3d_geometry(i, j, sect["xz_vector"])
        Ix = sect["Ix"]
        m = beam_3d_mass(e_x, e_y, e_z, h, rho, A, Ix)
    Mg = assemble.assemble(Mg, dof, m)
    return Mg


def beam_2d_mass(e_x, e_z, h, rho, A):
    r"""
    Compute beam mass matrix.

    The mass matrix is consistent, which means that it is computed from the
    discrete form of the kinetic energy of the element,

    .. math::

        \int \rho A \left(\dot u \cdot \dot u +  \dot w \cdot \dot
        w\right)dx

    where :math:`\dot u` and  :math:`\dot w` are the velocities in the
    :math:`x` and :math:`z` directions. Note that the rotational velocities of
    the cross sections do not play a role.

    The velocity :math:`\dot u` is assumed to vary linearly along the element,
    and the velocity :math:`\dot w` is assumed to vary according to the Hermite
    shape functions.

    Gauss quadrature is used to compute the mass matrix.

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
        Mass matrix of the beam.
    """
    xiG, WG = gauss.rule(4)
    n = (len(e_x) + 1) * 2
    m = zeros((n, n))
    for xi, W in zip(xiG, WG):
        N = geometry.lin_basis(xi)
        Nu = concatenate([N[0] * e_x, [0.0], N[1] * e_x, [0.0]])
        m += rho * A * outer(Nu, Nu) * W * (h / 2)
        N = geometry.herm_basis(xi)
        Nw = concatenate([N[0] * e_z, [(h / 2) * N[1]], N[2] * e_z, [(h / 2) * N[3]]])
        m += rho * A * outer(Nw, Nw) * W * (h / 2)
    return m


def beam_3d_mass(e_x, e_y, e_z, h, rho, A, Ix):
    r"""
    Compute beam mass matrix.

    The mass matrix is consistent, which means that it is computed from the
    discrete form of the kinetic energy of the element,

    .. math::

        \int \rho A \left(\dot u \cdot \dot u + \dot v \cdot \dot v + \dot w
        \cdot \dot    w\right)dx

    where :math:`\dot u`, :math:`\dot v`, and :math:`\dot w` are the velocities
    in the :math:`x`, :math:`y`, and :math:`z` directions.

    The velocity :math:`\dot u` is assumed to vary linearly along the element,
    and the velocity :math:`\dot v`, :math:`\dot w` is assumed to vary
    according to the Hermite shape functions. Rotations about :math:`y` and
    :math:`z` are ignored in the kinetic energy.

    For spinning of the beam about its axis, the kinetic energy is given by the
    formula

    .. math::

        \int \rho I_x \dot \theta_x \cdot \dot \theta_x dx.

    Gauss quadrature is used to compute the mass matrix.

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
    Ix
        Second moment of area of the cross section for rotation about x.

    Returns
    -------
    array
        Mass matrix of the beam.
    """
    xiG, WG = gauss.rule(4)
    n = len(e_x) * 4
    m = zeros((n, n))
    for xi, W in zip(xiG, WG):
        N = geometry.lin_basis(xi)
        # Axial translation
        extN = concatenate([N[0] * e_x, [0.0, 0.0, 0.0], N[1] * e_x, [0.0, 0.0, 0.0]])
        m += rho * A * outer(extN, extN) * W * (h / 2)
        # Torsion
        extN = concatenate([[0.0, 0.0, 0.0], N[0] * e_x, [0.0, 0.0, 0.0], N[1] * e_x])
        m += rho * Ix * outer(extN, extN) * W * (h / 2)
        # Transverse displacements and rotations about y and z
        N = beam_3d_xz_shape_fun(xi)
        extN = concatenate(
            [N[0] * e_z, (h / 2) * N[1] * e_y, N[2] * e_z, (h / 2) * N[3] * e_y]
        )
        m += rho * A * outer(extN, extN) * W * (h / 2)
        N = beam_3d_xy_shape_fun(xi)
        extN = concatenate(
            [N[0] * e_y, (h / 2) * N[1] * e_z, N[2] * e_y, (h / 2) * N[3] * e_z]
        )
        m += rho * A * outer(extN, extN) * W * (h / 2)

    return m


def beam_3d_end_forces(member, i, j):
    """
    Compute the end forces of a beam element in 3d.

    The end forces of the beam are forces acting on the joints ``i`` and ``j``
    by the beam.

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
    Dict
        Dictionary with the keys ``'Ni'``, ``'Qyi'``, ``'Qzi'``, ``'Ti'``,
        ``'Myi'``, ``'Mzi'``, ``'Nj'``, ``'Qyj'``, ``'Qzj'``, ``'Tj'``,
        ``'Myj'``, ``'Mzj'``,  is returned.
    """
    Ni = beam_3d_axial_force(member, i, j, 0.0)
    Nj = -beam_3d_axial_force(member, i, j, 0.0)
    Ti = beam_3d_torsion_moment(member, i, j, 0.0)
    Tj = -beam_3d_torsion_moment(member, i, j, 0.0)
    Myi = beam_3d_moment(member, i, j, "y", -1.0)
    Myj = -beam_3d_moment(member, i, j, "y", +1.0)
    Mzi = beam_3d_moment(member, i, j, "z", -1.0)
    Mzj = -beam_3d_moment(member, i, j, "z", +1.0)
    Qyi = beam_3d_shear_force(member, i, j, "y", -1.0)
    Qyj = -beam_3d_shear_force(member, i, j, "y", +1.0)
    Qzi = beam_3d_shear_force(member, i, j, "z", -1.0)
    Qzj = -beam_3d_shear_force(member, i, j, "z", +1.0)
    return dict(
        Ni=Ni[0],
        Qyi=Qyi[0],
        Qzi=Qzi[0],
        Ti=Ti[0],
        Myi=Myi[0],
        Mzi=Mzi[0],
        Nj=Nj[0],
        Qyj=Qyj[0],
        Qzj=Qzj[0],
        Tj=Tj[0],
        Myj=Myj[0],
        Mzj=Mzj[0],
    )


def beam_2d_end_forces(member, i, j):
    """
    Compute the end forces of a beam element in 3d.

    The end forces of the beam are forces acting on the joints ``i`` and ``j``
    by the beam.

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
    Dict
        Dictionary with the keys ``'Ni'``, ``'Qzi'``, ``'Myi'``,  ``'Nj'``,
        ``'Qzj'``, ``'Myj'``,  is returned.
    """
    Ni = beam_2d_axial_force(member, i, j, 0.0)
    Nj = -beam_2d_axial_force(member, i, j, 0.0)
    Myi = beam_2d_moment(member, i, j, -1.0)
    Myj = -beam_2d_moment(member, i, j, +1.0)
    Qzi = beam_2d_shear_force(member, i, j, -1.0)
    Qzj = -beam_2d_shear_force(member, i, j, +1.0)
    return dict(
        Ni=Ni[0],
        Qzi=Qzi[0],
        Myi=Myi[0],
        Nj=Nj[0],
        Qzj=Qzj[0],
        Myj=Myj[0],
    )

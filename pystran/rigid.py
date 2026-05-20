"""
Define rigid link mechanical quantities.
"""

from numpy import concatenate, zeros, dot, array, eye
from pystran import geometry
from pystran import assemble


def rigid_link_stiffness(e_x, h, Gamma):
    r"""
    Compute rigid link stiffness matrix.

    The stiffness matrix is computed as

    .. math::
        K =\begin{bmatrix} 
            C^T\Gamma C & -C^T\Gamma  \\
            -\Gamma C & \Gamma \\
        \end{bmatrix}.
    

    Here :math:`C` is a matrix computed from the vector  :math:`r = h e_x`, 
    which is the difference between the location of 
    the subordinate and the location of the master.

    In three dimensions
    
    .. math::
        C =\begin{bmatrix} 
            1 & \widetilde{r} \\
            0 & 1 \\
        \end{bmatrix}.
    
    Here :math:`\widetilde{r}` is a skew matrix corresponding to the 
    vector :math:`r`, and :math:`0` and :math:`1` stand for :math:`3\times3` zero 
    and identity matrices respectively.

    Further, :math:`\Gamma` is a diagonal matrix, such that 
    the diagonal entries provide penalty on the difference 
    between the individual degrees of freedom.

    Reference: APPLICATION OF RIGID LINKS  IN STRUCTURAL DESIGN MODELS,
    Sergey Yu. Fialko, International Journal for Computational Civil 
    and Structural Engineering, 13(3) 119-137 (2017). 

    Parameters
    ----------
    e_x
        Vector :math:`e_x` is the direction vector along the axis of the member.
    h
        Length of the rigid link (distance between the joints).
    Gamma
        Diagonal matrix of the penalty constants.

    Returns
    -------
    array
        Stiffness matrix.
    """
    if len(e_x) == 2:
        I = eye(2)
        C = zeros((3, 3))
        C[0:2, 0:2] = I
        C[2, 2] = 1.0
        rx, ry = e_x[0] * h, e_x[1] * h
        C[0, 2] = -ry
        C[1, 2] = +rx
    else:
        I = eye(3)
        C = zeros((6, 6))
        C[0:3, 0:3] = I
        C[3:6, 3:6] = I
        rx, ry, rz = e_x[0] * h, e_x[1] * h, e_x[2] * h
        C[0, 4] = +rz
        C[0, 5] = -ry
        C[1, 3] = -rz
        C[1, 5] = +rx
        C[2, 3] = +ry
        C[2, 4] = -rx
    k = concatenate(
        [
            concatenate([dot(C.T, dot(Gamma, C)), -dot(C.T, Gamma)], axis=1),
            concatenate([-dot(Gamma, C), Gamma], axis=1),
        ],
        axis=0,
    )
    return k


def assemble_stiffness(Kg, member, i, j):
    """
    Assemble rigid link stiffness matrix.

    Parameters
    ----------
    Kg
        Global structural stiffness matrix.
    member
        Dictionary that defines the data of the member.
    i
        Dictionary that defines the data of the first joint of the member. This
        is the master.
    j
        Dictionary that defines the data of the second joint of the member.
        This is the subordinate joint.

    Returns
    -------
    array
        Updated global matrix is returned.

    See Also
    --------
    :func:`pystran.assemble.assemble`
    """
    sect = member["section"]
    Gamma = sect["Gamma"]
    dim = len(i["coordinates"])
    if dim == 2:
        e_x, _, h = geometry.member_2d_geometry(i, j)
    else:
        e_x, _, _, h = geometry.member_3d_geometry(i, j, array([]))
    k = rigid_link_stiffness(e_x, h, Gamma)
    dof = concatenate([i["dof"], j["dof"]])
    return assemble.assemble(Kg, dof, k)

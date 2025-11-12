"""
Define the functions for defining and manipulating a model.
"""

from math import sqrt
import numpy
from numpy import array, zeros, dot, mean, concatenate
import scipy
import pystran
from pystran import freedoms


def create(dim=2):
    """
    Create a new model.

    Parameters
    ----------
    dim
        Supply the dimension of the model (2 or 3).

    Returns
    -------
    dict
    """
    m = {}
    m["dim"] = dim  # Dimension of the model
    m["joints"] = {}
    # Depending on the number of space dimensions, a set of degrees of freedom
    # will consist of either two translations and one rotation, or three
    # translations and three rotations.
    if m["dim"] == 2:
        freedoms.U1 = 0
        freedoms.U2 = 1
        freedoms.UR3 = 2
        freedoms.U3 = -1000  # invalid
        freedoms.UR1 = -1000  # invalid
        freedoms.UR2 = -1000  # invalid
    else:
        freedoms.U1 = 0
        freedoms.U2 = 1
        freedoms.U3 = 2
        freedoms.UR1 = 3
        freedoms.UR2 = 4
        freedoms.UR3 = 5
    return m


def add_joint(m, jid, coordinates, dof=None):
    """
    Add a joint to the model.

    Parameters
    ----------
    m
        Model.
    jid
        The joint identifier, which must be unique, but can be anything
        that is a legal dictionary key (integer, string, ...),
    coordinates
        The list (or a tuple) of coordinates of the joint; the
        input is converted to an array.
    dof
        Optional: the degrees of freedom of the joint as a list (or a
        tuple). If provided, do not use :func:`number_dofs` later on.

    Returns
    -------
    dict
        Newly created joint.
    """
    if jid in m["joints"]:
        raise RuntimeError("Joint already exists")
    coordinates = array(coordinates, dtype=numpy.float64)
    if coordinates.shape != (m["dim"],):
        raise RuntimeError("Coordinate dimension mismatch")
    m["joints"][jid] = {"jid": jid, "coordinates": coordinates}
    if dof is not None:
        m["joints"][jid]["dof"] = array(dof, dtype=numpy.int32)
    return m["joints"][jid]


def add_truss_member(m, mid, connectivity, sect):
    """
    Add a truss member to the model.

    Parameters
    ----------
    m
        Model.
    mid
        The member identifier, which must be unique, but can be anything
        that is a legal dictionary key (integer, string, ...).
    connectivity
        The list (or a tuple) of the joint identifiers.
    sect
        Section of appropriate type (i.e. a truss section).

    Returns
    -------
    dict
        Newly created member.
    """
    if "truss_members" not in m:
        m["truss_members"] = {}
    if mid in m["truss_members"]:
        raise RuntimeError("Truss member already exists")
    m["truss_members"][mid] = {
        "mid": mid,
        "connectivity": connectivity,
        "section": sect,
    }
    return m["truss_members"][mid]


def add_beam_member(m, mid, connectivity, sect):
    """
    Add a beam member to the model.

    Parameters
    ----------
    m
        Model.
    mid
        The member identifier, which must be unique, but can be anything
        that is a legal dictionary key (integer, string, ...).
    connectivity
        The list (or a tuple) of the joint identifiers.
    sect
        Section of appropriate type (2d or 3d beam section).

    Returns
    -------
    dict
        Newly created member.

    See Also
    --------
    :func:`section.beam_2d_section`
    :func:`section.beam_3d_section`
    """
    if "beam_members" not in m:
        m["beam_members"] = {}
    if mid in m["beam_members"]:
        raise RuntimeError("Beam member already exists")
    m["beam_members"][mid] = {
        "mid": mid,
        "connectivity": connectivity,
        "section": sect,
    }
    return m["beam_members"][mid]


def add_rigid_link_member(m, mid, connectivity, sect):
    """
    Add a rigid link member to the model.

    Parameters
    ----------
    m
        Model.
    mid
        The member identifier, which must be unique, but can be anything
        that is a legal dictionary key (integer, string, ...),
    connectivity
        The list (or a tuple) of the joint identifiers; the
        first is the *master* (its motion determines the motion of the
        subordinate), the second is the *subordinate* (its motion follows that of
        the master).
    sect
        Section of appropriate type (i.e. rigid link section).

    Returns
    -------
    dict
        Newly created member.

    See Also
    --------
    :func:`rigid_link_section`
    """
    if "rigid_link_members" not in m:
        m["rigid_link_members"] = {}
    if mid in m["rigid_link_members"]:
        raise RuntimeError("Rigid link member already exists")
    m["rigid_link_members"][mid] = {
        "mid": mid,
        "connectivity": connectivity,
        "section": sect,
    }
    return m["rigid_link_members"][mid]


def add_spring_member(m, mid, connectivity, sect):
    """
    Add a spring member to the model.

    Parameters
    ----------
    m
        Model.
    mid
        The member identifier, which must be unique, but can be anything
        that is a legal dictionary key (integer, string, ...),
    connectivity
        The list (or a tuple) of the joint identifiers.
    sect
        Section of appropriate type (i.e. a spring section).

    Returns
    -------
    dict
        Newly created member.

    See Also
    --------
    :func:`spring_section`
    """
    if "spring_members" not in m:
        m["spring_members"] = {}
    if mid in m["spring_members"]:
        raise RuntimeError("Spring member already exists")
    m["spring_members"][mid] = {
        "mid": mid,
        "connectivity": connectivity,
        "section": sect,
    }
    return m["spring_members"][mid]


def add_support(j, dof, value=0.0):
    """
    Add a support to a joint.

    Parameters
    ----------
    j
        The joint (obtained from the model as ``m["joints"][jid]``).
    dof
        The degree of freedom (0, 1, ...). Refer to the module ``freedoms``.
    value
        The signed magnitude of the support motion (default is zero).

    Returns
    -------
    None

    See Also
    --------
    `pystran.freedoms`
    """
    if "supports" not in j:
        j["supports"] = {}
    dim = len(j["coordinates"])
    for d, v in zip(*freedoms.prescribed_dofs_and_values(dim, dof, value)):
        j["supports"][d] = v


def add_load(j, dof, value):
    """
    Add a load to a joint.

    Parameters
    ----------
    j
        The joint (obtained from the model as ``m["joints"][jid]``).
    dof
        The degree of freedom (0, 1, ...).
    value
        The signed magnitude of the load.

    Returns
    -------
    None

    See Also
    --------
    `pystran.freedoms`
    """
    if "loads" not in j:
        j["loads"] = {}
    if dof not in j["loads"]:
        j["loads"][dof] = 0.0
    j["loads"][dof] += value


def add_mass(j, dof, value):
    """
    Add a mass to a joint.

    Parameters
    ----------
    j
        The joint (obtained from the model as ``m["joints"][jid]``).
    dof
        The degree of freedom (0, 1, ...). Refer to the module ``freedoms``.
    value
        The magnitude of the added mass.

    Returns
    -------
    None

    See Also
    --------
    `pystran.freedoms`
    """
    if "masses" not in j:
        j["masses"] = {}
    j["masses"][dof] = value


def add_dof_links(m, jids, dof):
    """
    Add degree-of-freedom links between all joints in the list ``jids`` in the
    direction ``dof``.

    Parameters
    ----------
    m
        The model.
    jids
        The list of joint identifiers.
    dof
        The degree of freedom at which the joints are to be linked.

    Returns
    -------
    None

    See Also
    --------
    `pystran.freedoms`
    """
    # Now add the mutual links between the joints
    for jid1 in jids:
        for jid2 in jids:
            if jid1 != jid2:
                j1 = m["joints"][jid1]
                if "links" not in j1:
                    j1["links"] = {}
                if jid2 not in j1["links"]:
                    j1["links"][jid2] = []
                for d, _ in zip(
                    *freedoms.prescribed_dofs_and_values(m["dim"], dof, 0.0)
                ):
                    j1["links"][jid2].append(d)


def bounding_box(m):
    """
    Compute the bounding box of the model.

    Parameters
    ----------
    m
        The model.

    Returns
    -------
    array
        Array of the lower and upper ranges (i.e. the bounding box that
        encloses all the joints).
    """
    dim = m["dim"]
    box = numpy.array(
        concatenate([[numpy.inf for i in range(dim)], [-numpy.inf for i in range(dim)]])
    )
    for j in m["joints"].values():
        cj = j["coordinates"]
        for i, v in enumerate(cj):
            box[i] = min(box[i], v)
            box[i + dim] = max(box[i + dim], v)
    return box


def characteristic_dimension(m):
    """
    Compute the characteristic dimension of the model.

    This is the average of the dimensions of the bounding box.

    Parameters
    ----------
    m
        The model.

    Returns
    -------
    float
        Characteristic dimension.
    """
    dim = m["dim"]
    box = bounding_box(m)
    dl = [box[i + dim] - box[i] for i in range(dim)]
    return mean(array(dl))


def _copy_dof_num_to_linked(m, j, d, n):
    if "links" in j:
        for k in j["links"].keys():
            o = m["joints"][k]
            if d in o["links"][j["jid"]]:
                o["dof"][d] = n


def _have_rotations(m):
    with_rotations = "beam_members" in m and m["beam_members"]
    if with_rotations:
        return True
    for j in m["joints"].values():
        if "supports" in j and j["supports"]:
            for dof in j["supports"].keys():
                if dof == freedoms.UR1 or dof == freedoms.UR2 or dof == freedoms.UR3:
                    return True
    return False


def ndof_per_joint(m):
    """
    How many degrees of freedom are there per joint?

    Parameters
    ----------
    m
        The model.

    Returns
    -------
    int
        Number of the degrees of freedom per joint. Depends on the space
        dimension of the model and the presence or absence of beams.
    """
    ndpn = m["dim"]
    with_rotations = _have_rotations(m)
    if with_rotations:
        if m["dim"] == 2:
            ndpn = 3
        else:
            ndpn = 6
    return ndpn


def number_dofs(m):
    """
    Number degrees of freedom.

    All current information about degrees of freedom will be replaced when this
    function is done.

    After this function returns, ``m["nfreedof"]`` will be the number of free
    degrees of freedom, and ``m["ntotaldof"]`` will be the total number of degrees
    of freedom.

    The degrees of freedom are numbered in the order of free and then
    prescribed.

    Parameters
    ----------
    m
        The model.

    Returns
    -------
    None
    """
    # Determine the number of degrees of freedom per joint
    ndpn = ndof_per_joint(m)
    # Generate arrays for storing the degrees of freedom
    for j in m["joints"].values():
        if "dof" not in j:
            j["dof"] = zeros((ndpn,), dtype=numpy.int32)
        j["dof"][:] = -1  # -1 means not yet numbered
    # For each linked pair of joints, make sure they share the same supports
    for j in m["joints"].values():
        if "links" in j and "supports" in j:
            for k in j["links"].keys():
                o = m["joints"][k]
                if not "supports" in o:
                    o["supports"] = j["supports"].copy()
                if o["supports"] != j["supports"]:
                    raise RuntimeError("Linked joints must have the same supports")

    # Number the free degrees of freedom first
    n = 0
    for j in m["joints"].values():
        for d in range(ndpn):
            if ("supports" not in j) or (d not in j["supports"]):
                if j["dof"][d] < 0:
                    j["dof"][d] = n
                    _copy_dof_num_to_linked(m, j, d, n)
                    n += 1
    m["nfreedof"] = n
    # Number all prescribed degrees of freedom
    for j in m["joints"].values():
        for d in range(ndpn):
            if "supports" in j and d in j["supports"]:
                if j["dof"][d] < 0:
                    j["dof"][d] = n
                    _copy_dof_num_to_linked(m, j, d, n)
                    n += 1
    m["ntotaldof"] = n


def _build_stiffness_matrix(m):
    nt = m["ntotaldof"]
    # Assemble global stiffness matrix and mass matrix
    K = zeros((nt, nt))
    if "truss_members" in m:
        for member in m["truss_members"].values():
            connectivity = member["connectivity"]
            i, j = m["joints"][connectivity[0]], m["joints"][connectivity[1]]
            pystran.truss.assemble_stiffness(K, member, i, j)
    if "beam_members" in m:
        for member in m["beam_members"].values():
            connectivity = member["connectivity"]
            i, j = m["joints"][connectivity[0]], m["joints"][connectivity[1]]
            pystran.beam.assemble_stiffness(K, member, i, j)
    if "rigid_link_members" in m:
        for member in m["rigid_link_members"].values():
            connectivity = member["connectivity"]
            i, j = m["joints"][connectivity[0]], m["joints"][connectivity[1]]
            pystran.rigid.assemble_stiffness(K, member, i, j)
    if "spring_members" in m:
        for member in m["spring_members"].values():
            connectivity = member["connectivity"]
            i, j = m["joints"][connectivity[0]], m["joints"][connectivity[1]]
            pystran.spring.assemble_stiffness(K, member, i, j)

    return K


def _build_mass_matrix(m):
    nt = m["ntotaldof"]
    M = zeros((nt, nt))
    if "truss_members" in m:
        for member in m["truss_members"].values():
            connectivity = member["connectivity"]
            i, j = m["joints"][connectivity[0]], m["joints"][connectivity[1]]
            pystran.truss.assemble_mass(M, member, i, j)
    if "beam_members" in m:
        for member in m["beam_members"].values():
            connectivity = member["connectivity"]
            i, j = m["joints"][connectivity[0]], m["joints"][connectivity[1]]
            pystran.beam.assemble_mass(M, member, i, j)
    for j in m["joints"].values():
        if "masses" in j:
            for dof, value in j["masses"].items():
                gr = j["dof"][dof]
                M[gr, gr] += value
    return M


def solve_statics(m):
    r"""
    Solve the static equilibrium of the discrete model.

    This function solves the equation of static equilibrium

    .. math::
        K \cdot U = F 

    Here :math:`K` is the stiffness matrix, :math:`U` is the displacement
    vector, and :math:`F` is the vector of forces acting on the joints.

    Note that the degrees of freedom can be partitioned into 'free' (unknown)
    and 'data' (given, i.e. prescribed).

    .. math::
        \left[ \begin{array}{cc} 
        K_{ff} & K_{fd} \\
        K_{df} & K_{dd} \\
        \end{array}\right] \cdot \left[ \begin{array}{cc} 
        U_{f} \\
        U_{d} \\
        \end{array}\right] = \left[ \begin{array}{cc} 
        L_{f} \\
        L_{d} + R\\
        \end{array}\right] 
    

    Here :math:`L_f` is the vector of active loads applied to the free degrees
    of freedom, and :math:`L_d`  is the vector of active loads applied to the
    data degrees of freedom. The reactions :math:`R` due to supports act on the
    prescribed (data) degrees of freedom.

    The system of equations is solved for the free degrees of freedom as

    .. math::
        K_{ff} \cdot U_{f} = -K_{fd} \cdot U_{d} +L_{f} 
    
    Note: :func:`number_dofs` must be called before this function to number the
    degrees of freedom, automatically. Alternatively, the user may specify the
    numbers of the degrees of freedom when defining the joints: the manual way.


    Parameters
    ----------
    m
        The model.

    Returns
    -------
    None

    See Also
    --------
    :func:`number_dofs`
    """
    if not ("ntotaldof" in m) or m["ntotaldof"] <= 0:
        raise RuntimeError(
            "No degrees of freedom: the numbers of degrees of freedom need to be generated"
        )
    if not ("nfreedof" in m) or m["nfreedof"] <= 0:
        raise RuntimeError("No free degrees of freedom: nothing to compute")
    nt, nf = m["ntotaldof"], m["nfreedof"]

    # Assemble global stiffness matrix
    K = _build_stiffness_matrix(m)

    m["K"] = K

    # Compute the active load vector
    F = zeros(nt)
    for joint in m["joints"].values():
        if "loads" in joint:
            for dof, value in joint["loads"].items():
                gr = joint["dof"][dof]
                F[gr] += value

    m["F"] = F

    U = zeros(m["ntotaldof"])
    for joint in m["joints"].values():
        if "supports" in joint:
            for dof, value in joint["supports"].items():
                if value != 0.0:
                    gr = joint["dof"][dof]
                    U[gr] = value
    # # Solve for displacements
    U[0:nf] = scipy.linalg.solve(K[0:nf, 0:nf], F[0:nf] - dot(K[0:nf, nf:nt], U[nf:nt]))

    m["U"] = U

    # # Assign displacements back to joints
    for joint in m["joints"].values():
        joint["displacements"] = U[joint["dof"]]


def statics_reactions(m):
    r"""
    Compute the reactions in the static equilibrium of the discrete model.

    The partitioned system of the balance equations reads

    .. math::
        \left[ \begin{array}{cc} 
        K_{ff} & K_{fd} \\
        K_{df} & K_{dd} \\
        \end{array}\right] \cdot \left[ \begin{array}{cc} 
        U_{f} \\
        U_{d} \\
        \end{array}\right] = \left[ \begin{array}{cc} 
        L_{f} \\
        L_{d} + R\\
        \end{array}\right] 
    

    Here :math:`L_f` is the vector of active loads applied to the free degrees
    of freedom, and :math:`L_d`  is the vector of active loads applied to the
    data degrees of freedom. The reactions :math:`R` due to supports act on the
    prescribed (data) degrees of freedom.

    The system of equations is solved for the reactions as

    .. math::
        R = K_{ff} \cdot U_{f} + K_{fd} \cdot U_{d} -L_{d} 
    
    once :math:`U_f` has been solved for in the :func:`solve_statics` step.

    The reactions are distributed to the joints, and can be retrieved from
    individual joint dictionaries ``j`` as ``j['reactions']``.

    Parameters
    ----------
    m
        The model.

    Returns
    -------
    None

    See Also
    --------
    :func:`solve_statics`
    """
    K = m["K"]
    U = m["U"]
    F = m["F"]

    # Compute reactions from the partitioned stiffness matrix and the
    # partitioned displacement vector
    # R = dot(K[nf:nt, 0:nf], U[0:nf]) + dot(K[nf:nt, nf:nt], U[nf:nt]) - F[nf:nt]
    # For convenience when working
    # with degrees of freedom, we compute this product and only use the rows
    # corresponding to fixed the degrees of freedom.
    R = dot(K, U) - F

    for joint in m["joints"].values():
        if "supports" in joint:
            reactions = {}
            for dof, _ in joint["supports"].items():
                gr = joint["dof"][dof]
                reactions[dof] = R[gr]
            joint["reactions"] = reactions


def solve_free_vibration(m):
    r"""
    Solve the free vibration of the discrete model.

    The free vibration eigenvalue problem is solved for the eigenvalues and
    eigenvectors (can be retrieved as  ``m["eigvals"]`` and ``m["eigvecs"]``). The
    frequencies are computed from the eigenvalues (can be retrieved as
    ``m["frequencies"]``).

    The equation of free vibration is

    .. math::
        K \cdot V = \omega^2 M \cdot V

    where :math:`M` is the mass matrix, :math:`V` is the eigenvector, and
    :math:`\omega` is the angular frequency.

    :func:`number_dofs` must be called before this function.

    Parameters
    ----------
    m
        The model.

    Returns
    -------
    None

    See Also
    --------
    :func:`number_dofs`
    """
    nf = m["nfreedof"]
    # Assemble global stiffness matrix and mass matrix
    K = _build_stiffness_matrix(m)
    M = _build_mass_matrix(m)

    m["K"] = K
    m["M"] = M

    U = zeros(m["ntotaldof"])
    for joint in m["joints"].values():
        if "supports" in joint:
            for dof, _ in joint["supports"].items():
                gr = joint["dof"][dof]
                U[gr] = 0.0
    m["U"] = U

    # Solved the eigenvalue problem
    eigvals, eigvecs = scipy.linalg.eigh(K[0:nf, 0:nf], M[0:nf, 0:nf])

    m["eigvals"] = eigvals
    m["frequencies"] = [sqrt(ev) / 2 / numpy.pi for ev in eigvals]
    m["eigvecs"] = eigvecs

    return


def set_solution(m, V):
    """
    Set the displacement solution from a vector.

    Parameters
    ----------
    m
        The model.
    V
        The displacement vector. Either of length ``m["nfreedof"]`` for only the
        free degrees of freedom, or of length ``m["ntotaldof"]`` for the total
        number of degrees of freedom.

    Returns
    -------
    None

    See Also
    --------
    :func:`number_dofs`
    """
    nf = m["nfreedof"]
    nt = m["ntotaldof"]
    if len(V) == nf:
        m["U"][0:nf] = V
    elif len(V) == nt:
        m["U"][0:nt] = V
    else:
        raise RuntimeError("Invalid vector length")
    for joint in m["joints"].values():
        joint["displacements"] = m["U"][joint["dof"]]


def free_body_check(m):
    """
    Check the balance of the structure as a free body.

    All the active forces and moments together with the reactions at all the
    supports should sum to zero.

    :func:`statics_reactions` must be called before this function as this calculation
    relies on the presence of reactions at the joints.

    Parameters
    ----------
    m
        The model.

    Returns
    -------
    array
        Array of resultant forces and moments.

    See Also
    --------
    :func:`statics_reactions`
    """
    if m["dim"] == 2:
        nrbm = 3  # Number of rigid body modes: assume 2 translations, 1 rotation
        MZ = 2
        allforces = zeros(nrbm)
        for joint in m["joints"].values():
            c = joint["coordinates"]
            x, y = c[0], c[1]
            if "loads" in joint:
                for dof, value in joint["loads"].items():
                    if dof < MZ:
                        # Add contributions of forces to the moment
                        if dof == 0:
                            allforces[MZ] += -value * y
                        elif dof == 1:
                            allforces[MZ] += +value * x
                    else:
                        # Add contributions of forces and moments
                        allforces[dof] += value
            if "reactions" in joint:
                for dof, value in joint["reactions"].items():
                    if dof < MZ:
                        # Add contributions of forces to the moment
                        if dof == 0:
                            allforces[MZ] += -value * y
                        elif dof == 1:
                            allforces[MZ] += +value * x
                    else:
                        # Add contributions of forces and moments
                        allforces[dof] += value
        return allforces
    else:
        nrbm = 6  # Number of rigid body modes: assume 3 translations, 3 rotations
        MX, MY, MZ = 3, 4, 5
        allforces = zeros(nrbm)
        for joint in m["joints"].values():
            c = joint["coordinates"]
            x, y, z = c[0], c[1], c[2]
            if "loads" in joint:
                for dof, value in joint["loads"].items():
                    if dof < MX:
                        # Add contributions of forces to the moment
                        if dof == 0:
                            allforces[MY] += +value * z
                            allforces[MZ] += -value * y
                        elif dof == 1:
                            allforces[MX] += -value * z
                            allforces[MZ] += +value * x
                        else:
                            allforces[MY] += -value * x
                            allforces[MX] += +value * y
                    else:
                        # Add contributions of forces and moments
                        allforces[dof] += value
            if "reactions" in joint:
                for dof, value in joint["reactions"].items():
                    if dof < MX:
                        # Add contributions of forces to the moment
                        if dof == 0:
                            allforces[MY] += +value * z
                            allforces[MZ] += -value * y
                        elif dof == 1:
                            allforces[MX] += -value * z
                            allforces[MZ] += +value * x
                        else:
                            allforces[MY] += -value * x
                            allforces[MX] += +value * y
                    else:
                        # Add contributions of forces and moments
                        allforces[dof] += value
        return allforces


def refine_member(m, mid, n):
    """
    Refine a beam member by replacing it with ``n`` new members.

    The new joints are numbered starting from zero, and the joint identifier is
    composed of the member identifier plus the serial number of the new joint.

    The new member identifiers are stored under the key ``"descendants"`` in
    the refined member. The refined member is removed from the list of beam
    members.

    Parameters
    ----------
    m
        The model.
    mid
        The identifier of the member to be refined.
    n
        The number of new beam members to replace the old member with.

    Returns
    -------
    None
    """
    if n < 2:
        raise RuntimeError("Number of new members must be at least 2")
    member = m["beam_members"][mid]
    connectivity = member["connectivity"]
    i, j = m["joints"][connectivity[0]], m["joints"][connectivity[1]]
    ci, cj = i["coordinates"], j["coordinates"]
    # Store the descendants
    descendants = []
    # First replacement member
    start = -1.0 + 2.0 / n
    c = (-1 + start) / (-2) * ci + (1 + start) / 2 * cj
    newjid = str(mid) + "j" + "0"
    add_joint(m, newjid, c)
    newmid = str(mid) + "m" + "0"
    add_beam_member(m, newmid, [i["jid"], newjid], member["section"])
    descendants.append(newmid)
    prevjid = newjid
    for k in range(n - 2):
        start += 2.0 / n
        c = (-1 + start) / (-2) * ci + (1 + start) / 2 * cj
        newjid = str(mid) + "j" + str(k + 1)
        add_joint(m, newjid, c)
        newmid = str(mid) + "m" + str(k + 1)
        add_beam_member(m, newmid, [prevjid, newjid], member["section"])
        descendants.append(newmid)
        prevjid = newjid
    # Last replacement member
    newmid = str(mid) + "m" + str(n - 1)
    add_beam_member(m, newmid, [newjid, j["jid"]], member["section"])
    descendants.append(newmid)
    # Remember the provenance of the new members
    member["descendants"] = descendants
    # Remove the old member
    del m["beam_members"][mid]


def remove_loads(m):
    """
    Remove all the nodal loads in the model.

    Parameters
    ----------
    m
        The model.
    """
    for joint in m["joints"].values():
        if "loads" in joint:
            joint["loads"] = {}


def remove_supports(m):
    """
    Remove all the nodal supports in the model.

    Parameters
    ----------
    m
        The model.
    """
    for joint in m["joints"].values():
        if "supports" in joint:
            joint["supports"] = {}

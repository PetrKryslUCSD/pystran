"""
Define section dictionaries.

A section defines material properties, geometrical properties, such as the
second moment of area, and also orientation of the cross section profile.
"""

from math import pi
import numpy


def truss_section(name, E=0.0, A=0.0, rho=0.0, CTE=0.0):
    """
    Define truss section.

    The parameters can be defined either as regular list of values or
    variables, or in the ``keyword=value`` style.

    Parameters
    ----------
    E
        Young's modulus.
    A
        Cross-sectional area.
    rho
        Mass density.
    CTE
        Coefficient of thermal expansion.

    Returns
    -------
    Dict
        Dictionary that holds the data for this section.

    Examples
    --------
    >>> s1 = section.truss_section("steel", E, A)

    >>> sr = section.truss_section("sr", E=E, A=Ar, CTE=0.0)

    >>> sr = section.truss_section("sr", E=E, A=Ar)
    """
    s = dict()
    s["name"] = name
    s["E"] = E
    s["rho"] = rho
    s["CTE"] = CTE
    s["A"] = A
    return s


def beam_2d_section(name, E=0.0, A=0.0, I=0.0, rho=0.0, CTE=0.0):
    r"""
    Define 2d beam section.

    The parameters can be defined either as regular list of values or
    variables, or in the ``keyword=value`` style.

    Parameters
    ----------
    E
        Young's modulus.
    A
        Cross-sectional area.
    I
        Central moment of inertia of the cross-section about the :math:`y`
        coordinate axis (i.e. the axis perpendicular to the plane of the
        bending, :math:`x-z`).
    rho
        Mass density.
    CTE
        Coefficient of thermal expansion.

    Returns
    -------
    Dict
        Dictionary that holds the data for this section.

    Examples
    --------
    >>> sbar = section.beam_2d_section("sbar", E=E, rho=rho, A=A, I=Iy)
    """
    s = dict()
    s["name"] = name
    s["E"] = E
    s["rho"] = rho
    s["CTE"] = CTE
    s["A"] = A
    s["I"] = I
    return s


def rigid_link_section(name, Gamma):
    """
    Define a rigid link section.

    The parameters can be defined either as regular list of values or
    variables, or in the ``keyword=value`` style.

    Parameters
    ----------
    Gamma
        Diagonal matrix of penalty coefficients; zero coefficient means
        the degrees of freedom are not linked.

    Returns
    -------
    Dict
        Dictionary that holds the data for this section.

    Examples
    --------
    >>> sr = section.rigid_link_section("sr", Gamma=1e8 * numpy.diagflat([1.0, 1.0, 1.0]))
    """
    s = dict()
    s["name"] = name
    s["Gamma"] = Gamma
    return s


def spring_section(name, kind, direction, stiffness_coefficient=1.0):
    """
    Define a section for a general extension or torsion spring.

    The parameters can be defined either as regular list of values or
    variables, or in the ``keyword=value`` style.

    Parameters
    ----------
    kind
        Either ``"extension"`` or ``"torsion"``. The connected joints either react
        to displacement or to rotation.
    direction
        The spring acts *along* this direction for extension springs, or about
        this direction for torsion springs. Supply as a vector.
    stiffness_coefficient
        Stiffness coefficient of the spring.

    Returns
    -------
    Dict
        Dictionary that holds the data for this section.

    Examples
    --------
    >>> section.spring_section("EXT_A", "extension", [0, 1, 0], K),

    """
    s = dict()
    s["name"] = name
    if kind not in ["extension", "torsion"]:
        raise ValueError("kind must be either 'extension' or 'torsion'")
    s["kind"] = kind
    s["direction"] = direction
    s["stiffness_coefficient"] = stiffness_coefficient
    return s


def beam_3d_section(
    name,
    E=0.0,
    G=0.0,
    A=0.0,
    Ix=0.0,
    Iy=0.0,
    Iz=0.0,
    J=0.0,
    rho=0.0,
    xy_vector=None,
    xz_vector=None,
    CTE=0.0,
):
    r"""
    Define 3d beam section.

    The parameters can be defined either as regular list of values or
    variables, or in the ``keyword=value`` style.

    Parameters
    ----------
    E
        Young's modulus.
    G
        Shear modulus.
    A
        Cross-sectional area.
    rho
        Mass density.
    A
        Cross-sectional area.
    Ix
        Central moment of inertia of the cross-section about the local
        :math:`x`. The local :math:`x` axis is the axis along the beam member, 
        pointing from the first joint to the second joint.
    Iy
        Central moment of inertia of the cross-section about the local
        :math:`y`. The first axis in the cross section.
    Iz
        Central moment of inertia of the cross-section about the local
        :math:`z`. The second axis in the cross section, perpendicular to the :math:`y` axis.
    J
        St Venant torsion constant.
    xy_vector
        Vector that lies in the local :math:`x-y` coordinate plane. Default `None`.
        Only one of the two vectors, `xy_vector` and `xz_vector`, can be supplied. 
    xz_vector
        Vector that lies in the local :math:`x-z` coordinate plane. Default `None`.
        Only one of the two vectors, `xy_vector` and `xz_vector`, can be supplied. 
    CTE
        Coefficient of thermal expansion.

    Returns
    -------
    Dict
        Dictionary that holds the data for this section.

    Examples
    --------
    >>> sb = section.beam_3d_section("sb", E=E, G=G, A=A, Ix=Ix, Iy=Iy, Iz=Iz, J=J)

    """
    s = dict()
    s["name"] = name
    s["E"] = E
    s["G"] = G
    s["rho"] = rho
    s["CTE"] = CTE
    s["A"] = A
    s["Ix"] = Ix
    s["Iy"] = Iy
    s["Iz"] = Iz
    s["J"] = J
    if xy_vector is not None and xz_vector is not None:
        raise ValueError("Only one of xy_vector and xz_vector can be supplied.")
    if xy_vector is not None:
        xy_vector = numpy.array(xy_vector)
    s["xy_vector"] = xy_vector
    if xz_vector is not None:
        xz_vector = numpy.array(xz_vector)
    s["xz_vector"] = xz_vector
    return s


def circular_tube(innerradius, outerradius):
    """
    Calculate cross section characteristics for a hollow circle (tube).

    Parameters
    ----------
    innerradius
        Inner radius of the tube.
    outerradius
        Outer radius of the tube.

    Returns
    -------
    tuple of A, Ix, Iy, Iz, J
        Area, moments of inertia and torsion constant.
    """
    Rext = outerradius
    Rint = innerradius
    A = pi * (Rext**2 - Rint**2)
    Iy = pi / 4 * (Rext**4 - Rint**4)
    Iz = pi / 4 * (Rext**4 - Rint**4)
    Ix = Iy + Iz
    J = pi / 2 * (Rext**4 - Rint**4)
    return A, Ix, Iy, Iz, J


def i_beam(H, B, tf, tw):
    """
    Calculate cross section characteristics for an I-beam.

    Parameters
    ----------
    H
        Height of the cross section, i.e. dimension along z. The axis parallel
        to the flanges is :math:`y`, the axis parallel to the web is :math:`z`.
    B
        Width of the flanges.
    tf
        Thickness of the flanges.
    tw
        Thickness of the web.

    Returns
    -------
    tuple of A, Ix, Iy, Iz, J
        Area, moments of inertia and torsion constant.
    """
    A = B * H - (B - tw) * (H - 2 * tf)
    Iy = (B / 12) * H**3 - ((B - tw) / 12) * (H - 2 * tf) ** 3
    Iz = (
        H * B**3 / 12
        - 2 * ((B - tw) / 2) ** 3 * (H - 2 * tf) / 12
        - 2 * ((B - tw) / 2) * (H - 2 * tf) * ((B - tw) / 4 + tw / 2) ** 2
    )
    Ix = Iy + Iz
    J = (2 * B * tf**3 + (H - 2 * tf) * tw**3) / 3
    return A, Ix, Iy, Iz, J


def rect_tube(H, B, th, tb):
    """
    Calculate cross section characteristics for an rectangular tube.

    Parameters
    ----------
    H
        Height of the cross section, i.e. dimension along :math:`z`.
    B
        Width of the cross section, i.e. dimension along :math:`y`.
    th
        Thickness of the walls along the height.
    tb
        Thickness of the walls along the width.

    Returns
    -------
    tuple of A, Ix, Iy, Iz, J
        Area, moments of inertia and torsion constant.
    """
    Bi, Hi = (B - 2 * tb), (H - 2 * th)
    A = B * H - Bi * Hi
    Iy = (B / 12) * H**3 - (Bi / 12) * Hi**3
    Iz = (B**3 / 12) * H - (Bi**3 / 12) * Hi
    Ix = Iy + Iz
    J = 2 * tb * th * Hi**2 * Bi**2 / (H * tb + B * th - tb**2 - th**2)
    return A, Ix, Iy, Iz, J


def rectangle(H, B):
    """
    Calculate cross section characteristics for a solid rectangle.

    Parameters
    ----------
    H
        Height of the cross section, i.e. dimension along :math:`z`.
    B
        Width of the cross section, i.e. dimension along :math:`y`.

    Returns
    -------
    tuple of A, Ix, Iy, Iz, J
        Area, moments of inertia and torsion constant.
    """
    a = max(H, B)
    b = min(H, B)
    A = B * H
    Iy = (B / 12) * H**3
    Iz = (B**3 / 12) * H
    Ix = Iy + Iz
    rs = numpy.array([1, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 10, 20, 40, 80, 200, 2000])
    coeff = numpy.array(
        [
            0.141,
            0.196,
            0.229,
            0.249,
            0.263,
            0.281,
            0.291,
            0.299,
            0.312,
            0.317,
            0.325,
            0.33,
            1 / 3,
            1 / 3,
        ]
    )
    c = numpy.interp(a / b, rs, coeff, coeff[0], coeff[-1])
    J = c * a * b**3
    return A, Ix, Iy, Iz, J

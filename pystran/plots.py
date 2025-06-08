"""
Implement simple plots for truss and beam structures.
"""

from math import sqrt
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import Arc, RegularPolygon
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy
from numpy import linspace, dot, zeros
from numpy import radians as rad
from numpy.linalg import norm
from pystran.model import ndof_per_joint, characteristic_dimension, bounding_box
from pystran.truss import (
    truss_axial_force,
)
from pystran.beam import (
    beam_2d_moment,
    beam_2d_shear_force,
    beam_2d_axial_force,
)
from pystran.beam import (
    beam_3d_xz_shape_fun,
    beam_3d_xy_shape_fun,
    beam_3d_moment,
    beam_3d_shear_force,
    beam_3d_torsion_moment,
    beam_3d_axial_force,
)
from pystran.geometry import (
    member_2d_geometry,
    member_3d_geometry,
    herm_basis,
    interpolate,
)

_myeps = numpy.finfo(float).eps

forc_mom_bbox_prop = dict(facecolor='white', boxstyle='round,pad=0.5')

# fig = plt.figure(figsize=(9,9))
# ax = plt.gca()
def _drawcirc(ax, radius, centX, centY, angle_, theta2_, sense, color_="black"):
    "A little circle helper"
    # ========Line
    arc = Arc(
        [centX, centY],
        radius,
        radius,
        angle=angle_,
        theta1=0,
        theta2=theta2_,
        capstyle="round",
        linestyle="-",
        lw=2,
        color=color_,
    )
    ax.add_patch(arc)

    # ========Create the arrow head
    if sense > 0:
        endX = centX + (radius / 2) * numpy.cos(
            rad(theta2_ + angle_)
        )  # Do trig to determine end position
        endY = centY + (radius / 2) * numpy.sin(rad(theta2_ + angle_))
    else:
        endX = centX + (radius / 2) * numpy.cos(
            rad(angle_)
        )  # Do trig to determine end position
        endY = centY + (radius / 2) * numpy.sin(rad(angle_))

    ax.add_patch(  # Create triangle as arrow head
        RegularPolygon(
            (endX, endY),  # (x,y)
            3,  # number of vertices
            radius=radius / 9,  # radius
            orientation=rad(angle_ + theta2_),  # orientation
            color=color_,
        )
    )
    # ax.set_xlim([centX - radius, centY + radius]) and ax.set_ylim(
    #     [centY - radius, centY + radius]
    # )
    # Make sure you keep the axes scaled or else arrow will distort


# _drawcirc(ax,1,1,1,0,250)
# _drawcirc(ax,2,1,1,90,330,color_='blue')
# plt.show()


# From: https://gist.github.com/WetHat/1d6cd0f7309535311a539b42cccca89c
class _Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)
        xs, ys, _ = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)
        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return numpy.min(zs)


def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    """Add an 3d arrow to an `Axes3D` instance."""

    arrow = _Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)


setattr(Axes3D, "arrow3D", _arrow3D)


def setup(m, set_limits=False):
    """
    Setup the plot.

    Parameters
    ----------
    m
        Model dictionary.
    set_limits
        Optional: set the limits of the graphics manually or not? Default is
        ``False``. If ``True``, the bounding box of the structure and its
        characteristic dimension are used to calculate the limits of the plot.

    Returns
    -------
    axes
        This function creates a figure and an axis object. The axes are
        returned.
    """
    fig = plt.figure()
    if m["dim"] == 3:
        ax = fig.add_subplot(projection="3d")
    else:
        ax = fig.gca()
        if set_limits:
            box = bounding_box(m)
            cd = characteristic_dimension(m)
            ax.set_xlim([box[0] - cd / 10, box[2] + cd / 10])
            ax.set_ylim([box[1] - cd / 10, box[3] + cd / 10])
    return ax


def _area_extrema(all_members):
    min_area = numpy.inf
    max_area = 0.0
    for members in all_members:
        for member in members:
            A = member["section"]["A"]
            min_area = min(A, min_area)
            max_area = max(A, max_area)
    return min_area, max_area


def _linewidth(a, min_area, max_area, minlw, maxlw):
    _a = sqrt(a)
    _mna = sqrt(min_area)
    _mxa = sqrt(max_area)
    if _mna >= _mxa:
        return minlw
    else:
        return (_a - _mna) / (_mxa - _mna) * maxlw + (_a - _mxa) / (_mna - _mxa) * minlw


def plot_members(m, max_area=0.0, max_linewidth=2, min_area=0.0, min_linewidth=2):
    """
    Plot the members of the structure.

    Parameters
    ----------
    m
        Model dictionary.
    max_area
        For the purposes of mapping the cross sectional area to the thickness,
        consider this to be the maximum. Default is 0.0 (which means it should be
        computed internally).
    max_linewidth
        Use this as the maximum line width corresponding to the maximum area
        (default 2).
    min_area
        For the purposes of mapping the cross sectional area to the thickness,
        consider this to be the minimum. Default is 0.0.
    min_linewidth
        Use this as the minimum line width corresponding to the minimum area
        (default 2).

    All truss, rigid link, and beam members will be included.
    """
    all_members = [m[k].values() for k in ["truss_members", "beam_members"] if k in m]
    if max_area == 0.0:
        _, max_area = _area_extrema(all_members)
    area = lambda member: member["section"]["A"]
    minlw, maxlw = min_linewidth, max_linewidth

    def lw(member):
        return _linewidth(area(member), min_area, max_area, minlw, maxlw)

    ax = plt.gca()
    if m["dim"] == 3:
        for members in all_members:
            for member in members:
                connectivity = member["connectivity"]
                i, j = m["joints"][connectivity[0]], m["joints"][connectivity[1]]
                ci, cj = i["coordinates"], j["coordinates"]
                plt.plot(
                    [ci[0], cj[0]],
                    [ci[1], cj[1]],
                    [ci[2], cj[2]],
                    "k-",
                    linewidth=lw(member),
                )
        if "rigid_link_members" in m:
            for member in m["rigid_link_members"].values():
                connectivity = member["connectivity"]
                i, j = m["joints"][connectivity[0]], m["joints"][connectivity[1]]
                ci, cj = i["coordinates"], j["coordinates"]
                plt.plot(
                    [ci[0], cj[0]], [ci[1], cj[1]], [ci[2], cj[2]], "k-", linewidth=3
                )
    else:
        for members in all_members:
            for member in members:
                connectivity = member["connectivity"]
                i, j = m["joints"][connectivity[0]], m["joints"][connectivity[1]]
                ci, cj = i["coordinates"], j["coordinates"]
                plt.plot([ci[0], cj[0]], [ci[1], cj[1]], "k-", linewidth=lw(member))
        if "rigid_link_members" in m:
            for member in m["rigid_link_members"].values():
                connectivity = member["connectivity"]
                i, j = m["joints"][connectivity[0]], m["joints"][connectivity[1]]
                ci, cj = i["coordinates"], j["coordinates"]
                plt.plot([ci[0], cj[0]], [ci[1], cj[1]], "k-", linewidth=3)
    return ax


def _plot_truss_or_rigid_link_deflection(ax, i, j, scale):
    di, dj = i["displacements"], j["displacements"]
    ci, cj = i["coordinates"], j["coordinates"]
    if len(ci) == 3:
        ax.plot(
            [ci[0] + scale * di[0], cj[0] + scale * dj[0]],
            [ci[1] + scale * di[1], cj[1] + scale * dj[1]],
            [ci[2] + scale * di[2], cj[2] + scale * dj[2]],
            "m-",
        )
    else:
        ax.plot(
            [ci[0] + scale * di[0], cj[0] + scale * dj[0]],
            [ci[1] + scale * di[1], cj[1] + scale * dj[1]],
            "m-",
        )


def _plot_2d_beam_deflection(ax, i, j, scale):
    di, dj = i["displacements"], j["displacements"]
    ci, cj = i["coordinates"], j["coordinates"]
    e_x, e_z, h = member_2d_geometry(i, j)
    ui = dot(di[0:2], e_x)
    uj = dot(dj[0:2], e_x)
    wi = dot(di[0:2], e_z)
    thi = di[2]
    wj = dot(dj[0:2], e_z)
    thj = dj[2]
    n = 20
    xs = zeros(n)
    ys = zeros(n)
    for s, xi in enumerate(linspace(-1, +1, n)):
        u = interpolate(xi, ui, uj)
        N = herm_basis(xi)
        w = N[0] * wi + (h / 2) * N[1] * thi + N[2] * wj + (h / 2) * N[3] * thj
        x = interpolate(xi, ci, cj)
        xs[s] = x[0]
        ys[s] = x[1]
        xs[s] += scale * u * e_x[0]
        ys[s] += scale * u * e_x[1]
        xs[s] += scale * w * e_z[0]
        ys[s] += scale * w * e_z[1]
    ax.plot(xs, ys, "m-")


def _plot_3d_beam_deflection(ax, member, i, j, scale):
    sect = member["section"]
    di, dj = i["displacements"], j["displacements"]
    ci, cj = i["coordinates"], j["coordinates"]
    e_x, e_y, e_z, h = member_3d_geometry(i, j, sect["xz_vector"])
    ui = dot(di[0:3], e_x)
    uj = dot(dj[0:3], e_x)
    wi = dot(di[0:3], e_z)
    thyi = dot(di[3:6], e_y)
    wj = dot(dj[0:3], e_z)
    thyj = dot(dj[3:6], e_y)
    vi = dot(di[0:3], e_y)
    thzi = dot(di[3:6], e_z)
    vj = dot(dj[0:3], e_y)
    thzj = dot(dj[3:6], e_z)
    n = 20
    xs = zeros(n)
    ys = zeros(n)
    zs = zeros(n)
    for s, xi in enumerate(linspace(-1, +1, n)):
        x = interpolate(xi, ci, cj)
        u = interpolate(xi, ui, uj)
        xs[s] = x[0]
        ys[s] = x[1]
        zs[s] = x[2]
        xs[s] += scale * u * e_x[0]
        ys[s] += scale * u * e_x[1]
        zs[s] += scale * u * e_x[2]
        N = beam_3d_xz_shape_fun(xi)
        w = N[0] * wi + (h / 2) * N[1] * thyi + N[2] * wj + (h / 2) * N[3] * thyj
        xs[s] += scale * w * e_z[0]
        ys[s] += scale * w * e_z[1]
        zs[s] += scale * w * e_z[2]
        N = beam_3d_xy_shape_fun(xi)
        v = N[0] * vi + (h / 2) * N[1] * thzi + N[2] * vj + (h / 2) * N[3] * thzj
        xs[s] += scale * v * e_y[0]
        ys[s] += scale * v * e_y[1]
        zs[s] += scale * v * e_y[2]
    ax.plot(xs, ys, zs, "m-")


def plot_deformations(m, scale=0.0):
    """
    Plot the deformation of the structure.

    Parameters
    ----------
    m
        Model dictionary.
    scale
        Optional: scale factor for the deformation. Default is 0.0, which
        means compute this internally.

    All truss, rigid links, and beam members will be included. Truss members
    and rigid links will be displayed as straight; beam members will be
    displayed using the cubic shape functions.
    """
    cd = characteristic_dimension(m)

    def fun(j):
        if "displacements" in j:
            return j["displacements"]
        else:
            return [0]

    if scale == 0.0:
        maxmag = _largest_mag_at_joints(m, fun)
        if maxmag != 0:
            scale = cd / 5 / maxmag

    ax = plt.gca()
    if "truss_members" in m:
        for member in m["truss_members"].values():
            connectivity = member["connectivity"]
            i, j = m["joints"][connectivity[0]], m["joints"][connectivity[1]]
            _plot_truss_or_rigid_link_deflection(ax, i, j, scale)
    if "rigid_link_members" in m:
        for member in m["rigid_link_members"].values():
            connectivity = member["connectivity"]
            i, j = m["joints"][connectivity[0]], m["joints"][connectivity[1]]
            _plot_truss_or_rigid_link_deflection(ax, i, j, scale)
    if "beam_members" in m:
        for member in m["beam_members"].values():
            connectivity = member["connectivity"]
            i, j = m["joints"][connectivity[0]], m["joints"][connectivity[1]]
            if m["dim"] == 3:
                _plot_3d_beam_deflection(ax, member, i, j, scale)
            else:
                _plot_2d_beam_deflection(ax, i, j, scale)
    return ax


def _plot_member_ids_2d(m):
    ax = plt.gca()
    if "truss_members" in m:
        for jid in m["truss_members"].keys():
            member = m["truss_members"][jid]
            connectivity = member["connectivity"]
            i, j = m["joints"][connectivity[0]], m["joints"][connectivity[1]]
            ci, cj = i["coordinates"], j["coordinates"]
            xm = (ci + cj) / 2.0
            ax.text(xm[0], xm[1], str(jid),
                    bbox=dict(facecolor='white', boxstyle='square'))
    if "beam_members" in m:
        for jid in m["beam_members"].keys():
            member = m["beam_members"][jid]
            connectivity = member["connectivity"]
            i, j = m["joints"][connectivity[0]], m["joints"][connectivity[1]]
            ci, cj = i["coordinates"], j["coordinates"]
            xm = (ci + cj) / 2.0
            ax.text(xm[0], xm[1], str(jid),
                    bbox=dict(facecolor='white', boxstyle='square'))
    return ax


def _plot_member_ids_3d(m):
    ax = plt.gca()
    if "truss_members" in m:
        for jid in m["truss_members"].keys():
            member = m["truss_members"][jid]
            connectivity = member["connectivity"]
            i, j = m["joints"][connectivity[0]], m["joints"][connectivity[1]]
            ci, cj = i["coordinates"], j["coordinates"]
            xm = (ci + cj) / 2.0
            ax.text(xm[0], xm[1], xm[2], str(jid), 
                    bbox=dict(facecolor='white', boxstyle='square'))
    if "beam_members" in m:
        for jid in m["beam_members"].keys():
            member = m["beam_members"][jid]
            connectivity = member["connectivity"]
            i, j = m["joints"][connectivity[0]], m["joints"][connectivity[1]]
            ci, cj = i["coordinates"], j["coordinates"]
            xm = (ci + cj) / 2.0
            ax.text(xm[0], xm[1], xm[2], str(jid), 
                    bbox=dict(facecolor='white', boxstyle='square'))
    return ax


def plot_member_ids(m):
    """
    Plot the member identifiers.

    Parameters
    ----------
    m
        Model dictionary.
    """
    if m["dim"] == 3:
        ax = _plot_member_ids_3d(m)
    else:
        ax = _plot_member_ids_2d(m)
    return ax


def plot_joint_ids(m, offsets = []):
    """
    Plot the joint identifiers.

    Parameters
    ----------
    m
        Model dictionary.
        
    offsets
        Optional: vector by which joint ID numbers should be offset 
        from the location of the joint. Default is
        an empty list, which means the offsets will be zero.
    """
    if len(offsets) == 0:
        offsets = [0, 0, 0]
    ax = plt.gca()
    for j in m["joints"].values():
        if m["dim"] == 3:
            ax.plot(j["coordinates"][0], j["coordinates"][1], j["coordinates"][2], "ro")
            ax.text(
                j["coordinates"][0] + offsets[0],
                j["coordinates"][1] + offsets[1],
                j["coordinates"][2] + offsets[2],
                str(j["jid"]),
                bbox=dict(facecolor='white', boxstyle='circle')
            )
        else:
            ax.plot(j["coordinates"][0], j["coordinates"][1], "ro")
            ax.text(j["coordinates"][0] + offsets[0], 
                    j["coordinates"][1] + offsets[1], 
                    str(j["jid"]),
                    bbox=dict(facecolor='white', boxstyle='circle')
                    )
    return ax


def _largest_mag_on_beam_members(m, fun):
    maxmag = 0.0
    if "beam_members" in m:
        for member in m["beam_members"].values():
            connectivity = member["connectivity"]
            i, j = m["joints"][connectivity[0]], m["joints"][connectivity[1]]
            for xi in linspace(-1, +1, 2):
                mmag = abs(fun(member, i, j, xi))
                maxmag = max(maxmag, mmag)
    return maxmag


def _largest_mag_on_truss_members(m, fun):
    maxmag = 0.0
    if "truss_members" in m:
        for member in m["truss_members"].values():
            connectivity = member["connectivity"]
            i, j = m["joints"][connectivity[0]], m["joints"][connectivity[1]]
            mmag = abs(fun(member, i, j))
            maxmag = max(maxmag, mmag)
    return maxmag


def _plot_2d_beam_moments(ax, member, i, j, scale, nearly_zero = 1000 * _myeps):
    _, e_z, _ = member_2d_geometry(i, j)
    ci, cj = i["coordinates"], j["coordinates"]
    n = 13
    for _, xi in enumerate(linspace(-1, +1, n)):
        M = beam_2d_moment(member, i, j, xi)
        x = interpolate(xi, ci, cj)
        # The convention: moment is plotted next to fibers in tension
        xs = zeros(2)
        ys = zeros(2)
        xs[0] = x[0]
        xs[1] = x[0] + scale * M * e_z[0]
        ys[0] = x[1]
        ys[1] = x[1] + scale * M * e_z[1]
        ax.plot(xs, ys, "r-" if (M > 0) else "b-")
        if abs(M) < nearly_zero / scale: M[0] = 0.0
        if xi == -1.0:
            ax.text(xs[1], ys[1], str(f"{M[0]:.5}"), bbox=forc_mom_bbox_prop)
        elif xi == +1.0:
            ax.text(xs[1], ys[1], str(f"{M[0]:.5}"), bbox=forc_mom_bbox_prop)
    return ax


def _plot_3d_beam_moments(ax, member, i, j, axis, scale, nearly_zero = 1000 * _myeps):
    sect = member["section"]
    _, e_y, e_z, _ = member_3d_geometry(i, j, sect["xz_vector"])
    ci, cj = i["coordinates"], j["coordinates"]
    n = 13
    # The moments are plotted so that they are adjacent to fibers in tension.
    dirv = -e_y  # y<0 are in tension for M>0
    if axis == "y":
        dirv = +e_z  # z>0 are in tension for M>0
    for _, xi in enumerate(linspace(-1, +1, n)):
        M = beam_3d_moment(member, i, j, axis, xi)
        x = interpolate(xi, ci, cj)
        xs = zeros(2)
        ys = zeros(2)
        zs = zeros(2)
        xs[0] = x[0]
        xs[1] = xs[0] + scale * M * dirv[0]
        ys[0] = x[1]
        ys[1] = ys[0] + scale * M * dirv[1]
        zs[0] = x[2]
        zs[1] = zs[0] + scale * M * dirv[2]
        ax.plot(xs, ys, zs, "r-" if (M > 0) else "b-")
        if abs(M) < nearly_zero / scale: M[0] = 0.0
        if xi == -1.0:
            ax.text(xs[1], ys[1], zs[1], str(f"{M[0]:.5}"), 
                    bbox=forc_mom_bbox_prop)
        elif xi == +1.0:
            ax.text(xs[1], ys[1], zs[1], str(f"{M[0]:.5}"), 
                    bbox=forc_mom_bbox_prop)
    return ax


def plot_bending_moments(m, axis="y", scale=0.0):
    """
    Plot the bending moments in the beam members.

    Parameters
    ----------
    m
        Model dictionary.
    axis
        Either "y" or "z" (default is "z", which is suitable for 2d beams).
    scale
        Optional: scale factor for the ordinate. Default is
        0.0, which means the scale will be computed internally.
    """

    def fun(member, i, j, xi):
        if m["dim"] == 3:
            return beam_3d_moment(member, i, j, axis, xi)
        else:
            return beam_2d_moment(member, i, j, xi)

    if scale == 0.0:
        cd = characteristic_dimension(m)
        maxmag = _largest_mag_on_beam_members(m, fun)
        if maxmag != 0:
            scale = cd / 5 / maxmag

    ax = plt.gca()
    for member in m["beam_members"].values():
        connectivity = member["connectivity"]
        i, j = m["joints"][connectivity[0]], m["joints"][connectivity[1]]
        if m["dim"] == 3:
            _plot_3d_beam_moments(ax, member, i, j, axis, scale)
        else:
            _plot_2d_beam_moments(ax, member, i, j, scale)
    if m["dim"] == 3:
        ax.set_xticklabels([]) 
        ax.set_yticklabels([]) 
        ax.set_zticklabels([]) 
    else:
        ax.set_xticklabels([]) 
        ax.set_yticklabels([]) 
    return ax 


def _plot_2d_beam_shear_forces(ax, member, i, j, scale, nearly_zero = 1000 * _myeps):
    _, e_z, _ = member_2d_geometry(i, j)
    ci, cj = i["coordinates"], j["coordinates"]
    n = 13
    for _, xi in enumerate(linspace(-1, +1, n)):
        Q = beam_2d_shear_force(member, i, j, 0.0)
        x = interpolate(xi, ci, cj)
        xs = zeros(2)
        ys = zeros(2)
        xs[0] = x[0]
        xs[1] = x[0] + scale * Q * e_z[0]
        ys[0] = x[1]
        ys[1] = x[1] + scale * Q * e_z[1]
        ax.plot(xs, ys, "r-" if (Q > 0) else "b-")
        if abs(Q) < nearly_zero / scale: Q[0] = 0.0
        if xi == 0.0:
            ax.text(xs[1], ys[1], str(f"{Q[0]:.5}"), bbox=forc_mom_bbox_prop)
    return ax


def _plot_3d_beam_shear_forces(ax, member, i, j, axis, scale, nearly_zero = 1000 * _myeps):
    sect = member["section"]
    _, e_y, e_z, _ = member_3d_geometry(i, j, sect["xz_vector"])
    ci, cj = i["coordinates"], j["coordinates"]
    n = 13
    dirv = e_z
    if axis == "y":
        dirv = e_y
    for _, xi in enumerate(linspace(-1, +1, n)):
        Q = beam_3d_shear_force(member, i, j, axis, xi)
        x = interpolate(xi, ci, cj)
        xs = zeros(2)
        ys = zeros(2)
        zs = zeros(2)
        xs[0] = x[0]
        xs[1] = xs[0] + scale * Q * dirv[0]
        ys[0] = x[1]
        ys[1] = ys[0] + scale * Q * dirv[1]
        zs[0] = x[2]
        zs[1] = zs[0] + scale * Q * dirv[2]
        ax.plot(xs, ys, zs, "r-" if (Q > 0) else "b-")
        if abs(Q) < nearly_zero / scale: Q[0] = 0.0
        if xi == -1.0:
            ax.text(xs[1], ys[1], zs[1], str(f"{Q[0]:.5}"), 
                    bbox=forc_mom_bbox_prop)
        elif xi == +1.0:
            ax.text(xs[1], ys[1], zs[1], str(f"{Q[0]:.5}"),
                    bbox=forc_mom_bbox_prop)
    return ax


def plot_shear_forces(m, axis="z", scale=0.0):
    """
    Plot the shear forces in the beam members.

    Parameters
    ----------
    m
        Model dictionary.
    axis
        Either "y" or "z" (default is "z", which is suitable for 2d beams).
    scale
        Optional: scale factor for the ordinate. Default is
        0.0, which means the scale will be computed internally.
    """

    def fun(member, i, j, _):
        if m["dim"] == 3:
            return beam_3d_shear_force(member, i, j, axis, 0.0)
        else:
            return beam_2d_shear_force(member, i, j, 0.0)

    if scale == 0.0:
        cd = characteristic_dimension(m)
        maxmag = _largest_mag_on_beam_members(m, fun)
        scale = cd / 5 / maxmag

    ax = plt.gca()
    for member in m["beam_members"].values():
        connectivity = member["connectivity"]
        i, j = m["joints"][connectivity[0]], m["joints"][connectivity[1]]
        if m["dim"] == 3:
            _plot_3d_beam_shear_forces(ax, member, i, j, axis, scale)
        else:
            _plot_2d_beam_shear_forces(ax, member, i, j, scale)
    if m["dim"] == 3:
        ax.set_xticklabels([]) 
        ax.set_yticklabels([]) 
        ax.set_zticklabels([]) 
    else:
        ax.set_xticklabels([]) 
        ax.set_yticklabels([]) 
    return ax


def _plot_2d_beam_axial_forces(ax, member, i, j, scale):
    _, e_z, _ = member_2d_geometry(i, j)
    ci, cj = i["coordinates"], j["coordinates"]
    n = 13
    for _, xi in enumerate(linspace(-1, +1, n)):
        N = beam_2d_axial_force(member, i, j, 0.0)
        x = interpolate(xi, ci, cj)
        xs = zeros(2)
        ys = zeros(2)
        xs[0] = x[0]
        xs[1] = x[0] + scale * N * e_z[0]
        ys[0] = x[1]
        ys[1] = x[1] + scale * N * e_z[1]
        ax.plot(xs, ys, "r-" if (N > 0) else "b-")
        if xi == 0.0:
            ax.text(xs[1], ys[1], str(f"{N[0]:.5}"),
                    bbox=forc_mom_bbox_prop)
    return ax


def _plot_2d_truss_axial_forces(ax, member, i, j, scale):
    _, e_z, _ = member_2d_geometry(i, j)
    ci, cj = i["coordinates"], j["coordinates"]
    N = truss_axial_force(member, i, j, 0.0)
    n = 13
    for _, xi in enumerate(linspace(-1, +1, n)):
        x = interpolate(xi, ci, cj)
        xs = zeros(2)
        ys = zeros(2)
        xs[0] = x[0]
        xs[1] = x[0] + scale * N * e_z[0]
        ys[0] = x[1]
        ys[1] = x[1] + scale * N * e_z[1]
        ax.plot(xs, ys, "r-" if (N > 0) else "b-")
        if xi == 0.0:
            ax.text(xs[1], ys[1], str(f"{N[0]:.5}"), 
                    bbox=forc_mom_bbox_prop)
    return ax


def _plot_3d_truss_beam_axial_forces(ax, member, i, j, scale):
    sect = member["section"]
    if 'xz_vector' in sect:
        xz_vector = sect["xz_vector"]
    else:
        xz_vector = []
    _, _, e_z, _ = member_3d_geometry(i, j, xz_vector)
    ci, cj = i["coordinates"], j["coordinates"]
    n = 13
    dirv = e_z
    for _, xi in enumerate(linspace(-1, +1, n)):
        N = beam_3d_axial_force(member, i, j, 0.0)
        x = interpolate(xi, ci, cj)
        xs = zeros(2)
        ys = zeros(2)
        zs = zeros(2)
        xs[0] = x[0]
        xs[1] = xs[0] + scale * N * dirv[0]
        ys[0] = x[1]
        ys[1] = ys[0] + scale * N * dirv[1]
        zs[0] = x[2]
        zs[1] = zs[0] + scale * N * dirv[2]
        ax.plot(xs, ys, zs, "r-" if (N > 0) else "b-")
        if xi == 0.0:
            ax.text(xs[1], ys[1], zs[1], str(f"{N[0]:.5}"),
                    bbox=forc_mom_bbox_prop)
    return ax


def plot_axial_forces(m, scale=0.0):
    """
    Plot the axial forces in the members.

    Parameters
    ----------
    m
        Model dictionary.
    scale
        Optional: scale factor for the ordinate. Default is
        0.0, which means the scale will be computed internally.

    """

    def funb(member, i, j, _):
        if m["dim"] == 3:
            return beam_3d_axial_force(member, i, j, 0.0)
        else:
            return beam_2d_axial_force(member, i, j, 0.0)

    def funt(member, i, j):
        return truss_axial_force(member, i, j, 0.0)

    if scale == 0.0:
        cd = characteristic_dimension(m)
        maxmag = max(
            _largest_mag_on_beam_members(m, funb),
            _largest_mag_on_truss_members(m, funt),
        )
        if maxmag != 0.0:
            scale = cd / 5 / maxmag

    ax = plt.gca()
    if "truss_members" in m:
        for member in m["truss_members"].values():
            connectivity = member["connectivity"]
            i, j = m["joints"][connectivity[0]], m["joints"][connectivity[1]]
            if m["dim"] == 3:
                _plot_3d_truss_beam_axial_forces(ax, member, i, j, scale)
            else:
                _plot_2d_truss_axial_forces(ax, member, i, j, scale)
    if "beam_members" in m:
        for member in m["beam_members"].values():
            connectivity = member["connectivity"]
            i, j = m["joints"][connectivity[0]], m["joints"][connectivity[1]]
            if m["dim"] == 3:
                _plot_3d_truss_beam_axial_forces(ax, member, i, j, scale)
            else:
                _plot_2d_beam_axial_forces(ax, member, i, j, scale)
    if m["dim"] == 3:
        ax.set_xticklabels([]) 
        ax.set_yticklabels([]) 
        ax.set_zticklabels([]) 
    else:
        ax.set_xticklabels([]) 
        ax.set_yticklabels([]) 
    return ax


def _plot_3d_beam_torsion_moments(ax, member, i, j, scale):
    sect = member["section"]
    _, _, e_z, _ = member_3d_geometry(i, j, sect["xz_vector"])
    ci, cj = i["coordinates"], j["coordinates"]
    n = 13
    dirv = e_z
    for _, xi in enumerate(linspace(-1, +1, n)):
        T = beam_3d_torsion_moment(member, i, j, 0.0)
        x = interpolate(xi, ci, cj)
        xs = zeros(2)
        ys = zeros(2)
        zs = zeros(2)
        xs[0] = x[0]
        xs[1] = xs[0] + scale * T * dirv[0]
        ys[0] = x[1]
        ys[1] = ys[0] + scale * T * dirv[1]
        zs[0] = x[2]
        zs[1] = zs[0] + scale * T * dirv[2]
        ax.plot(xs, ys, zs, "r-" if (T > 0) else "b-")
        if xi == 0.0:
            ax.text(xs[1], ys[1], zs[1], str(f"{T[0]:.5}"), 
                    bbox=forc_mom_bbox_prop)
    return ax


def plot_torsion_moments(m, scale=0.0):
    """
    Plot the torsion moments in the 3D beam members.

    Parameters
    ----------
    m
        Model dictionary.
    scale
        Optional: scale factor for the ordinate. Default is
        0.0, which means the scale will be computed internally.

    """

    def fun(member, i, j, _):
        if m["dim"] == 3:
            return beam_3d_torsion_moment(member, i, j, 0.0)
        else:
            return 0.0

    if scale == 0.0:
        cd = characteristic_dimension(m)
        maxmag = _largest_mag_on_beam_members(m, fun)
        scale = cd / 5 / maxmag

    ax = plt.gca()
    for member in m["beam_members"].values():
        connectivity = member["connectivity"]
        i, j = m["joints"][connectivity[0]], m["joints"][connectivity[1]]
        if m["dim"] == 3:
            _plot_3d_beam_torsion_moments(ax, member, i, j, scale)
    ax.set_xticklabels([]) 
    ax.set_yticklabels([]) 
    ax.set_zticklabels([]) 
    return ax


def plot_member_orientation(m, scale=0.0):
    """
    Plot the member orientations as cartesian triplets.

    The vectors are shown as red (x), green (y), and blue (z) lines that
    represent the basis vectors of a local cartesian coordinate system for each
    member.

    Parameters
    ----------
    m
        Model dictionary.
    scale
        Optional: scale factor for the Cartesian vector lengths. Default is
        0.0, which means the scale will be computed internally.

    """
    if scale == 0.0:
        cd = characteristic_dimension(m)
        scale = cd / 15

    ax = plt.gca()
    all_members = [m[k].values() for k in ["truss_members", "beam_members"] if k in m]
    for members in all_members:
        for member in members:
            sect = member["section"]
            connectivity = member["connectivity"]
            i, j = m["joints"][connectivity[0]], m["joints"][connectivity[1]]
            ci, cj = i["coordinates"], j["coordinates"]
            xm = (ci + cj) / 2.0
            if m["dim"] == 3:
                if "xz_vector" in sect:
                    xz_vector = sect['xz_vector']
                else:
                    xz_vector = []
                e_x, e_y, e_z, _ = member_3d_geometry(i, j, xz_vector)
                xs = zeros(2)
                ys = zeros(2)
                zs = zeros(2)
                xs[0] = xm[0]
                ys[0] = xm[1]
                zs[0] = xm[2]
                xs[1] = xs[0] + scale * e_x[0]
                ys[1] = ys[0] + scale * e_x[1]
                zs[1] = zs[0] + scale * e_x[2]
                ax.plot(xs, ys, zs, "r-", lw=3)
                xs = zeros(2)
                ys = zeros(2)
                zs = zeros(2)
                xs[0] = xm[0]
                ys[0] = xm[1]
                zs[0] = xm[2]
                xs[1] = xs[0] + scale * e_y[0]
                ys[1] = ys[0] + scale * e_y[1]
                zs[1] = zs[0] + scale * e_y[2]
                ax.plot(xs, ys, zs, "g-", lw=3)
                xs = zeros(2)
                ys = zeros(2)
                zs = zeros(2)
                xs[0] = xm[0]
                ys[0] = xm[1]
                zs[0] = xm[2]
                xs[1] = xs[0] + scale * e_z[0]
                ys[1] = ys[0] + scale * e_z[1]
                zs[1] = zs[0] + scale * e_z[2]
                ax.plot(xs, ys, zs, "b-", lw=3)
            else:
                e_x, e_z, _ = member_2d_geometry(i, j)
                xs = zeros(2)
                ys = zeros(2)
                xs[0] = xm[0]
                ys[0] = xm[1]
                xs[1] = xs[0] + scale * e_x[0]
                ys[1] = ys[0] + scale * e_x[1]
                ax.plot(xs, ys, "r-", lw=3)
                xs = zeros(2)
                ys = zeros(2)
                xs[0] = xm[0]
                ys[0] = xm[1]
                xs[1] = xs[0] + scale * e_z[0]
                ys[1] = ys[0] + scale * e_z[1]
                ax.plot(xs, ys, "b-", lw=3)
    return ax


def _largest_mag_at_joints(m, fun):
    """
    Find the maximum magnitude of a joint quantity.
    """
    maxmag = 0.0
    for j in m["joints"].values():
        jmag = 0.0
        for v in fun(j):
            if v:
                jmag = max(jmag, abs(v))
        maxmag = max(maxmag, jmag)
    return maxmag


def plot_applied_forces(m, scale=0.0):
    """
    Plot the applied forces at the joints.

    Parameters
    ----------
    m
        Model dictionary.
    scale
        Optional: scale factor for the arrows. Forces are rendered with single
        arrows. Default is 0.0, which means compute this internally.

    """
    ax = plt.gca()
    dim = m["dim"]
    cd = characteristic_dimension(m)

    def fun(j):
        if "loads" in j and j["loads"]:
            return [v for (d, v) in j["loads"].items() if d < dim]
        else:
            return [0]

    if scale == 0.0:
        maxmag = _largest_mag_at_joints(m, fun)
        if maxmag != 0:
            scale = cd / 2 / maxmag
    for j in m["joints"].values():
        if "loads" in j and j["loads"]:
            for d in j["loads"].keys():
                F = zeros((dim,))
                if d < dim:
                    F[d] = j["loads"][d]
                if norm(F) > 0:
                    if dim == 2:
                        x, y = j["coordinates"]
                        u, v = F
                        ax.arrow(
                            x,
                            y,
                            scale * u,
                            scale * v,
                            head_width=cd / 20,
                            head_length=cd / 20,
                            color="cyan",
                        )
                    else:
                        x, y, z = j["coordinates"]
                        u, v, w = F
                        ax.arrow3D(
                            x,
                            y,
                            z,
                            scale * u,
                            scale * v,
                            scale * w,
                            mutation_scale=20,
                            arrowstyle="-|>",
                            color="cyan",
                        )
    return ax


def plot_applied_moments(m, scale=0.0, radius=0.0):
    """
    Plot the applied moments at the joints.

    Parameters
    ----------
    m
        Model dictionary.
    scale
        Optional: scale factor for the arrows. Moments are rendered with double
        arrows. Default is 0.0, which means compute this internally.
    radius
        Radius of the circle to represent the moment (2D only). Default is 0.0,
        which means compute this internally.
    """
    ax = plt.gca()
    dim = m["dim"]
    ndpn = ndof_per_joint(m)
    cd = characteristic_dimension(m)

    def fun(j):
        if "loads" in j and j["loads"]:
            return [v for (d, v) in j["loads"].items() if d >= dim]
        else:
            return [0]

    if scale == 0.0:
        maxmag = _largest_mag_at_joints(m, fun)
        if maxmag != 0:
            scale = cd / 2 / maxmag
    if radius <= 0.0:
        radius = cd / 10
    for j in m["joints"].values():
        if "loads" in j and j["loads"]:
            for d in j["loads"].keys():
                M = zeros((ndpn - dim,))
                if d >= dim:
                    M[d - dim] = j["loads"][d]
                if norm(M) > 0:
                    if dim == 2:
                        x, y = j["coordinates"]
                        if M > 0:
                            st = -110
                            dl = 210
                            sense = +1
                        else:
                            st = 80
                            dl = 210
                            sense = -1
                        _drawcirc(ax, radius, x, y, st, dl, sense, color_="cyan")
                    else:
                        x, y, z = j["coordinates"]
                        u, v, w = M
                        ax.arrow3D(
                            x,
                            y,
                            z,
                            scale * u,
                            scale * v,
                            scale * w,
                            mutation_scale=20,
                            arrowstyle="-|>",
                            color="cyan",
                        )
                        ax.arrow3D(
                            x,
                            y,
                            z,
                            scale * 0.9 * u,
                            scale * 0.9 * v,
                            scale * 0.9 * w,
                            mutation_scale=20,
                            arrowstyle="-|>",
                            color="cyan",
                        )
    return ax


def plot_translation_supports(m, scale=0.0, shortest_arrow=1.0e-6):
    """
    Plot the translation supports at the joints.

    Parameters
    ----------
    m
        Model dictionary.
    scale
        Optional: scale factor for the arrows. Moments are rendered with double
        arrows. Default is 0.0, which means compute this internally.
    shortest_arrow
        How long should the shortest arrow be? Default is 1.0e-6.
    """
    ax = plt.gca()
    dim = m["dim"]
    cd = characteristic_dimension(m)

    def fun(j):
        if "supports" in j:
            return j["supports"]
        else:
            return [0]

    if scale == 0.0:
        maxmag = _largest_mag_at_joints(m, fun)
        if maxmag != 0:
            scale = cd / 10 / maxmag

    for j in m["joints"].values():
        if "supports" in j and j["supports"]:
            for d in j["supports"].keys():
                U = zeros((dim,))
                if d < dim:
                    v = j["supports"][d]
                    if v == 0:
                        v = shortest_arrow
                    U[d] = v
                if dim == 2:
                    x, y = j["coordinates"]
                    u, v = U
                    ax.arrow(
                        x,
                        y,
                        scale * u,
                        scale * v,
                        head_width=cd / 20,
                        head_length=cd / 20,
                        length_includes_head=True,
                        color="orange",
                    )
                else:
                    x, y, z = j["coordinates"]
                    u, v, w = U
                    ax.arrow3D(
                        x,
                        y,
                        z,
                        scale * u,
                        scale * v,
                        scale * w,
                        mutation_scale=20,
                        arrowstyle="-|>",
                        color="orange",
                    )
    return ax


def plot_rotation_supports(m, scale=0.0, radius=0.0, shortest_arrow=1.0e-6):
    """
    Plot the rotation supports at the joints.

    Parameters
    ----------
    m
        Model dictionary.
    scale
        Optional: scale factor for the arrows. Moments are rendered with double
        arrows. Default is 0.0, which means compute this internally.
    radius
        Radius of the circle (2D only). Default radius is 0.0, which means
        the scale is calculated internally.
    shortest_arrow
        How long should the shortest arrow be? Default is 1.0e-6.
    """
    ax = plt.gca()
    dim = m["dim"]
    ndpn = ndof_per_joint(m)
    if ndpn == dim:
        return
    cd = characteristic_dimension(m)

    def fun(j):
        if "supports" in j:
            return j["supports"]
        else:
            return [0]

    if scale == 0.0:
        maxmag = _largest_mag_at_joints(m, fun)
        if maxmag != 0:
            scale = cd / 10 / maxmag

    if radius <= 0.0:
        radius = cd / 10
    for j in m["joints"].values():
        if "supports" in j and j["supports"]:
            for d in j["supports"].keys():
                R = zeros((ndpn - dim,))
                if d >= dim:
                    v = j["supports"][d]
                    if v == 0:
                        v = shortest_arrow
                    R[d - dim] = v
                    if dim == 2:
                        x, y = j["coordinates"]
                        if R >= 0:
                            st = -110
                            dl = 210
                            sense = +1
                        else:
                            st = 80
                            dl = 210
                            sense = -1
                        _drawcirc(ax, radius, x, y, st, dl, sense, color_="blue")
                    else:
                        x, y, z = j["coordinates"]
                        u, v, w = scale * R
                        ax.arrow3D(
                            x,
                            y,
                            z,
                            u,
                            v,
                            w,
                            mutation_scale=20,
                            arrowstyle="-|>",
                            color="blue",
                        )
                        if norm(R) > 0:
                            ax.arrow3D(
                                x,
                                y,
                                z,
                                0.9 * u,
                                0.9 * v,
                                0.9 * w,
                                mutation_scale=20,
                                arrowstyle="-|>",
                                color="blue",
                            )
    return ax


def show(m):
    """
    Show the plot.

    Parameters
    ----------
    m
        Model dictionary.

    """
    ax = plt.gca()
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    if m["dim"] == 3:
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
    plt.show()


def plot_reaction_forces(m, scale=0.0):
    """
    Plot the reaction forces at the joints.

    Parameters
    ----------
    m
        Model dictionary.
    scale
        Optional: scale factor for the arrows. Forces are rendered with single
        arrows. Default is 0.0, which means compute this internally.

    """
    ax = plt.gca()
    dim = m["dim"]
    cd = characteristic_dimension(m)

    def fun(j):
        if "reactions" in j and j["reactions"]:
            return [v for (d, v) in j["reactions"].items() if d < dim]
        else:
            return [0]

    if scale == 0.0:
        maxmag = _largest_mag_at_joints(m, fun)
        if maxmag != 0:
            scale = cd / 2 / maxmag
    for j in m["joints"].values():
        if "reactions" in j and j["reactions"]:
            for d in j["reactions"].keys():
                F = zeros((dim,))
                if d < dim:
                    F[d] = j["reactions"][d]
                if norm(F) > 0:
                    if dim == 2:
                        x, y = j["coordinates"]
                        u, v = F
                        ax.arrow(
                            x,
                            y,
                            scale * u,
                            scale * v,
                            head_width=cd / 20,
                            head_length=cd / 20,
                            color="green",
                        )
                    else:
                        x, y, z = j["coordinates"]
                        u, v, w = F
                        ax.arrow3D(
                            x,
                            y,
                            z,
                            scale * u,
                            scale * v,
                            scale * w,
                            mutation_scale=20,
                            arrowstyle="-|>",
                            color="green",
                        )
    return ax


def plot_reaction_moments(m, scale=0.0, radius=0.0):
    """
    Plot the reaction moments at the joints.

    Parameters
    ----------
    m
        Model dictionary.
    scale
        Optional: scale factor for the arrows. Moments are rendered with double
        arrows. Default is 0.0, which means compute this internally.
    radius
        Radius of the circle to represent the moment (2D only). Default is 0.0,
        which means compute this internally.
    """
    ax = plt.gca()
    dim = m["dim"]
    ndpn = ndof_per_joint(m)
    cd = characteristic_dimension(m)

    def fun(j):
        if "reactions" in j and j["reactions"]:
            return [v for (d, v) in j["reactions"].items() if d >= dim]
        else:
            return [0]

    if scale == 0.0:
        maxmag = _largest_mag_at_joints(m, fun)
        if maxmag != 0:
            scale = cd / 2 / maxmag
    if radius <= 0.0:
        radius = cd / 10
    for j in m["joints"].values():
        if "reactions" in j and j["reactions"]:
            for d in j["reactions"].keys():
                M = zeros((ndpn - dim,))
                if d >= dim:
                    M[d - dim] = j["reactions"][d]
                if norm(M) > 0:
                    if dim == 2:
                        x, y = j["coordinates"]
                        if M > 0:
                            st = -110
                            dl = 210
                            sense = +1
                        else:
                            st = 80
                            dl = 210
                            sense = -1
                        _drawcirc(ax, radius, x, y, st, dl, sense, color_="green")
                    else:
                        x, y, z = j["coordinates"]
                        u, v, w = M
                        ax.arrow3D(
                            x,
                            y,
                            z,
                            scale * u,
                            scale * v,
                            scale * w,
                            mutation_scale=20,
                            arrowstyle="-|>",
                            color="green",
                        )
                        ax.arrow3D(
                            x,
                            y,
                            z,
                            scale * 0.9 * u,
                            scale * 0.9 * v,
                            scale * 0.9 * w,
                            mutation_scale=20,
                            arrowstyle="-|>",
                            color="green",
                        )
    return ax

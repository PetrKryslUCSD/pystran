"""
Define beam mechanical quantities.
"""

from stranalyzer import geometry
from stranalyzer import truss
from numpy import array, dot, reshape, transpose, hstack, vstack, arange, outer, concatenate, zeros
from numpy.linalg import norm
from math import sqrt

def beam_shape_functions(xi, h):
    return array([(2 - 3*xi + xi**3)/4,
                  (-1 + xi + xi**2 - xi**3)/4, 
                  (2 + 3*xi - xi**3)/4,
                  (+1 + xi - xi**2 - xi**3)/4])
    

def beam_2d_member_geometry(i, j):
    """
    Compute beam geometry.
    """
    e_x = geometry.delt(i['coordinates'], j['coordinates'])
    h = geometry.len(i['coordinates'], j['coordinates'])
    e_x /= h
    # The orientation here reflects the sign convention in the book.
    # The deflection is measured positive downwards, while the x coordinate is measured left to right.
    # So in two dimensions e_x and e_y form a left-handed coordinate system.
    e_y = array([e_x[1], -e_x[0]])
    return e_x, e_y, h

def stiffness_2d(e_x, e_y, h, E, I):
    """
    Compute beam stiffness matrix.
    """
    if abs(norm(e_x) - 1.0) > 1e-6:
        raise Exception("Direction vector must be a unit vector")
    xiG = [-1/sqrt(3), 1/sqrt(3)]
    WG = [1, 1]
    K = zeros((6, 6))
    for q in range(2):
        B = curvature_displacement_2d(e_x, e_y, h, xiG[q])
        K += E * I * outer(B.T, B) * WG[q] * (h/2)
    return  K
    
def curvature_displacement_2d(e_x, e_y, h, xi):
    """
    Compute beam curvature-displacement matrix.
    """
    B = zeros((1, 6))
    B[0, 0:2] = 6*xi/h**2*e_y
    B[0, 2] = (1 - 3*xi)/h
    B[0, 3:5] = -6*xi/h**2*e_y
    B[0, 5] = -(3*xi + 1)/h
    return B

def beam_2d_moment(member, i, j, xi):
    e_x, e_y, h = beam_2d_member_geometry(i, j)
    properties = member['properties']
    E, I = properties['E'], properties['I']
    ui, uj = i['displacements'], j['displacements']
    u = concatenate([ui, uj])
    B = curvature_displacement_2d(e_x, e_y, h, xi)
    return E * I * dot(B, u)

def _stiffness_2d(member, i, j):
    e_x, e_y, h = beam_2d_member_geometry(i, j)
    properties = member['properties']
    E, I = properties['E'], properties['I']
    return stiffness_2d(e_x, e_y, h, E, I)
    
def _stiffness_3d(member, i, j):
    e_x, e_y, h = beam_3d_member_geometry(i, j)
    properties = member['properties']
    E, I = properties['E'], properties['I']
    return stiffness_3d(e_x, e_y, h, E, I)

def _stiffness_truss(member, i, j):
    e_x, L = truss.truss_member_geometry(i, j)
    properties = member['properties']
    E, A = properties['E'], properties['A']
    return truss.stiffness(e_x, L, E, A)
    
def assemble_stiffness(Kg, member, i, j):
    """
    Assemble beam stiffness matrix.
    """
    beam_is_2d = len(i['coordinates']) == len(j['coordinates']) == 2
    if beam_is_2d:
        k = _stiffness_2d(member, i, j)
    else:
        k = _stiffness_3d(member, i, j)
    dof = concatenate([i['dof'], j['dof']])
    for r in arange(len(dof)):
        for c in arange(len(dof)):
            gr, gc = dof[r], dof[c]
            Kg[gr, gc] += k[r, c]
    k = _stiffness_truss(member, i, j)
    if beam_is_2d:
        dof = concatenate([i['dof'][0:2], j['dof'][0:2]])
    else:
        dof = concatenate([i['dof'][0:3], j['dof'][0:3]])
    for r in arange(len(dof)):
        for c in arange(len(dof)):
            gr, gc = dof[r], dof[c]
            Kg[gr, gc] += k[r, c]
    return Kg
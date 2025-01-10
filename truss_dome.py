# -*- coding: utf-8 -*-
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

Three bar example on page 32
"""
import stranalyzer
from stranalyzer import model
from stranalyzer import property
from stranalyzer import geometry
from stranalyzer import truss
from stranalyzer import plots
from numpy import array, dot, outer, concatenate

m = model.create(3)

model.add_joint(m, 1, [0.0, 0.0, .32346000e1])
model.add_joint(m, 2, [.49212500e1, .85239000e1, .24472000e1])
model.add_joint(m, 3, [-.49212500e1, .85239000e1, .24472000e1])
model.add_joint(m, 4, [-.98425000e1, 0.0, .24472000e1])
model.add_joint(m, 5, [-.49212500e1, -.85239000e1, .24472000e1])
model.add_joint(m, 6, [.49212500e1, -.85239000e1, .24472000e1])
model.add_joint(m, 7, [.98425000e1, 0.0, .24472000e1])
model.add_joint(m, 8, [0.0, .19685000e02, 0.0])
model.add_joint(m, 9, [-.17047200e02, .98425000e1, 0.0])
model.add_joint(m, 10, [-.17047200e02, -.98425000e1, 0.0])
model.add_joint(m, 11, [0.0, -.19685000e02, 0.0])
model.add_joint(m, 12, [.17047200e02, -.98425000e1, 0.0])
model.add_joint(m, 13, [.17047200e02, .98425000e1, 0.0])

E = 30000000.0
A = 0.0155
p1 = property.truss_property('steel', E, A)

for id, j in enumerate([
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
        [7, 2]
    ]):
    model.add_truss_member(m, id, j, p1)

for i in [8, 9, 10, 11, 12, 13]:
    for d in range(m['dim']):
        model.add_support(m['joints'][i], d)
        
model.add_load(m['joints'][1], 2, -220.46)

model.number_dofs(m)
print('Total Degrees of Freedom = ', m['ntotaldof'])
print('Free Degrees of Freedom = ', m['nfreedof'])

model.solve(m)

for j in m['joints'].values():
    print(j['displacements'])
    
print('Correct displacements: ', (-0.0033325938, -0.001591621))

for b in m['truss_members'].values():
    connectivity = b['connectivity']
    i, j = m['joints'][connectivity[0]], m['joints'][connectivity[1]]
    e_x, L = truss.truss_member_geometry(i, j)
    B = truss.strain_displacement(e_x, L)
    u = concatenate((i['displacements'], j['displacements']))
    eps = dot(B, u)
    print('Bar ' + str(connectivity) + ' force = ', E * A * eps[0])
    
print('Reference forces: ', -0.656854250e4, -0.48528137e4, -0.15685425e4)

plots.plot_setup(m)
plots.plot_members(m)
plots.plot_deformations(m, 10.0)
plots.show()
    

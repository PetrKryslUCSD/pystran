# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 17:34:32 2026

@author: pkonl
"""

import context
from pystran import Abaqus_import
file = './06_sennett_tut.inp'

nodes = Abaqus_import.read_abaqus_nodes(file)
# print(nodes)

for b in nodes:
    print('Block ', b['block_id'])
    for n in b['nodes']:
        print(n)

elements = Abaqus_import.read_abaqus_elements(file)
# print(elements)


for b in elements:
    print('Block ', b['block_id'])
    for e in b['elements']:
        print(e)

nsets = Abaqus_import.read_abaqus_nsets(file)
# print(nsets)

for b in nsets:
    print('Block ', b['block_id'])
    print('  Name: ', b['name'])
    for n in b['nodes']:
        print(n)

bgsects = Abaqus_import.read_abaqus_beam_general_sections(file)
print(bgsects)

# elsets = Abaqus_import.read_abaqus_elsets(file)
# print(elsets)

# cloads = Abaqus_import.read_abaqus_cloads(file)
# print(cloads)
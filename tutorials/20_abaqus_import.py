# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 17:34:32 2026

@author: pkonl
"""

import context
from pystran import Abaqus_import
file = 'tutorials/06_sennett_tut.inp'

nodes = Abaqus_import.read_abaqus_nodes(file)
print(nodes)

# elements = Abaqus_import.read_abaqus_elements(file)
# print(elements)

# nsets = Abaqus_import.read_abaqus_nsets(file)
# print(nsets)

# bgsects = Abaqus_import.read_abaqus_beam_general_sections(file)
# print(bgsects)

# elsets = Abaqus_import.read_abaqus_elsets(file)
# print(elsets)

# cloads = Abaqus_import.read_abaqus_cloads(file)
# print(cloads)
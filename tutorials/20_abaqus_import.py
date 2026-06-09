# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 17:34:32 2026

@author: pkonl
"""

from pystran import Abaqus_import
file = 'tutorials/06_sennett_tut.inp'
nodes = Abaqus_import.read_abaqus_nodes(file)
print(nodes)
elements = Abaqus_import.read_abaqus_elements(file)
print(elements)
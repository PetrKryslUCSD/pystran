# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 17:34:32 2026

@author: pkonl
"""

import context
from pystran import Abaqus_import
file = './tutorials/06_sennett_tut.inp'
file = './tutorials/06_sennett_tut_a.inp'

schema = Abaqus_import.read_abaqus_inp(file)
# print(schema["node_blocks"])
# print(schema["element_blocks"])
# print(schema["nset_blocks"])
# print(schema["elset_blocks"])
# print(schema["cload_blocks"])
print(schema["beam_general_section_blocks"])

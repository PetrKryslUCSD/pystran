# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 17:34:32 2026

@author: pkonl

NOTE: WORK IN PROGRESS
"""

import context
from pystran import Abaqus_import
file = './tutorials/06_sennett_tut.inp'
# file = './tutorials/06_sennett_tut_12el_bc.inp'

out = './tutorials/out.py'

model = Abaqus_import.read_abaqus_inp(file)
# print(model['part_blocks'])
print(model['assembly_blocks'])
# Abaqus_import.inp_to_pystran(file, out)
                             
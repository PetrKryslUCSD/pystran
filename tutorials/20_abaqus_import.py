# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 17:34:32 2026

@author: pkonl
"""

import context
from pystran import Abaqus_import
file = './tutorials/06_sennett_tut.inp'
# file = './tutorials/06_sennett_tut_a.inp'
# file = './tutorials/06_sennett_tut_b.inp'
out = './tutorials/out.py'

# schema = Abaqus_import.read_abaqus_inp(file)
# print(schema)
Abaqus_import.inp_to_pystran(file, out)
                             
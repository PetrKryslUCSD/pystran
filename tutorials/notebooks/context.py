"""
Mechanism for making python aware of the pystran package.
"""

import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))

sys.path.insert(0, os.path.abspath("."))
p = dir_path
while True:
    p = os.path.realpath(p)
    # If we find the pystran folder, presumably we have the module
    if os.path.exists(os.path.abspath(p + "/pystran")):
        break
    # If we can't go any higher, report an error
    if p == os.path.realpath(p + "/.."):
        print("Where am I?\n", os.getcwd())
        print("Where is context?\n", dir_path)
        raise Exception("Could not find the pystran package.")
    p = p + "/.."

sys.path.insert(0, os.path.abspath(p))

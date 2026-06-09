# -*- coding: utf-8 -*-
"""
Import selected sections from an Abaqus input file.

Created on Mon Jun  8 17:11:08 2026

@author: Petr Krysl
"""

import re
from typing import List, Dict, Optional, Iterable, Tuple


# Match a keyword line where:
#  - group(1) = the keyword phrase (the '*' plus one or more words, e.g. '*element' or '*element output')
#  - group(2) = the remainder (options) which starts after a comma or end of line
# The regex allows whitespace between words in the keyword phrase and requires that the keyword phrase
# is followed by either a comma or the end of line (possibly with trailing whitespace).
_KEYWORD_RE = re.compile(r'^\s*(\*(?:[A-Za-z0-9_]+(?:\s+[A-Za-z0-9_]+)*))\s*(?:,(.*)|$)', re.IGNORECASE)

def read_abaqus_nodes(inp_path: str) -> List[Tuple[int, Tuple[float, float, float]]]:
    """
    Parse the first *Node section found in an Abaqus .inp file and return a list of
    (node_id, (x, y, z)) tuples. If multiple *Node sections exist, this returns the first.
    Raises ValueError if no *Node section is found.
    
    # Example usage:
    # nodes = read_abaqus_nodes("model.inp")
    # print(nodes[:10])
    """
    
    record_re = re.compile(r'^\s*\*')  # start of a new record
    data_re = re.compile(r'^\s*(\d+)\s*,\s*([-\dEe+.]+)(?:\s*,\s*([-\dEe+.]+))?(?:\s*,\s*([-\dEe+.]+))?\s*$')

    nodes = []
    in_node_section = False

    with open(inp_path, 'r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith('**'):  # blank or comment line
                continue

            if not in_node_section:
                kwm = _KEYWORD_RE.match(line)
                if kwm:
                    # It's a keyword line
                    kw = kwm.group(1).lower()
                    rest = kwm.group(2)
                    if rest:
                        rest = rest.strip()
                    if kw == '*node':
                        in_node_section = True
                    continue

            # we are inside *Node section
            if record_re.match(line):  # new record starts; exit node section
                break

            # some files may split records with continuation commas or extra spaces; remove inline comments if any
            # Abaqus uses ** for full-line comments; inline comments are uncommon — ignore after '**' if present
            if '**' in line:
                line = line.split('**', 1)[0].strip()
                if not line:
                    continue

            m = data_re.match(line)
            if not m:
                # skip lines that don't look like node data
                continue
            nid = int(m.group(1))
            x = float(m.group(2))
            y = float(m.group(3)) if m.group(3) is not None else 0.0
            z = float(m.group(4)) if m.group(4) is not None else 0.0
            nodes.append((nid, (x, y, z)))

    if not nodes:
        raise ValueError("No *NODE section found or it contains no node data.")
    return nodes




def read_abaqus_elements(inp_path: str) -> List[Dict]:
    """
    Parse *Element sections from an Abaqus .inp file.

    Returns a list of dictionaries:
      {
        "block_id": int,            # sequential block number starting at 1
        "keyword_line": str,        # the original keyword line (e.g. "*Element, type=C3D8R")
        "type": Optional[str],      # element type if present in keyword options
        "elements": [               # list of elements in this block
            {"id": int, "nodes": [int, ...], "raw": "..."}, ...
        ]
      }

    Usage:
      with open('model.inp') as f:
          blocks = parse_elements_from_inp(f)

    Notes:
      - Stops collecting when it encounters the next keyword (line starting with '*').
      - Lines starting with '**' are Abaqus comments and ignored.
      - Element data lines are expected as: id, n1, n2, ..., (commas or spaces tolerated).
    """
    blocks = []
    current = None
    block_counter = 0

    def finish_block():
        nonlocal current
        if current is not None:
            blocks.append(current)
            current = None

    with open(inp_path, 'r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.rstrip('\n')
            stripped = line.strip()
            if not stripped:
                continue
            # Ignore comment lines that start with '**'
            if stripped.startswith('**'):
                continue
    
            kwm = _KEYWORD_RE.match(line)
            if kwm:
                # It's a keyword line
                kw = kwm.group(1).lower()
                rest = kwm.group(2)
                if rest:
                    rest = rest.strip()
                # rest = kwm.group(2).strip()
                if kw == '*element':
                    # start a new element block
                    finish_block()
                    block_counter += 1
                    # try to parse type option if present (e.g., type=C3D8R)
                    elem_type = None
                    # split options by comma, look for type=
                    if rest:
                        opts = [o.strip() for o in rest.split(',') if o.strip()]
                        for o in opts:
                            if o.lower().startswith('type='):
                                elem_type = o.split('=', 1)[1]
                                break
                    current = {
                        "block_id": block_counter,
                        "keyword_line": line.strip(),
                        "type": elem_type,
                        "elements": []
                    }
                    continue
                else:
                    # another keyword — end current element block if any
                    finish_block()
                    continue
    
            # If we reach here and are inside an element block, parse element lines
            if current is not None:
                # Clean inline comment after '!' if present (Abaqus sometimes uses '!'?),
                # also ignore trailing comments starting with '--' or ';' rarely used.
                dataline = re.split(r'!|--|;', line, maxsplit=1)[0].strip()
                if not dataline:
                    continue
                # Split by comma or whitespace
                parts = [p for p in re.split(r'[\s,]+', dataline) if p != '']
                if not parts:
                    continue
                # First token is element id
                try:
                    elem_id = int(parts[0])
                    nodes = [int(p) for p in parts[1:]]
                except ValueError:
                    # Skip malformed lines (could log or raise instead)
                    continue
                current["elements"].append((elem_id, nodes))
            else:
                # Not inside an element block; ignore other data lines
                continue

    # finalise last block
    finish_block()
    return blocks

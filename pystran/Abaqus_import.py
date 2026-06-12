# -*- coding: utf-8 -*-
"""
Import option blocks from an Abaqus input file.

Created on Mon Jun  8 17:11:08 2026

@author: Petr Krysl

Refer to the Section "Input Syntax Rules" in the Abaqus documentation 
for details on the syntax of Abaqus input files.
"""

import re
from typing import List, Dict, Optional, Iterable, Tuple
from collections import namedtuple

# Match a keyword line where:
#  - group(1) = the keyword phrase (the '*' plus one or more words, 
#   e.g. '*ELEMENT' or '*ELEMENT OUTPUT')
#  - group(2) = the remainder (options) which starts after a comma 
#   or end of line
# The regex allows whitespace between words in the keyword phrase and 
#   requires that the keyword phrase# is followed by either a comma 
#   or the end of line (possibly with trailing whitespace).
_KEYWORD_RE = re.compile(r'^\s*(\*(?:[A-Za-z0-9_]+(?:\s+[A-Za-z0-9_]+)*))\s*(?:,(.*)|$)', re.IGNORECASE)

def _split_tokens(line: str) -> List[str]:
    return [p for p in re.split(r'[\s,]+', line) if p != '']


# def read_abaqus_beam_general_sections(inp_path: str) -> List[Dict]:
#     """
#     Read *BEAM GENERAL SECTION option blocks from an Abaqus .inp file.

#     Returns a list of dicts:
#       {
#         "block_id": int,           # sequential block number starting at 1
#         "parameters": Dict[str,str],  # parsed keyword parameters (upper-cased keys)
#         "datalines": [             # list of beam property data lines found in the block
#             [float, ...], ...
#         ]
#       }

#     Required parameters recognized:
#     - ELSET: identifies the element set to which the beam section applies.
#     - MATERIAL: specifies the material name for the beam section.
#     - SECTION=GENERAL: only linear response is considered here.

#     Notes:
#       - Stops collecting when the next keyword line is encountered.
#       - Lines starting with '**' are ignored.
#       - BEAM GENERAL SECTION block content may include element-based 
#         property assignments, orientation lines, or property data; 
#         this function captures each data line as a list of numbers.
#     """
#     sections: List[Dict] = []
#     block_counter = 0

#     for kw, rest, keyword_line, lines in _iter_keyword_blocks(inp_path):
#         if kw != '*BEAM GENERAL SECTION':
#             continue

#         block_counter += 1
#         params = _parse_keyword_parameters(rest) if rest else {}
#         current: Dict = {
#             "block_id": block_counter,
#             "parameters": params,
#             "datalines": []
#         }

#         for line in lines:
#             dataline = _strip_inline_comment(line)
#             if not dataline:
#                 continue

#             data = [t.strip() for t in re.split(r'(?<!"),(?!")', dataline) if t.strip()]
#             if not data:
#                 data = _split_tokens(dataline)

#             current["datalines"].append(data)

#         sections.append(current)

#     return sections


# def read_abaqus_cloads(inp_path: str) -> List[Dict]:
#     """
#     Parse all *CLOAD sections from an Abaqus .inp file.

#     Returns a list of dicts:
#       {
#         "block_id": int,
#         "keyword_line": str,
#         "node": Optional[int],       # if a single node/id is placed on the keyword line (rare)
#         "options": Dict[str,str],    # parsed keyword options (upper-cased keys)
#         "loads": [                   # list of loads (node_or_nodepart, dof, value) or (node, dof, value)
#             {"node": int, "dof": int, "value": float, "raw": str}, ...
#         ]
#       }

#     Notes:
#       - Supports keyword options like 'amplitude=AMPL1' and 'create' etc.
#       - Lines are expected as: node, dof, magnitude
#       - Skips comment lines starting with '**' and inline comments after '!' or '--' or ';'
#       - Stops a block when a new keyword (line starting with '*') is encountered.
#     """
#     blocks: List[Dict] = []
#     block_counter = 0

#     for kw, rest, keyword_line, lines in _iter_keyword_blocks(inp_path):
#         if kw != '*CLOAD':
#             continue

#         block_counter += 1
#         opts = _parse_keyword_parameters(rest)
#         current: Dict = {
#             "block_id": block_counter,
#             "keyword_line": keyword_line,
#             "node": None,
#             "options": opts,
#             "loads": []
#         }

#         for line in lines:
#             dataline = _strip_inline_comment(line)
#             if not dataline:
#                 continue

#             parts = _split_tokens(dataline)
#             if not parts:
#                 continue

#             if len(parts) < 3:
#                 continue

#             try:
#                 node = int(parts[0])
#                 dof = int(parts[1])
#                 value = float(parts[2])
#             except ValueError:
#                 continue

#             current["loads"].append({
#                 "node": node,
#                 "dof": dof,
#                 "value": value,
#                 # "raw": dataline
#             })

#         blocks.append(current)

#     return blocks

def _is_keyword_line(line: str):
    kwm = _KEYWORD_RE.match(line)
    return kwm

def _is_comment_(line: str):
    return line.strip().startswith('**')

def _keyword_and_parameters(line: str) -> Tuple[str, Dict[str,str]]:
    kwm = _KEYWORD_RE.match(line)
    if not kwm:
        return line.strip().upper(), {}
    kw = kwm.group(1).upper()
    rest = kwm.group(2)
    parameters: Dict[str, str] = {}
    if rest:
        for part in [p.strip() for p in rest.split(',') if p.strip()]:
            if '=' in part:
                k, v = part.split('=', 1)
                parameters[k.strip().upper()] = v.strip()
            else:
                parameters[part.strip().upper()] = ''
    return kw, parameters

def _handle_nodes(schema: Dict, kw: str, parameters: Dict[str,str]):
    buffer = schema['buffer']
    kw_line = buffer['currentline']+1 # 1-based line number for reporting
    nodes = []
    buffer['currentline'] += 1
    while buffer['currentline'] < len(buffer['lines']):
        line = buffer['lines'][buffer['currentline']]
        if _is_keyword_line(line):
            buffer['currentline'] -= 1
            break
        if _is_comment_(line):
            buffer['currentline'] += 1
            continue
        parts = _split_tokens(line)
        if not parts:
            buffer['currentline'] += 1
            continue
        try:
            node_id = int(parts[0])
            coordinates = [float(p) for p in parts[1:]]
        except ValueError:
            continue
        nodes.append({
            "id": node_id,
            "coordinates": coordinates,
        })
        buffer['currentline'] += 1
    schema['node_blocks'].append({
        'kw_line': kw_line,
        "parameters": parameters,
        "nodes": nodes
    })
    return schema

def _handle_elements(schema: Dict, kw: str, parameters: Dict[str,str]):
    buffer = schema['buffer']
    kw_line = buffer['currentline']+1 # 1-based line number for reporting
    # Handle parameters
    elem_type = None
    if 'TYPE' in parameters:
        elem_type = parameters['TYPE']
    elements = []
    buffer['currentline'] += 1
    while buffer['currentline'] < len(buffer['lines']):
        line = buffer['lines'][buffer['currentline']]
        if _is_keyword_line(line):
            buffer['currentline'] -= 1
            break
        if _is_comment_(line):
            buffer['currentline'] += 1
            continue
        parts = _split_tokens(line)
        if not parts:
            buffer['currentline'] += 1
            continue
        try:
            element_id = int(parts[0])
            connectivity = [int(p) for p in parts[1:]]
        except ValueError:
            continue
        elements.append({
            "id": element_id,
            "connectivity": connectivity,
        })
        buffer['currentline'] += 1
    schema['element_blocks'].append({
        'kw_line': kw_line,
        "parameters": parameters,
        "elements": elements
    })
    return schema

def _handle_nsets(schema: Dict, kw: str, parameters: Dict[str,str]):
    buffer = schema['buffer']
    kw_line = buffer['currentline']+1 # 1-based line number for reporting
    # Handle parameters
    generate = parameters.get('GENERATE', None) is not None
    nodes = []
    buffer['currentline'] += 1
    while buffer['currentline'] < len(buffer['lines']):
        line = buffer['lines'][buffer['currentline']]
        if _is_keyword_line(line):
            buffer['currentline'] -= 1
            break
        if _is_comment_(line):
            buffer['currentline'] += 1
            continue
        parts = _split_tokens(line)
        if not parts:
            buffer['currentline'] += 1
            continue
        try:
            ns = [int(p) for p in parts[0:]]
        except ValueError:
            continue
        if generate and len(ns) == 3:
            start, end, step = ns
            ns = list(range(start, end+1, step))
        elif generate and len(ns) == 2:
            start, end = ns
            ns = list(range(start, end+1, 1))
        nodes.extend(ns)
        buffer['currentline'] += 1
    schema['nset_blocks'].append({
        'kw_line': kw_line,
        "parameters": parameters,
        "nodes": nodes
    })
    return schema

def _handle_elsets(schema: Dict, kw: str, parameters: Dict[str,str]):
    buffer = schema['buffer']
    kw_line = buffer['currentline']+1 # 1-based line number for reporting
    # Handle parameters
    generate = parameters.get('GENERATE', None) is not None
    elements = []
    buffer['currentline'] += 1
    while buffer['currentline'] < len(buffer['lines']):
        line = buffer['lines'][buffer['currentline']]
        if _is_keyword_line(line):
            buffer['currentline'] -= 1
            break
        if _is_comment_(line):
            buffer['currentline'] += 1
            continue
        parts = _split_tokens(line)
        if not parts:
            buffer['currentline'] += 1
            continue
        try:
            ns = [int(p) for p in parts[0:]]
        except ValueError:
            continue
        if generate and len(ns) == 3:
            start, end, step = ns
            ns = list(range(start, end+1, step))
        elif generate and len(ns) == 2:
            start, end = ns
            ns = list(range(start, end+1, 1))
        elements.extend(ns)
        buffer['currentline'] += 1
    schema['elset_blocks'].append({
        'kw_line': kw_line,
        "parameters": parameters,
        "elements": elements
    })
    return schema

def _handle_cloads(schema: Dict, kw: str, parameters: Dict[str,str]):
    buffer = schema['buffer']
    kw_line = buffer['currentline']+1 # 1-based line number for reporting
    buffer['currentline'] += 1
    while buffer['currentline'] < len(buffer['lines']):
        line = buffer['lines'][buffer['currentline']]
        if _is_keyword_line(line):
            buffer['currentline'] -= 1
            break
        if _is_comment_(line):
            buffer['currentline'] += 1
            continue
        parts = _split_tokens(line)
        if not parts:
            buffer['currentline'] += 1
            continue
        node_or_nset = parts[0]
        try:
            node_or_nset = int(parts[0])
        except ValueError:
            pass
        value = None
        dof = None
        try:
            dof = int(parts[1]) if len(parts) > 1 else None
            value = float(parts[2]) if len(parts) > 2 else None
        except ValueError:
            pass
        buffer['currentline'] += 1
    schema['cload_blocks'].append({
        'kw_line': kw_line,
        "node_or_nset": node_or_nset,
        "dof": dof,
        "value": value
    })
    return schema

_KEYWORD_HANDLERS = {
    '*NODE': _handle_nodes,
    '*ELEMENT': _handle_elements,
    '*NSET': _handle_nsets,
    '*ELSET': _handle_elsets,
    # '*BEAM GENERAL SECTION': _handle_beam_general_sections,
    '*CLOAD': _handle_cloads,
}

def _handler_for_keyword(kw: str):
    return _KEYWORD_HANDLERS.get(kw, None)

def read_abaqus_inp(inp_path: str) -> Dict:
    """
    
    """
    with open(inp_path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]
    buffer = dict(lines=lines, currentline=0)
    schema = {
        "inp_path": inp_path,
        "buffer": buffer,
        'node_blocks': [],
        'element_blocks': [],
        'nset_blocks': [],
        'elset_blocks': [],
        'cload_blocks': [],
        }
    buffer['currentline'] = 0
    while buffer['currentline'] < len(buffer['lines']):
        line = buffer['lines'][buffer['currentline']]
        if _is_keyword_line(line):
            kw, parameters = _keyword_and_parameters(line)
            print('Line ', buffer['currentline']+1, ', Keyword: ', kw, 'Parameters: ', parameters)
            handler = _handler_for_keyword(kw)
            if handler:
                schema = handler(schema, kw, parameters)
        buffer['currentline'] += 1

    return schema

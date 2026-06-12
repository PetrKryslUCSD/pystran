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

def _is_keyword_line(line: str):
    kwm = _KEYWORD_RE.match(line)
    return kwm

def _is_comment_(line: str):
    return line.strip().startswith('**')

def _skip_blank_and_comment_lines(buffer: Dict):
    while buffer['currentline'] < len(buffer['lines']):
        line = _current_line(buffer)
        if _is_comment_(line) or line.strip() == '':
            buffer['currentline'] += 1
        else:
            break

def _current_line(buffer: Dict) -> str:
    if buffer['currentline'] < len(buffer['lines']):
        return buffer['lines'][buffer['currentline']]
    else:
        return ''

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
        line = _current_line(buffer)
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
        line = _current_line(buffer)
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
        line = _current_line(buffer)
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
        line = _current_line(buffer)
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
        line = _current_line(buffer)
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

def _handle_beam_general_sections(schema: Dict, kw: str, parameters: Dict[str,str]):
    buffer = schema['buffer']
    kw_line = buffer['currentline']+1 # 1-based line number for reporting
    # Handle parameters
    elset = parameters.get('ELSET', None)
    material = parameters.get('MATERIAL', None)
    section = parameters.get('SECTION', None)
    if not section == 'GENERAL':
        raise ValueError(f"Only SECTION=GENERAL is supported in *BEAM GENERAL SECTION blocks, found: {section}")    
    buffer['currentline'] += 1
    while buffer['currentline'] < len(buffer['lines']):
        line = _current_line(buffer)
        if _is_keyword_line(line):
            buffer['currentline'] -= 1
            break
        if _is_comment_(line):
            buffer['currentline'] += 1
            continue
        # First the cross section properties
        parts = _split_tokens(line)
        if not parts:
            buffer['currentline'] += 1
            continue
        try:
            area = float(parts[0]) if len(parts) > 0 else None
            mominertia11 = float(parts[1]) if len(parts) > 1 else None
            mominertia12 = float(parts[2]) if len(parts) > 2 else None
            mominertia22 = float(parts[3]) if len(parts) > 3 else None
            torsionconst = float(parts[4]) if len(parts) > 4 else None
        except ValueError:
            pass
        buffer['currentline'] += 1
        # Is the orientation given? It is optional.
        _skip_blank_and_comment_lines(buffer)
        line = _current_line(buffer)
        if _is_keyword_line(line):
            buffer['currentline'] -= 1
            break
        parts = _split_tokens(line)
        if not parts:
            buffer['currentline'] += 1
            continue
        orientation = None
        try:
            cx = float(parts[0]) if len(parts) > 0 else None
            cy = float(parts[1]) if len(parts) > 1 else None
            cz = float(parts[2]) if len(parts) > 2 else None
            orientation = (cx, cy, cz)
        except ValueError:
            pass
        buffer['currentline'] += 1
    schema['beam_general_section_blocks'].append({
        'kw_line': kw_line,
        "elset": elset,
        "material": material,
        "section": section,
        "area": area,
        "mominertia11": mominertia11,
        "mominertia12": mominertia12,
        "mominertia22": mominertia22,
        "torsionconst": torsionconst,
        "orientation": orientation
    })
    return schema

_KEYWORD_HANDLERS = {
    '*NODE': _handle_nodes,
    '*ELEMENT': _handle_elements,
    '*NSET': _handle_nsets,
    '*ELSET': _handle_elsets,
    '*BEAM GENERAL SECTION': _handle_beam_general_sections,
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
        'beam_general_section_blocks': [],
        }
    buffer['currentline'] = 0
    while buffer['currentline'] < len(buffer['lines']):
        line = _current_line(buffer)
        if _is_keyword_line(line):
            kw, parameters = _keyword_and_parameters(line)
            print('Line ', buffer['currentline']+1, ', Keyword: ', kw, 'Parameters: ', parameters)
            handler = _handler_for_keyword(kw)
            if handler:
                schema = handler(schema, kw, parameters)
        buffer['currentline'] += 1

    return schema

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
from pystran import model, section

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
                parameters[k.strip().upper()] = re.sub(r'"', '', v.strip())
            else:
                parameters[part.strip().upper()] = ''
    return kw, parameters

def _is_tuple_of_ints(x) -> bool:
    return isinstance(x, tuple) and all(isinstance(i, int) for i in x)

def _is_int(x) -> bool:
    return isinstance(x, int)

def _translate_dofs(dof, dim, is_beam):
    if _is_tuple_of_ints(dof):
        dof = [d-1 for d in range(dof[0], dof[1], 1)]
    elif _is_int(dof):
        dof = [dof-1]
    else:
        if dof == 'ENCASTRE':
            if dim == 2:
                if is_beam:
                    dof = [0, 1, 2]
                else:
                    dof = [0, 1]
            else:
                if is_beam:
                    dof = [0, 1, 2, 3, 4, 5]
                else:
                    dof = [0, 1, 2]
        else:
            raise ValueError(f"Unsupported dof specification: {dof}")
    return dof

def _handle_nodes(schema: Dict, buffer: Dict, kw: str, parameters: Dict[str,str]):
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

def _handle_elements(schema: Dict, buffer: Dict, kw: str, parameters: Dict[str,str]):
    kw_line = buffer['currentline']+1 # 1-based line number for reporting
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

def _handle_nsets(schema: Dict, buffer: Dict, kw: str, parameters: Dict[str,str]):
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

def _handle_elsets(schema: Dict, buffer: Dict, kw: str, parameters: Dict[str,str]):
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

def _handle_cloads(schema: Dict, buffer: Dict, kw: str, parameters: Dict[str,str]):
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
        "parameters": parameters,
        "node_or_nset": node_or_nset,
        "dof": dof,
        "value": value
    })
    return schema

def _handle_beam_general_sections(schema: Dict, buffer: Dict, kw: str, parameters: Dict[str,str]):
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
        "parameters": parameters,
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

def _handle_step(schema: Dict, buffer: Dict, kw: str, parameters: Dict[str,str]):
    kw_line = buffer['currentline']+1 # 1-based line number for reporting
    step = {
        'kw_line': kw_line,
        "parameters": parameters,
        'boundary_blocks': [],
        'cload_blocks': [],
        }
    buffer['currentline'] += 1
    while buffer['currentline'] < len(buffer['lines']):
        line = _current_line(buffer)
        if _is_keyword_line(line):
            kw, parameters = _keyword_and_parameters(line)
            if kw == '*BOUNDARY':
                _handle_boundary(step, buffer, kw, parameters)
            elif kw == '*CLOAD':
                _handle_cloads(step, buffer, kw, parameters)
            elif kw == '*STATIC':    
                step['procedure'] = 'static'
            elif kw == '*FREQUENCY':    
                step['procedure'] = 'frequency'    
            else:
                buffer['currentline'] -= 1
                break
        buffer['currentline'] += 1
        
    schema['step_blocks'].append(step)
    return schema

def _handle_material(schema: Dict, buffer: Dict, kw: str, parameters: Dict[str,str]):
    kw_line = buffer['currentline']+1 # 1-based line number for reporting
    # Handle parameters
    name = parameters.get('NAME', None)
    E = 1.0
    nu = 0.0
    rho = 0.0
    buffer['currentline'] += 1
    while buffer['currentline'] < len(buffer['lines']):
        line = _current_line(buffer)
        if _is_keyword_line(line):
            kw, parameters = _keyword_and_parameters(line)
            if kw == '*ELASTIC':
                buffer['currentline'] += 1
                line = _current_line(buffer)
                try:
                    parts = _split_tokens(line)
                    if len(parts) >= 2:
                        E = float(parts[0])
                        nu = float(parts[1])
                except ValueError:
                    pass
            elif kw == '*DENSITY':
                buffer['currentline'] += 1
                line = _current_line(buffer)
                try:
                    parts = _split_tokens(line)
                    if len(parts) >= 1:
                        rho = float(parts[0])
                except ValueError:
                    pass
            else:
                buffer['currentline'] -= 1
                break
        buffer['currentline'] += 1
    schema['material_blocks'].append({
        'kw_line': kw_line,
        "parameters": parameters,
        "name": name,
        "E": E,
        "nu": nu,
        "rho": rho,
    })
    return schema

def _handle_boundary(schema: Dict, buffer: Dict, kw: str, parameters: Dict[str,str]):
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
        node_or_nset = None
        dof = None
        value = 0.0
        if len(parts) >= 1:
            try:
                node_or_nset = int(parts[0])
            except ValueError:
                node_or_nset = parts[0]
                pass
        if len(parts) < 3:
            try:
                dof = int(parts[1])
            except ValueError:
                dof = parts[1]
                pass
        else:
            try:
                dof = (int(parts[1]), int(parts[2]))
            except ValueError:
                pass
        if len(parts) > 3:
            try:
                value = float(parts[3]) if len(parts) > 3 else 0.0
            except ValueError:
                pass
        buffer['currentline'] += 1
    schema['boundary_blocks'].append({
        'kw_line': kw_line,
        "parameters": parameters,
        "node_or_nset": node_or_nset,
        "dof": dof,
        "value": value
    })
    return schema

def _handle_heading(schema: Dict, buffer: Dict, kw: str, parameters: Dict[str,str]):
    kw_line = buffer['currentline']+1 # 1-based line number for reporting
    text = []
    buffer['currentline'] += 1
    while buffer['currentline'] < len(buffer['lines']):
        line = _current_line(buffer)
        if _is_keyword_line(line):
            buffer['currentline'] -= 1
            break
        if _is_comment_(line):
            buffer['currentline'] += 1
            continue
        text.append(line)
        buffer['currentline'] += 1
    schema['heading_blocks'].append({
        'kw_line': kw_line,
        "parameters": {},
        "text": text,
    })
    return schema

def read_abaqus_inp(inp_path: str) -> Dict:
    """
    
    """
      
    _KEYWORD_HANDLERS = {
        '*HEADING': _handle_heading,
        '*NODE': _handle_nodes,
        '*ELEMENT': _handle_elements,
        '*NSET': _handle_nsets,
        '*ELSET': _handle_elsets,
        '*BEAM GENERAL SECTION': _handle_beam_general_sections,
        '*MATERIAL': _handle_material,
        '*STEP': _handle_step,
    }

    def _handler_for_keyword(kw: str):
        return _KEYWORD_HANDLERS.get(kw, None)

    with open(inp_path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]
    buffer = dict(lines=lines, currentline=0)
    schema = {
        "inp_path": inp_path,
        "buffer": buffer,
        'heading_blocks': [],
        'node_blocks': [],
        'element_blocks': [],
        'nset_blocks': [],
        'elset_blocks': [],
        'beam_general_section_blocks': [],
        'material_blocks': [],
        'step_blocks': [],
        }
    buffer['currentline'] = 0
    while buffer['currentline'] < len(buffer['lines']):
        line = _current_line(buffer)
        if _is_keyword_line(line):
            kw, parameters = _keyword_and_parameters(line)
            print('Line ', buffer['currentline']+1, ': ', kw, parameters)
            handler = _handler_for_keyword(kw)
            if handler:
                schema = handler(schema, buffer, kw, parameters)
        buffer['currentline'] += 1
    return schema

def inp_to_pystran(inp_path: str, out_path: str) -> Dict:
    schema = read_abaqus_inp(inp_path)
    def _named_elset_block(elset_name: str):
        for b in schema['elset_blocks']:
            if b['parameters'].get('ELSET', None) == elset_name:
                return b
        return None
    def _beam_element(e: int):
        for b in schema['element_blocks']:
            TYPE = b['parameters'].get('TYPE', None)
            if TYPE == 'B31' or TYPE == 'B33':
                for element in b['elements']:
                    if e  == element['id']:
                        return element
        return None
    def _named_nset_block(nset_name: str):
        for b in schema['nset_blocks']:
            if b['parameters'].get('NSET', None) == nset_name:
                return b
        return None
    with open(out_path, "w") as f:
        f.write(f"# PyStran model generated from Abaqus input file: {inp_path}\n")
        for b in schema['heading_blocks']:
            for line in b['text']:
                f.write(f"# {line}\n")
        f.write("from numpy import array\n")
        f.write("from numpy.linalg import norm\n")
        f.write("import context\n")
        f.write("from pystran import model\n")
        f.write("from pystran import section\n")
        f.write("from pystran import plots\n")  
        f.write("from pystran import beam\n")  
        dim = 3 # Assume 3D for now, as Abaqus does not have a clear 2D/3D distinction in the input file. We can determine the actual dimension from the node coordinates later if needed.
        is_beam = True # Assume beam elements for now, as we are only handling *BEAM GENERAL SECTION blocks. We can determine the actual element types later if needed.
        f.write(f"m = model.create({dim})\n")
        f.write("freedoms = m['freedoms']\n")
        for b in schema['node_blocks']:
            f.write(f"# Joints defined at line {b['kw_line']} with parameters: {b['parameters']}\n")
            for node in b['nodes']:
                f.write(f"model.add_joint(m, {node['id']}, {node['coordinates']})\n")
        materials = {}
        for b in schema['material_blocks']:
            f.write(f"# Material defined at line {b['kw_line']} with parameters: {b['parameters']}\n")
            name = b['name']
            E = b['E']
            nu = b['nu']
            rho = b['rho']
            materials[name] = dict(E=E, nu=nu, rho=rho)
        sects = {}
        for b in schema['beam_general_section_blocks']:
            f.write(f"# Beam general section defined at line {b['kw_line']} with parameters: {b['parameters']}\n")
            A = b['area']
            # PyStran's beam section orientation differs from the Abaqus convention.
            # Adjust by supplying xy_vector instead of xz_vector.
            I11 = b['mominertia11']
            I12 = b['mominertia12']
            I22 = b['mominertia22']
            J = b['torsionconst']
            xy_vector = b['orientation']
            if I12 != 0.0:
                raise ValueError(f"Non-zero product of inertia (I12={I12}) is not supported in *BEAM GENERAL SECTION blocks.")
            Iy = I11
            Iz = I22
            Ix = I11 + I22
            sect_name = f'sect_{b['kw_line']}_' + str(len(sects)+1)
            sects[b['parameters']['ELSET']] = sect_name
            material_name = b['parameters']['MATERIAL']
            E = materials[material_name]['E']
            nu = materials[material_name]['nu']  
            G = E / (2.0 * (1.0 + nu))        
            rho = materials[material_name]['rho']          
            f.write(f"{sect_name} = section.beam_3d_section('{sect_name}', \n")
            f.write(f"            E={E}, G={G}, rho={rho}, \n")
            f.write(f"            A={A}, Ix={Ix}, Iy={Iy}, Iz={Iz}, J={J}, \n")
            f.write(f"            xy_vector={xy_vector}, xz_vector=None)\n")
            elsetb = _named_elset_block(b['parameters']['ELSET'])
            for e in elsetb['elements']:
                element = _beam_element(e)
                f.write(f"model.add_beam_member(m, {element['id']}, {element['connectivity']}, {sect_name})\n")
        for sb in schema['step_blocks']:
            for b in sb['boundary_blocks']:
                f.write(f"# Boundary conditions defined at line {b['kw_line']} with parameters: {b['parameters']}\n")
                node_or_nset = b['node_or_nset']
                dof = b['dof']
                dof = _translate_dofs(dof, dim, is_beam) 
                value = b['value']
                if isinstance(node_or_nset, int):
                    for d in dof:
                        f.write(f"model.add_support(m['joints'][{node_or_nset}], {d}, {value})\n")
                else:
                    n = _named_nset_block(node_or_nset)
                    if n is None:
                        f.write(f"# Warning: nset '{node_or_nset}' not found for boundary condition at line {b['kw_line']}\n")
                    for j in n['nodes']:
                        for d in dof:
                            f.write(f"model.add_support(m['joints'][{j}], {d}, {value})\n")
            for b in sb['cload_blocks']:
                f.write(f"# Concentrated loads defined at line {b['kw_line']} with parameters: {b['parameters']}\n")
                node_or_nset = b['node_or_nset']
                dof = b['dof']
                dof = _translate_dofs(dof, dim, is_beam)
                value = b['value']
                if isinstance(node_or_nset, int):
                    for d in dof:
                        f.write(f"model.add_load(m['joints'][{node_or_nset}], {d}, {value})\n")
                else:
                    n = _named_nset_block(node_or_nset)
                    if n is None:
                        f.write(f"# Warning: nset '{node_or_nset}' not found for concentrated load at line {b['kw_line']}\n")
                    for j in n['nodes']:
                        for d in dof:
                            f.write(f"model.add_load(m['joints'][{j}], {d}, {value})\n")
            f.write(f"model.number_dofs(m)\n")
            if sb['procedure'] == 'static':
                f.write(f"model.solve_statics(m)\n")
            elif sb['procedure'] == 'frequency':
                f.write(f"model.solve_free_vibration(m)\n")
            else:
                f.write(f"Unknown procedure: {sb['procedure']}\n")
        f.write("# End of Abaqus model translation\n")

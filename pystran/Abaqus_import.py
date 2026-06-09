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

# Match any section keyword
_RECORD_RE = re.compile(r'^\s*\*')
    
def read_abaqus_nodes(inp_path: str) -> List[Tuple[int, Tuple[float, float, float]]]:
    """
    Parse the first *Node section found in an Abaqus .inp file and return a list of
    (node_id, (x, y, z)) tuples. If multiple *Node sections exist, this returns the first.
    Raises ValueError if no *Node section is found.
    
    # Example usage:
    # nodes = read_abaqus_nodes("model.inp")
    # print(nodes[:10])
    """
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
            if _RECORD_RE.match(line):  # new record starts; exit node section
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
                        # "keyword_line": line.strip(),
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

def _parse_keyword_options(keyword_rest: Optional[str]) -> Dict[str, str]:
    """Parse comma-separated keyword options like 'nset=SET1, generate' -> dict."""
    opts: Dict[str, str] = {}
    if not keyword_rest:
        return opts
    for part in [p.strip() for p in keyword_rest.split(',') if p.strip()]:
        if '=' in part:
            k, v = part.split('=', 1)
            opts[k.strip().lower()] = v.strip()
        else:
            opts[part.strip().lower()] = ''
    return opts


def read_abaqus_nsets(inp_path: str) -> List[Dict]:
    """
    Parse all *Nset sections from an Abaqus .inp file.

    Returns a list of dicts:
      {
        "block_id": int,           # sequential block number starting at 1
        "keyword_line": str,       # original keyword line
        "name": Optional[str],     # value of NSET= option if present (lower-cased)
        "generate": bool,          # True if 'generate' option present
        "nodes": [int, ...]        # list of node ids included in the set
      }
    """
    blocks: List[Dict] = []
    current = None
    block_counter = 0

    with open(inp_path, 'r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.rstrip('\n')
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith('**'):
                continue

            kwm = _KEYWORD_RE.match(line)
            if kwm:
                kw = kwm.group(1).lower()
                rest = kwm.group(2)
                rest = rest.strip() if rest else None
                if kw == '*nset':
                    # start new nset block
                    if current is not None:
                        blocks.append(current)
                        current = None
                    block_counter += 1
                    opts = _parse_keyword_options(rest)
                    name = opts.get('nset') or opts.get('name') or None
                    generate = 'generate' in opts
                    current = {
                        "block_id": block_counter,
                        # "keyword_line": line.strip(),
                        "name": name.lower() if name else None,
                        "generate": generate,
                        "nodes": []
                    }
                    continue
                else:
                    # other keyword ends current nset block
                    if current is not None:
                        blocks.append(current)
                        current = None
                    continue

            # inside an nset block
            if current is None:
                continue

            # remove inline comments if any
            dataline = re.split(r'!|--|;', line, maxsplit=1)[0].strip()
            if not dataline:
                continue

            # handle generate option lines like "1, 10, 1" or ranges "1,5,2"
            if current["generate"]:
                parts = [p.strip() for p in dataline.split(',') if p.strip()]
                try:
                    nums = [int(p) for p in parts]
                except ValueError:
                    continue
                if len(nums) == 3:
                    start, end, inc = nums
                    if inc == 0:
                        continue
                    step = inc
                    # inclusive end handling
                    if (step > 0 and end >= start) or (step < 0 and end <= start):
                        rng = list(range(start, end + (1 if step > 0 else -1), step))
                    else:
                        rng = []
                    current["nodes"].extend(rng)
                elif len(nums) == 2:
                    start, end = nums
                    step = 1 if end >= start else -1
                    rng = list(range(start, end + (1 if step > 0 else -1), step))
                    current["nodes"].extend(rng)
                else:
                    # single or multiple single values
                    for n in nums:
                        current["nodes"].append(n)
                continue

            # non-generate: comma or whitespace separated node ids
            parts = [p for p in re.split(r'[\s,]+', dataline) if p != '']
            for p in parts:
                try:
                    nid = int(p)
                except ValueError:
                    continue
                current["nodes"].append(nid)

    if current is not None:
        blocks.append(current)
    return blocks

def read_abaqus_elsets(inp_path: str) -> List[Dict]:
    """
    Parse all *Elset sections from an Abaqus .inp file.

    Returns a list of dicts:
      {
        "block_id": int,
        "keyword_line": str,
        "name": Optional[str],    # elset name lower-cased if present
        "generate": bool,         # True if 'generate' option present
        "elements": [int, ...]    # list of element ids in the set
      }
    """
    blocks: List[Dict] = []
    current = None
    block_counter = 0

    with open(inp_path, 'r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.rstrip('\n')
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith('**'):
                continue

            kwm = _KEYWORD_RE.match(line)
            if kwm:
                kw = kwm.group(1).lower()
                rest = kwm.group(2)
                rest = rest.strip() if rest else None
                if kw == '*elset' or kw == '*elset ':
                    # start new elset block
                    if current is not None:
                        blocks.append(current)
                        current = None
                    block_counter += 1
                    opts = _parse_keyword_options(rest)
                    name = opts.get('elset') or opts.get('name') or None
                    generate = 'generate' in opts
                    current = {
                        "block_id": block_counter,
                        # "keyword_line": line.strip(),
                        "name": name.lower() if name else None,
                        "generate": generate,
                        "elements": []
                    }
                    continue
                else:
                    # other keyword ends current elset block
                    if current is not None:
                        blocks.append(current)
                        current = None
                    continue

            # inside an elset block
            if current is None:
                continue

            # remove inline comments if any
            dataline = re.split(r'!|--|;', line, maxsplit=1)[0].strip()
            if not dataline:
                continue

            # handle generate option lines like "1, 10, 1"
            if current["generate"]:
                parts = [p.strip() for p in dataline.split(',') if p.strip()]
                try:
                    nums = [int(p) for p in parts]
                except ValueError:
                    continue
                if len(nums) == 3:
                    start, end, inc = nums
                    if inc == 0:
                        continue
                    step = inc
                    if (step > 0 and end >= start) or (step < 0 and end <= start):
                        rng = list(range(start, end + (1 if step > 0 else -1), step))
                    else:
                        rng = []
                    current["elements"].extend(rng)
                elif len(nums) == 2:
                    start, end = nums
                    step = 1 if end >= start else -1
                    rng = list(range(start, end + (1 if step > 0 else -1), step))
                    current["elements"].extend(rng)
                else:
                    for n in nums:
                        current["elements"].append(n)
                continue

            # non-generate: comma or whitespace separated element ids (or element-id, property combos — only first token parsed)
            # Abaqus elset lines can be "e1, e2, e3" or "1,2,3" or "elid, prop" forms; we parse integers only.
            parts = [p for p in re.split(r'[\s,]+', dataline) if p != '']
            for p in parts:
                try:
                    eid = int(p)
                except ValueError:
                    # skip non-integer tokens (e.g., field names); if token contains digits separated by '/', take digits
                    continue
                current["elements"].append(eid)

    if current is not None:
        blocks.append(current)
    return blocks


def read_abaqus_elset_by_name(inp_path: str, name: str) -> Dict:
    """
    Return the first *Elset block whose name matches `name` (case-insensitive).
    Raises KeyError if not found.
    """
    target = name.lower()
    for blk in read_abaqus_elsets(inp_path):
        if blk.get("name") == target:
            return blk
    raise KeyError(f"Elset named '{name}' not found.")

def read_abaqus_beam_general_sections(inp_path: str) -> List[Dict]:
    """
    Parse *Beam General sections from an Abaqus .inp file.

    Returns a list of dicts:
      {
        "block_id": int,           # sequential block number starting at 1
        "keyword_line": str,       # original keyword line (e.g., "*Beam Section, elset=EALL, material=STEEL")
        "options": Dict[str,str],  # parsed keyword options (lower-cased keys)
        "definitions": [           # list of beam property definitions found in the block
            {
              "raw": str,          # the raw data line (trimmed)
              "tokens": [str,...]  # tokens split by comma/whitespace
            }, ...
        ]
      }

    Notes:
      - Stops collecting when the next keyword line is encountered.
      - Lines starting with '**' are ignored.
      - Beam General block content may include element-based property assignments, orientation lines,
        or property definitions; this function captures each non-comment line as a tokenized record.
    """
    sections: List[Dict] = []
    current = None
    block_counter = 0

    with open(inp_path, 'r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.rstrip('\n')
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith('**'):
                continue

            kwm = _KEYWORD_RE.match(line)
            if kwm:
                kw = kwm.group(1).lower()
                rest = kwm.group(2)
                rest = rest.strip() if rest else None
                if kw == '*beam general section':
                    # finish previous
                    if current is not None:
                        sections.append(current)
                        current = None
                    block_counter += 1
                    # parse options into dict
                    opts = _parse_keyword_options(rest) if rest else {}
                    current = {
                        "block_id": block_counter,
                        # "keyword_line": line.strip(),
                        "options": opts,
                        "definitions": []
                    }
                    continue
                else:
                    # other keyword closes current block
                    if current is not None:
                        sections.append(current)
                        current = None
                    continue

            # inside a Beam General block
            if current is None:
                continue

            # remove inline comments if any
            dataline = re.split(r'!|--|;', line, maxsplit=1)[0].strip()
            if not dataline:
                continue

            # Tokenize by commas but preserve quoted strings if any (simple approach)
            tokens = [t.strip() for t in re.split(r'(?<!"),(?!")', dataline) if t.strip()]
            # Fallback: split by commas and whitespace
            if not tokens:
                tokens = [t for t in re.split(r'[\s,]+', dataline) if t != '']

            current["definitions"].append({
                # "raw": dataline,
                "data": tokens
            })

    if current is not None:
        sections.append(current)
    return sections

def read_abaqus_cloads(inp_path: str) -> List[Dict]:
    """
    Parse all *CLOAD sections from an Abaqus .inp file.

    Returns a list of dicts:
      {
        "block_id": int,
        "keyword_line": str,
        "node": Optional[int],       # if a single node/id is placed on the keyword line (rare)
        "options": Dict[str,str],    # parsed keyword options (lower-cased keys)
        "loads": [                   # list of loads (node_or_nodepart, dof, value) or (node, dof, value)
            {"node": int, "dof": int, "value": float, "raw": str}, ...
        ]
      }

    Notes:
      - Supports keyword options like 'amplitude=AMPL1' and 'create' etc.
      - Lines are expected as: node, dof, magnitude
      - Skips comment lines starting with '**' and inline comments after '!' or '--' or ';'
      - Stops a block when a new keyword (line starting with '*') is encountered.
    """
    blocks: List[Dict] = []
    current = None
    block_counter = 0

    with open(inp_path, 'r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.rstrip('\n')
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith('**'):
                continue

            kwm = _KEYWORD_RE.match(line)
            if kwm:
                kw = kwm.group(1).lower()
                rest = kwm.group(2)
                rest = rest.strip() if rest else None
                if kw == '*cload':
                    # finish previous
                    if current is not None:
                        blocks.append(current)
                        current = None
                    block_counter += 1
                    opts = _parse_keyword_options(rest)
                    current = {
                        "block_id": block_counter,
                        "keyword_line": line.strip(),
                        "node": None,
                        "options": opts,
                        "loads": []
                    }
                    continue
                else:
                    # other keyword closes current CLOAD block
                    if current is not None:
                        blocks.append(current)
                        current = None
                    continue

            # inside a CLOAD block
            if current is None:
                continue

            # remove inline comments
            dataline = re.split(r'!|--|;', line, maxsplit=1)[0].strip()
            if not dataline:
                continue

            # CLOAD data lines expected: node, dof, magnitude
            # Tokens can be separated by commas or whitespace; allow scientific notation for magnitude
            parts = [p for p in re.split(r'[\s,]+', dataline) if p != '']
            if not parts:
                continue

            # Some variants may include node ranges or references; we only parse numeric node ids and dof/value here.
            # Handle lines with at least 3 tokens; if more tokens exist, we take the first three.
            # If magnitude is missing, skip the line.
            if len(parts) < 3:
                # skip malformed or partial lines
                continue

            try:
                node = int(parts[0])
            except ValueError:
                # skip non-integer node tokens
                node = parts[0]

            try:
                dof = int(parts[1])
            except ValueError:
                # sometimes DOF may be given as a field name; skip such lines
                continue

            try:
                value = float(parts[2])
            except ValueError:
                # skip if magnitude not parseable as float
                continue

            current["loads"].append({
                "node": node,
                "dof": dof,
                "value": value,
                # "raw": dataline
            })

    if current is not None:
        blocks.append(current)
    return blocks


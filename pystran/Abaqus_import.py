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
_NODE_DATA_RE = re.compile(r'^\s*(\d+)\s*,\s*([\-\dEe+.]+)(?:\s*,\s*([\-\dEe+.]+))?(?:\s*,\s*([\-\dEe+.]+))?\s*$')
_COMMENT_SPLIT_RE = re.compile(r'!|--|;')

def _strip_inline_comment(line: str) -> str:
    return _COMMENT_SPLIT_RE.split(line, maxsplit=1)[0].strip()


def _split_tokens(line: str) -> List[str]:
    return [p for p in re.split(r'[\s,]+', line) if p != '']


def _iter_inp_lines(inp_path: str):
    with open(inp_path, 'r', encoding='utf-8') as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if not stripped or stripped.startswith('**'):
                continue
            yield raw_line.rstrip('\n')


def _iter_keyword_blocks(inp_path: str):
    current_keyword = None
    current_rest = None
    current_keyword_line = None
    current_lines: List[str] = []

    for line in _iter_inp_lines(inp_path):
        kwm = _KEYWORD_RE.match(line)
        if kwm:
            if current_keyword is not None:
                yield current_keyword, current_rest, current_keyword_line, current_lines
            current_keyword = kwm.group(1).lower()
            current_rest = kwm.group(2)
            if current_rest:
                current_rest = current_rest.strip()
            current_keyword_line = line.strip()
            current_lines = []
            continue
        if current_keyword is not None:
            current_lines.append(line)

    if current_keyword is not None:
        yield current_keyword, current_rest, current_keyword_line, current_lines


def _parse_generate_line(dataline: str) -> List[int]:
    parts = [p.strip() for p in dataline.split(',') if p.strip()]
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return []

    if len(nums) == 3:
        start, end, inc = nums
        if inc == 0:
            return []
        step = inc
        if (step > 0 and end >= start) or (step < 0 and end <= start):
            return list(range(start, end + (1 if step > 0 else -1), step))
        return []

    if len(nums) == 2:
        start, end = nums
        step = 1 if end >= start else -1
        return list(range(start, end + (1 if step > 0 else -1), step))

    return nums


def read_abaqus_nodes(inp_path: str) -> List[Tuple[int, Tuple[float, float, float]]]:
    """
    Parse the first *Node section found in an Abaqus .inp file and return a list of
    (node_id, (x, y, z)) tuples. If multiple *Node sections exist, this returns the first.
    Raises ValueError if no *Node section is found.
    
    # Example usage:
    # nodes = read_abaqus_nodes("model.inp")
    # print(nodes[:10])
    """
    nodes = []
    for kw, rest, keyword_line, lines in _iter_keyword_blocks(inp_path):
        if kw != '*node':
            continue
        for line in lines:
            dataline = _strip_inline_comment(line)
            if not dataline:
                continue
            m = _NODE_DATA_RE.match(dataline)
            if not m:
                continue
            nid = int(m.group(1))
            x = float(m.group(2))
            y = float(m.group(3)) if m.group(3) is not None else 0.0
            z = float(m.group(4)) if m.group(4) is not None else 0.0
            nodes.append((nid, (x, y, z)))
        break

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
    blocks: List[Dict] = []
    block_counter = 0

    for kw, rest, keyword_line, lines in _iter_keyword_blocks(inp_path):
        if kw != '*element':
            continue

        block_counter += 1
        elem_type = None
        if rest:
            opts = [o.strip() for o in rest.split(',') if o.strip()]
            for o in opts:
                if o.lower().startswith('type='):
                    elem_type = o.split('=', 1)[1]
                    break

        current: Dict = {
            "block_id": block_counter,
            "keyword_line": keyword_line,
            "type": elem_type,
            "elements": []
        }

        for line in lines:
            dataline = _strip_inline_comment(line)
            if not dataline:
                continue
            parts = _split_tokens(dataline)
            if not parts:
                continue
            try:
                elem_id = int(parts[0])
                nodes = [int(p) for p in parts[1:]]
            except ValueError:
                continue
            current["elements"].append({
                "id": elem_id,
                "nodes": nodes,
                "raw": dataline
            })

        blocks.append(current)

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
    block_counter = 0

    for kw, rest, keyword_line, lines in _iter_keyword_blocks(inp_path):
        if kw != '*nset':
            continue

        block_counter += 1
        opts = _parse_keyword_options(rest)
        name = opts.get('nset') or opts.get('name') or None
        generate = 'generate' in opts

        current: Dict = {
            "block_id": block_counter,
            "keyword_line": keyword_line,
            "name": name.lower() if name else None,
            "generate": generate,
            "nodes": []
        }

        for line in lines:
            dataline = _strip_inline_comment(line)
            if not dataline:
                continue

            if generate:
                current["nodes"].extend(_parse_generate_line(dataline))
                continue

            for p in _split_tokens(dataline):
                try:
                    nid = int(p)
                except ValueError:
                    continue
                current["nodes"].append(nid)

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
    block_counter = 0

    for kw, rest, keyword_line, lines in _iter_keyword_blocks(inp_path):
        if kw != '*elset':
            continue

        block_counter += 1
        opts = _parse_keyword_options(rest)
        name = opts.get('elset') or opts.get('name') or None
        generate = 'generate' in opts

        current: Dict = {
            "block_id": block_counter,
            "keyword_line": keyword_line,
            "name": name.lower() if name else None,
            "generate": generate,
            "elements": []
        }

        for line in lines:
            dataline = _strip_inline_comment(line)
            if not dataline:
                continue

            if generate:
                current["elements"].extend(_parse_generate_line(dataline))
                continue

            for p in _split_tokens(dataline):
                try:
                    eid = int(p)
                except ValueError:
                    continue
                current["elements"].append(eid)

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
    block_counter = 0

    for kw, rest, keyword_line, lines in _iter_keyword_blocks(inp_path):
        if kw != '*beam general section':
            continue

        block_counter += 1
        opts = _parse_keyword_options(rest) if rest else {}
        current: Dict = {
            "block_id": block_counter,
            "keyword_line": keyword_line,
            "options": opts,
            "definitions": []
        }

        for line in lines:
            dataline = _strip_inline_comment(line)
            if not dataline:
                continue

            tokens = [t.strip() for t in re.split(r'(?<!"),(?!")', dataline) if t.strip()]
            if not tokens:
                tokens = _split_tokens(dataline)

            current["definitions"].append({
                "raw": dataline,
                "tokens": tokens
            })

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
    block_counter = 0

    for kw, rest, keyword_line, lines in _iter_keyword_blocks(inp_path):
        if kw != '*cload':
            continue

        block_counter += 1
        opts = _parse_keyword_options(rest)
        current: Dict = {
            "block_id": block_counter,
            "keyword_line": keyword_line,
            "node": None,
            "options": opts,
            "loads": []
        }

        for line in lines:
            dataline = _strip_inline_comment(line)
            if not dataline:
                continue

            parts = _split_tokens(dataline)
            if not parts:
                continue

            if len(parts) < 3:
                continue

            try:
                node = int(parts[0])
                dof = int(parts[1])
                value = float(parts[2])
            except ValueError:
                continue

            current["loads"].append({
                "node": node,
                "dof": dof,
                "value": value,
                # "raw": dataline
            })

        blocks.append(current)

    return blocks


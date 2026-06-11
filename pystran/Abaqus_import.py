# -*- coding: utf-8 -*-
"""
Import option blocks from an Abaqus input file.

Created on Mon Jun  8 17:11:08 2026

@author: Petr Krysl

This section describes the syntax rules that govern an Abaqus input file.

All data definitions in Abaqus are accomplished with option blocks—sets 
of data describing a part of the problem definition. You choose those 
options that are relevant for a particular application. Options are 
defined by lines in the input file. Three types of input lines are 
used in an Abaqus input file: keyword lines, data lines, and comment lines. 
Only 7-bit ASCII characters are supported in keyword lines and data lines, 
and a line feed is required at the end of each line in an input file.

Keyword lines introduce options and often have parameters, which appear 
as words or phrases separated by commas on the keyword line. Parameters 
are used to define the behavior of an option. Parameters can stand 
alone or have a value, and they may be required or optional.

Data lines, which are used to provide numeric or alphanumeric entries, 
follow most keyword lines.

Any line that begins with stars in columns 1 and 2 (**) is a comment line. 
Such lines can be placed anywhere in the file. They are ignored by 
Abaqus, so they will be printed only in the initial listing of the file. 
There is no restriction on how many or where such lines occur in the file.

Relevant parameters and data lines (including the number of entries 
per data line) are described in the sections of the Abaqus Keywords 
Guide describing each option. This section describes the general 
rules that apply to all keyword and data lines.
"""

import re
from typing import List, Dict, Optional, Iterable, Tuple


# Match a keyword line where:
#  - group(1) = the keyword phrase (the '*' plus one or more words, 
#   e.g. '*ELEMENT' or '*ELEMENT OUTPUT')
#  - group(2) = the remainder (options) which starts after a comma 
#   or end of line
# The regex allows whitespace between words in the keyword phrase and 
#   requires that the keyword phrase# is followed by either a comma 
#   or the end of line (possibly with trailing whitespace).
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
            current_keyword = kwm.group(1).upper()
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

def read_abaqus_nodes(inp_path: str) -> List[Dict]:
    """
    Read *NODE option blocks from an Abaqus .inp file.

    Returns a list of dictionaries, one for each *NODE block found:
      {
        "block_id": int,         # sequential block number starting at 1
        "nodes": [               # list of elements in this block
            {"id": int, "coordinates": [float, ...], "raw": "..."}, ...
        ]
      }

    Usage:
      with open('model.inp') as f:
          blocks = read_abaqus_nodes(f)

    Notes:
      - Stops collecting when it encounters the next keyword (line starting with '*').
      - Lines starting with '**' are Abaqus comments and ignored.
      - Node data lines are expected as: id, x, y [, z] (commas or spaces tolerated).
      - Parameters such as INPUT, NSET, etc. are not parsed from the keyword line 
      in this function; it focuses on node data.
    """
    blocks: List[Dict] = []
    block_counter = 0

    for kw, rest, keyword_line, lines in _iter_keyword_blocks(inp_path):
        if kw != '*NODE':
            continue

        block_counter += 1
        
        current: Dict = {
            "block_id": block_counter,
            "keyword_line": keyword_line,
            "nodes": []
        }

        for line in lines:
            dataline = _strip_inline_comment(line)
            if not dataline:
                continue
            parts = _split_tokens(dataline)
            if not parts:
                continue
            try:
                node_id = int(parts[0])
                coordinates = [float(p) for p in parts[1:]]
            except ValueError:
                continue
            current["nodes"].append({
                "id": node_id,
                "coordinates": coordinates,
                "raw": dataline
            })

        blocks.append(current)

    return blocks


def read_abaqus_elements(inp_path: str) -> List[Dict]:
    """
    Read *ELEMENT option blocks from an Abaqus .inp file.

    Returns a list of dictionaries, one for each *ELEMENT block found:
      {
        "block_id": int,            # sequential block number starting at 1
        "type": Optional[str],      # element type if present in keyword options
        "elements": [               # list of elements in this block
            {"id": int, "nodes": [int, ...]}, ...
        ]
      }

    Usage:
      with open('model.inp') as f:
          blocks = read_abaqus_elements(f)

    Notes:
      - Stops collecting when it encounters the next keyword (line starting with '*').
      - Lines starting with '**' are Abaqus comments and ignored.
      - Element data lines are expected as: id, n1, n2, ..., (commas or spaces tolerated).
    """
    blocks: List[Dict] = []
    block_counter = 0

    for kw, rest, keyword_line, lines in _iter_keyword_blocks(inp_path):
        if kw != '*ELEMENT':
            continue

        block_counter += 1
        elem_type = None
        if rest:
            opts = [o.strip() for o in rest.split(',') if o.strip()]
            for o in opts:
                if o.upper().startswith('TYPE='):
                    elem_type = o.split('=', 1)[1]
                    break

        current: Dict = {
            "block_id": block_counter,
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
            })

        blocks.append(current)

    return blocks

def _parse_keyword_options(keyword_rest: Optional[str]) -> Dict[str, str]:
    """Parse comma-separated keyword options like 'NSET=SET1, GENERATE' -> dict."""
    opts: Dict[str, str] = {}
    if not keyword_rest:
        return opts
    for part in [p.strip() for p in keyword_rest.split(',') if p.strip()]:
        if '=' in part:
            k, v = part.split('=', 1)
            opts[k.strip().upper()] = v.strip()
        else:
            opts[part.strip().upper()] = ''
    return opts


def read_abaqus_nsets(inp_path: str) -> List[Dict]:
    """
    Read all *NSET option blocks from an Abaqus .inp file.

    Returns a list of dictionaries, one for each *NSET block found:
      {
        "block_id": int,           # sequential block number starting at 1
        "name": Optional[str],     # value of NSET= option if present (upper-cased)
        "generate": bool,          # True if 'generate' option present
        "nodes": [int, ...]        # list of node ids included in the set
      }
    """
    blocks: List[Dict] = []
    block_counter = 0

    for kw, rest, keyword_line, lines in _iter_keyword_blocks(inp_path):
        if kw != '*NSET':
            continue

        block_counter += 1
        opts = _parse_keyword_options(rest)
        name = opts.get('NSET') or opts.get('NAME') or None
        generate = 'generate' in opts

        current: Dict = {
            "block_id": block_counter,
            "name": name.upper() if name else None,
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
    Read all *Elset option blocks from an Abaqus .inp file.

    Returns a list of dicts:
      {
        "block_id": int,          # sequential block number starting at 1
        "name": Optional[str],    # elset name upper-cased if present
        "generate": bool,         # True if 'generate' option present
        "elements": [int, ...]    # list of element ids in the set
      }
    """
    blocks: List[Dict] = []
    block_counter = 0

    for kw, rest, keyword_line, lines in _iter_keyword_blocks(inp_path):
        if kw != '*ELSET':
            continue

        block_counter += 1
        opts = _parse_keyword_options(rest)
        name = opts.get('ELSET') or opts.get('NAME') or None
        generate = 'generate' in opts

        current: Dict = {
            "block_id": block_counter,
            "name": name.upper() if name else None,
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

def read_abaqus_beam_general_sections(inp_path: str) -> List[Dict]:
    """
    Read *BEAM GENERAL SECTION option blocks from an Abaqus .inp file.

    Returns a list of dicts:
      {
        "block_id": int,           # sequential block number starting at 1
        "options": Dict[str,str],  # parsed keyword options (upper-cased keys)
        "datalines": [             # list of beam property data lines found in the block
            [float, ...], ...
        ]
      }

    Required parameters recognized:
    - ELSET: identifies the element set to which the beam section applies.
    - MATERIAL: specifies the material name for the beam section.
    - SECTION=GENERAL: only linear response is considered here.

    Notes:
      - Stops collecting when the next keyword line is encountered.
      - Lines starting with '**' are ignored.
      - BEAM GENERAL SECTION block content may include element-based 
        property assignments, orientation lines, or property data; 
        this function captures each data line as a list of numbers.
    """
    sections: List[Dict] = []
    block_counter = 0

    for kw, rest, keyword_line, lines in _iter_keyword_blocks(inp_path):
        if kw != '*BEAM GENERAL SECTION':
            continue

        block_counter += 1
        opts = _parse_keyword_options(rest) if rest else {}
        current: Dict = {
            "block_id": block_counter,
            "options": opts,
            "datalines": []
        }

        for line in lines:
            dataline = _strip_inline_comment(line)
            if not dataline:
                continue

            data = [t.strip() for t in re.split(r'(?<!"),(?!")', dataline) if t.strip()]
            if not data:
                data = _split_tokens(dataline)

            current["datalines"].append(data)

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
        "options": Dict[str,str],    # parsed keyword options (upper-cased keys)
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
        if kw != '*CLOAD':
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


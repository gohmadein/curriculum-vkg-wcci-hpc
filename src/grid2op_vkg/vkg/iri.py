"""IRI conventions shared by the RDF backend and the query decoders.

Instance IRIs are index-based (``grid:line_3``) rather than name-based so they
stay stable when a data source renames things -- the grid2op name is attached as
``rdfs:label``. Decoding an IRI back to an array index therefore never needs a
lookup table.
"""

from __future__ import annotations

import re

from rdflib import Namespace

#: TBox vocabulary.
G2O = Namespace("https://grid2op-vkg.org/ontology#")
#: ABox instances.
GRID = Namespace("https://grid2op-vkg.org/resource/")

NAMESPACES = {"g2o": str(G2O), "grid": str(GRID)}

#: SPARQL prefix block, prepended to every query.
PREFIXES = "\n".join(f"PREFIX {p}: <{u}>" for p, u in NAMESPACES.items()) + "\n"

_LOCAL = re.compile(rf"^{re.escape(str(GRID))}(?P<kind>[a-z_]+?)_(?P<idx>\d+)$")


def sub(i: int):
    return GRID[f"sub_{i}"]


def busbar(sub_id: int, bb: int):
    return GRID[f"busbar_{sub_id * 100 + bb}"]


def bus(global_bus_id: int):
    return GRID[f"bus_{global_bus_id}"]


def line(i: int):
    return GRID[f"line_{i}"]


def gen(i: int):
    return GRID[f"gen_{i}"]


def load(i: int):
    return GRID[f"load_{i}"]


def storage(i: int):
    return GRID[f"storage_{i}"]


def terminal(i: int):
    return GRID[f"term_{i}"]


def action(i: int):
    return GRID[f"action_{i}"]


def assignment(action_idx: int, term_idx: int):
    """Named (not blank) node for one terminal->busbar pair of an action."""
    return GRID[f"assign_{action_idx}_{term_idx}"]


def remedial(action_idx: int, line_idx: int):
    """Named node for one catalogued (action, line) remedial effect."""
    return GRID[f"remedial_{action_idx}_{line_idx}"]


def index_of(iri) -> int:
    """Recover the numeric index from an instance IRI.

    >>> index_of(line(7))
    7
    """
    m = _LOCAL.match(str(iri))
    if m is None:
        raise ValueError(f"not an indexed grid resource IRI: {iri!r}")
    return int(m.group("idx"))


def kind_of(iri) -> str:
    """Recover the resource kind (``line``, ``sub``, ``term``, ...) from an IRI."""
    m = _LOCAL.match(str(iri))
    if m is None:
        raise ValueError(f"not an indexed grid resource IRI: {iri!r}")
    return m.group("kind")

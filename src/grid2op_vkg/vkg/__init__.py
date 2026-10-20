"""Virtual Knowledge Graph semantic feedback for Grid2Op.

The package is organised around one idea: a *query* has exactly one semantic
definition (SPARQL over the grid ontology) and two execution paths -- a
reference path that really evaluates the SPARQL, and a compiled numpy kernel
fast enough to sit inside the RL training loop. A conformance test asserts the
two agree.

Layout::

    schema.py     GridSchema  -- static structure, built once per environment
    state.py      GridState   -- one step of dynamic state, cheap to build
    ontology/     the TBox (grid.ttl) and, later, the R2RML mappings
    queries/      Query objects: SPARQL + compiled kernel + signature
    backends/     execution paths (compiled numpy, rdflib, later Ontop)
    providers/    FeedbackProvider: the interface Handcrafted and VKG share
    channels/     the four feedback channels (state, reward, mask, aux)

See ``docs/vkg-module-design.md``.
"""

from grid2op_vkg.vkg.schema import GridSchema, ObjKind
from grid2op_vkg.vkg.state import GridState

__all__ = ["GridSchema", "GridState", "ObjKind"]

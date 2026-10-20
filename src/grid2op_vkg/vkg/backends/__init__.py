"""Execution paths for VKG queries.

``compiled``  numpy kernels, microseconds, used during RL training
``rdf``       rdflib materialisation + real SPARQL, the reference semantics
``ontop``     SPARQL over relational views through Ontop -- the graph is never
              stored; optional, needs the Ontop CLI (see backends/ontop.py)
"""

from grid2op_vkg.vkg.backends.base import Timings, VkgBackend
from grid2op_vkg.vkg.backends.compiled import CompiledBackend
from grid2op_vkg.vkg.backends.ontop import OntopBackend, OntopUnavailable, ontop_available
from grid2op_vkg.vkg.backends.rdf import RdfBackend, load_tbox

#: Name -> backend class, for config-driven construction.
BACKENDS: dict[str, type[VkgBackend]] = {
    CompiledBackend.name: CompiledBackend,
    RdfBackend.name: RdfBackend,
    OntopBackend.name: OntopBackend,
}

__all__ = [
    "BACKENDS",
    "CompiledBackend",
    "OntopBackend",
    "OntopUnavailable",
    "RdfBackend",
    "Timings",
    "VkgBackend",
    "load_tbox",
    "ontop_available",
]

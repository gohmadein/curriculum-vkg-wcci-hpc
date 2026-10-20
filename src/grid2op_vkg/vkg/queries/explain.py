"""Provenance: *why* is this action on the table?

The relevance mask says which actions the semantics admit. This query says, for
one admitted action, which facts admit it -- the witness: each (risk line,
substation) pair such that the line is at risk, the substation is an endpoint
of a line within the relevance radius of it, and the action reconfigures that
substation. Returned as an (n_line, n_sub) boolean matrix so it fits the
:class:`Query` contract and inherits the conformance machinery; on the RDF and
Ontop paths it is literally the SPARQL solution set, i.e. the explanation is a
query over the same graph the decision came from, not a log line written next
to it.

The design document asks for feedback that carries "provenance, time and
explanation" (survey §10.3). This is the provenance half: a decision can be
traced to the triples that justified it, in the ontology's vocabulary, and the
trace is checked against the compiled path like every other answer.
"""

from __future__ import annotations

import numpy as np

from grid2op_vkg.vkg import iri
from grid2op_vkg.vkg.queries.base import Kernel, Query, QuerySig
from grid2op_vkg.vkg.queries.risk import _risk_mask, _risk_pattern
from grid2op_vkg.vkg.queries.structure import adjacency_path, k_hop_over, substation_adjacency
from grid2op_vkg.vkg.schema import GridSchema


class ExplainAction(Query):
    """Witness pairs (risk line, substation) for one topology action."""

    base_name = "explain_action"
    required_obs_attrs = frozenset({"topo_vect", "line_status", "rho", "timestep_overflow"})
    uses_reasoning = True

    def __init__(self, action: int, tau: float = 0.95, k: int = 1, adjacency: str = "substation") -> None:
        super().__init__(action=int(action), tau=float(tau), k=int(k), adjacency=str(adjacency))

    def sparql(self, schema: GridSchema) -> str:
        tau, k, adj = self.params["tau"], self.params["k"], self.params["adjacency"]
        action = iri.action(self.params["action"])
        # The BIND must sit in the same group as the pattern that binds ?risk:
        # UNION branches are evaluated independently, so a BIND in a group of
        # its own sees ?risk unbound and ?near becomes a free variable, which
        # made *every* line touching the substation a witness.
        return f"""
        SELECT DISTINCT ?risk ?sub WHERE {{
            {{
                {{{_risk_pattern('risk', tau)}
                }}
                BIND(?risk AS ?near)
            }}
            UNION
            {{
                {{{_risk_pattern('risk', tau)}
                }}
                ?risk {adjacency_path(k, adj)} ?near .
            }}
            ?near g2o:hasTerminal ?nt . ?nt g2o:atSubstation ?sub .
            <{action}> a g2o:TopologyAction ; g2o:actsOn ?sub .
        }}"""

    def decode(self, rows, schema):
        out = np.zeros((schema.n_line, schema.n_sub), dtype=bool)
        for row in rows:
            out[iri.index_of(row[0]), iri.index_of(row[1])] = True
        return out

    def sig(self, schema):
        return QuerySig(shape=(schema.n_line, schema.n_sub), dtype=np.dtype(bool), low=0.0, high=1.0)

    def kernel(self, schema: GridSchema) -> Kernel:
        tau, k, adj = self.params["tau"], self.params["k"], self.params["adjacency"]
        a = schema.require_actions()
        idx = self.params["action"]
        sub = int(a.sub[idx]) if a.is_set_bus[idx] else -1
        static_adj = substation_adjacency(schema) if adj == "substation" else None
        line_sub = np.zeros((schema.n_line, schema.n_sub), dtype=bool)
        line_sub[np.arange(schema.n_line), schema.line_subs[:, 0]] = True
        line_sub[np.arange(schema.n_line), schema.line_subs[:, 1]] = True

        def run(state) -> np.ndarray:
            out = np.zeros((schema.n_line, schema.n_sub), dtype=bool)
            if sub < 0:
                return out
            risk = _risk_mask(state, tau)
            if not risk.any():
                return out
            A = static_adj if static_adj is not None else state.line_adjacency
            for r in np.flatnonzero(risk):
                # the SPARQL: ?risk (self | 1..k hops) ?near, ?near has an end at ?sub
                near = k_hop_over(A, np.array([r]), k)
                if line_sub[near, sub].any():
                    out[r, sub] = True
            return out

        return run

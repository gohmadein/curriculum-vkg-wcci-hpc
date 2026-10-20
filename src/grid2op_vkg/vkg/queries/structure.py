"""Structural queries: what is attached to what, and what that implies.

These are the queries that pay for the ontology. ``topo_vect`` already tells you
that terminal 24 is on busbar 2; it does not tell you that line 7 and line 12
are therefore no longer electrically adjacent, which is the fact that decides
whether an action on substation 5 can relieve line 12 at all.
"""

from __future__ import annotations

import numpy as np

from grid2op_vkg.vkg.queries.base import Kernel, LineMaskQuery, LinePairQuery, SubMaskQuery
from grid2op_vkg.vkg.schema import GridSchema

#: Predicate for each adjacency notion the relevance radius may walk.
ADJACENCY_PREDICATE = {"electrical": "g2o:electricallyAdjacent", "substation": "g2o:sharesSubstationWith"}


def adjacency_path(k: int, adjacency: str = "electrical") -> str:
    """SPARQL property path matching 1..k steps of an adjacency predicate.

    SPARQL 1.1 dropped the ``{n,m}`` range syntax, so bounded reachability has
    to be written as an alternation of fixed-length paths. ``+`` would be
    unbounded, which is a different question.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    pred = ADJACENCY_PREDICATE[adjacency]
    return "|".join("/".join([pred] * i) for i in range(1, k + 1))


def k_hop_over(adj: np.ndarray, sources: np.ndarray, k: int) -> np.ndarray:
    """Boolean mask of nodes within ``k`` hops of ``sources`` over a boolean adjacency matrix."""
    reach = np.zeros(adj.shape[0], dtype=bool)
    reach[np.asarray(sources, dtype=np.int64)] = True
    frontier = reach.copy()
    for _ in range(k):
        if not frontier.any():
            break
        nxt = adj[frontier].any(axis=0) & ~reach
        reach |= nxt
        frontier = nxt
    return reach


def substation_adjacency(schema: GridSchema) -> np.ndarray:
    """(n_line, n_line) bool: lines meeting at a substation, whatever its configuration."""
    n = schema.n_line
    adj = np.zeros((n, n), dtype=bool)
    for end_a in (0, 1):
        for end_b in (0, 1):
            a = schema.line_subs[:, end_a][:, None]
            b = schema.line_subs[:, end_b][None, :]
            adj |= a == b
    np.fill_diagonal(adj, False)
    adj.flags.writeable = False
    return adj


class SubIsSplit(SubMaskQuery):
    """Substations currently spread over more than one electrical node."""

    base_name = "sub_is_split"
    required_obs_attrs = frozenset({"topo_vect", "line_status"})

    def sparql(self, schema: GridSchema) -> str:
        return """
        SELECT DISTINCT ?sub WHERE {
            ?ta g2o:atSubstation ?sub ; g2o:connectedTo ?ba .
            ?tb g2o:atSubstation ?sub ; g2o:connectedTo ?bb .
            FILTER(?ba != ?bb)
        }"""

    def kernel(self, schema: GridSchema) -> Kernel:
        def run(state) -> np.ndarray:
            return state.sub_is_split

        return run


class ControllableSubs(SubMaskQuery):
    """Substations the agent may act on right now.

    Cooldown is a game rule, not semantics; this query exists so the *legality*
    half of the mask can be given to every ablation arm (see the fairness split
    in ``docs/vkg-module-design.md`` 5.2).
    """

    base_name = "controllable_subs"
    required_obs_attrs = frozenset({"time_before_cooldown_sub"})

    def __init__(self, min_terminals: int = 3) -> None:
        super().__init__(min_terminals=min_terminals)

    def sparql(self, schema: GridSchema) -> str:
        return f"""
        SELECT ?sub WHERE {{
            ?sub a g2o:Substation ; g2o:cooldown ?cd ; g2o:terminalCount ?n .
            FILTER(?cd = 0 && ?n >= {self.params["min_terminals"]})
        }}"""

    def kernel(self, schema: GridSchema) -> Kernel:
        big_enough = schema.sub_info >= self.params["min_terminals"]

        def run(state) -> np.ndarray:
            return (state.cooldown_sub == 0) & big_enough

        return run


class ReconnectableLines(LineMaskQuery):
    """Out-of-service lines whose cooldown has expired."""

    base_name = "reconnectable_lines"
    required_obs_attrs = frozenset({"line_status", "time_before_cooldown_line"})

    def sparql(self, schema: GridSchema) -> str:
        return """
        SELECT ?line WHERE {
            ?line a g2o:PowerLine ; g2o:inService ?svc ; g2o:cooldown ?cd .
            FILTER(?svc = false && ?cd = 0)
        }"""

    def kernel(self, schema: GridSchema) -> Kernel:
        def run(state) -> np.ndarray:
            return (~state.line_status) & (state.cooldown_line == 0)

        return run


class LineElectricalAdjacency(LinePairQuery):
    """In-service lines with an end on the same electrical node.

    The reference path reads the ``g2o:electricallyAdjacent`` triples the
    backend materialises, so this query also serves as the conformance check on
    that reasoning step itself.
    """

    base_name = "line_adjacency"
    required_obs_attrs = frozenset({"topo_vect", "line_status"})
    uses_reasoning = True

    def sparql(self, schema: GridSchema) -> str:
        return """
        SELECT DISTINCT ?a ?b WHERE {
            ?a g2o:electricallyAdjacent ?b .
        }"""

    def kernel(self, schema: GridSchema) -> Kernel:
        def run(state) -> np.ndarray:
            return state.line_adjacency

        return run


class LinesSharingSubstation(LinePairQuery):
    """Lines meeting at a substation, *regardless* of how it is currently split.

    The purely topographic relation, materialised as ``g2o:sharesSubstationWith``
    by every backend. Comparing it with :class:`LineElectricalAdjacency` is what
    makes a bus split legible: the difference between the two is exactly what
    the agent bought by splitting.
    """

    base_name = "lines_sharing_substation"
    required_obs_attrs = frozenset()
    static = True

    def sparql(self, schema: GridSchema) -> str:
        return """
        SELECT DISTINCT ?a ?b WHERE {
            ?a g2o:sharesSubstationWith ?b .
        }"""

    def kernel(self, schema: GridSchema) -> Kernel:
        adj = substation_adjacency(schema)

        def run(state) -> np.ndarray:
            return adj

        return run

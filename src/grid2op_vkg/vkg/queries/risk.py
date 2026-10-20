"""Risk queries: where the grid hurts, and what is near enough to help.

Thresholds and weights are parameters, never constants -- the values belong in
``VkgConfig`` and are still open pending the third design document. What is
fixed here is the *shape* of the predicates.
"""

from __future__ import annotations

import numpy as np

from grid2op_vkg.vkg.queries.base import Kernel, LineMaskQuery, LineValueQuery, ScalarQuery
from grid2op_vkg.vkg.queries.structure import adjacency_path, k_hop_over, substation_adjacency
from grid2op_vkg.vkg.schema import GridSchema

#: Observation fields every risk predicate reads.
_RISK_ATTRS = frozenset({"rho", "timestep_overflow", "line_status"})


def _risk_pattern(var: str, tau: float) -> str:
    """Triple pattern binding ``?var`` to an in-service line that is at risk.

    "At risk" is deliberately broader than "overloaded": a line whose overflow
    counter has already started is on a protection timer even if rho has since
    dropped back, and it is the timer that ends the episode.
    """
    return f"""
            ?{var} a g2o:PowerLine ;
                   g2o:inService ?{var}_svc ;
                   g2o:rho ?{var}_rho ;
                   g2o:timestepOverflow ?{var}_ov .
            FILTER(?{var}_svc = true && (?{var}_rho >= {tau:.12g} || ?{var}_ov > 0))"""


def _risk_mask(state, tau: float) -> np.ndarray:
    return state.line_status & ((state.rho >= tau) | (state.timestep_overflow > 0))


def near_risk_kernel(schema: GridSchema, tau: float, k: int, adjacency: str):
    """Shared kernel: lines at risk plus everything within k hops over the chosen adjacency."""
    static_adj = substation_adjacency(schema) if adjacency == "substation" else None

    def run(state) -> np.ndarray:
        risk = _risk_mask(state, tau)
        if not risk.any():
            return risk
        adj = static_adj if static_adj is not None else state.line_adjacency
        return k_hop_over(adj, np.flatnonzero(risk), k)

    return run


class LinesAtRisk(LineMaskQuery):
    """In-service lines at or above ``tau``, or already counting down to a trip."""

    base_name = "lines_at_risk"
    required_obs_attrs = _RISK_ATTRS

    def __init__(self, tau: float = 0.95) -> None:
        super().__init__(tau=float(tau))

    def sparql(self, schema: GridSchema) -> str:
        return f"SELECT DISTINCT ?line WHERE {{{_risk_pattern('line', self.params['tau'])}\n        }}"

    def kernel(self, schema: GridSchema) -> Kernel:
        tau = self.params["tau"]

        def run(state) -> np.ndarray:
            return _risk_mask(state, tau)

        return run


class LinesNearRisk(LineMaskQuery):
    """Lines at risk, plus everything within ``k`` hops of them.

    This is the relevance set the action mask is built from: an action on a
    substation that touches none of these lines cannot plausibly help.
    ``adjacency`` chooses the relation the hops walk (see ``VkgConfig``).
    """

    base_name = "lines_near_risk"
    required_obs_attrs = _RISK_ATTRS | {"topo_vect"}
    uses_reasoning = True

    def __init__(self, tau: float = 0.95, k: int = 2, adjacency: str = "electrical") -> None:
        super().__init__(tau=float(tau), k=int(k), adjacency=str(adjacency))

    def sparql(self, schema: GridSchema) -> str:
        tau, k, adj = self.params["tau"], self.params["k"], self.params["adjacency"]
        return f"""
        SELECT DISTINCT ?line WHERE {{
            {{{_risk_pattern('line', tau)}
            }}
            UNION
            {{{_risk_pattern('risk', tau)}
                ?risk {adjacency_path(k, adj)} ?line .
            }}
        }}"""

    def kernel(self, schema: GridSchema) -> Kernel:
        return near_risk_kernel(schema, self.params["tau"], self.params["k"], self.params["adjacency"])


class LineMargin(LineValueQuery):
    """Per-line headroom ``tau - rho``, clipped at 0 below; 0 for dead lines.

    A dense signal, unlike the binary risk mask -- useful as a state feature and
    as the per-line term of the potential function.
    """

    base_name = "line_margin"
    required_obs_attrs = frozenset({"rho", "line_status"})
    value_low = 0.0
    value_high = 2.0

    def __init__(self, tau: float = 1.0) -> None:
        super().__init__(tau=float(tau))

    def sparql(self, schema: GridSchema) -> str:
        tau = self.params["tau"]
        return f"""
        SELECT ?line ?margin WHERE {{
            ?line a g2o:PowerLine ; g2o:inService ?svc ; g2o:rho ?rho .
            FILTER(?svc = true && ?rho < {tau:.12g})
            BIND({tau:.12g} - ?rho AS ?margin)
        }}"""
        # lines that fall through the FILTER are unbound and decode to the fill value 0;
        # written this way rather than with IF() because Ontop cannot push a conditional
        # BIND into SQL, and both engines have to run the same text

    def kernel(self, schema: GridSchema) -> Kernel:
        tau = self.params["tau"]

        def run(state) -> np.ndarray:
            return np.where(state.line_status, np.maximum(tau - state.rho, 0.0), 0.0)

        return run


class OverloadPressure(ScalarQuery):
    """``sum over in-service lines of max(rho - tau, 0)``.

    The natural main term of the shaping potential: zero while the grid is
    comfortable, growing smoothly as lines approach their limit, so it gives the
    agent gradient in exactly the region where the raw reward gives none.
    """

    base_name = "overload_pressure"
    required_obs_attrs = frozenset({"rho", "line_status"})
    value_low = 0.0
    value_high = np.inf

    def __init__(self, tau: float = 0.95) -> None:
        super().__init__(tau=float(tau))

    def sparql(self, schema: GridSchema) -> str:
        tau = self.params["tau"]
        return f"""
        SELECT (SUM(?excess) AS ?value) WHERE {{
            ?line a g2o:PowerLine ; g2o:inService ?svc ; g2o:rho ?rho .
            FILTER(?svc = true && ?rho > {tau:.12g})
            BIND(?rho - {tau:.12g} AS ?excess)
        }}"""
        # SUM over no rows is 0 (or unbound, which decodes to the fill value 0)

    def kernel(self, schema: GridSchema) -> Kernel:
        tau = self.params["tau"]

        def run(state) -> np.ndarray:
            excess = np.where(state.line_status, np.maximum(state.rho - tau, 0.0), 0.0)
            return np.array(excess.sum(), dtype=np.float64)

        return run


class HeadroomNearRisk(ScalarQuery):
    """Upward redispatch margin of generators electrically close to a risk line.

    Free topology actions are preferred, but when they run out the agent has to
    pay -- and whether paying will even help depends on whether there is a
    generator in the right *place*. That is a graph question, not a vector one,
    and it is the clearest example of a feature the raw observation cannot
    express without the network first learning the grid's structure.
    """

    base_name = "headroom_near_risk"
    required_obs_attrs = _RISK_ATTRS | {"topo_vect", "gen_p"}
    uses_reasoning = True
    value_low = 0.0  # a sum of upward margins; declared so, since the Box is what the policy sees
    value_high = np.inf

    def __init__(self, tau: float = 0.95, k: int = 2, adjacency: str = "electrical") -> None:
        super().__init__(tau=float(tau), k=int(k), adjacency=str(adjacency))

    def sparql(self, schema: GridSchema) -> str:
        tau, k, adj = self.params["tau"], self.params["k"], self.params["adjacency"]
        # the inner SELECT DISTINCT stops a generator reachable by several paths
        # from being counted more than once
        return f"""
        SELECT (SUM(?margin) AS ?value) WHERE {{
            {{
                SELECT DISTINCT ?gen ?margin WHERE {{
                    ?gen a g2o:Generator ; g2o:marginUp ?margin ; g2o:hasTerminal ?gt .
                    ?gt g2o:connectedTo ?bus .
                    ?line a g2o:PowerLine ; g2o:hasTerminal ?lt .
                    ?lt g2o:connectedTo ?bus .
                    {{{_risk_pattern('line', tau)}
                    }}
                    UNION
                    {{{_risk_pattern('risk', tau)}
                        ?risk {adjacency_path(k, adj)} ?line .
                    }}
                }}
            }}
        }}"""

    def kernel(self, schema: GridSchema) -> Kernel:
        near_fn = near_risk_kernel(schema, self.params["tau"], self.params["k"], self.params["adjacency"])
        gen_terms = schema.gen_terminals
        line_terms = schema.line_terminals

        def run(state) -> np.ndarray:
            near = near_fn(state)
            if not near.any():
                return np.array(0.0)
            buses = state.terminal_bus[line_terms[near]].ravel()
            buses = buses[buses >= 0]
            if buses.size == 0:
                return np.array(0.0)
            gen_bus = state.terminal_bus[gen_terms]
            close = np.isin(gen_bus, buses) & (gen_bus >= 0)
            return np.array(state.gen_margin_up[close].sum(), dtype=np.float64)

        return run

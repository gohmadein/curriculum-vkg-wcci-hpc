"""Queries over the remedial-action catalogue (``g2o:RemedialEffect``).

The catalogue stores, per (action, line), the mean signed change of the line's
loading the action caused in the reference topology. Joined with the live
loading of every in-service line it predicts, per candidate, the loading
profile after the action; its maximum is the number the search cares about.

``predicted_max_rho``  per action, max over in-service lines of rho_l - effect[a, l]
                       (a zero effect -- stored as such in the graph, absent in the
                       file -- keeps the line's loading; do-nothing and line-status
                       actions predict the current maximum)
``remedial_score``     per action, the predicted *reduction* of the maximum loading,
                       clipped at 0: max(0, max rho now - predicted_max_rho); 0 when
                       the grid is not at risk. Clipping loses nothing a decision
                       uses (every consumer thresholds it at theta > 0) and lets the
                       kernel skip the ~99 % of actions that cannot help
``remedial_actions``   actions whose predicted reduction is at least ``theta`` while
                       some line is at risk -- electrical relevance, wherever the
                       action's substation is

All three are zero / False when the schema carries no catalogue, so every
agent that did not load one is unchanged.
"""

from __future__ import annotations

import numpy as np

from grid2op_vkg.vkg.queries.base import ActionMaskQuery, ActionValueQuery, Kernel
from grid2op_vkg.vkg.queries.risk import _RISK_ATTRS, _risk_mask, _risk_pattern
from grid2op_vkg.vkg.schema import GridSchema

_PRED_ATTRS = frozenset({"rho", "line_status"})

#: predicted post-action maximum loading per action, as a SPARQL group. The
#: RemedialEffect relation is total over (action, line) in the graph (zero = no
#: measurable effect), so this is a plain join -- no OPTIONAL.
_PRED_MAX = """
            SELECT ?action (MAX(?rho - ?eff) AS ?pred) WHERE {
                ?e a g2o:RemedialEffect ; g2o:ofAction ?action ; g2o:onLine ?line ; g2o:effect ?eff .
                ?line g2o:inService ?svc ; g2o:rho ?rho . FILTER(?svc = true)
            }
            GROUP BY ?action"""

#: current maximum loading over in-service lines, as a SPARQL group
_MAX_NOW = """
            SELECT (MAX(?r) AS ?rmax) WHERE {
                ?l a g2o:PowerLine ; g2o:inService ?s ; g2o:rho ?r . FILTER(?s = true)
            }"""


def _predicted_max(schema: GridSchema, state) -> np.ndarray:
    """The catalogue's predicted max loading per action, computed once per state.

    Four queries (and the relevance branch) need it every stressed step; on
    wcci_2022 it is a 73k x 186 subtraction, so it is memoised on the state's
    derived-view cache like the other per-step views.
    """
    cat = schema.remedial
    n = schema.require_actions().n
    if cat is None:
        live = state.line_status
        now = float(state.rho[live].max()) if live.any() else 0.0
        return np.full(n, now, dtype=np.float64)
    key = ("predicted_max_rho", id(cat))
    cache = getattr(state, "_cache", None)
    if cache is not None and key in cache:
        return cache[key]
    out = cat.predicted_max_rho(state.rho, state.line_status)
    if cache is not None:
        cache[key] = out
    return out


def _max_now(state) -> float:
    live = state.line_status
    return float(state.rho[live].max()) if live.any() else 0.0


def remedial_score_of(schema: GridSchema, state, tau: float) -> np.ndarray:
    """Clipped predicted reduction per action, memoised on the state; zeros without a catalogue or risk.

    Shared by ``remedial_score``, ``remedial_actions`` and the electrical branch
    of ``relevant_actions``, so a stressed step pays for it once.
    """
    n = schema.require_actions().n
    cat = schema.remedial
    if cat is None or not _risk_mask(state, tau).any():
        return np.zeros(n, dtype=np.float64)
    key = ("remedial_score", id(cat))
    cache = getattr(state, "_cache", None)
    if cache is not None and key in cache:
        return cache[key]
    out = cat.reduction_score(state.rho, state.line_status)
    if cache is not None:
        cache[key] = out
    return out


def _check_theta(theta: float) -> float:
    theta = float(theta)
    if not theta > 0:
        raise ValueError(f"remedial theta must be > 0 (the score is clipped at 0), got {theta}")
    return theta


class PredictedMaxRho(ActionValueQuery):
    """Per action, the catalogue's predicted maximum loading after it."""

    base_name = "predicted_max_rho"
    required_obs_attrs = _PRED_ATTRS
    uses_reasoning = True
    fill = 0.0

    def sparql(self, schema: GridSchema) -> str:
        return f"""SELECT ?action (?pred AS ?value) WHERE {{
            {{{_PRED_MAX}
            }}
        }}"""

    def kernel(self, schema: GridSchema) -> Kernel:
        def run(state) -> np.ndarray:
            return _predicted_max(schema, state)

        return run


class RemedialScore(ActionValueQuery):
    """Per action, the predicted reduction of the maximum loading, clipped at 0; 0 unless a line is at risk."""

    base_name = "remedial_score"
    required_obs_attrs = _RISK_ATTRS | _PRED_ATTRS
    uses_reasoning = True
    fill = 0.0

    def __init__(self, tau: float = 0.95) -> None:
        super().__init__(tau=float(tau))

    def sparql(self, schema: GridSchema) -> str:
        tau = self.params["tau"]
        # max(x, 0) written as (x + |x|) / 2: exact in floating point, and ABS is
        # supported by every engine we target (Ontop rewrites IF less reliably)
        return f"""SELECT ?action (((?rmax - ?pred) + ABS(?rmax - ?pred)) / 2 AS ?value) WHERE {{
            {{ SELECT (COUNT(?any) AS ?nrisk) WHERE {{{_risk_pattern("any", tau)}
            }} }}
            FILTER(?nrisk > 0)
            {{{_MAX_NOW}
            }}
            {{{_PRED_MAX}
            }}
        }}"""

    def kernel(self, schema: GridSchema) -> Kernel:
        tau = self.params["tau"]

        def run(state) -> np.ndarray:
            return remedial_score_of(schema, state, tau)

        return run


class RemedialActions(ActionMaskQuery):
    """Actions the catalogue expects to lower the maximum loading by at least ``theta`` while a line is at risk."""

    base_name = "remedial_actions"
    required_obs_attrs = _RISK_ATTRS | _PRED_ATTRS
    uses_reasoning = True

    def __init__(self, tau: float = 0.95, theta: float = 0.02) -> None:
        super().__init__(tau=float(tau), theta=_check_theta(theta))

    def sparql(self, schema: GridSchema) -> str:
        tau, theta = self.params["tau"], self.params["theta"]
        return f"""SELECT ?action WHERE {{
            {{ SELECT (COUNT(?any) AS ?nrisk) WHERE {{{_risk_pattern("any", tau)}
            }} }}
            FILTER(?nrisk > 0)
            {{{_MAX_NOW}
            }}
            {{{_PRED_MAX}
            }}
            FILTER(?rmax - ?pred >= {theta:.12g})
        }}"""

    def kernel(self, schema: GridSchema) -> Kernel:
        tau, theta = self.params["tau"], self.params["theta"]
        n = schema.require_actions().n

        def run(state) -> np.ndarray:
            if schema.remedial is None:
                return np.zeros(n, dtype=bool)
            return remedial_score_of(schema, state, tau) >= theta

        return run

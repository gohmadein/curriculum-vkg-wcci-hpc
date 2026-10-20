"""Budget queries: how to spend a limited number of ``simulate()`` calls.

These exist because of a measurement, not a hunch. On case14 the candidate set
is ~44 actions against a 30-simulation budget, so the *order* already decides
what gets looked at -- spending the whole budget on substation 5 (57
configurations on its own) is what used to hide ExpertAgent's cooldown fallback
at substation 1. On a 118-substation grid the candidate set is orders of
magnitude larger than any affordable budget, and ordering stops being an
optimisation and becomes the thing that makes search possible at all.

The split of responsibility is deliberate:

* **here** live the facts that depend on the grid and therefore on the data
  layer -- which substation an action acts on, how many terminals it really
  moves, which substations sit at the most loaded line, how much topological
  debt is already outstanding, which actions restore the reference topology;
* **in the search** lives the arithmetic that combines them -- group by
  substation, interleave round-robin, stop at the budget. That code mentions no
  grid vocabulary at all, which is the point: after a schema change the
  mappings move and the search does not.

``ActionDisruption`` counts only *live* terminals. ``onBusbar`` is asserted for
connected terminals alone in both backends (``rdf.py`` skips ``tb[t] < 0``, the
R2RML source has ``WHERE bus IS NOT NULL``), and counting a terminal of an
out-of-service line as "moved" would be wrong anyway -- it is not on a busbar
to move from.
"""

from __future__ import annotations

import numpy as np

from grid2op_vkg.vkg.queries.base import ActionMaskQuery, ActionValueQuery, Kernel, SubMaskQuery
from grid2op_vkg.vkg.schema import GridSchema


class ActionDisruption(ActionValueQuery):
    """How many live terminals a topology action actually reassigns.

    Zero for do-nothing and for line-status actions, and zero for a topology
    action that re-asserts the configuration already in place. Cheap
    reconfigurations are probed first: they are the ones an operator would
    rather perform, and they are the ones most likely to be undoable later.
    """

    base_name = "action_disruption"
    required_obs_attrs = frozenset({"topo_vect", "line_status"})

    def sparql(self, schema: GridSchema) -> str:
        return """
        SELECT ?action (COUNT(?t) AS ?value) WHERE {
            ?action a g2o:TopologyAction ; g2o:assigns ?as .
            ?as g2o:assignsTerminal ?t ; g2o:assignsBusbar ?b .
            ?t g2o:onBusbar ?cur .
            FILTER(?cur != ?b)
        }
        GROUP BY ?action"""

    def kernel(self, schema: GridSchema) -> Kernel:
        a = schema.require_actions()
        assigns = a.set_bus != 0

        def run(state) -> np.ndarray:
            live = state.terminal_bus >= 0
            moved = assigns & live[None, :] & (a.set_bus != state.topo_vect[None, :])
            return moved.sum(axis=1).astype(np.float64)

        return run


class PrioritySubstations(SubMaskQuery):
    """Substations at an end of a most-loaded line -- probe these first.

    The relevance mask already answers *which* substations may be touched; this
    answers which of them deserves the budget first. It is the endpoints of the
    line closest to its thermal limit, which is where a reconfiguration has the
    most direct effect on the quantity about to cause a cascade.
    """

    base_name = "priority_substations"
    required_obs_attrs = frozenset({"rho"})

    def sparql(self, schema: GridSchema) -> str:
        return """
        SELECT DISTINCT ?sub WHERE {
            { SELECT (MAX(?anyrho) AS ?peak) WHERE { ?any a g2o:PowerLine ; g2o:rho ?anyrho . } }
            ?line a g2o:PowerLine ; g2o:rho ?rho ; g2o:hasTerminal ?t .
            FILTER(?rho = ?peak)
            ?t g2o:atSubstation ?sub .
        }"""

    def kernel(self, schema: GridSchema) -> Kernel:
        line_subs = schema.line_subs

        def run(state) -> np.ndarray:
            out = np.zeros(schema.n_sub, dtype=bool)
            rho = state.rho
            if rho.size:
                out[line_subs[rho >= rho.max()].ravel()] = True
            return out

        return run


class AffordableActions(ActionMaskQuery):
    """Actions that do not take on topological debt we cannot service.

    Every split locks its substation for the cooldown and moves the grid one
    step away from the reference topology; a split we cannot merge back is a
    substation unavailable for the *next* overload, and there will be one. Past
    ``max_split_subs`` outstanding splits, stop opening new ones -- actions that
    reconfigure an already-split substation, restore it, or touch line status
    stay available.

    Semantic, not a game rule: grid2op happily lets you split every substation.
    """

    base_name = "affordable_actions"
    required_obs_attrs = frozenset({"topo_vect", "line_status"})

    def __init__(self, max_split_subs: int = 2) -> None:
        super().__init__(max_split_subs=int(max_split_subs))

    def sparql(self, schema: GridSchema) -> str:
        cap = self.params["max_split_subs"]
        return f"""
        SELECT DISTINCT ?action WHERE {{
            {{
                # under the cap everything is affordable
                {{ SELECT (COUNT(?s) AS ?ndebt) WHERE {{
                    ?s a g2o:Substation ; g2o:isSplit ?sp . FILTER(?sp = true)
                }} }}
                FILTER(?ndebt < {cap})
                ?action a ?cls . FILTER(?cls IN (g2o:DoNothingAction, g2o:TopologyAction, g2o:LineStatusAction))
            }}
            UNION
            {{
                # at or over the cap, keep everything that does not open a new split
                {{ SELECT (COUNT(?s) AS ?ndebt) WHERE {{
                    ?s a g2o:Substation ; g2o:isSplit ?sp . FILTER(?sp = true)
                }} }}
                FILTER(?ndebt >= {cap})
                {{ ?action a ?cls . FILTER(?cls IN (g2o:DoNothingAction, g2o:LineStatusAction)) }}
                UNION
                {{
                    ?action a g2o:TopologyAction ; g2o:actsOn ?sub .
                    ?sub g2o:isSplit ?issplit . FILTER(?issplit = true)
                }}
                UNION
                {{
                    ?action a g2o:TopologyAction ; g2o:actsOn ?sub .
                    ?sub g2o:isSplit ?issplit . FILTER(?issplit = false)
                    FILTER NOT EXISTS {{
                        ?action g2o:assigns ?as . ?as g2o:assignsBusbar ?nb . FILTER(?nb != 1)
                    }}
                }}
            }}
        }}"""

    def kernel(self, schema: GridSchema) -> Kernel:
        cap = self.params["max_split_subs"]
        a = schema.require_actions()
        sb = a.is_set_bus
        # a topology action "opens" a substation when it puts something on a
        # busbar other than the first
        opens = sb & (a.set_bus == 2).any(axis=1)
        sub_of = np.maximum(a.sub, 0)

        def run(state) -> np.ndarray:
            split = state.sub_is_split
            if int(split.sum()) < cap:
                return np.ones(a.n, dtype=bool)
            return ~(opens & ~split[sub_of])

        return run


class RecoveryActions(ActionMaskQuery):
    """Merge a split substation back to its reference topology.

    Restorative rather than corrective, and worth a little rho: ExpertAgent
    merges back at rho ~0.89 precisely because holding a split costs it the
    substation next time. Judged against "does the grid stay calm", not against
    "does this beat doing nothing" -- which is why it is its own query and not
    part of the relevance mask.
    """

    base_name = "recovery_actions"
    required_obs_attrs = frozenset({"topo_vect", "line_status", "time_before_cooldown_sub"})

    def sparql(self, schema: GridSchema) -> str:
        return """
        SELECT DISTINCT ?action WHERE {
            ?sub a g2o:Substation ; g2o:isSplit ?split ; g2o:cooldown ?cd .
            FILTER(?split = true && ?cd = 0)
            ?action a g2o:TopologyAction ; g2o:actsOn ?sub .
            FILTER NOT EXISTS {
                ?action g2o:assigns ?as . ?as g2o:assignsBusbar ?b . FILTER(?b != 1)
            }
        }"""

    def kernel(self, schema: GridSchema) -> Kernel:
        a = schema.require_actions()
        sb = a.is_set_bus
        all_bus1 = sb & ~(a.set_bus == 2).any(axis=1)
        sub_of = np.maximum(a.sub, 0)

        def run(state) -> np.ndarray:
            ok = state.sub_is_split & (state.cooldown_sub == 0)
            return all_bus1 & ok[sub_of]

        return run

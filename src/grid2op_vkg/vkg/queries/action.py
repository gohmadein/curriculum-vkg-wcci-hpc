"""Action queries: which candidate actions are legal, relevant, or harmful.

This is the control layer earning its keep. Because the candidate actions are
in the graph, "actions that could relieve the overloaded line" is a join
between the action's assignments, the terminals they move, the substation
those terminals sit at, and the lines currently at risk -- one SPARQL pattern
instead of a hand-maintained lookup.

Version 1 of the relevance rule is lifted directly from what ExpertAgent does
on case14 (results/README.md): with line 17 at rho 1.01 it splits substation
4, moving two of the three inflows to the other busbar, then merges back once
rho is comfortable. Generalised:

  at risk      -> reconfigure an endpoint substation of a line near the risk,
                  provided the action actually changes something there;
                  or drop a line electrically adjacent to the overloaded one
  not at risk  -> merge split substations back, reconnect what can be
  always       -> do nothing

Islanding is kept separate from relevance so the paper can report how often
the semantic filter and the structural filter disagree.
"""

from __future__ import annotations

import numpy as np

from grid2op_vkg.vkg.queries.base import ActionMaskQuery, Kernel
from grid2op_vkg.vkg.queries.risk import _risk_mask, _risk_pattern, near_risk_kernel
from grid2op_vkg.vkg.queries.structure import adjacency_path
from grid2op_vkg.vkg.schema import GridSchema

_ACTION_ATTRS = frozenset({"topo_vect", "line_status", "time_before_cooldown_line", "time_before_cooldown_sub"})


class LegalActions(ActionMaskQuery):
    """Allowed by the game rules right now.

    * do-nothing: always
    * set_bus: the substation is off cooldown, and none of the terminals moved
      belongs to an out-of-service line (grid2op would treat that as an implicit
      reconnection; keep reconnection explicit)
    * line status: the line is off cooldown, and the change is a change

    Rules, not semantics. Given to every arm (see design doc 5.2).
    """

    base_name = "legal_actions"
    required_obs_attrs = _ACTION_ATTRS

    def sparql(self, schema: GridSchema) -> str:
        return """
        SELECT DISTINCT ?action WHERE {
            { ?action a g2o:DoNothingAction . }
            UNION
            {
                ?sub a g2o:Substation ; g2o:cooldown ?cd . FILTER(?cd = 0)
                ?action a g2o:TopologyAction ; g2o:actsOn ?sub .
                # subtract the actions that would implicitly reconnect a dead line:
                # start from the (few) dead lines, not from the (many) actions
                MINUS {
                    ?dead a g2o:PowerLine ; g2o:inService ?dsvc ; g2o:hasTerminal ?t . FILTER(?dsvc = false)
                    ?as g2o:assignsTerminal ?t .
                    ?action g2o:assigns ?as .
                }
            }
            UNION
            {
                ?action a g2o:LineStatusAction ; g2o:actsOn ?line ; g2o:setsLineStatus ?st .
                ?line g2o:cooldown ?cd ; g2o:inService ?svc .
                FILTER(?cd = 0 && ((?st = 1 && ?svc = false) || (?st = -1 && ?svc = true)))
            }
        }"""

    def kernel(self, schema: GridSchema) -> Kernel:
        a = schema.require_actions()
        touches = a.set_bus != 0  # (n, dim_topo)
        line_of_term = np.full(schema.dim_topo, -1, dtype=np.int32)
        line_of_term[schema.line_terminals[:, 0]] = np.arange(schema.n_line)
        line_of_term[schema.line_terminals[:, 1]] = np.arange(schema.n_line)

        def run(state) -> np.ndarray:
            out = a.is_nothing.copy()
            # set_bus
            sb = a.is_set_bus
            sub_ok = np.zeros(a.n, dtype=bool)
            sub_ok[sb] = state.cooldown_sub[a.sub[sb]] == 0
            dead_term = np.zeros(schema.dim_topo, dtype=bool)
            dead_lines = np.flatnonzero(~state.line_status)
            dead_term[schema.line_terminals[dead_lines].ravel()] = True
            touches_dead = (touches & dead_term[None, :]).any(axis=1)
            out |= sb & sub_ok & ~touches_dead
            # line status
            ls = a.is_line_status
            ln = a.line[ls]
            ok = (state.cooldown_line[ln] == 0) & (
                ((a.status[ls] == 1) & ~state.line_status[ln]) | ((a.status[ls] == -1) & state.line_status[ln])
            )
            out[ls] = ok
            return out

        return run


class RelevantActions(ActionMaskQuery):
    """Actions the semantics say could help *now* -- the VKG-Mask increment."""

    base_name = "relevant_actions"
    required_obs_attrs = _ACTION_ATTRS | {"rho", "timestep_overflow"}
    uses_reasoning = True

    def __init__(
        self, tau: float = 0.95, k: int = 2, adjacency: str = "electrical", remedial_theta: float = -1.0
    ) -> None:
        # remedial_theta >= 0 switches the electrical branch on: actions whose
        # catalogued relief on a line at risk is at least that much are relevant
        # wherever their substation is. -1 (default) keeps membership topographic.
        remedial_theta = float(remedial_theta)
        if remedial_theta != -1.0 and not remedial_theta > 0:
            raise ValueError(f"remedial_theta must be -1 (off) or > 0, got {remedial_theta}")
        super().__init__(tau=float(tau), k=int(k), adjacency=str(adjacency), remedial_theta=remedial_theta)

    def sparql(self, schema: GridSchema) -> str:
        tau, k, adj = self.params["tau"], self.params["k"], self.params["adjacency"]
        theta = self.params["remedial_theta"]
        remedial = (
            f"""
            UNION
            {{
                # electrical relevance: joined with the live loading of every line, the
                # remedial catalogue (signed effects, compiled offline in the reference
                # topology) predicts this action lowers the *maximum* loading by at
                # least {theta:.12g} -- side effects included, wherever its substation is
                {{ SELECT (COUNT(?any) AS ?nrisk) WHERE {{{_risk_pattern('any', tau)}
                }} }}
                FILTER(?nrisk > 0)
                {{
                    SELECT (MAX(?r) AS ?rmax) WHERE {{
                        ?l a g2o:PowerLine ; g2o:inService ?s ; g2o:rho ?r . FILTER(?s = true)
                    }}
                }}
                {{
                    SELECT ?action (MAX(?rho - ?eff) AS ?pred) WHERE {{
                        ?e a g2o:RemedialEffect ; g2o:ofAction ?action ; g2o:onLine ?line ; g2o:effect ?eff .
                        ?line g2o:inService ?svc ; g2o:rho ?rho . FILTER(?svc = true)
                    }}
                    GROUP BY ?action
                }}
                ?action a g2o:TopologyAction .
                FILTER(?rmax - ?pred >= {theta:.12g})
                ?action g2o:assigns ?as .
                ?as g2o:assignsTerminal ?t ; g2o:assignsBusbar ?b .
                ?t g2o:onBusbar ?cur . FILTER(?cur != ?b)
            }}"""
            if theta >= 0
            else ""
        )
        # Each branch starts from the small, selective set (the risk lines) and
        # only then joins the 200-odd actions. Written the other way round,
        # rdflib's left-to-right evaluation builds an action x assignment x
        # terminal cross product before filtering it -- seconds instead of ms.
        near_subs = f"""
                SELECT DISTINCT ?sub WHERE {{
                    {{{_risk_pattern('near', tau)}
                    }}
                    UNION
                    {{{_risk_pattern('risk', tau)}
                        ?risk {adjacency_path(k, adj)} ?near .
                    }}
                    ?near g2o:hasTerminal ?nt . ?nt g2o:atSubstation ?sub .
                }}"""
        return f"""
        SELECT DISTINCT ?action WHERE {{
            {{ ?action a g2o:DoNothingAction . }}
            UNION
            {{ ?action a g2o:LineStatusAction ; g2o:setsLineStatus ?st . FILTER(?st = 1) }}
            UNION
            {{
                # calm grid: merging a split substation back is the restorative move.
                # "calm" is a count of risk lines equal to zero -- a variable-free
                # NOT EXISTS is legal SPARQL but Ontop's rewriter rejects it
                {{ SELECT (COUNT(?any) AS ?nrisk) WHERE {{{_risk_pattern('any', tau)}
                }} }}
                FILTER(?nrisk = 0)
                ?sub g2o:isSplit ?split . FILTER(?split = true)
                ?action a g2o:TopologyAction ; g2o:actsOn ?sub .
                FILTER NOT EXISTS {{ ?action g2o:assigns/g2o:assignsBusbar ?b . FILTER(?b != 1) }}
            }}
            UNION
            {{
                # stressed grid: reconfigure an endpoint substation of a line near the risk,
                # but only if the action changes the current configuration there
                {{{near_subs}
                }}
                ?action a g2o:TopologyAction ; g2o:actsOn ?sub ; g2o:assigns ?as .
                ?as g2o:assignsTerminal ?t ; g2o:assignsBusbar ?b .
                ?t g2o:onBusbar ?cur . FILTER(?cur != ?b)
            }}
            UNION
            {{
                # stressed grid: drop a line that shares a node with the overloaded one
                {{
                    SELECT DISTINCT ?line WHERE {{{_risk_pattern('risk', tau)}
                        ?risk g2o:electricallyAdjacent ?line .
                        FILTER NOT EXISTS {{{_risk_pattern('line', tau)}
                        }}
                    }}
                }}
                ?action a g2o:LineStatusAction ; g2o:actsOn ?line ; g2o:setsLineStatus ?st . FILTER(?st = -1)
            }}{remedial}
        }}"""

    def kernel(self, schema: GridSchema) -> Kernel:
        tau = self.params["tau"]
        theta = self.params["remedial_theta"]
        cat = None if (theta < 0 or schema.remedial is None) else schema.remedial
        near_fn = near_risk_kernel(schema, tau, self.params["k"], self.params["adjacency"])
        a = schema.require_actions()
        # substations touched by each line: (n_line, n_sub) bool
        line_sub = np.zeros((schema.n_line, schema.n_sub), dtype=bool)
        line_sub[np.arange(schema.n_line), schema.line_subs[:, 0]] = True
        line_sub[np.arange(schema.n_line), schema.line_subs[:, 1]] = True
        sb, ls = a.is_set_bus, a.is_line_status
        all_bus1 = sb & ~(a.set_bus == 2).any(axis=1)

        def run(state) -> np.ndarray:
            out = a.is_nothing.copy()
            out |= ls & (a.status == 1)
            risk = _risk_mask(state, tau)
            if not risk.any():
                out |= all_bus1 & state.sub_is_split[np.maximum(a.sub, 0)]
                return out
            near = near_fn(state)
            near_subs = line_sub[near].any(axis=0)  # (n_sub,)
            live = state.terminal_bus >= 0
            cur = np.where(live, state.topo_vect, -999)
            changes = ((a.set_bus != 0) & (a.set_bus != cur[None, :]) & live[None, :]).any(axis=1)
            out |= sb & near_subs[np.maximum(a.sub, 0)] & changes
            adj_to_risk = state.line_adjacency[risk].any(axis=0) & ~risk
            out |= ls & (a.status == -1) & adj_to_risk[np.maximum(a.line, 0)]
            if cat is not None:
                from grid2op_vkg.vkg.queries.remedial import remedial_score_of

                out |= sb & changes & (remedial_score_of(schema, state, tau) >= theta)
            return out

        return run


class IslandingActions(ActionMaskQuery):
    """Bus splits that would strand something.

    Two structural faults, both invisible to a one-step power-flow look-ahead
    because the flow *does* drop when a line quietly disappears:

    * a busbar carrying injections but no line -- the generator or load has
      nowhere to send or draw power and grid2op ends the episode;
    * a busbar carrying a single terminal -- if it is a line end, the line is
      dangling and effectively disconnected without a line-status action (the
      greedy agent found exactly this move and it looked good for one step).

    Rule: every busbar an action uses must carry at least two terminals and at
    least one line. Static, so it never depends on the state -- but it is
    semantic knowledge, not a game rule, so it sits on the relevance side of
    the fairness split, not the legality side.
    """

    base_name = "islanding_actions"
    required_obs_attrs = frozenset()
    static = True

    def sparql(self, schema: GridSchema) -> str:
        return """
        SELECT DISTINCT ?action WHERE {
            {
                # a busbar with an injection but no line
                ?inj a ?cls . FILTER(?cls IN (g2o:Generator, g2o:Load, g2o:Storage))
                ?inj g2o:hasTerminal ?t1 .
                ?as1 g2o:assignsTerminal ?t1 ; g2o:assignsBusbar ?b .
                ?action g2o:assigns ?as1 .
                FILTER NOT EXISTS {
                    ?action g2o:assigns ?as2 . ?as2 g2o:assignsTerminal ?t2 ; g2o:assignsBusbar ?b .
                    ?line a g2o:PowerLine ; g2o:hasTerminal ?t2 .
                }
            }
            UNION
            {
                # a busbar with a single terminal
                SELECT ?action WHERE {
                    ?action a g2o:TopologyAction ; g2o:assigns ?as .
                    ?as g2o:assignsTerminal ?t ; g2o:assignsBusbar ?b .
                }
                GROUP BY ?action ?b
                HAVING (COUNT(?t) = 1)
            }
        }"""

    def kernel(self, schema: GridSchema) -> Kernel:
        a = schema.require_actions()
        is_line_term = np.zeros(schema.dim_topo, dtype=bool)
        is_line_term[schema.line_terminals.ravel()] = True
        out = np.zeros(a.n, dtype=bool)
        for i in np.flatnonzero(a.is_set_bus):
            cfg = a.set_bus[i]
            for b in np.unique(cfg[cfg != 0]):
                on_b = cfg == b
                if on_b.sum() < 2 or not (on_b & is_line_term).any():
                    out[i] = True
                    break
        out.flags.writeable = False

        def run(state) -> np.ndarray:
            return out

        return run

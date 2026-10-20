"""Action-mask plumbing shared by every arm.

The fairness split (design doc 5.2) puts two different things under "mask":

``legality_mask``  the game's own rules -- cooldowns, no implicit reconnection,
                   no reconnecting a live line. Not knowledge, so every arm gets
                   it, and it lives here rather than in a provider.
``compose_mask``   AND-in whatever *semantic* filter a provider returns. Only
                   the VKG-Mask / Handcrafted / VKG-Full arms have one.

Both leave action 0 (do nothing) available: a policy must always have a move.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from grid2op_vkg.vkg.schema import GridSchema


def legality_mask(obs: Any, schema: GridSchema) -> np.ndarray:
    """Rule-level legality of every candidate action for this observation.

    Deliberately written against the raw observation with no VKG machinery, so
    the Raw arm can use it without importing anything semantic. The
    ``legal_actions`` query must agree with it; a test checks that.
    """
    a = schema.require_actions()
    line_status = np.asarray(obs.line_status, dtype=bool)
    cd_sub = np.asarray(obs.time_before_cooldown_sub)
    cd_line = np.asarray(obs.time_before_cooldown_line)

    out = np.zeros(a.n, dtype=bool)
    out[0] = True

    # set_bus: substation off cooldown, and no terminal of a dead line touched
    dead_terms = np.zeros(schema.dim_topo, dtype=bool)
    dead_terms[schema.line_terminals[~line_status].ravel()] = True
    sb = a.is_set_bus
    touches_dead = ((a.set_bus != 0) & dead_terms[None, :]).any(axis=1)
    out[sb] = (cd_sub[a.sub[sb]] == 0) & ~touches_dead[sb]

    # line status: off cooldown, and the change is a change
    ls = a.is_line_status
    ln = a.line[ls]
    wants_on = a.status[ls] == 1
    out[ls] = (cd_line[ln] == 0) & np.where(wants_on, ~line_status[ln], line_status[ln])
    return out


def rewritable_actions(obs: Any, schema: GridSchema, legal: np.ndarray) -> dict[int, np.ndarray]:
    """set_bus candidates that are illegal *only* because they touch a dead line's terminal.

    The frozen candidate set holds whole-substation configurations, so once a
    line is out every configuration of its two substations touches a dead
    terminal and :func:`legality_mask` rejects it -- exactly where a cascade
    lands. Returned is ``{index: set_bus vector}`` with the dead terminals left
    alone (0), restricted to candidates that still move a live terminal. Not
    knowledge -- the game's own rules again -- so it lives here.

    Same answer as the loop it replaces on ``feat/schedule``, vectorised and
    restricted to the substations of dead lines (a set_bus candidate only ever
    touches terminals of its own substation), so a step with a dead line at
    the 65 535-configuration substation costs milliseconds, not seconds.
    """
    a = schema.require_actions()
    line_status = np.asarray(obs.line_status, dtype=bool)
    if line_status.all():
        return {}
    dead = np.zeros(schema.dim_topo, dtype=bool)
    dead[schema.line_terminals[~line_status].ravel()] = True
    cd_sub = np.asarray(obs.time_before_cooldown_sub)
    topo = np.asarray(obs.topo_vect)
    dead_subs = np.unique(schema.topo_sub[dead])
    rows = np.flatnonzero(
        a.is_set_bus & np.isin(a.sub, dead_subs) & ~np.asarray(legal, dtype=bool) & (cd_sub[np.maximum(a.sub, 0)] == 0)
    )
    if rows.size == 0:
        return {}
    sb = a.set_bus[rows]
    touches = ((sb != 0) & dead[None, :]).any(axis=1)
    rows, eff = rows[touches], sb[touches].copy()
    eff[:, dead] = 0
    live = ~dead & (topo >= 0)
    keep = ((eff != 0) & live[None, :] & (eff != topo[None, :])).any(axis=1)
    return {int(i): eff[j] for j, i in enumerate(rows) if keep[j]}


def compose_mask(legal: np.ndarray, semantic: np.ndarray | None) -> np.ndarray:
    """``legal AND semantic``, with do-nothing forced on."""
    mask = np.asarray(legal, dtype=bool).copy()
    if semantic is not None:
        mask &= np.asarray(semantic, dtype=bool)
    mask[0] = True
    return mask

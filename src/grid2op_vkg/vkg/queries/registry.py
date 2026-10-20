"""Name -> query-class registry, and the default instantiated query set.

Queries are parameterised (``tau``, ``k``), so the registry maps *class names*
and the concrete parameter values come from ``VkgConfig``. Nothing here fixes a
threshold; :data:`DEFAULT_QUERIES` only exists so tests and smoke runs have
something to chew on.
"""

from __future__ import annotations

from typing import Any

from grid2op_vkg.vkg.queries.action import IslandingActions, LegalActions, RelevantActions
from grid2op_vkg.vkg.queries.base import Query
from grid2op_vkg.vkg.queries.budget import (
    ActionDisruption,
    AffordableActions,
    PrioritySubstations,
    RecoveryActions,
)
from grid2op_vkg.vkg.queries.explain import ExplainAction
from grid2op_vkg.vkg.queries.remedial import PredictedMaxRho, RemedialActions, RemedialScore
from grid2op_vkg.vkg.queries.risk import (
    HeadroomNearRisk,
    LineMargin,
    LinesAtRisk,
    LinesNearRisk,
    OverloadPressure,
)
from grid2op_vkg.vkg.queries.structure import (
    ControllableSubs,
    LineElectricalAdjacency,
    LinesSharingSubstation,
    ReconnectableLines,
    SubIsSplit,
)

#: Every query the module knows about, keyed by ``base_name``.
QUERY_TYPES: dict[str, type[Query]] = {
    q.base_name: q
    for q in (
        SubIsSplit,
        ControllableSubs,
        ReconnectableLines,
        LineElectricalAdjacency,
        LinesSharingSubstation,
        LinesAtRisk,
        LinesNearRisk,
        LineMargin,
        OverloadPressure,
        HeadroomNearRisk,
        LegalActions,
        RelevantActions,
        IslandingActions,
        ActionDisruption,
        PrioritySubstations,
        AffordableActions,
        RecoveryActions,
        ExplainAction,
        RemedialActions,
        RemedialScore,
        PredictedMaxRho,
    )
}


def build_queries(spec: dict[str, dict[str, Any]] | None = None) -> list[Query]:
    """Instantiate queries from a ``{base_name: kwargs}`` spec.

    ``spec`` is what a config file hands over, so the active query set and its
    thresholds are recorded with the run rather than baked into the code.
    """
    if spec is None:
        return list(DEFAULT_QUERIES)
    out = []
    for name, kwargs in spec.items():
        if name not in QUERY_TYPES:
            raise KeyError(f"unknown query {name!r}; known: {sorted(QUERY_TYPES)}")
        out.append(QUERY_TYPES[name](**(kwargs or {})))
    return out


#: Placeholder parameter values. The real ones are pending the third design
#: document; do not read anything into these numbers.
DEFAULT_QUERIES: tuple[Query, ...] = (
    SubIsSplit(),
    ControllableSubs(),
    ReconnectableLines(),
    LineElectricalAdjacency(),
    LinesSharingSubstation(),
    LinesAtRisk(tau=0.95),
    LinesNearRisk(tau=0.95, k=2, adjacency="electrical"),
    LineMargin(tau=1.0),
    OverloadPressure(tau=0.95),
    HeadroomNearRisk(tau=0.95, k=2, adjacency="electrical"),
    LegalActions(),
    RelevantActions(tau=0.95, k=2, adjacency="electrical"),
    IslandingActions(),
    ActionDisruption(),
    PrioritySubstations(),
    AffordableActions(max_split_subs=2),
    RecoveryActions(),
    # provenance for one action (80 = the substation-4 split on case14), both walks
    ExplainAction(action=80, tau=0.95, k=1, adjacency="substation"),
    ExplainAction(action=80, tau=0.95, k=2, adjacency="electrical"),
    # the substation-adjacency variants, so conformance covers both walks
    LinesNearRisk(tau=0.95, k=1, adjacency="substation"),
    HeadroomNearRisk(tau=0.95, k=1, adjacency="substation"),
    RelevantActions(tau=0.95, k=1, adjacency="substation"),
    # the remedial-action catalogue, and the relevance query with its electrical branch live
    RemedialActions(tau=0.95, theta=0.02),
    RemedialScore(tau=0.95),
    PredictedMaxRho(),
    RelevantActions(tau=0.95, k=1, adjacency="substation", remedial_theta=0.02),
)

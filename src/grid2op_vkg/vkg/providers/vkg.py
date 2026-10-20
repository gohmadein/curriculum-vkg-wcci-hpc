"""Feedback derived from the knowledge graph.

The provider is deliberately thin: it owns a :class:`VkgEngine` and maps the
four channels onto queries. All of the semantics live in the query definitions,
which is what makes them reusable across backbones and inspectable in SPARQL.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from grid2op_vkg.vkg.config import VkgConfig
from grid2op_vkg.vkg.engine import VkgEngine
from grid2op_vkg.vkg.providers.base import FeedbackProvider, SearchGuidance
from grid2op_vkg.vkg.queries.explain import ExplainAction
from grid2op_vkg.vkg.schema import GridSchema

#: Queries used as auxiliary prediction targets. All are functions of the
#: *current* observation, so the auxiliary head is pure self-supervision: it
#: asks the network to recover structure that the flat observation encodes only
#: implicitly. Nothing here leaks a future state.
DEFAULT_AUX_TARGETS: tuple[str, ...] = ("lines_near_risk", "sub_is_split")


class VkgProvider(FeedbackProvider):
    """Wires :class:`VkgEngine` into the :class:`FeedbackProvider` contract."""

    name = "vkg"

    def __init__(
        self,
        schema: GridSchema,
        cfg: VkgConfig,
        engine: VkgEngine | None = None,
        aux_targets: tuple[str, ...] = DEFAULT_AUX_TARGETS,
    ) -> None:
        self.cfg = cfg
        self.engine = engine if engine is not None else VkgEngine(schema, cfg)
        self._aux = aux_targets
        self._explain_kernels: dict[int, Any] = {}
        # only validate the channels this arm actually uses: an arm should be
        # free to carry the minimal query set its own channels need
        if cfg.aux:
            for name in self._aux:
                if name not in self.engine.queries:
                    raise KeyError(f"aux target {name!r} is not in the active query set")
        if cfg.reward and cfg.potential_query not in self.engine.queries:
            raise KeyError(f"potential query {cfg.potential_query!r} is not in the active query set")
        if cfg.mask and not {"relevant_actions", "islanding_actions"} <= set(self.engine.queries):
            raise KeyError("mask channel needs relevant_actions and islanding_actions in the active query set")

    # ------------------------------------------------------------------ #
    @property
    def required_obs_attrs(self) -> frozenset[str]:
        return self.engine.required_obs_attrs

    def begin_step(self, obs: Any) -> None:
        self.engine.begin_step(obs)

    def end_step(self) -> float:
        return self.engine.end_step()

    # -- channels ------------------------------------------------------- #
    def features(self) -> np.ndarray | None:
        if not self.cfg.state:
            return None
        return self.engine.feature_vector()

    def potential(self) -> float | None:
        if not self.cfg.reward:
            return None
        return self.engine.potential()

    def action_mask(self, action_set: Any = None) -> np.ndarray | None:
        if not self.cfg.mask:
            return None
        return self.engine.semantic_mask()

    def search_guidance(self) -> SearchGuidance | None:
        """Five queries, no arithmetic -- the ordering policy is the caller's."""
        ask = self.engine.ask
        return SearchGuidance(
            disruption=np.asarray(ask("action_disruption"), dtype=np.float64),
            priority_subs=np.asarray(ask("priority_substations"), dtype=bool),
            affordable=np.asarray(ask("affordable_actions"), dtype=bool),
            recovery=np.asarray(ask("recovery_actions"), dtype=bool),
            split=np.asarray(ask("sub_is_split"), dtype=bool),
            reconnectable=np.asarray(ask("reconnectable_lines"), dtype=bool),
            at_risk=np.asarray(ask("lines_at_risk"), dtype=bool),
            remedial=np.asarray(ask("remedial_actions"), dtype=bool),
            remedial_score=np.asarray(ask("remedial_score"), dtype=np.float64),
        )

    def justify(self, action_idx: int) -> np.ndarray:
        """Run ExplainAction on the current state, compiled path; one kernel per action, cached."""
        kern = self._explain_kernels.get(action_idx)
        if kern is None:
            q = ExplainAction(
                action_idx, tau=self.cfg.risk_tau, k=self.cfg.relevance_k, adjacency=self.cfg.relevance_adjacency
            )
            kern = self._explain_kernels[action_idx] = q.kernel(self.engine.schema)
        return np.asarray(kern(self.engine.state), dtype=bool)

    def aux_targets(self) -> dict[str, np.ndarray] | None:
        if not self.cfg.aux:
            return None
        return {name: np.asarray(self.engine.ask(name), dtype=np.float32) for name in self._aux}

    # -- reporting ------------------------------------------------------ #
    @property
    def feature_names(self) -> list[str]:
        return self.engine.feature_names if self.cfg.state else []

    def feature_bounds(self):
        if not self.cfg.state:
            return super().feature_bounds()
        return self.engine.feature_bounds()

    @property
    def feature_dim(self) -> int:
        return self.engine.feature_dim if self.cfg.state else 0

    def timings(self) -> dict[str, float]:
        return self.engine.timings()

    def close(self) -> None:
        self.engine.close()

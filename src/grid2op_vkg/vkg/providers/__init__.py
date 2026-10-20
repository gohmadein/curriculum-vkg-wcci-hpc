"""Feedback providers -- the seam the ablation turns on.

``none``         Raw: same plumbing, no signal
``vkg``          feedback derived from the knowledge graph
``handcrafted``  the same feedback written as plain Python (pending the third
                 design document, which defines what "the same" means)
"""

from __future__ import annotations

from typing import Any

from grid2op_vkg.vkg.config import VkgConfig
from grid2op_vkg.vkg.providers.base import FeedbackProvider
from grid2op_vkg.vkg.providers.null import NullProvider
from grid2op_vkg.vkg.providers.vkg import VkgProvider
from grid2op_vkg.vkg.schema import GridSchema


def make_provider(schema: GridSchema, cfg: VkgConfig, **kwargs: Any) -> FeedbackProvider:
    """Build the provider an arm's config asks for."""
    if cfg.provider == "none":
        return NullProvider()
    if cfg.provider == "vkg":
        return VkgProvider(schema, cfg, **kwargs)
    if cfg.provider == "handcrafted":
        from grid2op_vkg.vkg.providers.handcrafted import HandcraftedProvider

        return HandcraftedProvider(schema, cfg, **kwargs)
    raise KeyError(f"unknown provider {cfg.provider!r}")


__all__ = ["FeedbackProvider", "NullProvider", "VkgProvider", "make_provider"]

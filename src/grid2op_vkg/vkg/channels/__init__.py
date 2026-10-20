"""The feedback channels' shared plumbing -- boolean algebra, no semantics."""

from grid2op_vkg.vkg.channels.mask import compose_mask, legality_mask

__all__ = ["compose_mask", "legality_mask"]

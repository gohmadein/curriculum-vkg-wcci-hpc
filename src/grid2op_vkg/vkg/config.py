"""One configuration object, seven ablation arms.

The ablation is not seven code paths, it is one code path and seven configs.
That is deliberate: if ``Raw`` and ``VKG-State`` went through different gym
wrappers, any difference between them would be confounded with the difference
between the wrappers.

The arm names are fixed by the design documents and must not drift:
``Raw``, ``Handcrafted``, ``VKG-State``, ``VKG-Reward``, ``VKG-Mask``,
``VKG-Aux``, ``VKG-Full``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal

ProviderName = Literal["none", "handcrafted", "vkg"]

#: Queries whose results are concatenated into the observation by default.
#: Pair-valued queries are excluded -- an (n_line, n_line) block would dwarf the
#: rest of the observation and is better consumed by the GNN backbone later.
DEFAULT_STATE_QUERIES: tuple[str, ...] = (
    "lines_at_risk",
    "lines_near_risk",
    "line_margin",
    "overload_pressure",
    "headroom_near_risk",
    "sub_is_split",
    "controllable_subs",
)


@dataclass(frozen=True)
class VkgConfig:
    """Everything that distinguishes one ablation arm from another.

    Every threshold lives here rather than in the query code, so a run's
    semantics are recorded alongside its results. The numeric defaults are
    placeholders pending the third design document -- do not read anything into
    them.
    """

    #: Where feedback comes from. ``none`` is the Raw arm.
    provider: ProviderName = "none"

    # -- the four channels ------------------------------------------- #
    state: bool = False
    reward: bool = False
    mask: bool = False
    aux: bool = False

    # -- query layer -------------------------------------------------- #
    backend: str = "compiled"
    #: ``{base_name: kwargs}``; ``None`` means the registry's default set.
    queries: dict[str, dict[str, Any]] | None = None
    state_queries: tuple[str, ...] = DEFAULT_STATE_QUERIES

    # -- shared thresholds -------------------------------------------- #
    #: Every arm with a provider reads these; the VKG side builds its default
    #: query set from them, so Handcrafted and VKG cannot disagree by accident.
    risk_tau: float = 0.95
    #: Electrical hops from a risk line that still count as "near". On case14 a
    #: radius of 2 from line 17 already touches 13 of 14 substations, i.e. the
    #: relevance mask stops selecting anything; 1 touches 8. Grid-dependent, so
    #: revisit for larger environments.
    relevance_k: int = 1
    #: Which adjacency the relevance radius walks. ``electrical`` (lines sharing
    #: a live bus) changes when a substation splits -- and thereby hides the
    #: alternative substation that ExpertAgent uses once the first one is on
    #: cooldown. ``substation`` (lines meeting at a substation, however it is
    #: configured) does not. The electrical relation stays what the state
    #: features report; this only steers the relevance radius.
    relevance_adjacency: Literal["electrical", "substation"] = "electrical"
    margin_tau: float = 1.0
    #: Outstanding substation splits past which the budget queries stop
    #: offering to open a new one. A split locks its substation for the
    #: cooldown and is a substation unavailable for the next overload, so it is
    #: a debt; 2 is what the greedy agent was measured at.
    max_split_subs: int = 2
    #: Remedial-action catalogue (``grid2op_vkg.vkg.remedial``): an action is
    #: "remedial" when the catalogue predicts it lowers the maximum loading by
    #: at least this (units of rho). Drives ``remedial_actions`` and the
    #: electrical branch of ``relevant_actions``.
    remedial_theta: float = 0.02
    #: Whether ``relevant_actions`` admits catalogue hits for the lines at risk
    #: (electrical membership) on top of the endpoint substations. Off, the
    #: candidate set is exactly the one measured before the catalogue existed.
    remedial_membership: bool = False

    # -- reward shaping (VKG-Reward) ---------------------------------- #
    #: r' = r + shaping_lambda * (gamma**k * Phi(s') - Phi(s)), k = grid2op
    #: steps elapsed inside one gym step. Sweep this; report the sensitivity.
    shaping_lambda: float = 0.0
    gamma: float = 0.99
    #: Query supplying the potential. Negated: less pressure is a better state.
    potential_query: str = "overload_pressure"

    # -- action masking (VKG-Mask) ------------------------------------ #
    #: Rule-level legality (cooldowns, no implicit reconnection). Not
    #: semantics, so it is handed to *every* arm, provider or not; the ``mask``
    #: channel above is then purely the semantic relevance filter.
    legality_mask: bool = True

    # -- bookkeeping --------------------------------------------------- #
    arm: str = "Raw"
    seed: int = 0
    notes: str = ""

    def __post_init__(self) -> None:
        if self.provider == "none" and (self.state or self.reward or self.aux or self.mask):
            raise ValueError(f"arm {self.arm!r}: channels are on but provider is 'none'")
        if self.reward and self.shaping_lambda == 0.0:
            raise ValueError(f"arm {self.arm!r}: reward shaping enabled but shaping_lambda is 0")

    @property
    def uses_provider(self) -> bool:
        return self.provider != "none"

    def with_(self, **kw: Any) -> VkgConfig:
        """Copy with overrides -- the way to build a lambda sweep."""
        return replace(self, **kw)

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


def _arm(name: str, **kw: Any) -> VkgConfig:
    return VkgConfig(arm=name, **kw)


#: The seven arms of the ablation, all on one backbone.
#:
#: ``Handcrafted`` mirrors ``VKG-Full``: same four channels, same thresholds,
#: plain Python instead of the ontology. It is the comparison the paper turns
#: on, so it gets the full treatment rather than a reduced one.
def ablation_arms(shaping_lambda: float = 0.1) -> dict[str, VkgConfig]:
    """Build the seven arms. ``shaping_lambda`` is the value being swept."""
    full = {"state": True, "reward": True, "mask": True, "aux": True, "shaping_lambda": shaping_lambda}
    return {
        "Raw": _arm("Raw", provider="none"),
        "Handcrafted": _arm("Handcrafted", provider="handcrafted", **full),
        "VKG-State": _arm("VKG-State", provider="vkg", state=True),
        "VKG-Reward": _arm("VKG-Reward", provider="vkg", reward=True, shaping_lambda=shaping_lambda),
        "VKG-Mask": _arm("VKG-Mask", provider="vkg", mask=True),
        "VKG-Aux": _arm("VKG-Aux", provider="vkg", aux=True),
        "VKG-Full": _arm("VKG-Full", provider="vkg", **full),
    }


#: Arms whose total scores may be compared in one table. VKG-Mask changes the
#: action space, i.e. the MDP, so it is reported separately with mask size and
#: the rate at which it hides a key action (environment note, section 9).
COMPARABLE_ARMS: frozenset[str] = frozenset({"Raw", "Handcrafted", "VKG-State", "VKG-Reward", "VKG-Aux", "VKG-Full"})
MASK_ARMS: frozenset[str] = frozenset({"VKG-Mask"})

ARM_NAMES: tuple[str, ...] = (
    "Raw",
    "Handcrafted",
    "VKG-State",
    "VKG-Reward",
    "VKG-Mask",
    "VKG-Aux",
    "VKG-Full",
)


@dataclass(frozen=True)
class FairnessContract:
    """What every arm is promised and held to.

    Pinned by ``configs/base.yaml`` on the main branch; mirrored here so the
    VKG side can assert against it rather than assume it.
    """

    #: Observation attributes the Raw arm sees. Every provider's
    #: ``required_obs_attrs`` must be a subset (see :mod:`..fairness`).
    obs_attr_to_keep: tuple[str, ...] = ()
    #: obs.simulate() calls allowed per step. 0 means the VKG side may not use
    #: the simulator at all, which is the default: queries read observations.
    simulate_budget_per_step: int = 0
    #: Wall clock ceiling for one decision, milliseconds.
    step_time_limit_ms: float = 1000.0
    #: Fixed candidate action set, identified by content hash.
    action_set_hash: str = ""
    n_seeds: int = 5
    field_notes: dict[str, Any] = field(default_factory=dict)

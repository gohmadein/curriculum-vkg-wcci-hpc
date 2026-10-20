"""Executable versions of the fairness protocol.

The design documents state the rules; this module turns the two that can be
checked mechanically into assertions:

1. **Information parity.** A VKG arm may transform what the baseline sees. It
   may not see more. If a query reads ``gen_margin_up`` then ``gen_margin_up``
   has to be in the Raw arm's observation too, otherwise a score difference is
   explained by extra information rather than by better representation of the
   same information.

2. **Handcrafted isolation.** The Handcrafted arm is the comparison the paper
   turns on. If it can reach into the ontology package it is no longer an
   independent reimplementation, and the comparison quietly becomes VKG against
   itself.
"""

from __future__ import annotations

import ast
from pathlib import Path

from grid2op_vkg.vkg.config import FairnessContract
from grid2op_vkg.vkg.providers.base import FeedbackProvider


class FairnessViolation(AssertionError):
    """A protocol rule that can be checked in code was broken."""


#: Modules the Handcrafted provider must not reach for, directly or otherwise.
FORBIDDEN_FOR_HANDCRAFTED = (
    "rdflib",
    "grid2op_vkg.vkg.ontology",
    "grid2op_vkg.vkg.backends",
    "grid2op_vkg.vkg.queries",
    "grid2op_vkg.vkg.engine",
    "grid2op_vkg.vkg.state",  # its derived views (adjacency, k-hop) are the reasoning step
    "grid2op_vkg.vkg.providers.vkg",
)


#: grid2op exposes the same field under two names; the gym attr list uses the
#: ``prod_*`` spelling, the observation object the ``gen_*`` one.
OBS_ALIASES = {"prod_p": "gen_p", "prod_q": "gen_q", "prod_v": "gen_v"}


def canonical_obs_attrs(names) -> set[str]:
    return {OBS_ALIASES.get(n, n) for n in names}


def check_information_parity(provider: FeedbackProvider, contract: FairnessContract) -> None:
    """Every observation field the provider reads must be visible to Raw too."""
    if not contract.obs_attr_to_keep:
        raise FairnessViolation(
            "the fairness contract carries no obs_attr_to_keep, so parity cannot be "
            "checked. Fill it from configs/base.yaml before running an arm."
        )
    visible = canonical_obs_attrs(contract.obs_attr_to_keep)
    needed = canonical_obs_attrs(provider.required_obs_attrs)
    missing = sorted(needed - visible)
    if missing:
        raise FairnessViolation(
            f"provider {provider.name!r} reads {missing}, which the Raw arm does not see. "
            "Either add these to the shared observation list for every arm, or stop "
            "reading them: as it stands the VKG arm is given more information, not a "
            "better representation of the same information."
        )


def check_simulate_budget(provider: FeedbackProvider, contract: FairnessContract) -> None:
    """A provider may not spend more simulator calls than every arm is granted."""
    if provider.simulate_calls_per_step > contract.simulate_budget_per_step:
        raise FairnessViolation(
            f"provider {provider.name!r} wants {provider.simulate_calls_per_step} "
            f"obs.simulate() calls per step but the contract grants "
            f"{contract.simulate_budget_per_step}. Simulator calls are compute, and "
            "compute is what the protocol equalises."
        )


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
    return found


def check_handcrafted_isolation(path: Path | None = None) -> None:
    """Static check that the Handcrafted arm shares no code with the VKG arm."""
    path = path or Path(__file__).resolve().parent / "providers" / "handcrafted.py"
    if not path.exists():
        return  # not written yet; pending the third design document
    for mod in sorted(_imported_modules(path)):
        for forbidden in FORBIDDEN_FOR_HANDCRAFTED:
            if mod == forbidden or mod.startswith(forbidden + "."):
                raise FairnessViolation(
                    f"{path.name} imports {mod!r}. The Handcrafted arm must be an "
                    "independent plain-Python reimplementation; sharing the ontology "
                    "makes it a comparison of VKG against itself."
                )


def check_all(provider: FeedbackProvider, contract: FairnessContract) -> None:
    """Run every mechanical check. Call this before training an arm."""
    check_information_parity(provider, contract)
    check_simulate_budget(provider, contract)
    check_handcrafted_isolation()

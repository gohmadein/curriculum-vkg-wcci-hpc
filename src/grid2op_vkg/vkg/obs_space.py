"""State injection, via ``BoxGymObsSpace(functs=...)``.

Why here and not in the gym environment, which would be the obvious place:

* ``ppo_sb3.train.build_gym_env`` constructs ``gymenv_class(env)`` and *then*
  overwrites ``env_gym.observation_space`` with a fresh ``BoxGymObsSpace``.
  Anything the wrapper did to the observation is discarded.
* ``ppo_sb3.evaluate`` never calls the wrapper's ``step()`` at all --
  ``GymAgent.act()`` goes straight to ``observation_space.to_gym(obs)``. A
  wrapper-based augmentation would therefore be present during training and
  absent during evaluation, which is the worst possible failure mode: it would
  not crash, it would just quietly score badly.

Putting the features in the observation space fixes both. As a bonus,
``evaluate()`` restores ``obs_attr_to_keep`` from the JSON written at training
time, so the extra block reappears at evaluation without anyone remembering to
ask for it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: Name of the synthetic observation attribute holding the semantic block.
VKG_ATTR = "vkg_feat"


def build_functs(provider: Any) -> dict[str, tuple]:
    """Build the ``functs`` mapping ``BoxGymObsSpace`` expects.

    The callable closes over the provider, so it carries whatever per-step state
    the provider needs. ``BoxGymObsSpace`` calls it once per observation
    conversion, which happens once per agent decision in both training and
    evaluation -- so the provider sees exactly the states the agent acts on.

    Returns an empty dict when the arm has no state channel, which keeps the
    caller free of ``if`` statements.
    """
    dim = provider.feature_dim
    if dim == 0:
        return {}

    low, high = provider.feature_bounds()

    def extract(obs) -> np.ndarray:
        provider.begin_step(obs)
        feats = provider.features()
        provider.end_step()
        if feats is None:
            return np.zeros(dim, dtype=np.float32)
        return np.asarray(feats, dtype=np.float32)

    return {VKG_ATTR: (extract, low, high, (dim,), np.float32)}


def extend_obs_attrs(obs_attr_to_keep: list[str], provider: Any) -> list[str]:
    """Append the semantic block to an observation attribute list, once."""
    out = list(obs_attr_to_keep)
    if provider.feature_dim and VKG_ATTR not in out:
        out.append(VKG_ATTR)
    return out

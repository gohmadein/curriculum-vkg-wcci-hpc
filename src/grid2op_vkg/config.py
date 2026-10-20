"""The single entry point from configs/base.yaml to a grid2op environment.

    cfg = load_config()                    # configs/base.yaml by default
    env = make_env(cfg, seed=0)            # backend, action class, rewards,
                                           # Parameters and opponent from cfg
    ids = chronic_ids(cfg, "test", n=20)   # from configs/chronics_split.json

Run ``python -m grid2op_vkg.config`` to (re)generate the chronic split file.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "base.yaml"
SPLIT_FILE = ROOT / "configs" / "chronics_split.json"


def load_config(path: str | os.PathLike | None = None) -> dict:
    path = Path(path or DEFAULT_CONFIG)
    cfg = yaml.safe_load(path.read_text())
    cfg["_path"] = str(path)
    cfg["_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    return cfg


def make_parameters(cfg: dict):
    from grid2op.Parameters import Parameters

    p = Parameters()
    for key, value in cfg["env"]["parameters"].items():
        if not hasattr(p, key):
            raise KeyError(f"unknown grid2op Parameters field {key!r} in {cfg['_path']}")
        setattr(p, key, value)
    return p


def _reward_cls(name: str):
    import grid2op.Reward as rewards

    return getattr(rewards, name)


def make_env(cfg: dict, seed: int | None = None, **overrides):
    """Build the environment exactly as the config describes. ``overrides`` go to ``grid2op.make``."""
    import grid2op
    import grid2op.Action as actions

    env_cfg = cfg["env"]
    kwargs = {
        "param": make_parameters(cfg),
        "action_class": getattr(actions, env_cfg["action_class"]),
        "reward_class": _reward_cls(env_cfg["reward_class"]),
        "other_rewards": {k: _reward_cls(v) for k, v in (env_cfg.get("other_rewards") or {}).items()},
    }
    if env_cfg.get("backend") == "lightsim2grid":
        from lightsim2grid import LightSimBackend

        kwargs["backend"] = LightSimBackend()

    profile = env_cfg.get("opponent", "none")
    if profile == "none":
        # Passing nothing does NOT mean "no opponent": grid2op.make falls back to
        # the dataset's own config.py, and l2rpn_wcci_2022 ships a
        # GeometricOpponent there (96-step line attacks). case14 ships none, so
        # this only ever mattered once a second grid was run. Say it explicitly.
        from grid2op.Opponent import BaseOpponent, NeverAttackBudget

        kwargs.update(
            opponent_class=BaseOpponent,
            opponent_budget_class=NeverAttackBudget,
            opponent_init_budget=0.0,
            opponent_budget_per_ts=0.0,
            opponent_attack_duration=0,
            opponent_attack_cooldown=99999,
        )
    else:
        import grid2op.Opponent as opponent

        op = dict(cfg["opponent_profiles"][profile])
        lines = op.pop("lines_attacked")
        if not lines:
            raise ValueError(f"opponent profile {profile!r} has no lines_attacked")
        kwargs.update(
            opponent_class=getattr(opponent, op["opponent_class"]),
            opponent_action_class=getattr(actions, op["opponent_action_class"]),
            opponent_budget_class=getattr(opponent, op["opponent_budget_class"]),
            opponent_attack_duration=op["opponent_attack_duration"],
            opponent_attack_cooldown=op["opponent_attack_cooldown"],
            opponent_budget_per_ts=op["opponent_budget_per_ts"],
            opponent_init_budget=op["opponent_init_budget"],
            kwargs_opponent={"lines_attacked": lines},
        )
    kwargs.update(overrides)
    env = grid2op.make(env_cfg["name"], **kwargs)
    if seed is not None:
        env.seed(seed)
    return env


# --------------------------------------------------------------------------- #
# chronic split
# --------------------------------------------------------------------------- #
def build_split(cfg: dict) -> dict:
    env = make_env(cfg)
    names = sorted(os.listdir(env.chronics_handler.path))
    env.close()
    ch = cfg["chronics"]
    rng = np.random.default_rng(ch["seed"])
    perm = rng.permutation(len(names))
    n_test, n_val = ch["test"], ch["val"]
    return {
        "env": cfg["env"]["name"],
        "seed": ch["seed"],
        "n_total": len(names),
        "test": sorted(perm[:n_test].tolist()),
        "val": sorted(perm[n_test : n_test + n_val].tolist()),
        "train": sorted(perm[n_test + n_val :].tolist()),
        "names": names,
    }


def write_split(cfg: dict, path: Path = SPLIT_FILE) -> dict:
    split = build_split(cfg)
    path.write_text(json.dumps(split, indent=1))
    return split


def load_split(path: Path = SPLIT_FILE) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run `python -m grid2op_vkg.config` to generate it")
    return json.loads(path.read_text())


def chronic_ids(cfg: dict, split: str = "test", n: int | None = None) -> list[int]:
    s = load_split()
    if s["env"] != cfg["env"]["name"] or s["seed"] != cfg["chronics"]["seed"]:
        raise ValueError("configs/chronics_split.json was generated for a different env/seed — regenerate it")
    ids = s[split]
    return ids[:n] if n else ids


def main():
    ap = argparse.ArgumentParser(description="inspect the frozen config / regenerate the chronic split")
    ap.add_argument("--config", default=None)
    ap.add_argument("--write-split", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    print(f"config {cfg['_path']} sha {cfg['_sha256']}  env={cfg['env']['name']} opponent={cfg['env']['opponent']}")
    if args.write_split:
        s = write_split(cfg)
        print(f"split -> {SPLIT_FILE}: test {len(s['test'])}, val {len(s['val'])}, train {len(s['train'])}")
        print(f"  test[:20] = {s['test'][:20]}")
    env = make_env(cfg, seed=0)
    p = env.parameters
    print(
        f"env check: cooldown sub/line={p.NB_TIMESTEP_COOLDOWN_SUB}/{p.NB_TIMESTEP_COOLDOWN_LINE} "
        f"simulate/step={p.MAX_SIMULATE_PER_STEP} reward={type(env.get_reward_instance()).__name__} "
        f"other={list(env.other_rewards)}"
    )


if __name__ == "__main__":
    main()

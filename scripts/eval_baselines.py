"""Evaluate agents on full episodes under the frozen config and write metrics.

Every agent is built by ``grid2op_vkg.registry`` inside a worker whose env comes
from ``grid2op_vkg.config.make_env`` (configs/base.yaml), on chronics from the
committed split. Other packages plug in with ``--agent-module`` (imported in
every worker so their ``@register_agent`` factories are available).

Usage:
    uv run scripts/eval_baselines.py                          # 4 P0 baselines, 20 test chronics
    uv run scripts/eval_baselines.py --ppo-path checkpoints/ppo/ppo_raw_200k
    uv run scripts/eval_baselines.py --agent-module grid2op_vkg.vkg.agents \\
        --agents Raw Handcrafted VKG-Reward --agent-kwargs '{"VKG-Reward": {"lam": 0.5}}'
"""

import argparse
import importlib
import json
import logging
import os
import platform
import subprocess
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_AGENTS = ["DoNothing", "Reconnect", "ExpertAgent", "OptimCVXPY"]


# --------------------------------------------------------------------------- #
# worker
# --------------------------------------------------------------------------- #
def run_episode(agent_name, chronic_id, cfg, seed, max_steps, agent_kwargs, agent_modules):
    warnings.filterwarnings("ignore")
    logging.getLogger("alphaDeesp").setLevel(logging.ERROR)  # ExpertAgent dep logs a WARNING per simulate
    from grid2op.Observation import BaseObservation

    import grid2op_vkg.agents  # noqa: F401  registers the P0 baselines
    from grid2op_vkg.config import make_env
    from grid2op_vkg.registry import make_agent

    for mod in agent_modules:
        importlib.import_module(mod)

    # Count obs.simulate() calls: the hidden compute budget of rule agents.
    n_sim = {"n": 0}
    orig_sim = BaseObservation.simulate

    def counted(self, *a, **k):
        n_sim["n"] += 1
        return orig_sim(self, *a, **k)

    BaseObservation.simulate = counted

    env = make_env(cfg, seed=seed)
    # "Name" or "Name:label" — the registry key is before the colon, kwargs are
    # looked up by the full label first (so PPO:s0 / PPO:s1 can differ), then by name.
    reg_name = agent_name.split(":", 1)[0]
    kwargs = agent_kwargs.get(agent_name, agent_kwargs.get(reg_name, {}))
    agent = make_agent(reg_name, env, cfg, seed, **kwargs)
    env.set_id(chronic_id)
    obs = env.reset()
    chronic_name = os.path.basename(env.chronics_handler.get_id())
    horizon = env.chronics_handler.max_timestep()
    if max_steps > 0:
        horizon = min(max_steps, horizon)
    act_limit = float(cfg["evaluation"].get("max_act_seconds", 0) or 0)

    reward, done, step = 0.0, False, 0
    cum_reward, cum_l2rpn, max_rho = 0.0, 0.0, 0.0
    act_times = []
    kinds = {"topo": 0, "line": 0, "redispatch": 0, "curtail": 0, "storage": 0, "noop": 0}
    n_illegal = n_ambiguous = n_overflow_steps = n_over_limit = n_agent_errors = 0
    redispatch_mw = 0.0
    error = None
    t_ep = time.perf_counter()
    while not done and step < horizon:
        t0 = time.perf_counter()
        try:
            act = agent.act(obs, reward, done)
        except Exception as e:
            # Protocol rule: an agent that fails to produce an action (simulate
            # budget exceeded, solver failure, ...) plays do-nothing this step.
            # Counted so it shows up in the table; the episode goes on.
            n_agent_errors += 1
            if error is None:
                error = f"{type(e).__name__}: {str(e)[:160]}"
            act = env.action_space({})
        dt = time.perf_counter() - t0
        act_times.append(dt)
        n_over_limit += int(act_limit > 0 and dt > act_limit)

        d = act.as_dict()
        k = 0
        if "set_bus_vect" in d or "change_bus_vect" in d:
            kinds["topo"] += 1
            k += 1
        if "set_line_status" in d or "change_line_status" in d:
            kinds["line"] += 1
            k += 1
        if "redispatch" in d:
            kinds["redispatch"] += 1
            k += 1
            redispatch_mw += float(np.abs(act.redispatch).sum())
        if "curtailment" in d:
            kinds["curtail"] += 1
            k += 1
        if "storage_power" in d or "set_storage" in d:
            kinds["storage"] += 1
            k += 1
        if k == 0:
            kinds["noop"] += 1

        obs, reward, done, info = env.step(act)
        step += 1
        cum_reward += float(reward)
        cum_l2rpn += float(info.get("rewards", {}).get("l2rpn", 0.0))
        max_rho = max(max_rho, float(obs.rho.max()))
        n_overflow_steps += int((obs.rho > 1.0).any())
        n_illegal += int(info.get("is_illegal", False))
        n_ambiguous += int(info.get("is_ambiguous", False))

    wall = time.perf_counter() - t_ep
    at = np.asarray(act_times) * 1000.0 if act_times else np.zeros(1)
    BaseObservation.simulate = orig_sim
    env.close()
    return {
        "agent": agent_name,
        "chronic_id": chronic_id,
        "chronic": chronic_name,
        "seed": seed,
        "horizon": horizon,
        "steps": step,
        "survival": step / horizon,
        "blackout": bool(done and step < horizon),
        "agent_error": error,
        "n_agent_errors": n_agent_errors,
        "cum_reward": cum_reward,
        "cum_reward_l2rpn": cum_l2rpn,
        "max_rho": max_rho,
        "overflow_steps": n_overflow_steps,
        **{f"n_{k}": v for k, v in kinds.items()},
        "redispatch_mw": redispatch_mw,
        "n_illegal": n_illegal,
        "n_ambiguous": n_ambiguous,
        "n_simulate": n_sim["n"],
        "act_ms_p50": float(np.percentile(at, 50)),
        "act_ms_p95": float(np.percentile(at, 95)),
        "act_ms_max": float(at.max()),
        "act_over_limit": n_over_limit,
        "wall_s": wall,
    }


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def git_rev():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return "unknown"


def summarise(df, order):
    g = df.groupby("agent")
    s = pd.DataFrame(
        {
            "episodes": g.size(),
            "survival_mean": g["survival"].mean(),
            "survival_std": g["survival"].std().fillna(0),
            "steps_mean": g["steps"].mean(),
            "blackout_rate": g["blackout"].mean(),
            "agent_errors_mean": g["n_agent_errors"].mean(),
            "cum_reward_mean": g["cum_reward"].mean(),
            "cum_l2rpn_mean": g["cum_reward_l2rpn"].mean(),
            "max_rho_mean": g["max_rho"].mean(),
            "overflow_steps_mean": g["overflow_steps"].mean(),
            "n_topo_mean": g["n_topo"].mean(),
            "n_line_mean": g["n_line"].mean(),
            "n_redispatch_mean": g["n_redispatch"].mean(),
            "redispatch_mw_mean": g["redispatch_mw"].mean(),
            "n_illegal_mean": g["n_illegal"].mean(),
            "n_simulate_mean": g["n_simulate"].mean(),
            "act_ms_p50": g["act_ms_p50"].median(),
            "act_ms_p95": g["act_ms_p95"].median(),
            "act_over_limit_mean": g["act_over_limit"].mean(),
            "wall_s_mean": g["wall_s"].mean(),
        }
    )
    return s.reindex([a for a in order if a in s.index])


def main():
    from grid2op_vkg.config import chronic_ids, load_config

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="default configs/base.yaml")
    ap.add_argument("--agents", nargs="+", default=DEFAULT_AGENTS, help="registry names, optionally 'Name:label'")
    ap.add_argument(
        "--agent-module", action="append", default=[], help="module to import (registers agents); repeatable"
    )
    ap.add_argument("--agent-kwargs", default="{}", help='JSON: {"AgentName": {"kw": value}}')
    ap.add_argument("--ppo-path", help="shortcut: adds PPO with model_dir=<save_path>/<name>")
    ap.add_argument("--split", default="test", choices=["test", "val", "train"])
    ap.add_argument("--n-chronics", type=int, default=None, help="default: chronics.eval_default from config")
    ap.add_argument("--chronics", nargs="*", type=int, help="explicit chronic ids (overrides split)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0, help="0 = full episode (reported runs must use 0)")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 2))
    ap.add_argument("--out", default="results/eval")
    args = ap.parse_args()

    cfg = load_config(args.config)
    for mod in args.agent_module:
        importlib.import_module(mod)
    agent_kwargs = json.loads(args.agent_kwargs)
    agents = list(args.agents)
    if args.ppo_path:
        agent_kwargs.setdefault("PPO", {})["model_dir"] = str(Path(args.ppo_path).resolve())
        if "PPO" not in agents:
            agents.append("PPO")
    for a in agents:
        if a.split(":", 1)[0] == "PPO" and "model_dir" not in agent_kwargs.get(a, agent_kwargs.get("PPO", {})):
            ap.error(f'{a} needs model_dir: use --ppo-path or --agent-kwargs \'{{"{a}": {{"model_dir": ...}}}}\'')

    if args.chronics:
        ids = args.chronics
    else:
        ids = chronic_ids(cfg, args.split, args.n_chronics or cfg["chronics"]["eval_default"])

    run_dir = Path(args.out) / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "config": cfg["_path"],
        "config_sha": cfg["_sha256"],
        "config_version": cfg.get("version"),
        "env": cfg["env"]["name"],
        "opponent": cfg["env"]["opponent"],
        "split": None if args.chronics else args.split,
        "chronic_ids": ids,
        "agents": agents,
        "agent_kwargs": agent_kwargs,
        "agent_modules": args.agent_module,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "git": git_rev(),
        "host": platform.node(),
        "python": sys.version.split()[0],
        "started": datetime.now().isoformat(timespec="seconds"),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    jobs = [(a, c) for a in agents for c in ids]
    print(f"{len(jobs)} episodes ({len(agents)} agents x {len(ids)} chronics) on {args.workers} workers -> {run_dir}")
    print(f"config {cfg['_path']} v{cfg.get('version')} sha {cfg['_sha256']}  split={meta['split']}")

    rows, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(run_episode, a, c, cfg, args.seed, args.max_steps, agent_kwargs, args.agent_module): (a, c)
            for a, c in jobs
        }
        for i, f in enumerate(as_completed(futs), 1):
            a, c = futs[f]
            try:
                r = f.result()
                rows.append(r)
                flag = "BLACKOUT" if r["blackout"] else "ok      "
                if r["n_agent_errors"]:
                    flag += f" err×{r['n_agent_errors']}"
                print(
                    f"[{i:>3}/{len(jobs)}] {a:<12} chronic {c:<4} {r['steps']:>5}/{r['horizon']} steps  {flag}"
                    f"  reward {r['cum_reward']:>10.1f}  sim {r['n_simulate']:>5}  {r['wall_s']:6.0f}s",
                    flush=True,
                )
            except Exception as e:
                rows.append({"agent": a, "chronic_id": c, "error": repr(e)})
                print(f"[{i:>3}/{len(jobs)}] {a:<12} chronic {c:<4} FAILED: {e!r}", flush=True)
            pd.DataFrame(rows).to_csv(run_dir / "per_episode.csv", index=False)

    df = pd.DataFrame(rows)
    ok = df[df["error"].isna()] if "error" in df else df
    if len(ok):
        summ = summarise(ok, agents)
        summ.to_csv(run_dir / "summary.csv")
        with pd.option_context(
            "display.width", 220, "display.max_columns", 50, "display.float_format", "{:.3g}".format
        ):
            print("\n" + summ.to_string())
    print(f"\ndone in {time.time() - t0:.0f}s -> {run_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Durable experiment supervisor; delegates episodes to the unchanged shared harness.

Writes status.json, report.md, cells.csv and contrasts.csv without an active AI
session. Starts the frozen confirmation phase only after every technical check
in the existing-60 phase passes, never in response to performance. Interrupted
jobs remain explicit partial results; rerunning this command creates new attempt
directories and does not silently pool partial and complete attempts.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent


def atomic_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def source_digest():
    digest = hashlib.sha256()
    for base in ("src", "scripts", "configs", "experiments"):
        for p in sorted((REPO / base).rglob("*")):
            if p.suffix in (".py", ".yaml", ".json") and p.is_file():
                digest.update(str(p.relative_to(REPO)).encode())
                digest.update(p.read_bytes())
    digest.update((REPO / "uv.lock").read_bytes())
    return digest.hexdigest()


def read_rows(attempt):
    if not attempt:
        return []
    files = sorted(Path(attempt).glob("*/per_episode.csv"))
    if not files:
        return []
    try:
        return list(csv.DictReader(files[-1].open()))
    except (OSError, csv.Error):
        return []  # the harness is rewriting this small file


def inspect_complete(job, rows, audit_dir):
    expected = {(a, str(c)) for a in job["agents"] for c in job["chronics"]}
    got = [(r.get("agent"), r.get("chronic_id")) for r in rows]
    if len(got) != len(expected) or set(got) != expected:
        return "missing or duplicate episodes"
    for r in rows:
        if not r.get("survival") or r.get("error") or int(r.get("n_agent_errors") or 0):
            return "episode/agent error; excluded from conclusions"
        if int(r.get("n_illegal") or 0) or int(r.get("n_ambiguous") or 0):
            return "illegal/ambiguous executed action; inspect before confirmation"
        label = job["agent_kwargs"][r["agent"]]["audit_label"]
        p = audit_dir / f"{label}__{r['chronic']}.json"
        if not p.exists():
            return "missing budget audit"
        a = json.loads(p.read_text())
        if a["max_simulate_per_step"] > job["budget"] or a["budget_exhaustions"] or a["probe_exceptions"]:
            return "budget/probe audit failed"
        if a["simulate_calls"] != int(r["n_simulate"]):
            return "harness/audit simulate totals differ"
    return None


def write_csv(path, rows):
    if not rows:
        return
    temp = path.with_suffix(".tmp")
    with temp.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def summarise(plan, state, out):
    cells, per_chronic = [], {}
    for job in plan["jobs"]:
        s = state["jobs"][job["id"]]
        if s["status"] != "complete":
            continue
        rows = read_rows(s["attempt"])
        values = np.asarray([float(r["survival"]) for r in rows])
        means, by_chronic = {}, {}
        for a in job["agents"]:
            means[a] = np.mean([float(r["survival"]) for r in rows if r["agent"] == a])
        for c in job["chronics"]:
            by_chronic[c] = np.mean([float(r["survival"]) for r in rows if int(r["chronic_id"]) == c])
        per_chronic[job["id"]] = by_chronic
        cells.append(
            {
                "phase": job["phase"],
                "family": job["family"],
                "cell": job["cell"],
                "episodes": len(rows),
                "chronics": len(by_chronic),
                "draws": len(means),
                "survival_pct": 100 * float(values.mean()),
                "draw_sd_pp": 100 * float(np.std(list(means.values()), ddof=1)) if len(means) > 1 else 0.0,
                "full_episodes": sum(int(r["steps"]) == int(r["horizon"]) for r in rows),
                "simulations_per_episode": float(np.mean([float(r["n_simulate"]) for r in rows])),
                "episode_p95_ms_median": float(np.median([float(r["act_ms_p95"]) for r in rows])),
                "over_1s_per_episode": float(np.mean([float(r["act_over_limit"]) for r in rows])),
                "illegal_actions": sum(int(r["n_illegal"]) for r in rows),
                "agent_errors": sum(int(r["n_agent_errors"]) for r in rows),
                "source": s["attempt"],
            }
        )
    contrasts = []
    for phase in plan["phases"]:
        for family in ("random", "hybrid"):
            jobs = {j["cell"]: j for j in plan["jobs"] if j["phase"] == phase and j["family"] == family}
            for x, y in (("B", "A"), ("C", "A"), ("D", "C"), ("D", "B")):
                if x not in jobs or y not in jobs:
                    continue
                a, b = per_chronic.get(jobs[x]["id"]), per_chronic.get(jobs[y]["id"])
                if a is None or b is None:
                    continue
                assert a.keys() == b.keys()
                diff = np.asarray([a[c] - b[c] for c in sorted(a)])
                wins, losses = int((diff > 0).sum()), int((diff < 0).sum())
                p = min(
                    1.0,
                    2 * sum(math.comb(wins + losses, i) for i in range(min(wins, losses) + 1)) / 2 ** (wins + losses),
                )
                contrasts.append(
                    {
                        "phase": phase,
                        "family": family,
                        "contrast": f"{x}-{y}",
                        "difference_pp": 100 * float(diff.mean()),
                        "wins": wins,
                        "losses": losses,
                        "ties": int((diff == 0).sum()),
                        "sign_p_raw": p,
                        "sign_p_bonferroni_4": min(1.0, 4 * p),
                        "unit": "chronic mean over draws; seeds are not additional chronics",
                    }
                )
    write_csv(out / "cells.csv", cells)
    write_csv(out / "contrasts.csv", contrasts)
    lines = [
        "# VKG × terminal rewrite factorial experiment",
        "",
        f"Updated: {state['updated']}",
        "",
        "A = neither; B = VKG filtering; C = rewriting; D = both.",
        "All cells: total simulate budget 30/step; thresholds frozen; 1 s is recorded, not enforced.",
        "Existing-60 is exploratory. Fresh-40 is a separately reported confirmation cohort.",
        "",
        "| Phase / family | A baseline | B +VKG | D +VKG +rewrite | C +rewrite |",
        "|---|---:|---:|---:|---:|",
    ]
    for phase in plan["phases"]:
        for family in ("random", "hybrid"):
            row = {r["cell"]: r for r in cells if r["phase"] == phase and r["family"] == family}
            vals = [f"{row[c]['survival_pct']:.2f}%" if c in row else "pending" for c in ("A", "B", "D", "C")]
            lines.append(f"| {phase} / {family} | " + " | ".join(vals) + " |")
    lines += ["", "## Job status", "", "| Job | Status | Episodes | Estimated end |", "|---|---|---:|---|"]
    for key, s in state["jobs"].items():
        lines.append(
            f"| {key} | {s['status']} {s.get('reason', '')} | "
            f"{s.get('completed', 0)}/{s['expected']} | {s.get('eta', 'pending')} |"
        )
    lines += [
        "",
        "Pair counts, corrected sign-test p values, per-episode latency and simulation costs are in the CSVs.",
        "Sign tests use chronic means; they do not establish seed-generalisation on their own.",
        "Only complete jobs passing coverage, action-error and total-budget audits enter the table.",
    ]
    tmp = out / "report.tmp"
    tmp.write_text("\n".join(lines) + "\n")
    tmp.replace(out / "report.md")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--poll", type=int, default=30)
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / "supervisor.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = json.loads(Path(args.plan).read_text())
    digest = source_digest()
    saved_plan = out / "plan.json"
    if saved_plan.exists() and json.loads(saved_plan.read_text()) != plan:
        raise SystemExit("Refusing to change the plan of an existing experiment")
    atomic_json(saved_plan, plan)
    status_file = out / "status.json"
    state = (
        json.loads(status_file.read_text())
        if status_file.exists()
        else {
            "source_digest": digest,
            "git": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
            "jobs": {},
        }
    )
    if state["source_digest"] != digest:
        raise SystemExit("Source differs from the frozen experiment; use a new output directory")
    for previous in state["jobs"].values():
        pid = previous.get("pid")
        if previous.get("status") == "running" and pid and Path(f"/proc/{pid}").exists():
            raise SystemExit(f"Previous worker {pid} is still alive; refusing duplicate evaluation")
    for job in plan["jobs"]:
        state["jobs"].setdefault(
            job["id"], {"status": "pending", "expected": len(job["agents"]) * len(job["chronics"])}
        )
    state["supervisor_pid"] = os.getpid()
    state["started"] = datetime.now().isoformat(timespec="seconds")
    running = {}
    for phase in plan["phases"]:
        jobs = [j for j in plan["jobs"] if j["phase"] == phase]
        for job in jobs:
            s = state["jobs"][job["id"]]
            if s["status"] == "complete":
                continue
            attempt = out / job["id"] / datetime.now().strftime("attempt_%Y%m%d_%H%M%S")
            attempt.mkdir(parents=True, exist_ok=False)
            env = dict(
                os.environ,
                OMP_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
                NUMEXPR_NUM_THREADS="1",
                PYTHONUNBUFFERED="1",
                VKG_FACTORIAL_AUDIT_DIR=str(attempt / "audit"),
            )
            cmd = [
                sys.executable,
                "scripts/eval_baselines.py",
                "--config",
                plan["config"],
                "--agent-module",
                "grid2op_vkg.vkg.factorial",
                "--agents",
                *job["agents"],
                "--agent-kwargs",
                json.dumps(job["agent_kwargs"]),
                "--chronics",
                *map(str, job["chronics"]),
                "--seed",
                "0",
                "--max-steps",
                str(plan.get("max_steps", 0)),
                "--workers",
                str(job["workers"]),
                "--out",
                str(attempt),
            ]
            log = (attempt / "harness.log").open("w")
            proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
            log.close()
            running[job["id"]] = proc
            s.update(
                status="running",
                pid=proc.pid,
                attempt=str(attempt),
                started=datetime.now().isoformat(timespec="seconds"),
            )
            atomic_json(attempt / "command.json", {"argv": cmd, "source_digest": digest, "git": state["git"]})
            print(f"started {job['id']}: pid={proc.pid}, expected={s['expected']}", flush=True)
        while True:
            for job in jobs:
                s = state["jobs"][job["id"]]
                if s["status"] != "running":
                    continue
                rows = read_rows(s["attempt"])
                s["completed"] = len(rows)
                times = [float(r["wall_s"]) for r in rows if r.get("wall_s")]
                estimate = float(np.mean(times)) if times else job["estimated_episode_seconds"]
                # Early finishes are biased toward short episodes: retain the historical floor.
                estimate = max(estimate, job["estimated_episode_seconds"])
                remaining = max(1, s["expected"] - len(rows))
                seconds = math.ceil(remaining / job["workers"]) * estimate
                s["eta"] = (datetime.now() + timedelta(seconds=seconds)).isoformat(timespec="minutes")
                proc = running[job["id"]]
                rc = proc.poll()
                if rc is not None:
                    reason = f"harness exit {rc}" if rc else inspect_complete(job, rows, Path(s["attempt"]) / "audit")
                    s.update(
                        status="failed" if reason else "complete",
                        reason=reason or "",
                        returncode=rc,
                        finished=datetime.now().isoformat(timespec="seconds"),
                        eta="finished",
                    )
                    print(f"{job['id']}: {s['status']} {s['reason']}", flush=True)
            state["updated"] = datetime.now().isoformat(timespec="seconds")
            atomic_json(status_file, state)
            summarise(plan, state, out)
            if all(state["jobs"][j["id"]]["status"] != "running" for j in jobs):
                break
            time.sleep(args.poll)
        if any(state["jobs"][j["id"]]["status"] != "complete" for j in jobs):
            state["stopped_reason"] = "Technical check failed; confirmation not started"
            atomic_json(status_file, state)
            return 1
        if source_digest() != digest:
            state["stopped_reason"] = "Source changed during experiment; confirmation not started"
            atomic_json(status_file, state)
            return 1
    state["finished"] = datetime.now().isoformat(timespec="seconds")
    atomic_json(status_file, state)
    print("All frozen experiments and reports completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

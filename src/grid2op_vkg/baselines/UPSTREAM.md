# Vendored code: l2rpn-baselines

| | |
|---|---|
| Upstream | https://github.com/Grid2op/l2rpn-baselines |
| Version | 0.8.0, commit `2ec125b` (2025-07-25) |
| License | MPL-2.0 — see `LICENSE.MPL-2.0` in this directory |
| Vendored on | 2026-09-19 |

Only the modules needed for the P0 baselines are copied; the TensorFlow/Keras
legacy agents, RLlib, CurriculumAgent, TopoOracleAgent and the uninitialised
submodules are deliberately left out. Upstream file names are kept so the copies
can be diffed against the source.

| Here | Upstream |
|---|---|
| `do_nothing/` | `l2rpn_baselines/DoNothing/` (minus the `main.py` CLI) |
| `expert_agent/` | `l2rpn_baselines/ExpertAgent/` |
| `optim_cvxpy/` | `l2rpn_baselines/OptimCVXPY/` |
| `ppo_sb3/` | `l2rpn_baselines/PPO_SB3/` |
| `utils/` | `gymenv_custom.py`, `gymAgent.py`, `save_log_gif.py` from `l2rpn_baselines/utils/` |
| `../../tests/test_optim_cvxpy.py` | `l2rpn_baselines/test/test_optimcvxpy.py` |

## Local modifications

1. **Import prefix** `l2rpn_baselines.<Pkg>` → `grid2op_vkg.baselines.<pkg>` in every file.
2. **`optim_cvxpy/optimCVXPY.py` — numpy 2 / cvxpy ≥ 1.4 compatibility**
   - `x[self.bus_*.value]` → `x[self.bus_*.value.astype(int)]` at 9 sites, plus
     the two `idx_gen = self.bus_gen.value[...]` assignments in `to_grid2op()`.
     cvxpy stores the value of a `cp.Parameter(integer=True)` as `float64`, so
     using it directly as an index raises `IndexError`.
   - `np.NaN` → `np.nan` (5 sites); the alias was removed in numpy 2.0.
   Neither is reported upstream as of 2026-09.
3. **`expert_agent/evaluate.py`** — added the missing
   `from grid2op.Reward import RedispReward` and
   `from grid2op.Action import TopologyChangeAction` used by the `__main__` block.
4. **`utils/__init__.py`** rewritten to export only the vendored pieces.
5. **`../../tests/test_optim_cvxpy.py`** — `np.NaN` → `np.nan`, dropped unused
   `pickle.FALSE` / `pdb` imports, sorted imports (ruff).
6. **`optim_cvxpy/optimCVXPY.py` — `_update_topo_param` mutated grid2op's class arrays.**
   For loads, generators and storage it did `tmp_ = obs.load_to_subid` (an alias of the
   *class-level* `load_to_subid` array, shared by every observation and by the
   environment) and then `tmp_[obs.load_bus == 2] += obs.n_sub`. The line branches use
   `1 * obs.line_or_to_subid`, a copy, so the bug only fires when a load / generator /
   storage unit sits on busbar 2 -- which OptimCVXPY alone never causes, but any agent
   that splits a substation around it does. Each call then adds `n_sub` again to the
   shared array (observed indices 41 + 2·118, 41 + 6·118, ...), `_mask_theta_zero`
   raises `IndexError`, and the environment's own metadata is corrupted for the rest of
   the process. Fixed by copying (`1 * obs.<x>_to_subid`) at the three sites; regression
   test in `tests/test_optim_cvxpy.py`. Not reported upstream as of 2026-09.


---

# Vendored code: curriculumagent (CurriculumAgent, senior IEEE14)

| | |
|---|---|
| Upstream | https://github.com/FraunhoferIEE/curriculumagent, PyPI `curriculumagent` 1.1.2 (wheel sha256 `e679b1cc…6ce`) |
| Pretrained model | `l2rpn_baselines/CurriculumAgent/model_IEEE14/` of l2rpn-baselines `2ec125b` |
| License | MPL-2.0, "commercial use is not allowed" — see `curriculum_agent/LICENSE` |
| Vendored on | 2026-09-27 |

The l2rpn-baselines `CurriculumAgent` class is a thin wrapper around
`curriculumagent.submission.my_agent.MyAgent`, which needs TensorFlow 2.12 / ray 2.5 /
Python ≤ 3.11. Only the inference path is ported.

| Here | Upstream |
|---|---|
| `curriculum_agent/utilities.py` | `curriculumagent/submission/utilities.py`, **verbatim** (sha256 `e8e38907…5cc0`) |
| `curriculum_agent/my_agent.py` | `curriculumagent/submission/my_agent.py`, Keras functional-model branch only |
| `curriculum_agent/model_IEEE14/actions/actions.npy` | `model_IEEE14/actions/actions.npy`, verbatim (sha256 `504e1ddb…4111`) |
| `data/curriculum/senior_ieee14_policy.npz` (not in git) | `model_IEEE14/model` policy head, exported by `scripts/export_curriculum_policy.py` (sha256 `eb261846…4d52`) |

## Local modifications (`my_agent.py`, every site marked `# LOCAL`)

1. **TensorFlow → numpy.** The SavedModel is a Keras 2.12 functional model: 467 inputs,
   four 1000-unit ReLU layers, a 125-way policy head (`action_out`) and a value head.
   The policy head is evaluated in numpy float32. Checked against the SavedModel's serving
   signature on 600 real case14 observations: max |logit error| 5.7e-6 and the full
   125-action ranking identical on 600/600.
2. **Observation layout.** Input = `obs.to_vect()[:467]`. grid2op 1.12 has 493 entries;
   the first 467 are, attribute by attribute, the layout the model was trained on
   (`gen_p_delta` and `timestep_protection_engaged` were appended later).
3. **Simulation budget.** `obs.simulate` goes through a per-step proxy that serves repeated
   identical calls from a cache (grid2op's simulate is deterministic, so behaviour is
   unchanged) and makes at most `budget` distinct calls; the candidate loop stops when the
   budget is spent. Upstream has no budget (default `max_action_sim=50`).
4. **Hooks** `_candidate_vect`, `_search_budget`, `_after_search`, identity by default, for
   the matched add-ons in `grid2op_vkg/vkg/curriculum.py`.
5. Dropped: the torch / RLlib-signature / Sequential branches, `scaler`, `subset`,
   `topo=True` (topology reversion), and the broken `from tests.data... import config`.

Upstream quirks kept on purpose: `if idx_chosen:` treats action index 0 as "no action";
each candidate is simulated after a reconnection search, so a step with `n` reconnectable
lines costs about `n` calls per candidate.

## 2026-10-20: standalone118-node Curriculum training package

The existing MyAgent NumPy deployment port is retained. Its VKG wrapper now accepts
explicit policy/actions paths and hashes and can restrict semantic retrieval to
the387train-derived policy actions. No default IEEE14 behaviour is removed.
The official curriculumagent1.1.2Torch Senior class and three environment methods
are extracted to training/ without TF imports; original source/license retained
in upstream/. PPO_Reward only gains optional logger compatibility. Junior
architecture/loss/Adam are ported into a pureTorch loop; budget100epochs and
Senior50RLlib iterations are explicit deployment adaptation choices. See
README.md and docs/ADAPTATIONS.md for all provenance and methodological limits.

"""Schema shifts: the same facts, laid out the way a *different* operator would.

Roadmap step 7. The claim under test is the one that makes "virtual" mean
something: when the source tables change, the knowledge graph is repaired by
editing the mapping, and everything above the mapping -- 17 ``Query.sparql``
strings, the compiled kernels, the providers, the search -- is provably
untouched. "Provably" is a conformance run: on the shifted source, Ontop must
return exactly what the compiled kernels return on every sampled state.

Each shift is a pure function on the DataFrames :class:`RelationalSource`
would have written, plus the mapping file that reads the result. Nothing else
in the package knows a shift exists.

The four shifts are the four things that actually happen to operational data:

``rename``        another vendor's column and table names
``split``         a normalised database: flows in one table, switching state
                  in another; line topology and line ratings apart
``units``         loading stored as a percentage, generator power in kW,
                  thermal ratings in kA
``granularity``   a source that models *elements* rather than terminals: one
                  row per line end, one row per injection; the terminal
                  identity the ontology needs has to be reconstructed by a join
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

Frames = dict[str, pd.DataFrame]
MAPPINGS = Path(__file__).parent / "ontology" / "mappings"


@dataclass(frozen=True)
class SchemaShift:
    name: str
    description: str
    mapping: Path
    static: Callable[[Frames], Frames]
    state: Callable[[Frames], Frames]


def _identity(frames: Frames) -> Frames:
    return frames


BASE = SchemaShift(
    name="base",
    description="the layout RelationalSource writes natively",
    mapping=MAPPINGS / "grid.obda",
    static=_identity,
    state=_identity,
)


# --------------------------------------------------------------------------- #
# rename: another vendor's vocabulary
# --------------------------------------------------------------------------- #
_RENAME_STATIC = {
    "substation": ("station", {"id": "station_id", "name": "station_name", "n_terminals": "terminal_count"}),
    "line": (
        "branch",
        {
            "id": "branch_id",
            "name": "branch_name",
            "sub_or": "from_station",
            "sub_ex": "to_station",
            "term_or": "from_terminal",
            "term_ex": "to_terminal",
            "thermal_limit": "rating_a",
        },
    ),
    "remedial_effect": (
        "remedial_catalogue",
        {"action": "action_id", "line": "branch_id", "effect": "loading_effect", "frac_pos": "sign_stability"},
    ),
}
_RENAME_STATE = {
    "line_state": (
        "branch_telemetry",
        {
            "step": "ts",
            "line": "branch_id",
            "rho": "loading",
            "in_service": "status_flag",
            "overflow": "overload_steps",
            "cooldown": "lockout_steps",
        },
    ),
    "sub_state": (
        "station_status",
        {"step": "ts", "sub": "station_id", "cooldown": "lockout_steps", "is_split": "sectionalised"},
    ),
    "terminal_state": ("connection_status", {"step": "ts", "terminal": "conn_id", "busbar": "bar", "bus": "node"}),
    "gen_state": (
        "unit_telemetry",
        {
            "step": "ts",
            "gen": "unit_id",
            "p": "mw_out",
            "margin_up": "headroom_up",
            "margin_down": "headroom_down",
            "dispatch": "redispatch_mw",
        },
    ),
}


def _apply_renames(frames: Frames, table: dict[str, tuple[str, dict[str, str]]]) -> Frames:
    out = dict(frames)
    for old, (new, cols) in table.items():
        if old in out:
            out[new] = out.pop(old).rename(columns=cols)
    return out


RENAME = SchemaShift(
    name="rename",
    description="every table and column the queries touch carries another vendor's name",
    mapping=MAPPINGS / "shifts" / "rename.obda",
    static=lambda f: _apply_renames(f, _RENAME_STATIC),
    state=lambda f: _apply_renames(f, _RENAME_STATE),
)


# --------------------------------------------------------------------------- #
# split: a normalised database
# --------------------------------------------------------------------------- #
def _split_static(frames: Frames) -> Frames:
    out = dict(frames)
    line = out.pop("line")
    out["line_topology"] = line[["id", "sub_or", "sub_ex", "term_or", "term_ex"]].copy()
    out["line_rating"] = line[["id", "name", "thermal_limit"]].copy()
    return out


def _split_state(frames: Frames) -> Frames:
    out = dict(frames)
    ls = out.pop("line_state")
    out["line_flow"] = ls[["step", "line", "rho", "overflow"]].copy()
    out["line_switching"] = ls[["step", "line", "in_service", "cooldown"]].copy()
    return out


SPLIT = SchemaShift(
    name="split",
    description="line facts normalised into flow / switching and topology / rating tables",
    mapping=MAPPINGS / "shifts" / "split.obda",
    static=_split_static,
    state=_split_state,
)


# --------------------------------------------------------------------------- #
# units: the numbers mean the same thing, in other units
# --------------------------------------------------------------------------- #
def _units_static(frames: Frames) -> Frames:
    out = dict(frames)
    line = out["line"].copy()
    line["thermal_limit_ka"] = line.pop("thermal_limit") / 1000.0  # A -> kA
    out["line"] = line
    if "remedial_effect" in out:
        re_ = out["remedial_effect"].copy()
        re_["effect_percent"] = re_.pop("effect") * 100.0  # rho -> percent of the thermal limit
        out["remedial_effect"] = re_
    return out


def _units_state(frames: Frames) -> Frames:
    out = dict(frames)
    ls = out["line_state"].copy()
    ls["loading_percent"] = ls.pop("rho") * 100.0
    out["line_state"] = ls
    gs = out["gen_state"].copy()
    for col in ("p", "margin_up", "margin_down", "dispatch"):
        gs[f"{col}_kw"] = gs.pop(col) * 1000.0  # MW -> kW
    out["gen_state"] = gs
    return out


UNITS = SchemaShift(
    name="units",
    description="loading in percent, generator power in kW, thermal rating in kA",
    mapping=MAPPINGS / "shifts" / "units.obda",
    static=_units_static,
    state=_units_state,
)


# --------------------------------------------------------------------------- #
# granularity: a source that never heard of terminals
# --------------------------------------------------------------------------- #
def granularity_shift(terminal: pd.DataFrame) -> SchemaShift:
    """The granularity shift, bound to the grid's terminal table.

    The transform has to know which line/injection each terminal belongs to,
    which is static structure, so the shift is constructed per grid.
    """
    kind_of = terminal.set_index("id")["kind"]
    obj_of = terminal.set_index("id")["obj_id"]

    def state(frames: Frames) -> Frames:
        out = dict(frames)
        ts = out.pop("terminal_state")
        kind = ts["terminal"].map(kind_of)
        obj = ts["terminal"].map(obj_of).astype(int)
        is_line = kind.isin(["line_or", "line_ex"])
        line_rows = ts[is_line].reset_index(drop=True)
        out["line_end_state"] = pd.DataFrame(
            {
                "step": line_rows["step"],
                "line": obj[is_line].reset_index(drop=True),
                "side": kind[is_line].str.replace("line_", "", regex=False).reset_index(drop=True),
                "busbar": line_rows["busbar"],
                "bus": line_rows["bus"],
            }
        )
        inj_rows = ts[~is_line].reset_index(drop=True)
        out["injection_state"] = pd.DataFrame(
            {
                "step": inj_rows["step"],
                "kind": kind[~is_line].reset_index(drop=True),
                "element": obj[~is_line].reset_index(drop=True),
                "busbar": inj_rows["busbar"],
                "bus": inj_rows["bus"],
            }
        )
        return out

    return SchemaShift(
        name="granularity",
        description="connection state per line end and per injection; terminal identity rebuilt by a join",
        mapping=MAPPINGS / "shifts" / "granularity.obda",
        static=_identity,
        state=state,
    )


def all_shifts(terminal: pd.DataFrame) -> list[SchemaShift]:
    return [RENAME, SPLIT, UNITS, granularity_shift(terminal)]


def mapping_lines_changed(shift: SchemaShift, base: SchemaShift = BASE) -> int:
    """Lines added or removed in the mapping relative to the base layout."""
    import difflib

    a = base.mapping.read_text().splitlines()
    b = shift.mapping.read_text().splitlines()
    return sum(
        1
        for ln in difflib.unified_diff(a, b, n=0)
        if (ln.startswith("+") or ln.startswith("-")) and not ln.startswith(("+++", "---"))
    )

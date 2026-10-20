"""The relational *source* the knowledge graph is virtual over.

Phase C (roadmap step 7) is where "VKG" stops being a figure of speech. Until
now the ABox was materialised in memory from the observation each step; here
the same facts are laid out the way an operator's database would hold them --
one table per entity kind, one row per element, state keyed by step -- and the
graph is only ever *defined*, by the mapping in ``ontology/mappings/grid.obda``,
never stored.

Two-process layout, because DuckDB allows one writer and Ontop is a JVM:

    <root>/static.duckdb        structure + candidate actions; built once, then
                                closed, so Ontop can open it alone
    <root>/state/*.parquet      one file per state table, rewritten atomically
                                every step; the DuckDB file holds *views* over
                                them, so Ontop's next query simply sees the
                                new rows

The schema-shift experiments change these tables (split, rename, re-unit) and
the mapping, and nothing else. That "nothing else" is the claim being tested.
"""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from grid2op_vkg.vkg.actions import ActKind
from grid2op_vkg.vkg.schema import GridSchema, ObjKind
from grid2op_vkg.vkg.shifts import BASE, SchemaShift
from grid2op_vkg.vkg.state import GridState

STATE_TABLES = ("line_state", "sub_state", "terminal_state", "gen_state", "bus", "line_adjacency")

_KIND_NAME = {
    ObjKind.LINE_OR: "line_or",
    ObjKind.LINE_EX: "line_ex",
    ObjKind.GEN: "gen",
    ObjKind.LOAD: "load",
    ObjKind.STORAGE: "storage",
}


class RelationalSource:
    def __init__(self, schema: GridSchema, root: str | os.PathLike, shift: SchemaShift = BASE) -> None:
        self.schema = schema
        self.root = Path(root)
        self.state_dir = self.root / "state"
        self.static_dir = self.root / "static"
        self.db_path = self.root / "static.duckdb"
        #: the layout the tables are written in; BASE is the native one
        self.shift = shift
        #: table names *after* the shift; fixed at build time from a real state
        self.state_tables: tuple[str, ...] = STATE_TABLES
        self.static_tables: tuple[str, ...] = ()

    # ------------------------------------------------------------------ #
    # static side
    # ------------------------------------------------------------------ #
    def static_frames(self) -> dict[str, pd.DataFrame]:
        s = self.schema
        frames = {
            "substation": pd.DataFrame(
                {"id": np.arange(s.n_sub), "name": list(s.name_sub), "n_terminals": s.sub_info.astype(int)}
            ),
            "busbar": pd.DataFrame(
                {
                    "id": [sub * 100 + b for sub in range(s.n_sub) for b in range(1, s.n_busbar_per_sub + 1)],
                    "sub": [sub for sub in range(s.n_sub) for _ in range(s.n_busbar_per_sub)],
                    "number": [b for _ in range(s.n_sub) for b in range(1, s.n_busbar_per_sub + 1)],
                }
            ),
            "terminal": pd.DataFrame(
                {
                    "id": np.arange(s.dim_topo),
                    "sub": s.topo_sub.astype(int),
                    "kind": [_KIND_NAME[ObjKind(int(k))] for k in s.topo_kind],
                    "obj_id": s.topo_obj.astype(int),
                    "label": [s.terminal_label(t) for t in range(s.dim_topo)],
                }
            ),
            "line": pd.DataFrame(
                {
                    "id": np.arange(s.n_line),
                    "name": list(s.name_line),
                    "sub_or": s.line_subs[:, 0].astype(int),
                    "sub_ex": s.line_subs[:, 1].astype(int),
                    "term_or": s.line_terminals[:, 0].astype(int),
                    "term_ex": s.line_terminals[:, 1].astype(int),
                    "thermal_limit": [None if not np.isfinite(x) else float(x) for x in s.thermal_limit],
                }
            ),
            "generator": pd.DataFrame(
                {
                    "id": np.arange(s.n_gen),
                    "name": list(s.name_gen),
                    "terminal": s.gen_terminals.astype(int),
                    "pmax": s.gen_pmax,
                    "pmin": s.gen_pmin,
                    "ramp_up": s.gen_max_ramp_up,
                    "ramp_down": s.gen_max_ramp_down,
                    "redispatchable": s.gen_redispatchable,
                }
            ),
            "load": pd.DataFrame(
                {"id": np.arange(s.n_load), "name": list(s.name_load), "terminal": s.load_terminals.astype(int)}
            ),
            "storage": pd.DataFrame(
                {
                    "id": np.arange(s.n_storage),
                    "name": list(s.name_storage),
                    "terminal": s.storage_terminals.astype(int),
                }
            ),
        }
        # topographic neighbours, so the virtual graph can walk sharesSubstationWith
        pairs = [
            (int(x), int(y))
            for x in range(s.n_line)
            for y in range(s.n_line)
            if x != y and set(s.sub_of_line(x)) & set(s.sub_of_line(y))
        ]
        frames["line_neighbour"] = pd.DataFrame({"a": [x for x, _ in pairs], "b": [y for _, y in pairs]})
        if s.actions is not None:
            a = s.actions
            kind_name = {ActKind.NOTHING: "nothing", ActKind.SET_BUS: "set_bus", ActKind.LINE_STATUS: "line_status"}
            frames["candidate_action"] = pd.DataFrame(
                {
                    "id": np.arange(a.n),
                    "kind": [kind_name[ActKind(int(k))] for k in a.kind],
                    # nullable *integer* columns: a float NaN column would template
                    # the IRI as grid:sub_4.0 and silently join with nothing
                    "sub": pd.Series(a.sub, dtype="Int64").where(a.sub >= 0),
                    "line": pd.Series(a.line, dtype="Int64").where(a.line >= 0),
                    "status": pd.Series(a.status, dtype="Int64").where(a.status != 0),
                    "label": [a.describe(i, s) for i in range(a.n)],
                }
            )
            rows = np.argwhere(a.set_bus != 0)
            frames["action_assignment"] = pd.DataFrame(
                {
                    "action_id": rows[:, 0].astype(int),
                    "terminal": rows[:, 1].astype(int),
                    "busbar": a.set_bus[rows[:, 0], rows[:, 1]].astype(int),
                }
            )
        # the remedial-action catalogue: always a table (Ontop reads the mapping
        # at start-up), empty unless a catalogue was attached to the schema
        pairs = s.remedial.pairs(dense=True) if s.remedial is not None else np.zeros((0, 2), dtype=int)
        ai, li = pairs[:, 0], pairs[:, 1]
        effect = s.remedial.relief[ai, li].astype(np.float64) if pairs.size else []
        frac = s.remedial.frac_pos[ai, li].astype(np.float64) if pairs.size else []
        frames["remedial_effect"] = pd.DataFrame(
            {
                "action": pd.Series(ai, dtype="int64"),
                "line": pd.Series(li, dtype="int64"),
                "effect": pd.Series(effect, dtype="float64"),
                "frac_pos": pd.Series(frac, dtype="float64"),
            }
        )
        return frames

    def build_static(self, initial_state: GridState) -> None:
        """Create ``static.duckdb``: structure tables plus views over the state files, then close it.

        DuckDB resolves a ``read_parquet`` view's columns from the file when
        the view is created, so a first state has to be on disk before the
        views can exist; Ontop needs the same thing to read table metadata at
        start-up.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.static_dir.mkdir(parents=True, exist_ok=True)
        self.state_tables = tuple(self.shift.state(self.state_frames(initial_state)))
        self.write_state(initial_state)
        if self.db_path.exists():
            self.db_path.unlink()
        static = self.shift.static(self.static_frames())
        self.static_tables = tuple(static)
        con = duckdb.connect(str(self.db_path))
        try:
            for name, df in static.items():
                con.register(f"_{name}", df)
                con.execute(f"CREATE TABLE {name} AS SELECT * FROM _{name}")
                # a Parquet copy too, so readers other than Ontop (which holds
                # the DuckDB file lock for as long as it runs) can see structure
                con.execute(f"COPY _{name} TO '{(self.static_dir / f'{name}.parquet').resolve()}' (FORMAT PARQUET)")
                con.unregister(f"_{name}")
            for name in self.state_tables:
                path = (self.state_dir / f"{name}.parquet").resolve()
                con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{path}')")
        finally:
            con.close()

    # ------------------------------------------------------------------ #
    # dynamic side
    # ------------------------------------------------------------------ #
    def state_frames(self, state: GridState) -> dict[str, pd.DataFrame]:
        s, step = self.schema, int(state.step)
        tb = state.terminal_bus
        connected = tb >= 0
        adj = np.argwhere(state.line_adjacency)
        return {
            "line_state": pd.DataFrame(
                {
                    "step": step,
                    "line": np.arange(s.n_line),
                    "rho": state.rho,
                    "in_service": state.line_status,
                    "overflow": state.timestep_overflow.astype(int),
                    "cooldown": state.cooldown_line.astype(int),
                }
            ),
            "sub_state": pd.DataFrame(
                {
                    "step": step,
                    "sub": np.arange(s.n_sub),
                    "cooldown": state.cooldown_sub.astype(int),
                    "is_split": state.sub_is_split,
                }
            ),
            "terminal_state": pd.DataFrame(
                {
                    "step": step,
                    "terminal": np.arange(s.dim_topo),
                    # NULL when disconnected: no busbar, no bus, hence no triple
                    "busbar": pd.Series(state.topo_vect, dtype="Int64").where(connected),
                    "bus": pd.Series(tb, dtype="Int64").where(connected),
                }
            ),
            "gen_state": pd.DataFrame(
                {
                    "step": step,
                    "gen": np.arange(s.n_gen),
                    "p": state.gen_p,
                    "margin_up": state.gen_margin_up,
                    "margin_down": state.gen_margin_down,
                    "dispatch": state.actual_dispatch,
                }
            ),
            "bus": pd.DataFrame(
                {"step": step, "id": state.active_buses.astype(int), "sub": (state.active_buses % s.n_sub).astype(int)}
            ),
            "line_adjacency": pd.DataFrame({"step": step, "a": adj[:, 0].astype(int), "b": adj[:, 1].astype(int)}),
        }

    def write_state(self, state: GridState) -> None:
        """Rewrite every state file atomically (write to a temp name, then rename)."""
        con = duckdb.connect()
        try:
            for name, df in self.shift.state(self.state_frames(state)).items():
                final = self.state_dir / f"{name}.parquet"
                tmp = self.state_dir / f".{name}.{os.getpid()}.tmp.parquet"
                con.register("_df", df)
                con.execute(f"COPY _df TO '{tmp}' (FORMAT PARQUET)")
                con.unregister("_df")
                os.replace(tmp, final)
        finally:
            con.close()

    # ------------------------------------------------------------------ #
    def connect(self, read_only: bool = True) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self.db_path), read_only=read_only)

    def connect_replica(self) -> duckdb.DuckDBPyConnection:
        """An in-memory connection exposing every table as a view over its Parquet file.

        This is what a second consumer gets in practice -- a read replica --
        and it is the only thing that works while Ontop holds the file lock.
        Table names are the *shifted* names, so hand-written SQL against this
        connection sees exactly the layout the mapping sees.
        """
        con = duckdb.connect()
        for name in self.static_tables:
            con.execute(
                f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{(self.static_dir / f'{name}.parquet').resolve()}')"
            )
        for name in self.state_tables:
            con.execute(
                f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{(self.state_dir / f'{name}.parquet').resolve()}')"
            )
        return con

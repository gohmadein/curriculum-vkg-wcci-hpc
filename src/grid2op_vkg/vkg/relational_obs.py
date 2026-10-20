"""What the hand-written rules read when the data comes from a database.

The schema-shift experiment has to be fair to the hand-written side.
:class:`HandcraftedProvider` reads seven arrays off a grid2op observation; in a
deployment those arrays come from the operations database, through SQL that
somebody wrote by hand. This module *is* that SQL -- one reader per source
layout -- so the repair a schema shift forces on the hand-written side can be
counted the same way the mapping repair is counted on the VKG side: in lines.

Neither side touches its rules. What differs is where the repair goes and what
verifies it: on the VKG side a mapping file and a conformance run against the
compiled kernels; here a Python class and whatever tests its author wrote.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass

import duckdb
import numpy as np


@dataclass(frozen=True)
class RelationalObservation:
    """The seven attributes ``HandcraftedProvider`` reads, grid2op layout and conventions."""

    rho: np.ndarray
    line_status: np.ndarray
    timestep_overflow: np.ndarray
    time_before_cooldown_line: np.ndarray
    time_before_cooldown_sub: np.ndarray
    topo_vect: np.ndarray
    gen_p: np.ndarray


class BaseReader:
    """Hand-written SQL against the native layout."""

    LINE = "SELECT line, rho, in_service, overflow, cooldown FROM line_state ORDER BY line"
    SUB = "SELECT sub, cooldown FROM sub_state ORDER BY sub"
    TERMINAL = "SELECT terminal, busbar FROM terminal_state ORDER BY terminal"
    GEN = "SELECT gen, p FROM gen_state ORDER BY gen"

    def read(self, con: duckdb.DuckDBPyConnection) -> RelationalObservation:
        line = con.execute(self.LINE).fetchdf()
        sub = con.execute(self.SUB).fetchdf()
        term = con.execute(self.TERMINAL).fetchdf()
        gen = con.execute(self.GEN).fetchdf()
        return RelationalObservation(
            rho=line["rho"].to_numpy(dtype=np.float32),
            line_status=line["in_service"].to_numpy(dtype=bool),
            timestep_overflow=line["overflow"].to_numpy(dtype=np.int32),
            time_before_cooldown_line=line["cooldown"].to_numpy(dtype=np.int32),
            time_before_cooldown_sub=sub["cooldown"].to_numpy(dtype=np.int32),
            # grid2op: -1 for a disconnected element, the busbar otherwise
            topo_vect=term["busbar"].fillna(-1).to_numpy(dtype=np.int32),
            gen_p=gen["p"].to_numpy(dtype=np.float32),
        )


class RenameReader(BaseReader):
    LINE = (
        "SELECT branch_id AS line, loading AS rho, status_flag AS in_service, "
        "overload_steps AS overflow, lockout_steps AS cooldown FROM branch_telemetry ORDER BY branch_id"
    )
    SUB = "SELECT station_id AS sub, lockout_steps AS cooldown FROM station_status ORDER BY station_id"
    TERMINAL = "SELECT conn_id AS terminal, bar AS busbar FROM connection_status ORDER BY conn_id"
    GEN = "SELECT unit_id AS gen, mw_out AS p FROM unit_telemetry ORDER BY unit_id"


class SplitReader(BaseReader):
    LINE = (
        "SELECT f.line, f.rho, w.in_service, f.overflow, w.cooldown "
        "FROM line_flow f JOIN line_switching w ON w.step = f.step AND w.line = f.line ORDER BY f.line"
    )


class UnitsReader(BaseReader):
    LINE = (
        "SELECT line, loading_percent / 100.0 AS rho, in_service, overflow, cooldown "
        "FROM line_state ORDER BY line"
    )
    GEN = "SELECT gen, p_kw / 1000.0 AS p FROM gen_state ORDER BY gen"


class GranularityReader(BaseReader):
    TERMINAL = (
        "SELECT t.id AS terminal, s.busbar FROM line_end_state s "
        "JOIN terminal t ON t.obj_id = s.line AND t.kind = 'line_' || s.side "
        "UNION ALL "
        "SELECT t.id AS terminal, s.busbar FROM injection_state s "
        "JOIN terminal t ON t.obj_id = s.element AND t.kind = s.kind "
        "ORDER BY terminal"
    )


READERS: dict[str, type[BaseReader]] = {
    "base": BaseReader,
    "rename": RenameReader,
    "split": SplitReader,
    "units": UnitsReader,
    "granularity": GranularityReader,
}


def reader_lines_changed(cls: type[BaseReader]) -> int:
    """Lines the hand-written side had to write for this layout.

    A reader subclass contains nothing but its overrides, so its body *is* the
    change. Blank lines and the class statement are not counted.
    """
    if cls is BaseReader:
        return 0
    body = inspect.getsource(cls).splitlines()[1:]
    return sum(1 for ln in body if ln.strip())

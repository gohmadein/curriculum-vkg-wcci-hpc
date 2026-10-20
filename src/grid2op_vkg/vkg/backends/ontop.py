"""Ontop backend: SPARQL over the relational source, no graph stored anywhere.

This is the path that makes the "V" in VKG literal. The facts live in DuckDB
tables and Parquet files (:mod:`grid2op_vkg.vkg.relational`); the mapping in
``ontology/mappings/grid.obda`` defines the graph; Ontop rewrites each SPARQL
query into SQL at query time. The *same* ``Query.sparql`` strings run here as
run against the materialised rdflib graph -- so the conformance test gains a
third participant, and the schema-shift experiments of roadmap step 7 can
change tables and mapping while the queries, the compiled kernels and the
Handcrafted arm stay exactly as they are.

Process layout: Ontop is a JVM, started once as ``ontop endpoint`` and talked
to over HTTP. Python only ever writes the Parquet state files; DuckDB views
re-read them on every query, so a step is "write six files, ask".

Requirements: Java 17+, the Ontop CLI (``ONTOP_HOME``, default
``~/.local/opt/ontop-cli-5.5.0``) with the DuckDB JDBC driver in its ``jdbc/``
directory. Everything degrades to ``OntopUnavailable`` when missing.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from grid2op_vkg.vkg.backends.base import VkgBackend
from grid2op_vkg.vkg.queries.base import Query
from grid2op_vkg.vkg.relational import RelationalSource
from grid2op_vkg.vkg.schema import GridSchema
from grid2op_vkg.vkg.shifts import BASE, SchemaShift
from grid2op_vkg.vkg.state import GridState

ONTOLOGY_DIR = Path(__file__).resolve().parent.parent / "ontology"
DEFAULT_MAPPING = ONTOLOGY_DIR / "mappings" / "grid.obda"
DEFAULT_TBOX = ONTOLOGY_DIR / "grid.ttl"


class OntopUnavailable(RuntimeError):
    """The Ontop CLI or its JDBC driver is not installed here."""


def ontop_home() -> Path:
    return Path(os.environ.get("ONTOP_HOME", Path.home() / ".local" / "opt" / "ontop-cli-5.5.0"))


def ontop_available() -> bool:
    home = ontop_home()
    has_driver = any((home / "jdbc").glob("duckdb_jdbc*.jar"))
    return (home / "ontop").exists() and has_driver and shutil.which("java") is not None


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class OntopBackend(VkgBackend):
    """Virtual knowledge graph over :class:`RelationalSource`, queried through Ontop."""

    name = "ontop"

    def __init__(
        self,
        schema: GridSchema,
        queries: Iterable[Query],
        initial_state: GridState,
        root: str | os.PathLike | None = None,
        mapping: Path | None = None,
        tbox: Path = DEFAULT_TBOX,
        shift: SchemaShift = BASE,
        port: int | None = None,
        startup_timeout_s: float = 120.0,
        keep_root: bool = False,
    ) -> None:
        if not ontop_available():
            raise OntopUnavailable(f"Ontop CLI with the DuckDB JDBC driver not found under {ontop_home()}")
        super().__init__(schema, queries)
        self._tmp = None
        if root is None:
            self._tmp = tempfile.mkdtemp(prefix="vkg_ontop_")
            root = self._tmp
        self.root = Path(root)
        self.keep_root = keep_root
        # a shift brings its own mapping; an explicit mapping= still wins
        mapping = Path(mapping) if mapping is not None else shift.mapping
        self.shift = shift
        self.source = RelationalSource(schema, self.root / "source", shift=shift)
        self.source.build_static(initial_state)

        self.port = port or _free_port()
        self.endpoint = f"http://127.0.0.1:{self.port}/sparql"
        props = self.root / "grid.properties"
        props.write_text(
            f"jdbc.url=jdbc:duckdb:{self.source.db_path.resolve()}\n"
            "jdbc.driver=org.duckdb.DuckDBDriver\njdbc.user=\njdbc.password=\n"
        )
        self._log = open(self.root / "ontop.log", "w")  # noqa: SIM115  lives as long as the process
        self._proc = subprocess.Popen(
            [
                str(ontop_home() / "ontop"),
                "endpoint",
                "-m",
                str(mapping),
                "-t",
                str(tbox),
                "-p",
                str(props),
                "--port",
                str(self.port),
                "--disable-portal-page",
            ],
            stdout=self._log,
            stderr=subprocess.STDOUT,
            cwd=str(self.root),
        )
        self._wait_ready(startup_timeout_s)
        self._sparql_text = {name: q.full_sparql(schema) for name, q in self.queries.items()}

    # ------------------------------------------------------------------ #
    def _wait_ready(self, timeout_s: float) -> None:
        deadline = time.time() + timeout_s
        probe = "SELECT * WHERE { ?s ?p ?o } LIMIT 1"
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(f"ontop endpoint exited early; see {self.root / 'ontop.log'}")
            try:
                self._post(probe, timeout=5)
                return
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
                time.sleep(0.5)
        self.close()
        raise TimeoutError(f"ontop endpoint did not come up in {timeout_s}s; see {self.root / 'ontop.log'}")

    def _post(self, sparql: str, timeout: float = 60.0) -> dict:
        data = urllib.parse.urlencode({"query": sparql}).encode()
        req = urllib.request.Request(
            self.endpoint,
            data=data,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/sparql-results+json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:2000]
            raise RuntimeError(f"ontop returned HTTP {e.code} for query:\n{sparql}\n--- response ---\n{body}") from None

    @staticmethod
    def _rows(result: dict) -> list[list]:
        """SPARQL JSON results -> positional rows in SELECT order, None for unbound."""
        variables = result["head"]["vars"]
        out = []
        for binding in result["results"]["bindings"]:
            row = []
            for v in variables:
                cell = binding.get(v)
                row.append(None if cell is None else cell["value"])
            out.append(row)
        return out

    # ------------------------------------------------------------------ #
    def _prepare(self, state: GridState) -> None:
        t0 = time.perf_counter()
        self.source.write_state(state)
        t1 = time.perf_counter()
        # the adjacency table *is* the reasoning output; it is written above
        self.timings.prepare_ms.append((t1 - t0) * 1e3)
        self.timings.reason_ms.append(0.0)

    def _run(self, query: Query, state: GridState) -> np.ndarray:
        t0 = time.perf_counter()
        rows = self._rows(self._post(self._sparql_text[query.name]))
        res = query.decode(rows, self.schema)
        self.timings.record_query(query.name, (time.perf_counter() - t0) * 1e3)
        return np.asarray(res)

    def close(self) -> None:
        proc = getattr(self, "_proc", None)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
        log = getattr(self, "_log", None)
        if log is not None and not log.closed:
            log.close()
        if self._tmp is not None and not self.keep_root:
            shutil.rmtree(self._tmp, ignore_errors=True)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001  best effort at interpreter exit
            pass

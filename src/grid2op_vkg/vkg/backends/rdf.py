"""RDF reference backend: materialise the ABox, then really run the SPARQL.

This path defines what the queries *mean*. It is milliseconds per step rather
than microseconds, so it does not sit in the training loop -- but it is what the
conformance test measures the compiled kernels against, what produces the
honest query-latency numbers for the paper, and what the Ontop backend will
slot in beside once the R2RML mappings land (roadmap step 7).

The triples are split in three, and timed separately, because they are
three different claims:

  static     structure that holds for the whole run, built once
  state      the current observation, rewritten every step
  inferred   facts no source row contains (electrical adjacency), derived
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from rdflib import RDF, RDFS, Graph, Literal
from rdflib.plugins.sparql import prepareQuery

from grid2op_vkg.vkg import iri
from grid2op_vkg.vkg.actions import ActKind
from grid2op_vkg.vkg.backends.base import VkgBackend
from grid2op_vkg.vkg.iri import G2O
from grid2op_vkg.vkg.queries.base import Query
from grid2op_vkg.vkg.queries.structure import substation_adjacency
from grid2op_vkg.vkg.schema import GridSchema, ObjKind
from grid2op_vkg.vkg.state import GridState

ONTOLOGY_PATH = Path(__file__).resolve().parent.parent / "ontology" / "grid.ttl"

_TERMINAL_CLASS = {
    ObjKind.LINE_OR: G2O.LineOrigin,
    ObjKind.LINE_EX: G2O.LineExtremity,
    ObjKind.GEN: G2O.GenTerminal,
    ObjKind.LOAD: G2O.LoadTerminal,
    ObjKind.STORAGE: G2O.StorageTerminal,
}


def load_tbox() -> Graph:
    """Parse the ontology. Useful on its own to check the TBox is well formed."""
    g = Graph()
    g.parse(ONTOLOGY_PATH, format="turtle")
    return g


class RdfBackend(VkgBackend):
    """SPARQL over an in-memory materialisation of the grid.

    Parameters
    ----------
    include_tbox:
        Load ``grid.ttl`` into the same graph. Off by default: the queries are
        all ABox queries, and rdflib does no RDFS inference anyway, so the
        schema triples would only slow matching down. Turn it on to run SHACL
        or to inspect the graph by hand.
    """

    name = "rdf"

    def __init__(self, schema: GridSchema, queries: Iterable[Query], include_tbox: bool = False) -> None:
        super().__init__(schema, queries)
        self.graph = Graph()
        for prefix, uri in iri.NAMESPACES.items():
            self.graph.bind(prefix, uri)
        if include_tbox:
            self.graph += load_tbox()
        self._add_static()
        self._volatile: list[tuple] = []
        self._prepared = {
            name: prepareQuery(q.full_sparql(schema), initNs={"g2o": G2O}) for name, q in self.queries.items()
        }
        self._needs_reasoning = any(q.uses_reasoning for q in self.queries.values())

    # ------------------------------------------------------------------ #
    # ABox: structure
    # ------------------------------------------------------------------ #
    def _add_static(self) -> None:
        s, add = self.schema, self.graph.add

        for i in range(s.n_sub):
            node = iri.sub(i)
            add((node, RDF.type, G2O.Substation))
            add((node, RDFS.label, Literal(s.name_sub[i])))
            add((node, G2O.elementIndex, Literal(i)))
            add((node, G2O.terminalCount, Literal(int(s.sub_info[i]))))
            for b in range(1, s.n_busbar_per_sub + 1):
                bb = iri.busbar(i, b)
                add((bb, RDF.type, G2O.Busbar))
                add((bb, G2O.busbarNumber, Literal(b)))
                add((node, G2O.hasBusbar, bb))

        for i in range(s.n_line):
            node = iri.line(i)
            add((node, RDF.type, G2O.PowerLine))
            add((node, RDFS.label, Literal(s.name_line[i])))
            add((node, G2O.elementIndex, Literal(i)))
            if np.isfinite(s.thermal_limit[i]):
                add((node, G2O.thermalLimit, Literal(float(s.thermal_limit[i]))))
            o, e = (int(x) for x in s.line_terminals[i])
            add((node, G2O.originTerminal, iri.terminal(o)))
            add((node, G2O.extremityTerminal, iri.terminal(e)))
            # emitted explicitly: rdflib does not chase rdfs:subPropertyOf
            add((node, G2O.hasTerminal, iri.terminal(o)))
            add((node, G2O.hasTerminal, iri.terminal(e)))

        for cls, names, terms, maker in (
            (G2O.Generator, s.name_gen, s.gen_terminals, iri.gen),
            (G2O.Load, s.name_load, s.load_terminals, iri.load),
            (G2O.Storage, s.name_storage, s.storage_terminals, iri.storage),
        ):
            for i, term in enumerate(terms):
                node = maker(i)
                add((node, RDF.type, cls))
                add((node, RDFS.label, Literal(names[i])))
                add((node, G2O.elementIndex, Literal(i)))
                add((node, G2O.hasTerminal, iri.terminal(int(term))))

        for t in range(s.dim_topo):
            node = iri.terminal(t)
            add((node, RDF.type, G2O.Terminal))
            add((node, RDF.type, _TERMINAL_CLASS[ObjKind(int(s.topo_kind[t]))]))
            add((node, G2O.topoVectIndex, Literal(t)))
            add((node, G2O.atSubstation, iri.sub(int(s.topo_sub[t]))))
            add((node, RDFS.label, Literal(s.terminal_label(t))))

        # the topographic line relation is static structure, so it is emitted once
        adj = np.argwhere(substation_adjacency(s))
        for a_, b_ in adj:
            add((iri.line(int(a_)), G2O.sharesSubstationWith, iri.line(int(b_))))

        if s.actions is not None:
            self._add_actions()
        if s.remedial is not None:
            self._add_remedial()

    def _add_remedial(self) -> None:
        """Knowledge layer: the remedial-action catalogue as reified (action, line) effects."""
        cat, add = self.schema.remedial, self.graph.add
        for a_, l_ in cat.pairs(dense=True):
            node = iri.remedial(int(a_), int(l_))
            add((node, RDF.type, G2O.RemedialEffect))
            add((node, G2O.ofAction, iri.action(int(a_))))
            add((node, G2O.onLine, iri.line(int(l_))))
            add((node, G2O.effect, Literal(float(cat.relief[a_, l_]))))
            add((node, G2O.signStability, Literal(float(cat.frac_pos[a_, l_]))))

    def _add_actions(self) -> None:
        """Control layer: every candidate action, decoded into the graph."""
        a, add = self.schema.require_actions(), self.graph.add
        for i in range(a.n):
            node = iri.action(i)
            add((node, RDF.type, G2O.CandidateAction))
            add((node, G2O.actionIndex, Literal(i)))
            add((node, RDFS.label, Literal(a.describe(i, self.schema))))
            kind = ActKind(int(a.kind[i]))
            if kind == ActKind.NOTHING:
                add((node, RDF.type, G2O.DoNothingAction))
                add((node, G2O.actionCost, Literal(0.0)))
            elif kind == ActKind.LINE_STATUS:
                add((node, RDF.type, G2O.LineStatusAction))
                add((node, G2O.actsOn, iri.line(int(a.line[i]))))
                add((node, G2O.setsLineStatus, Literal(int(a.status[i]))))
                add((node, G2O.actionCost, Literal(0.0)))
            else:
                add((node, RDF.type, G2O.TopologyAction))
                add((node, G2O.actsOn, iri.sub(int(a.sub[i]))))
                add((node, G2O.actionCost, Literal(0.0)))
                for t in np.flatnonzero(a.set_bus[i]):
                    asg = iri.assignment(i, int(t))
                    add((asg, RDF.type, G2O.BusAssignment))
                    add((asg, G2O.assignsTerminal, iri.terminal(int(t))))
                    add((asg, G2O.assignsBusbar, Literal(int(a.set_bus[i, t]))))
                    add((node, G2O.assigns, asg))

    # ------------------------------------------------------------------ #
    # ABox: state + inference, rewritten every step
    # ------------------------------------------------------------------ #
    def _state_triples(self, state: GridState):
        s = self.schema
        tb = state.terminal_bus

        for i in range(s.n_line):
            node = iri.line(i)
            yield (node, G2O.rho, Literal(float(state.rho[i])))
            yield (node, G2O.inService, Literal(bool(state.line_status[i])))
            yield (node, G2O.timestepOverflow, Literal(int(state.timestep_overflow[i])))
            yield (node, G2O.cooldown, Literal(int(state.cooldown_line[i])))

        split = state.sub_is_split
        for i in range(s.n_sub):
            yield (iri.sub(i), G2O.cooldown, Literal(int(state.cooldown_sub[i])))
            yield (iri.sub(i), G2O.isSplit, Literal(bool(split[i])))

        for i in range(s.n_gen):
            node = iri.gen(i)
            yield (node, G2O.marginUp, Literal(float(state.gen_margin_up[i])))
            yield (node, G2O.marginDown, Literal(float(state.gen_margin_down[i])))
            yield (node, G2O.actualDispatch, Literal(float(state.actual_dispatch[i])))

        for bus_id in state.active_buses:
            node = iri.bus(int(bus_id))
            yield (node, RDF.type, G2O.Bus)
            yield (node, G2O.busAt, iri.sub(int(bus_id) % s.n_sub))

        for t in range(s.dim_topo):
            if tb[t] < 0:
                continue
            yield (iri.terminal(t), G2O.connectedTo, iri.bus(int(tb[t])))
            yield (iri.terminal(t), G2O.onBusbar, Literal(int(state.topo_vect[t])))

    def _inferred_triples(self, state: GridState):
        """Electrical adjacency -- the fact the raw observation does not contain."""
        adj = state.line_adjacency
        for a, b in zip(*np.nonzero(adj), strict=True):
            yield (iri.line(int(a)), G2O.electricallyAdjacent, iri.line(int(b)))

    def _prepare(self, state: GridState) -> None:
        for t in self._volatile:
            self.graph.remove(t)

        t0 = time.perf_counter()
        fresh = list(self._state_triples(state))
        for t in fresh:
            self.graph.add(t)
        t1 = time.perf_counter()

        if self._needs_reasoning:
            inferred = list(self._inferred_triples(state))
            for t in inferred:
                self.graph.add(t)
            fresh.extend(inferred)
        t2 = time.perf_counter()

        self._volatile = fresh
        self.timings.prepare_ms.append((t1 - t0) * 1e3)
        self.timings.reason_ms.append((t2 - t1) * 1e3)

    # ------------------------------------------------------------------ #
    def _run(self, query: Query, state: GridState) -> np.ndarray:
        t0 = time.perf_counter()
        rows = list(self.graph.query(self._prepared[query.name]))
        res = query.decode(rows, self.schema)
        self.timings.record_query(query.name, (time.perf_counter() - t0) * 1e3)
        return np.asarray(res)

    @property
    def n_triples(self) -> int:
        return len(self.graph)

    def serialize(self, path: str | Path, fmt: str = "turtle") -> None:
        """Dump the current graph -- handy when a conformance test disagrees."""
        self.graph.serialize(destination=str(path), format=fmt)

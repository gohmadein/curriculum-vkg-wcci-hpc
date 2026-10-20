"""The :class:`Query` contract.

A query has exactly one semantic definition -- its SPARQL -- and two execution
paths:

``sparql()``  the reference. Slow (milliseconds), but it is what the ontology
              actually says, and it is what survives a schema change in the
              source once the R2RML mappings are in place.
``kernel()``  a compiled numpy closure, microseconds, used inside the RL loop.

``tests/test_vkg_queries.py`` asserts the two agree on sampled states. If you
change one you must change the other, and the test will tell you if you didn't.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from grid2op_vkg.vkg import iri
from grid2op_vkg.vkg.schema import GridSchema
from grid2op_vkg.vkg.state import GridState

Kernel = Callable[[GridState], np.ndarray]


@dataclass(frozen=True)
class QuerySig:
    """Shape and bounds of a query result, needed to declare gym spaces."""

    shape: tuple[int, ...]
    dtype: np.dtype
    low: float
    high: float

    @property
    def size(self) -> int:
        return int(np.prod(self.shape)) if self.shape else 1


class Query(ABC):
    """Base class for every VKG query.

    Subclasses set :attr:`base_name`, declare which observation fields they
    depend on, and implement :meth:`sparql`, :meth:`decode` and :meth:`kernel`.
    """

    #: Stable name without parameters.
    base_name: str = ""
    #: Observation attributes this query reads. Checked by the fairness guard.
    required_obs_attrs: frozenset[str] = frozenset()
    #: True when part of the semantics comes from an analytic reasoning service
    #: (graph algorithms SPARQL cannot express). Such queries share that step
    #: between both paths, so their conformance test only covers the query part.
    uses_reasoning: bool = False
    #: True when the answer depends on structure only, never on the state.
    #: Backends evaluate such queries once and reuse the result every step.
    static: bool = False

    #: Parameters that make this instance distinct, e.g. ``{"tau": 0.95}``.
    params: dict[str, Any]

    def __init__(self, **params: Any) -> None:
        self.params = dict(params)

    @property
    def name(self) -> str:
        """Cache key: base name plus the parameters that change the answer."""
        if not self.params:
            return self.base_name
        tail = ",".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.base_name}({tail})"

    # -- the two execution paths -------------------------------------- #
    @abstractmethod
    def sparql(self, schema: GridSchema) -> str:
        """The reference semantics, as a SELECT query (prefixes prepended)."""

    @abstractmethod
    def decode(self, rows: Iterable[Sequence[Any]], schema: GridSchema) -> np.ndarray:
        """Turn SPARQL solution bindings into the dense array the RL side wants."""

    @abstractmethod
    def kernel(self, schema: GridSchema) -> Kernel:
        """Compile to a numpy closure. Must agree with SPARQL + decode."""

    @abstractmethod
    def sig(self, schema: GridSchema) -> QuerySig:
        """Result shape and bounds."""

    def full_sparql(self, schema: GridSchema) -> str:
        return iri.PREFIXES + self.sparql(schema)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name}>"


# --------------------------------------------------------------------------- #
# decoders for the shapes that keep recurring
# --------------------------------------------------------------------------- #
class _MaskQuery(Query):
    """SELECT ?x -- a boolean mask over one element kind."""

    _dim_attr: str = ""

    def _dim(self, schema: GridSchema) -> int:
        return int(getattr(schema, self._dim_attr))

    def decode(self, rows, schema):
        out = np.zeros(self._dim(schema), dtype=bool)
        for row in rows:
            out[iri.index_of(row[0])] = True
        return out

    def sig(self, schema):
        return QuerySig(shape=(self._dim(schema),), dtype=np.dtype(bool), low=0.0, high=1.0)


class LineMaskQuery(_MaskQuery):
    _dim_attr = "n_line"


class ActionMaskQuery(_MaskQuery):
    """SELECT ?action -- a boolean mask over the fixed candidate action set."""

    def _dim(self, schema: GridSchema) -> int:
        return schema.require_actions().n


class SubMaskQuery(_MaskQuery):
    _dim_attr = "n_sub"


class ActionValueQuery(Query):
    """SELECT ?action ?value -- a float per candidate action, with a fill.

    Used by the budget queries: a candidate set larger than the simulation
    budget has to be *ordered*, and the ordering keys are data-layer facts
    (which substation, how many terminals actually move) rather than search
    policy. Keeping them as queries is what lets the search itself stay free
    of any grid vocabulary.
    """

    fill: float = 0.0
    value_low: float = 0.0
    value_high: float = np.inf

    def decode(self, rows, schema):
        out = np.full(schema.require_actions().n, self.fill, dtype=np.float64)
        for row in rows:
            out[iri.index_of(row[0])] = float(row[1])
        return out

    def sig(self, schema):
        return QuerySig(
            shape=(schema.require_actions().n,),
            dtype=np.dtype(np.float64),
            low=self.value_low,
            high=self.value_high,
        )


class LineValueQuery(Query):
    """SELECT ?line ?value -- a float per line, with a fill for unbound lines."""

    fill: float = 0.0
    value_low: float = 0.0
    value_high: float = np.inf

    def decode(self, rows, schema):
        out = np.full(schema.n_line, self.fill, dtype=np.float64)
        for row in rows:
            out[iri.index_of(row[0])] = float(row[1])
        return out

    def sig(self, schema):
        return QuerySig(
            shape=(schema.n_line,), dtype=np.dtype(np.float64), low=self.value_low, high=self.value_high
        )


class LinePairQuery(Query):
    """SELECT ?a ?b -- a boolean line-by-line relation."""

    def decode(self, rows, schema):
        out = np.zeros((schema.n_line, schema.n_line), dtype=bool)
        for row in rows:
            out[iri.index_of(row[0]), iri.index_of(row[1])] = True
        return out

    def sig(self, schema):
        return QuerySig(shape=(schema.n_line, schema.n_line), dtype=np.dtype(bool), low=0.0, high=1.0)


class ScalarQuery(Query):
    """SELECT (… AS ?value) -- one number for the whole grid."""

    value_low: float = -np.inf
    value_high: float = np.inf
    fill: float = 0.0

    def decode(self, rows, schema):
        rows = list(rows)
        if not rows or rows[0][0] is None:
            return np.array(self.fill, dtype=np.float64)
        return np.array(float(rows[0][0]), dtype=np.float64)

    def sig(self, schema):
        return QuerySig(shape=(), dtype=np.dtype(np.float64), low=self.value_low, high=self.value_high)

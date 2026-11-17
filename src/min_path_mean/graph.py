"""Sparse neighbour tables for minimal, information, and native NP-GC scores."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class NeighborTable:
    """Fixed-width neighbour IDs and their cosine weights.

    Missing neighbours have ID ``-1`` and weight zero.  Callers must mask them
    before indexing another array.
    """

    ids: np.ndarray
    weights: np.ndarray

    def __post_init__(self) -> None:
        if self.ids.ndim != 2 or self.ids.shape != self.weights.shape:
            raise ValueError("neighbor IDs and weights must be aligned matrices")
        if self.ids.dtype.kind not in "iu":
            raise TypeError("neighbor IDs must be integers")
        if not np.isfinite(self.weights).all() or (self.weights < 0).any():
            raise ValueError("neighbor weights must be finite and non-negative")


@dataclass(frozen=True)
class GraphScorerConfig:
    """Configuration for one independently cached graph scorer."""

    family: str = "minimal"
    k: int = 2
    idf: bool = False
    similarity_shrinkage: bool = False
    budget_prefix: bool = False
    tau: float = 2.0
    budget: float = 1.0
    idf_smoothing: float = 1.0
    norm_floor: float = 1e-12
    native_exponent: float = -0.5
    native_degree_epsilon: float = 1e-8
    padding_id: Optional[int] = 0
    block_rows: int = 256

    @classmethod
    def preset(cls, name: str, **overrides) -> "GraphScorerConfig":
        presets = {
            "minimal": {},
            "idf": {"family": "information", "idf": True},
            "information": {
                "family": "information",
                "idf": True,
                "similarity_shrinkage": True,
                "budget_prefix": True,
            },
            "nonparamgc": {"family": "nonparamgc"},
        }
        if name not in presets:
            raise ValueError(f"unknown graph scorer preset: {name}")
        return cls(**{**presets[name], **overrides})

    def validate(self) -> None:
        if self.family not in {"minimal", "information", "nonparamgc"}:
            raise ValueError(f"unknown graph family: {self.family}")
        if self.k < 1 or self.block_rows < 1:
            raise ValueError("k and block_rows must be positive")
        if min(
            self.tau,
            self.budget,
            self.idf_smoothing,
            self.norm_floor,
            self.native_degree_epsilon,
        ) <= 0:
            raise ValueError("graph numerical constants must be positive")
        if self.family in {"minimal", "nonparamgc"} and (
            self.idf or self.similarity_shrinkage or self.budget_prefix
        ):
            raise ValueError(f"{self.family} cannot enable information components")


def training_graph(dataset, padding_id: Optional[int] = 0) -> sparse.csr_matrix:
    """Create a binary user-item CSR matrix from the supplied train dataset.

    The caller controls which split/rows enter ``dataset``.  No label-based
    filtering occurs here: this preserves RecBole's prepared-train semantics.
    Duplicate rows represent one structural edge after binarization.
    """

    users = np.asarray(dataset.inter_feat[dataset.uid_field].cpu(), dtype=np.int64)
    items = np.asarray(dataset.inter_feat[dataset.iid_field].cpu(), dtype=np.int64)
    if users.shape != items.shape or users.ndim != 1:
        raise ValueError("unaligned user/item columns in training interactions")
    valid = np.ones(users.shape, dtype=bool)
    if padding_id is not None:
        valid &= (users != padding_id) & (items != padding_id)
    graph = sparse.csr_matrix(
        (np.ones(int(valid.sum()), dtype=np.float64), (users[valid], items[valid])),
        shape=(dataset.user_num, dataset.item_num),
    )
    graph.sum_duplicates()
    graph.data.fill(1.0)
    graph.eliminate_zeros()
    graph.sort_indices()
    return graph


def graph_fingerprint(graph: sparse.csr_matrix) -> str:
    """Content key that prevents accidental reuse with another train split."""

    graph = sparse.csr_matrix(graph)
    digest = hashlib.blake2b(digest_size=20)
    digest.update(np.asarray(graph.shape, dtype=np.int64).tobytes())
    digest.update(graph.indptr.astype(np.int64, copy=False).tobytes())
    digest.update(graph.indices.astype(np.int64, copy=False).tobytes())
    digest.update(graph.data.astype(np.float64, copy=False).tobytes())
    return digest.hexdigest()


def build_minimal_neighbors(
    rows: sparse.spmatrix,
    k: int = 2,
    *,
    padding_id: Optional[int] = 0,
    block_rows: int = 256,
) -> NeighborTable:
    """Compute exact sparse binary-cosine top-k neighbours row by row.

    ``rows`` is the user-item matrix for user neighbours and its transpose for
    item neighbours.  Similarity ties are resolved by smaller node ID so cache
    generation is deterministic across sparse block sizes.
    """

    return build_neighbors(
        rows,
        GraphScorerConfig(
            family="minimal",
            k=k,
            padding_id=padding_id,
            block_rows=block_rows,
        ),
    )


def build_neighbors(rows: sparse.spmatrix, config: GraphScorerConfig) -> NeighborTable:
    """Build minimal, information-weighted, or native NP-GC neighbours."""

    config.validate()
    matrix = sparse.csr_matrix(rows, dtype=np.float64).copy()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    if matrix.nnz and not np.all(matrix.data == 1):
        raise ValueError("graph scorers expect an already-binary graph")

    count = matrix.shape[0]
    feature_degree = np.asarray(matrix.sum(axis=0)).reshape(-1)
    weighted = matrix.copy()
    if config.family == "information" and config.idf:
        idf = np.log1p(
            (count + config.idf_smoothing)
            / (feature_degree + config.idf_smoothing)
        )
        weighted.data *= np.sqrt(idf[weighted.indices])

    if config.family == "nonparamgc":
        # Native NP-GC normalizes with the row mass of A A^T, not cosine degree.
        row_mass = np.asarray(matrix @ feature_degree).reshape(-1)
        normalizer = np.power(
            row_mass + config.native_degree_epsilon, config.native_exponent
        )
    else:
        information = np.asarray(weighted.multiply(weighted).sum(axis=1)).reshape(-1)
        normalizer = np.power(np.maximum(information, config.norm_floor), -0.5)

    ids = np.full((count, config.k), -1, dtype=np.int64)
    weights = np.zeros((count, config.k), dtype=np.float64)
    transpose = weighted.T.tocsc()

    for begin in range(0, count, config.block_rows):
        end = min(begin + config.block_rows, count)
        common = (weighted[begin:end] @ transpose).tocsr()
        for local_row, node in enumerate(range(begin, end)):
            if config.padding_id is not None and node == config.padding_id:
                continue
            left, right = common.indptr[local_row : local_row + 2]
            candidates = common.indices[left:right]
            overlap = common.data[left:right]
            keep = overlap > 0
            if config.family != "nonparamgc":
                keep &= candidates != node
            if config.padding_id is not None:
                keep &= candidates != config.padding_id
            candidates, overlap = candidates[keep], overlap[keep]
            if not candidates.size:
                continue
            cosine = overlap * normalizer[node] * normalizer[candidates]
            reliability = overlap / (overlap + config.tau)
            ranking_weight = (
                cosine * reliability if config.similarity_shrinkage else cosine
            )
            order = np.lexsort((candidates, -ranking_weight))[: config.k]
            if config.budget_prefix:
                before = np.cumsum(reliability[order]) - reliability[order]
                order = order[(before < config.budget) & (ranking_weight[order] > 0)]
            size = len(order)
            ids[node, :size] = candidates[order]
            chosen = ranking_weight[order]
            if config.family == "nonparamgc" and size:
                chosen = chosen / ranking_weight.max()
            weights[node, :size] = chosen
    return NeighborTable(ids=ids, weights=weights)


def load_or_build_tables(
    graph: sparse.csr_matrix,
    cache_path: Path,
    *,
    k: int = 2,
    padding_id: Optional[int] = 0,
    block_rows: int = 256,
    rebuild: bool = False,
    scorer_config: Optional[GraphScorerConfig] = None,
) -> Tuple[NeighborTable, NeighborTable]:
    """Load a split-safe cache or build both sides and save it atomically."""

    cache_path = Path(cache_path)
    config = scorer_config or GraphScorerConfig(
        k=k, padding_id=padding_id, block_rows=block_rows
    )
    config.validate()
    expected = {
        "version": 2,
        "shape": list(graph.shape),
        "nnz": int(graph.nnz),
        "fingerprint": graph_fingerprint(graph),
        "scorer_config": asdict(config),
    }
    if cache_path.exists() and not rebuild:
        with np.load(cache_path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata"].item()))
            if metadata != expected:
                raise ValueError(
                    f"neighbor cache does not match this train graph: {cache_path}"
                )
            return (
                NeighborTable(saved["user_ids"], saved["user_weights"]),
                NeighborTable(saved["item_ids"], saved["item_weights"]),
            )

    users = build_neighbors(graph, config)
    items = build_neighbors(graph.T, config)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # Multiple controller jobs can request the same scorer cache at once.
    # Give each builder a private temporary file and publish atomically.
    temporary = cache_path.with_suffix(cache_path.suffix + f".tmp.{os.getpid()}.npz")
    np.savez_compressed(
        temporary,
        metadata=np.asarray(json.dumps(expected, sort_keys=True)),
        user_ids=users.ids,
        user_weights=users.weights,
        item_ids=items.ids,
        item_weights=items.weights,
    )
    temporary.replace(cache_path)
    return users, items

"""A minimal one-layer NumPy message-passing regression baseline."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral, Real
from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from toposc_lab.ml.graph import GraphInput


class MinimalGNNRegressor:
    """One trainable message-passing layer with invariant mean/sum pooling.

    This deliberately small baseline has no architecture search and no physics
    claim. It consumes only the basis-independent :class:`GraphInput` contract.
    """

    def __init__(
        self,
        *,
        hidden_dimension: int = 12,
        epochs: int = 500,
        learning_rate: float = 0.01,
        l2: float = 1e-5,
        seed: int = 1207,
    ) -> None:
        self.hidden_dimension = _positive_integer(hidden_dimension, "hidden_dimension")
        self.epochs = _positive_integer(epochs, "epochs")
        self.learning_rate = _positive_float(learning_rate, "learning_rate")
        self.l2 = _nonnegative_float(l2, "l2")
        self.seed = _nonnegative_integer(seed, "seed")
        self._local_mean: NDArray[np.float64] | None = None
        self._local_scale: NDArray[np.float64] | None = None
        self._target_mean: float | None = None
        self._target_scale: float | None = None
        self._message_weights: NDArray[np.float64] | None = None
        self._message_bias: NDArray[np.float64] | None = None
        self._readout_weights: NDArray[np.float64] | None = None
        self._readout_bias: float | None = None
        self._dimensions: tuple[int, int, int] | None = None

    def fit(self, graphs: Sequence[GraphInput], targets: ArrayLike) -> Self:
        selected = _graphs(graphs)
        values = np.asarray(targets, dtype=float)
        if values.shape != (len(selected),) or not np.all(np.isfinite(values)):
            raise ValueError("targets must be one finite value per graph")
        local = tuple(_local_message_inputs(graph) for graph in selected)
        stacked = np.vstack(local)
        local_mean = np.mean(stacked, axis=0)
        local_scale = np.std(stacked, axis=0)
        local_scale = np.where(local_scale > 1e-12, local_scale, 1.0)
        standardized = tuple((matrix - local_mean) / local_scale for matrix in local)
        target_mean = float(np.mean(values))
        target_scale = float(np.std(values))
        if target_scale <= 1e-12:
            target_scale = 1.0
        normalized_targets = (values - target_mean) / target_scale

        generator = np.random.default_rng(self.seed)
        message_weights = generator.normal(
            0.0,
            1.0 / np.sqrt(stacked.shape[1]),
            size=(stacked.shape[1], self.hidden_dimension),
        )
        message_bias = np.zeros(self.hidden_dimension, dtype=float)
        readout_width = 2 * self.hidden_dimension + 1
        readout_weights = generator.normal(
            0.0, 1.0 / np.sqrt(readout_width), size=readout_width
        )
        readout_bias = 0.0

        for _ in range(self.epochs):
            message_gradient = np.zeros_like(message_weights)
            bias_gradient = np.zeros_like(message_bias)
            readout_gradient = np.zeros_like(readout_weights)
            readout_bias_gradient = 0.0
            for matrix, target in zip(standardized, normalized_targets, strict=True):
                hidden = np.tanh(matrix @ message_weights + message_bias)
                pooled = np.concatenate(
                    (np.mean(hidden, axis=0), np.sum(hidden, axis=0), [np.log1p(len(hidden))])
                )
                prediction = float(pooled @ readout_weights + readout_bias)
                output_gradient = 2.0 * (prediction - target) / len(selected)
                readout_gradient += output_gradient * pooled
                readout_bias_gradient += output_gradient
                pooled_hidden_gradient = (
                    output_gradient
                    * (
                        readout_weights[: self.hidden_dimension] / len(hidden)
                        + readout_weights[
                            self.hidden_dimension : 2 * self.hidden_dimension
                        ]
                    )
                )
                activation_gradient = (
                    np.ones((len(hidden), 1))
                    * pooled_hidden_gradient
                    * (1.0 - np.square(hidden))
                )
                message_gradient += matrix.T @ activation_gradient
                bias_gradient += np.sum(activation_gradient, axis=0)
            message_gradient += self.l2 * message_weights
            readout_gradient += self.l2 * readout_weights
            message_weights -= self.learning_rate * np.clip(message_gradient, -10.0, 10.0)
            message_bias -= self.learning_rate * np.clip(bias_gradient, -10.0, 10.0)
            readout_weights -= self.learning_rate * np.clip(readout_gradient, -10.0, 10.0)
            readout_bias -= self.learning_rate * float(
                np.clip(readout_bias_gradient, -10.0, 10.0)
            )

        self._local_mean = local_mean
        self._local_scale = local_scale
        self._target_mean = target_mean
        self._target_scale = target_scale
        self._message_weights = message_weights
        self._message_bias = message_bias
        self._readout_weights = readout_weights
        self._readout_bias = readout_bias
        first = selected[0]
        self._dimensions = (
            first.node_features.shape[1],
            first.edge_features.shape[1],
            len(first.global_features),
        )
        return self

    def predict(self, graphs: Sequence[GraphInput]) -> NDArray[np.float64]:
        selected = _graphs(graphs)
        if (
            self._local_mean is None
            or self._local_scale is None
            or self._target_mean is None
            or self._target_scale is None
            or self._message_weights is None
            or self._message_bias is None
            or self._readout_weights is None
            or self._readout_bias is None
            or self._dimensions is None
        ):
            raise RuntimeError("GNN must be fitted before prediction")
        predictions: list[float] = []
        for graph in selected:
            dimensions = (
                graph.node_features.shape[1],
                graph.edge_features.shape[1],
                len(graph.global_features),
            )
            if dimensions != self._dimensions:
                raise ValueError("graph feature dimensions do not match fitted GNN")
            matrix = (_local_message_inputs(graph) - self._local_mean) / self._local_scale
            hidden = np.tanh(matrix @ self._message_weights + self._message_bias)
            pooled = np.concatenate(
                (
                    np.mean(hidden, axis=0),
                    np.sum(hidden, axis=0),
                    [np.log1p(len(hidden))],
                )
            )
            normalized = float(pooled @ self._readout_weights + self._readout_bias)
            predictions.append(self._target_mean + self._target_scale * normalized)
        return np.asarray(predictions, dtype=float)


def _local_message_inputs(graph: GraphInput) -> NDArray[np.float64]:
    node_count, node_width = graph.node_features.shape
    edge_width = graph.edge_features.shape[1]
    neighbor_sum = np.zeros((node_count, node_width), dtype=float)
    edge_sum = np.zeros((node_count, edge_width), dtype=float)
    counts = np.zeros(node_count, dtype=float)
    for edge_index in range(graph.edge_index.shape[1]):
        source = int(graph.edge_index[0, edge_index])
        target = int(graph.edge_index[1, edge_index])
        neighbor_sum[target] += graph.node_features[source]
        edge_sum[target] += graph.edge_features[edge_index]
        counts[target] += 1.0
    divisor = np.where(counts > 0.0, counts, 1.0)[:, None]
    return np.column_stack(
        (
            graph.node_features,
            neighbor_sum / divisor,
            edge_sum / divisor,
            np.repeat(graph.global_features[None, :], node_count, axis=0),
            (counts == 0.0).astype(float),
        )
    )


def _graphs(values: Sequence[GraphInput]) -> tuple[GraphInput, ...]:
    selected = tuple(values)
    if not selected or any(not isinstance(graph, GraphInput) for graph in selected):
        raise TypeError("graphs must be a non-empty sequence of GraphInput values")
    first_dimensions = (
        selected[0].node_features.shape[1],
        selected[0].edge_features.shape[1],
        len(selected[0].global_features),
    )
    if any(
        (
            graph.node_features.shape[1],
            graph.edge_features.shape[1],
            len(graph.global_features),
        )
        != first_dimensions
        for graph in selected
    ):
        raise ValueError("all graphs must use one GraphInputSchema")
    return selected


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _nonnegative_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _positive_float(value: object, name: str) -> float:
    result = _nonnegative_float(value, name)
    if result == 0.0:
        raise ValueError(f"{name} must be positive")
    return result

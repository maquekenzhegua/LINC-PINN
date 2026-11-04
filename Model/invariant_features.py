from __future__ import annotations

"""Label-free invariant representation from the AdaptiveInvariantPINN paper."""

from typing import Final

import torch
from torch import Tensor, nn


DEFAULT_EPSILON: Final = 1e-6


class FeatureShapeError(Exception):
    """Raised when a feature tensor does not match the configured schema."""

    def __init__(self, expected_columns: int, actual_shape: tuple[int, ...]) -> None:
        self.expected_columns = expected_columns
        self.actual_shape = actual_shape
        super().__init__()

    def __str__(self) -> str:
        return (
            f"expected a 2-D tensor with {self.expected_columns} columns, "
            f"received shape {self.actual_shape}"
        )


class InvariantFeatureMap(nn.Module):
    """Apply row invariance and a source-fitted feature standardization.

    With ``include_row_mean`` the row mean of the raw (per-battery min-maxed)
    features is appended as one extra coordinate.  The row-standardized and
    anchored channels are blind to the common-mode aging level (rows that
    co-move produce identical shape and near-zero anchors regardless of where
    the row sits in the battery's life); the row mean restores that signal
    while staying bounded in [-1, 1] for every battery.  Boundedness is a
    property of the affine row transform only; it does not imply that unseen
    batteries share the source distribution.
    """

    def __init__(
        self,
        raw_feature_dim: int = 16,
        epsilon: float = DEFAULT_EPSILON,
        include_row_mean: bool = False,
        include_anchored: bool = True,
        map_mode: str = "invariant",
        temporal_delta_dim: int = 0,
    ) -> None:
        super().__init__()
        self.raw_feature_dim = raw_feature_dim
        self.epsilon = epsilon
        self.include_row_mean = include_row_mean
        self.include_anchored = include_anchored
        self.map_mode = map_mode
        self.temporal_delta_dim = temporal_delta_dim
        transformed_dim = self.output_dim
        self.register_buffer("scaler_mean", torch.zeros(transformed_dim))
        self.register_buffer("scaler_scale", torch.ones(transformed_dim))
        self.register_buffer("is_fitted", torch.tensor(False))

    @property
    def output_dim(self) -> int:
        if self.map_mode == "identity":
            return self.raw_feature_dim + 1 + self.temporal_delta_dim
        anchored_dim = self.raw_feature_dim if self.include_anchored else 0
        return (
            self.raw_feature_dim
            + anchored_dim
            + 1
            + int(self.include_row_mean)
            + self.temporal_delta_dim
        )

    def _check_shape(self, inputs: Tensor) -> None:
        expected_columns = self.raw_feature_dim + 1 + self.temporal_delta_dim
        if inputs.ndim != 2 or inputs.shape[1] != expected_columns:
            raise FeatureShapeError(expected_columns, tuple(inputs.shape))

    def transform(self, inputs: Tensor) -> Tensor:
        """Compute normalized, anchor-subtracted, and progress coordinates."""
        self._check_shape(inputs)
        if self.map_mode == "identity":
            return inputs
        delta = None
        if self.temporal_delta_dim:
            delta = inputs[:, self.raw_feature_dim + 1:
                           self.raw_feature_dim + 1 + self.temporal_delta_dim]
        features = inputs[:, :self.raw_feature_dim]
        progress = inputs[:, self.raw_feature_dim:self.raw_feature_dim + 1]
        centered = features - features.mean(dim=1, keepdim=True)
        scale = torch.sqrt(
            features.var(dim=1, keepdim=True, unbiased=False) + self.epsilon**2
        )
        normalized = centered / scale
        coordinates = [normalized]
        if self.include_anchored:
            coordinates.append(features - features[:, :1])
        coordinates.append(progress)
        if self.include_row_mean:
            coordinates.append(features.mean(dim=1, keepdim=True))
        if delta is not None:
            coordinates.append(delta)
        return torch.cat(coordinates, dim=1)

    @torch.no_grad()
    def fit(self, source_inputs: Tensor) -> None:
        """Fit post-transform scaling statistics on source rows only."""
        transformed = self.transform(source_inputs)
        mean = transformed.mean(dim=0)
        scale = transformed.std(dim=0, unbiased=False).clamp_min(self.epsilon)
        self.set_scaler(mean, scale)

    @torch.no_grad()
    def set_scaler(self, mean: Tensor, scale: Tensor) -> None:
        """Store source statistics computed by a streaming caller."""
        self.scaler_mean.copy_(mean)
        self.scaler_scale.copy_(scale.clamp_min(self.epsilon))
        self.is_fitted.fill_(True)

    def forward(self, inputs: Tensor) -> Tensor:
        transformed = self.transform(inputs)
        return (transformed - self.scaler_mean) / self.scaler_scale

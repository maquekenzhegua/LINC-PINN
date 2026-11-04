from __future__ import annotations

"""Trainable AdaptiveInvariantPINN for the measured PINN4SOH datasets."""

from typing import Iterable, NamedTuple, Tuple

import torch
from torch import Tensor, nn

from Model.adaptive_networks import SOHPredictor, SineMLP
from Model.invariant_features import DEFAULT_EPSILON, InvariantFeatureMap


Batch = Tuple[Tensor, Tensor, Tensor, Tensor]
BatchLoader = Iterable[Batch]


class EmptyBatchLoaderError(Exception):
    def __str__(self) -> str:
        return "the batch loader produced no rows"


class AdaptivePINNConfig(NamedTuple):
    """Immutable architecture and objective configuration.

    ``monotone_weight`` scales the prediction-side penalty
    ``relu((u2 - u1) * (y1 - y2)).sum()``, the same form as the original
    PINN4SOH physics loss (loss3 in Model/Model.py).  The row-mean variance of
    the raw features is kept only as a logged diagnostic: it has no gradient,
    and the invariance itself is enforced architecturally by
    ``InvariantFeatureMap``.
    """

    input_dim: int = 17
    latent_dim: int = 8
    hidden_dim: int = 60
    operator_layers: int = 3
    operator_hidden_dim: int = 60
    learning_rate: float = 2e-3
    operator_learning_rate: float = 1e-3
    operator_weight: float = 0.5
    # 0.2 = the headline XJTU/HUST value; reproduce.py overrides per dataset.
    monotone_weight: float = 0.2
    encoder_dropout: float = 0.2
    # "identity" (default): the row transform is self-normalizing (row-standardized
    # channel has per-row zero mean / unit variance, anchored channel is bounded in
    # [-2, 2], progress and row mean in [-1, 1]), so no affine map is fitted.  A
    # source-fitted z-score ("source") distorts inputs from batteries whose row
    # statistics deviate from the few source cells and can amplify near-constant
    # source dimensions by 1/epsilon.
    feature_scaler: str = "identity"
    # The row-standardized and anchored channels are blind to the common-mode
    # aging level; the appended row mean restores it while staying bounded.
    include_row_mean: bool = True
    # Ablation switch for the anchored channel (each feature's offset from
    # the first feature of the same row); kept True in the final configuration.
    include_anchored: bool = True
    # Ladder variants: "invariant" (paper row map) or "identity" (raw row
    # passthrough, PINN4SOH-style input).
    map_mode: str = "invariant"
    # When True the dataloader row layout is [features, progress, delta(16)];
    # the progress column moves to index input_dim - 1 - 16.
    temporal_delta: bool = False
    epsilon: float = DEFAULT_EPSILON
    base_lr: float = 1e-2
    final_lr: float = 2e-4
    warmup_epochs: int = 30


class LossReport(NamedTuple):
    data: float
    operator: float
    invariant: float
    monotone: float
    total: float


class _LossTensors(NamedTuple):
    data: Tensor
    operator: Tensor
    invariant: Tensor
    monotone: Tensor
    total: Tensor


class InvariantSOHNetwork(nn.Module):
    """Encode invariant coordinates into degradation state and SOH."""

    def __init__(
        self,
        transformed_dim: int,
        latent_dim: int,
        hidden_dim: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = SineMLP(
            input_dim=transformed_dim,
            output_dim=latent_dim,
            layers_num=3,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.predictor = SOHPredictor(input_dim=latent_dim)

    def forward(self, transformed: Tensor) -> Tuple[Tensor, Tensor]:
        latent = self.encoder(transformed)
        return self.predictor(latent), latent


class AdaptiveInvariantPINN(nn.Module):
    """Joint SOH regressor and shared cycle-to-cycle degradation operator."""

    def __init__(
        self,
        config: AdaptivePINNConfig,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.config = config
        self.device = device
        delta_dim = (config.input_dim - 17) if config.temporal_delta else 0
        self.progress_index = config.input_dim - 1 - delta_dim
        self.feature_map = InvariantFeatureMap(
            raw_feature_dim=config.input_dim - 1 - delta_dim,
            epsilon=config.epsilon,
            include_row_mean=config.include_row_mean,
            include_anchored=config.include_anchored,
            map_mode=config.map_mode,
            temporal_delta_dim=delta_dim,
        )
        self.solution_u = InvariantSOHNetwork(
            transformed_dim=self.feature_map.output_dim,
            latent_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.encoder_dropout,
        )
        self.shared_operator = SineMLP(
            input_dim=config.latent_dim + 1,
            output_dim=config.latent_dim,
            layers_num=config.operator_layers,
            hidden_dim=config.operator_hidden_dim,
            dropout=0.0,
        )
        self.to(device)
        self.solution_optimizer = torch.optim.Adam(
            self.solution_u.parameters(), lr=config.learning_rate
        )
        self.operator_optimizer = torch.optim.Adam(
            self.shared_operator.parameters(), lr=config.operator_learning_rate
        )
        self.loss_func = nn.MSELoss()

    @torch.no_grad()
    def fit_feature_scaler(self, loader: BatchLoader) -> None:
        """Fit source scaling with bounded-memory streaming moments.

        No-op in ``identity`` mode: the buffers keep their default
        mean-0 / scale-1 values, so ``forward`` returns the raw transform.
        """
        if self.config.feature_scaler != "source":
            return
        feature_sum = torch.zeros(self.feature_map.output_dim, device=self.device)
        square_sum = torch.zeros_like(feature_sum)
        row_count = 0
        for x1, x2, _, _ in loader:
            for inputs in (x1, x2):
                transformed = self.feature_map.transform(inputs.to(self.device))
                feature_sum += transformed.sum(dim=0)
                square_sum += (transformed**2).sum(dim=0)
                row_count += transformed.shape[0]
        if row_count == 0:
            raise EmptyBatchLoaderError()
        mean = feature_sum / row_count
        variance = (square_sum / row_count - mean**2).clamp_min(0.0)
        self.feature_map.set_scaler(mean, variance.sqrt())

    def _encode(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        return self.solution_u(self.feature_map(inputs))

    def predict(self, inputs: Tensor) -> Tensor:
        # No [0, 1] clamp: measured SOH labels exceed 1.0 on HUST (initial
        # capacity above the 1.1 Ah rating), so clamping would pin a large
        # share of predictions to 1.0.
        prediction, _ = self._encode(inputs.to(self.device))
        return prediction

    def _losses(self, batch: Batch) -> _LossTensors:
        x1, x2, y1, y2 = (value.to(self.device) for value in batch)
        u1, z1 = self._encode(x1)
        u2, z2 = self._encode(x2)
        data = 0.5 * self.loss_func(u1, y1) + 0.5 * self.loss_func(u2, y2)
        predicted_delta = self.shared_operator(
            torch.cat((z1, x1[:, self.progress_index:self.progress_index + 1]), dim=1))
        operator = self.loss_func(predicted_delta, z2 - z1)
        raw_rows = torch.cat(
            (x1[:, :self.feature_map.raw_feature_dim],
             x2[:, :self.feature_map.raw_feature_dim]), dim=0)
        # Diagnostic only: a function of inputs alone, it carries no gradient.
        invariant = raw_rows.mean(dim=1).var(unbiased=False)
        # Prediction-side monotonicity, matching Model/Model.py loss3: penalize
        # a predicted SOH increase in proportion to the measured decrease.
        monotone = torch.relu((u2 - u1) * (y1 - y2)).sum()
        total = (
            data
            + self.config.operator_weight * operator
            + self.config.monotone_weight * monotone
        )
        return _LossTensors(data, operator, invariant, monotone, total)

    def train_one_epoch(self, loader: BatchLoader) -> LossReport:
        self.train()
        totals = torch.zeros(5, device=self.device)
        batches = 0
        for batch in loader:
            losses = self._losses(batch)
            self.solution_optimizer.zero_grad()
            self.operator_optimizer.zero_grad()
            losses.total.backward()
            self.solution_optimizer.step()
            self.operator_optimizer.step()
            totals += torch.stack(tuple(losses)).detach()
            batches += 1
        if batches == 0:
            raise EmptyBatchLoaderError()
        averages = (totals / batches).cpu().tolist()
        return LossReport(*averages)

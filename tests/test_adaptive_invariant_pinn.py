from __future__ import annotations

"""Behavior tests for the adaptive invariant PINN.

Run from the repository root with:
    python tests/test_adaptive_invariant_pinn.py
"""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from Model.AdaptiveInvariantPINN import AdaptiveInvariantPINN, AdaptivePINNConfig
from Model.adaptive_trainer import (
    AdaptivePINNTrainer,
    AdaptiveRuntimeConfig,
    UnsafeLogPathError,
    _resolve_log_path,
)
from Model.invariant_features import InvariantFeatureMap


def test_normalized_block_cancels_positive_affine_nuisance() -> None:
    # Given: two charge-window rows differing only by offset and positive scale.
    feature_map = InvariantFeatureMap(raw_feature_dim=4, epsilon=1e-6)
    source = torch.tensor([[1.0, 2.0, 4.0, 8.0, 0.25]])
    shifted = torch.cat((3.0 + 2.5 * source[:, :-1], source[:, -1:]), dim=1)

    # When: the paper's row-wise invariant transform is applied.
    source_phi = feature_map.transform(source)
    shifted_phi = feature_map.transform(shifted)

    # Then: normalized coordinates and cycle progress are nuisance invariant.
    assert torch.allclose(source_phi[:, :4], shifted_phi[:, :4])
    assert torch.allclose(source_phi[:, -1:], shifted_phi[:, -1:])


def test_scaler_is_fitted_from_source_rows_and_reused() -> None:
    # Given: a source batch and an unseen target batch with a large shift.
    feature_map = InvariantFeatureMap(raw_feature_dim=4, epsilon=1e-6)
    source = torch.tensor(
        [[1.0, 2.0, 4.0, 8.0, 0.0], [2.0, 3.0, 5.0, 9.0, 1.0]]
    )
    target = source + torch.tensor([100.0, 100.0, 100.0, 100.0, 0.0])

    # When: only source transformed rows fit the feature scaler.
    feature_map.fit(source)
    source_scaled = feature_map(source)
    mean_before_target = feature_map.scaler_mean.clone()
    _ = feature_map(target)

    # Then: source coordinates are centered and target inference changes no state.
    assert torch.allclose(
        source_scaled.mean(dim=0), torch.zeros(9), atol=1e-5, rtol=0.0
    )
    assert torch.allclose(feature_map.scaler_mean, mean_before_target)


def test_training_step_optimizes_joint_operator_objective() -> None:
    # Given: paired cycles whose SOH decreases and a fully specified model config.
    generator = torch.Generator().manual_seed(7)
    x1 = torch.randn(12, 5, generator=generator)
    x1[:, -1] = torch.linspace(0.0, 0.8, 12)
    x2 = x1.clone()
    x2[:, :-1] += 0.02 * torch.randn(12, 4, generator=generator)
    x2[:, -1] += 0.05
    y1 = torch.linspace(1.0, 0.8, 12).unsqueeze(1)
    y2 = y1 - 0.01
    loader = DataLoader(TensorDataset(x1, x2, y1, y2), batch_size=6)
    config = AdaptivePINNConfig(
        input_dim=5,
        latent_dim=3,
        hidden_dim=12,
        operator_layers=2,
        operator_hidden_dim=12,
        learning_rate=1e-3,
        operator_learning_rate=1e-3,
        operator_weight=0.5,
        monotone_weight=0.1,
        epsilon=1e-6,
    )
    model = AdaptiveInvariantPINN(config=config, device=torch.device("cpu"))
    model.fit_feature_scaler(loader)

    # When: one adaptive invariant training epoch is executed.
    report = model.train_one_epoch(loader)

    # Then: every paper objective is finite and contributes to the report.
    assert report.data >= 0.0
    assert report.operator >= 0.0
    assert report.invariant >= 0.0
    assert report.monotone >= 0.0
    assert report.total >= report.data


def test_monotone_penalty_reaches_model_parameters() -> None:
    # Regression: the monotonicity penalty must act on predictions (which
    # carry gradient), not on labels alone (which are constants).
    torch.manual_seed(3)  # keep the train-mode dropout draw deterministic
    generator = torch.Generator().manual_seed(7)
    x1 = torch.randn(6, 5, generator=generator)
    x1[:, -1] = torch.linspace(0.0, 0.8, 6)
    x2 = x1.clone()
    x2[:, -1] += 0.05
    y1 = torch.linspace(1.0, 0.8, 6).unsqueeze(1)
    y2 = y1 - 0.01
    config = AdaptivePINNConfig(input_dim=5, latent_dim=2, hidden_dim=8)
    model = AdaptiveInvariantPINN(config=config, device=torch.device("cpu"))
    model.feature_map.fit(torch.cat((x1, x2), dim=0))
    losses = model._losses((x1, x2, y1, y2))
    encoder_weight = model.solution_u.encoder.net[0].weight
    (monotone_gradient,) = torch.autograd.grad(
        losses.monotone, encoder_weight, allow_unused=True
    )
    assert losses.monotone.item() > 0.0  # penalty is active for this batch
    assert monotone_gradient is not None
    assert monotone_gradient.abs().sum() > 0.0


def test_invariant_statistic_stays_gradient_free() -> None:
    # The row-mean variance is a diagnostic: it must not depend on parameters.
    x1 = torch.randn(4, 5)
    x2 = x1 + 0.01
    y1 = torch.ones(4, 1)
    y2 = y1 - 0.01
    config = AdaptivePINNConfig(input_dim=5, latent_dim=2, hidden_dim=8)
    model = AdaptiveInvariantPINN(config=config, device=torch.device("cpu"))
    losses = model._losses((x1, x2, y1, y2))
    assert losses.invariant.requires_grad is False
    # Sanity: the trainable terms do require grad.
    assert losses.data.requires_grad is True
    assert losses.operator.requires_grad is True
    assert losses.monotone.requires_grad is True


def test_prediction_has_soh_shape_and_finite_values() -> None:
    # Given: a fitted invariant feature map and a batch of cycle features.
    generator = torch.Generator().manual_seed(11)
    x = torch.randn(7, 5, generator=generator)
    x[:, -1] = torch.linspace(0.0, 1.0, 7)
    config = AdaptivePINNConfig(input_dim=5, latent_dim=2, hidden_dim=8)
    model = AdaptiveInvariantPINN(config=config, device=torch.device("cpu"))
    model.feature_map.fit(x)

    # When: SOH is predicted through the invariant representation.
    prediction = model.predict(x)

    # Then: one finite SOH estimate is returned per row (no [0, 1] clamp:
    # measured HUST SOH exceeds 1.0).
    assert prediction.shape == (7, 1)
    assert torch.isfinite(prediction).all()


def test_real_xjtu_minibatch_runs_end_to_end() -> Optional[bool]:
    # Given: adjacent cycles from an actual PINN4SOH XJTU battery file.
    csv_path = PROJECT_ROOT / "data" / "XJTU data" / "2C_battery-1.csv"
    if not csv_path.exists():
        # The datasets are public downloads and not part of the repository;
        # skip instead of failing on a fresh clone.
        print("SKIP test_real_xjtu_minibatch_runs_end_to_end "
              "(data/XJTU data/2C_battery-1.csv not found; download the XJTU dataset first)")
        return False
    raw = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    raw = raw[np.isfinite(raw).all(axis=1)]
    charge_features = raw[:, :-1]
    progress = np.arange(len(raw), dtype=np.float64)[:, None]
    model_inputs = np.column_stack((charge_features, progress))
    minimum = model_inputs.min(axis=0, keepdims=True)
    span = np.maximum(model_inputs.max(axis=0, keepdims=True) - minimum, 1e-8)
    model_inputs = 2.0 * (model_inputs - minimum) / span - 1.0
    soh = raw[:, -1:] / 2.0
    x1 = torch.from_numpy(model_inputs[:-1]).float()
    x2 = torch.from_numpy(model_inputs[1:]).float()
    y1 = torch.from_numpy(soh[:-1]).float()
    y2 = torch.from_numpy(soh[1:]).float()
    loader = DataLoader(TensorDataset(x1, x2, y1, y2), batch_size=64)
    config = AdaptivePINNConfig(input_dim=17, latent_dim=3, hidden_dim=16)
    model = AdaptiveInvariantPINN(config=config, device=torch.device("cpu"))
    model.fit_feature_scaler(loader)

    # When: one training epoch and target-style inference are performed.
    report = model.train_one_epoch(loader)
    prediction = model.predict(x1[:8])

    # Then: the measured-data path produces a finite objective and SOH.
    assert np.isfinite(report.total)
    assert prediction.shape == (8, 1)
    assert torch.isfinite(prediction).all()


def test_trainer_saves_complete_best_checkpoint() -> None:
    # Given: a one-epoch experiment with validation and test loaders.
    x1 = torch.tensor(
        [[1.0, 2.0, 4.0, 8.0, 0.0], [1.1, 2.1, 3.9, 7.8, 0.5]]
    )
    x2 = x1 + torch.tensor([0.01, -0.01, 0.02, -0.02, 0.1])
    y1 = torch.tensor([[1.0], [0.95]])
    y2 = torch.tensor([[0.99], [0.94]])
    loader = DataLoader(TensorDataset(x1, x2, y1, y2), batch_size=2)
    config = AdaptivePINNConfig(input_dim=5, latent_dim=2, hidden_dim=8)
    model = AdaptiveInvariantPINN(config=config, device=torch.device("cpu"))

    # When: the complete trainer lifecycle saves its best state.
    with TemporaryDirectory() as directory:
        runtime = AdaptiveRuntimeConfig(
            epochs=1,
            early_stop=1,
            save_folder=directory,
            log_dir="training.log",
        )
        trainer = AdaptivePINNTrainer(model=model, runtime=runtime)
        trainer.Train(loader, testloader=loader, validloader=loader)
        checkpoint_path = Path(directory) / "model.pth"
        checkpoint = torch.load(checkpoint_path)

        # Then: predictor, shared operator, scaler, and predictions are persisted.
        assert set(checkpoint) == {"solution_u", "shared_operator", "feature_map"}
        assert (Path(directory) / "true_label.npy").exists()
        assert (Path(directory) / "pred_label.npy").exists()


def test_operator_loss_reaches_shared_operator() -> None:
    # Regression: the operator-consistency loss must backpropagate into the
    # shared degradation operator, not only into the encoder that produces
    # the z2 - z1 target.
    torch.manual_seed(3)
    generator = torch.Generator().manual_seed(7)
    x1 = torch.randn(6, 5, generator=generator)
    x1[:, -1] = torch.linspace(0.0, 0.8, 6)
    x2 = x1.clone()
    x2[:, :-1] += 0.02 * torch.randn(6, 4, generator=generator)
    x2[:, -1] += 0.05
    y1 = torch.linspace(1.0, 0.8, 6).unsqueeze(1)
    y2 = y1 - 0.01
    config = AdaptivePINNConfig(input_dim=5, latent_dim=2, hidden_dim=8)
    model = AdaptiveInvariantPINN(config=config, device=torch.device("cpu"))
    model.feature_map.fit(torch.cat((x1, x2), dim=0))
    losses = model._losses((x1, x2, y1, y2))
    operator_weight = model.shared_operator.net[0].weight
    (operator_gradient,) = torch.autograd.grad(
        losses.operator, operator_weight, allow_unused=True
    )
    assert operator_gradient is not None
    assert operator_gradient.abs().sum() > 0.0


def test_checkpoint_restores_into_a_fresh_model() -> None:
    # Round-trip: a checkpoint saved from one trainer must reproduce the same
    # predictions after loading into a freshly constructed model.
    torch.manual_seed(5)
    x1 = torch.tensor(
        [[1.0, 2.0, 4.0, 8.0, 0.0], [1.1, 2.1, 3.9, 7.8, 0.5]]
    )
    x2 = x1 + torch.tensor([0.01, -0.01, 0.02, -0.02, 0.1])
    y1 = torch.tensor([[1.0], [0.95]])
    y2 = torch.tensor([[0.99], [0.94]])
    loader = DataLoader(TensorDataset(x1, x2, y1, y2), batch_size=2)
    config = AdaptivePINNConfig(input_dim=5, latent_dim=2, hidden_dim=8)
    model = AdaptiveInvariantPINN(config=config, device=torch.device("cpu"))
    with TemporaryDirectory() as directory:
        runtime = AdaptiveRuntimeConfig(
            epochs=1, early_stop=1, save_folder=directory, log_dir=None
        )
        AdaptivePINNTrainer(model=model, runtime=runtime).Train(
            loader, testloader=loader, validloader=loader
        )
        checkpoint = torch.load(Path(directory) / "model.pth")
    fresh = AdaptiveInvariantPINN(config=config, device=torch.device("cpu"))
    fresh.solution_u.load_state_dict(checkpoint["solution_u"])
    fresh.shared_operator.load_state_dict(checkpoint["shared_operator"])
    fresh.feature_map.load_state_dict(checkpoint["feature_map"])
    model.eval()  # deterministic prediction (dropout disabled) for the comparison
    fresh.eval()
    reference = model.predict(x1)
    assert torch.allclose(fresh.predict(x1), reference)


def test_log_file_cannot_escape_experiment_directory() -> None:
    # Given: a log path containing parent-directory traversal.
    experiment = PROJECT_ROOT / "results" / "AdaptiveInvariantPINN"

    # When: the trainer resolves that untrusted CLI value.
    try:
        _resolve_log_path(experiment, "../outside.log")
    except UnsafeLogPathError as error:
        message = str(error)
    else:
        raise AssertionError("path traversal was accepted")

    # Then: resolution is rejected before a file handler is opened.
    assert "without directory components" in message


def main() -> None:
    tests = (
        test_normalized_block_cancels_positive_affine_nuisance,
        test_scaler_is_fitted_from_source_rows_and_reused,
        test_training_step_optimizes_joint_operator_objective,
        test_monotone_penalty_reaches_model_parameters,
        test_operator_loss_reaches_shared_operator,
        test_invariant_statistic_stays_gradient_free,
        test_prediction_has_soh_shape_and_finite_values,
        test_real_xjtu_minibatch_runs_end_to_end,
        test_trainer_saves_complete_best_checkpoint,
        test_checkpoint_restores_into_a_fresh_model,
        test_log_file_cannot_escape_experiment_directory,
    )
    for test in tests:
        skipped = test() is False
        if not skipped:
            print(f"PASS {test.__name__}")


if __name__ == "__main__":
    main()

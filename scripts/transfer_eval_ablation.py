"""Transfer-phase ablation: adaptive WITHOUT the operator-consistency loss.

Zero-modification companion to ``scripts/transfer_eval.py``.  This file does
not touch any existing code: it imports the original module, swaps in a
variant of ``run_adaptive_finetune`` whose source training runs with operator
weight ``alpha = 0.0``, and reuses the original ``PAIRS`` / ``run_one`` so the
two-stage protocol (source train -> freeze operator -> fine-tune solution_u
on 1 target battery) stays 1:1 identical.  Fine-tuning itself is unchanged --
the shared operator is frozen during Adaptation, so the ablation isolates the
operator loss's contribution during source training.

Outputs go to ``results/ablation/no_operator/transfer/<pair>/Experiment<k>/``
with the same npy / ``finetune_done.txt`` layout (idempotent resume).

Run:
    python scripts/transfer_eval_ablation.py --repeats 10
Smoke (one pair, one repeat):
    python scripts/transfer_eval_ablation.py --repeats 1 --pairs XJTU-2C-to-HUST
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import transfer_eval  # original, unmodified module
from Model.adaptive_trainer import AdaptivePINNTrainer  # noqa: E402
from scripts.reproduce import build_adaptive_args, make_loaders_for_phase  # noqa: E402


def run_adaptive_finetune_no_operator(src_data, src_idx, tgt_data, tgt_idx, adaptation_lr, exp_dir: Path):
    """Identical to transfer_eval.run_adaptive_finetune except alpha = 0.0
    during source training (no operator-consistency loss)."""
    src_args = build_adaptive_args(src_data, transfer_eval.batch_arg(src_data, src_idx),
                                   **transfer_eval.MAP_KWARGS)
    src_args.batch_size = 256
    src_args.alpha = 0.0  # the ablation: no operator-consistency loss
    src_loaders = make_loaders_for_phase(src_data, transfer_eval.batch_arg(src_data, src_idx), src_args)
    tgt_args = build_adaptive_args(tgt_data, transfer_eval.batch_arg(tgt_data, tgt_idx),
                                   **transfer_eval.MAP_KWARGS)
    tgt_args.batch_size = 128
    tgt_loaders = make_loaders_for_phase(tgt_data, transfer_eval.batch_arg(tgt_data, tgt_idx), tgt_args, small_sample=1)

    src_args.save_folder = str(exp_dir)
    src_args.log_dir = "logging.txt"
    trainer = AdaptivePINNTrainer.from_args(src_args)
    trainer.Train(
        trainloader=src_loaders["train"],
        validloader=src_loaders["valid"],
        testloader=tgt_loaders["test"],
    )
    shutil.copy(exp_dir / "true_label.npy", exp_dir / "src_only_true_label.npy")
    shutil.copy(exp_dir / "pred_label.npy", exp_dir / "src_only_pred_label.npy")

    trainer.Adaptation(
        trainloader=tgt_loaders["train"],
        validloader=tgt_loaders["valid"],
        testloader=tgt_loaders["test"],
        adaptation_lr=adaptation_lr,
        adaptation_epochs=200,
        early_stop=10,
    )


transfer_eval.run_adaptive_finetune = run_adaptive_finetune_no_operator


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--pairs", type=str, default=None,
                    help="comma-separated pair labels to restrict, e.g. XJTU-2C-to-HUST")
    ap.add_argument("--out", default="results/ablation/no_operator/transfer",
                    help="output root for the ablation transfer runs")
    ap.add_argument("--map_mode", choices=["invariant", "identity"], default="invariant")
    ap.add_argument("--temporal_delta", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--include_anchored", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--include_row_mean", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--operator_weight", type=float, default=None)
    ap.add_argument("--latent_dim", type=int, default=None)
    ap.add_argument("--seed_base", type=int, default=0,
                    help="seed of repeat k is seed_base + k (default 0, "
                         "same deterministic reruns as transfer_eval)")
    args = ap.parse_args()
    import argparse as _ap
    transfer_eval.MAP_KWARGS.update(dict(
        map_mode=getattr(args, "map_mode", "invariant"),
        temporal_delta=getattr(args, "temporal_delta", False),
        include_anchored=getattr(args, "include_anchored", True),
        include_row_mean=getattr(args, "include_row_mean", True),
        operator_weight=getattr(args, "operator_weight", None),
        latent_dim=getattr(args, "latent_dim", None)))
    out_root = PROJECT_ROOT / args.out
    out_root.mkdir(parents=True, exist_ok=True)
    from scripts.provenance import record_provenance
    record_provenance(out_root, args)
    wanted = set(args.pairs.split(",")) if args.pairs else None
    for pair in transfer_eval.PAIRS:
        src_data, src_idx, tgt_data, tgt_idx, _ = pair
        label = f"{transfer_eval.batch_label(src_data, src_idx)}-to-{transfer_eval.batch_label(tgt_data, tgt_idx)}"
        if wanted is not None and label not in wanted:
            continue
        for exp in range(1, args.repeats + 1):
            transfer_eval._set_seed(args.seed_base + exp)
            transfer_eval.run_one(pair, exp, "adaptive", out_root)
    print("transfer ablation done")


if __name__ == "__main__":
    main()

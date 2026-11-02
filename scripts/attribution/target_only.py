"""Step-2 attribution control: target-only training (no source pretraining).

Companion to ``scripts/transfer_eval.py``.  For each transfer pair, this
trains ONE model from scratch on the single target battery (exactly the
adaptation-stage data: 1 target battery for train/valid, full target test),
with the same batch size / loss weights / epochs the transfer protocol uses.
Comparing against the transfer results isolates how much of the transfer
score comes from target data alone versus source pretraining.

Both branches reuse the entry code paths 1:1:
  adaptive build_adaptive_args(tgt) + make_loaders_for_phase(small_sample=1)
           + AdaptivePINNTrainer.Train (same as transfer_eval's fine-tune stage)
  orig     build_orig_args(tgt) + make_loaders_for_phase(small_sample=1)
           + run_original_pinn (same trainer used by reproduce.py)

Outputs (same layout and idempotent resume as transfer_eval):
  <out>/<model>/<pair-label>/ExperimentK/{true_label,pred_label,model.pth,
  finetune_done.txt}

Run:
    python scripts/attribution/target_only.py --repeats 10
Smoke (one pair, one repeat):
    python scripts/attribution/target_only.py --repeats 1 --pairs XJTU-2C-to-HUST
Aggregate:
    python scripts/evaluate.py --phase transfer --in results/target_only --out evaluation/target_only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Patch torch.save for unicode paths (same as scripts/transfer_eval.py)
_orig_save = torch.save
def _patched(obj, f, *a, **kw):
    try:
        return _orig_save(obj, f, *a, **kw)
    except RuntimeError as e:
        if "Parent directory" in str(e):
            kw.setdefault("_use_new_zipfile_serialization", False)
            return _orig_save(obj, f, *a, **kw)
        raise
torch.save = _patched

import transfer_eval  # reuse PAIRS, batch_arg, batch_label, _set_seed
from scripts.reproduce import (  # noqa: E402
    build_adaptive_args,
    build_orig_args,
    make_loaders_for_phase,
    run_original_pinn,
)
from Model.adaptive_trainer import AdaptivePINNTrainer  # noqa: E402


def run_target_only(pair, exp: int, model: str, out_root: Path, seed_base: int) -> None:
    src_data, src_idx, tgt_data, tgt_idx, adaptation_lr = pair
    label = f"{transfer_eval.batch_label(src_data, src_idx)}-to-{transfer_eval.batch_label(tgt_data, tgt_idx)}"
    exp_dir = out_root / model / label / f"Experiment{exp}"
    if (exp_dir / "finetune_done.txt").exists():
        print(f"  skip {model} {label} Experiment{exp}")
        return
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [{model}] {label} Experiment{exp} (target-only, no pretraining)")
    transfer_eval._set_seed(seed_base + exp)
    if model == "orig":
        args = build_orig_args(tgt_data, transfer_eval.batch_arg(tgt_data, tgt_idx))
        args.batch_size = 128
        loaders = make_loaders_for_phase(tgt_data, transfer_eval.batch_arg(tgt_data, tgt_idx),
                                         args, small_sample=1)
        run_original_pinn(loaders, exp_dir, args)
    else:
        args = build_adaptive_args(tgt_data, transfer_eval.batch_arg(tgt_data, tgt_idx))
        args.batch_size = 128
        loaders = make_loaders_for_phase(tgt_data, transfer_eval.batch_arg(tgt_data, tgt_idx),
                                         args, small_sample=1)
        args.save_folder = str(exp_dir)
        args.log_dir = "logging.txt"
        trainer = AdaptivePINNTrainer.from_args(args)
        trainer.Train(
            trainloader=loaders["train"],
            validloader=loaders["valid"],
            testloader=loaders["test"],
        )
    (exp_dir / "finetune_done.txt").write_text("done", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--pairs", type=str, default=None,
                    help="comma-separated pair labels to restrict, e.g. XJTU-2C-to-HUST")
    ap.add_argument("--model", choices=["orig", "adaptive", "both"], default="both")
    ap.add_argument("--out", default="results/target_only",
                    help="output root for the target-only control")
    ap.add_argument("--seed_base", type=int, default=0,
                    help="seed of repeat k is seed_base + k (default 0)")
    args = ap.parse_args()
    from scripts.provenance import record_provenance
    out_root = PROJECT_ROOT / args.out
    out_root.mkdir(parents=True, exist_ok=True)
    record_provenance(out_root, args)
    models = ["orig", "adaptive"] if args.model == "both" else [args.model]
    wanted = set(args.pairs.split(",")) if args.pairs else None
    for model in models:
        for pair in transfer_eval.PAIRS:
            src_data, src_idx, tgt_data, tgt_idx, _ = pair
            label = f"{transfer_eval.batch_label(src_data, src_idx)}-to-{transfer_eval.batch_label(tgt_data, tgt_idx)}"
            if wanted is not None and label not in wanted:
                continue
            for exp in range(1, args.repeats + 1):
                run_target_only(pair, exp, model, out_root, args.seed_base)
    print("target-only done")


if __name__ == "__main__":
    main()

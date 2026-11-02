"""Protocol-matched transfer comparison (paper Table 4 fine-tuning protocol).

Pairs: XJTU-2C->HUST, HUST->XJTU-2C, XJTU-2C->TJU-0, TJU-0->XJTU-2C

Both models now follow the same two-stage protocol:

  orig     load the pretrained source PINN (``pretrained model/``), freeze
           dynamical_F, fine-tune solution_u on 1 target battery
           (AdaPINN, per-pair adaptation_lr from the paper fine-tune script)
  adaptive train AdaptiveInvariantPINN on the source (source-fitted scaler),
           freeze shared_operator + feature scaler, fine-tune solution_u on
           the same 1 target battery with the same adaptation_lr

Per Experiment folder:
  true_label.npy / pred_label.npy   post-fine-tuning predictions (headline)
  src_only_true_label.npy / ...     predictions before adaptation (reference)
  finetune_done.txt                 resume marker

Run:
    python scripts/transfer_eval.py --repeats 10
Aggregate:
    python scripts/evaluate.py --in results/transfer --phase transfer --out evaluation

Runs are seeded (``--seed_base``, default 0) with cuDNN in deterministic
mode, so reruns are reproducible on the same hardware/PyTorch build.

Note: the ``orig`` branch loads the source PINN from ``pretrained model/``.
Those weights are the public PINN4SOH pre-trained models (Wang et al.,
Nature Communications 2024) and are NOT redistributed with this repository.
Download them from the original PINN4SOH repository, or train your own with
the bundled ``main_*.py`` scripts, before running ``--model orig``/``both``.
"""
import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for _p in (str(SCRIPT_DIR), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Patch torch.save for unicode paths
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

XJTU_BATCH_NAMES = ["2C", "3C", "R2.5", "R3", "RW", "satellite"]


def _set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# (src_data, src_batch_idx, tgt_data, tgt_batch_idx, adaptation_lr)
# adaptation_lr defaults follow the paper fine-tune script where the same
# target exists: HUST target <- MIT2HUST (2e-4); XJTU-2C target <- TJU2XJTU
# target_batch=0 (4e-4); TJU-0 target <- XJTU2TJU target_batch=0 (3e-3).
PAIRS = [
    ("XJTU", 0, "HUST", None, 0.0002),
    ("HUST", None, "XJTU", 0, 0.0004),
    ("XJTU", 0, "TJU", 0, 0.003),
    ("TJU", 0, "XJTU", 0, 0.0004),
]

# Original PINN4SOH loss weights (alpha, beta), per target dataset
# (upstream PINN4SOH main_*.py defaults).
ORIG_ALPHA_BETA = {"XJTU": (0.7, 0.2), "TJU": (1.0, 0.05), "HUST": (0.5, 0.2), "MIT": (1.0, 0.02)}


def batch_arg(dataset: str, idx):
    """Convert a numeric batch index to the argument make_loaders_for_phase expects."""
    if idx is None:
        return None
    if dataset == "XJTU":
        return XJTU_BATCH_NAMES[idx]
    return idx  # TJU 0..2; HUST/MIT unused


def batch_label(dataset: str, idx) -> str:
    if idx is None:
        return dataset
    if dataset == "XJTU":
        return f"{dataset}-{XJTU_BATCH_NAMES[idx]}"
    return f"{dataset}-{idx}"


def _load_finetune_module():
    # The release renamed the original "main_adaptation - fine-tuning.py"
    # (spaces break importlib on some toolchains) to this hyphen-free name.
    path = PROJECT_ROOT / "main_adaptation_fine-tuning.py"
    spec = importlib.util.spec_from_file_location(
        "main_adaptation_fine_tuning",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_orig_finetune(src_data, src_idx, tgt_data, tgt_idx, adaptation_lr, exp_dir: Path):
    module = _load_finetune_module()
    alpha, beta = ORIG_ALPHA_BETA[tgt_data]
    if src_data in ("XJTU", "TJU"):
        pretrain = f"model_{src_data}_{src_idx}.pth"
    else:
        pretrain = f"model_{src_data}.pth"
    pretrain_path = PROJECT_ROOT / "pretrained model" / pretrain
    if not pretrain_path.exists():
        raise FileNotFoundError(
            f"{pretrain_path} not found. The orig transfer branch needs the public "
            "PINN4SOH pre-trained weights, which are not redistributed here. "
            "Download them from https://github.com/wang-fujin/PINN4SOH (folder "
            "'pretrained model'), or train your own with the bundled main_*.py "
            "scripts, or run only the adaptive branch with --model adaptive."
        )
    args = argparse.Namespace(
        batch_size=128, normalization_method="min-max",
        epochs=200, early_stop=10,
        warmup_epochs=30, warmup_lr=0.002, lr=0.01, final_lr=0.0002, lr_F=0.001,
        F_layers_num=3, F_hidden_dim=60,
        alpha=alpha, beta=beta,
        log_dir="logging.txt", save_folder=str(exp_dir),
        pretrain_model=str(pretrain_path),
        adaptation_lr=adaptation_lr, adaptation_epochs=200,
        target_data=tgt_data,
        target_batch=-1 if tgt_idx is None else tgt_idx,
        source_batch=-1 if src_idx is None else src_idx,
    )
    target_loader = getattr(module, f"load_{tgt_data}_data")(args, small_sample=1)
    model = module.AdaPINN(args)

    # Source-only reference before adaptation
    true_label, pred_label = model.Test(target_loader["test"])
    np.save(exp_dir / "src_only_true_label.npy", true_label)
    np.save(exp_dir / "src_only_pred_label.npy", pred_label)

    model.Adaptation(
        trainloader=target_loader["train"],
        validloader=target_loader["valid"],
        testloader=target_loader["test"],
    )


def run_adaptive_finetune(src_data, src_idx, tgt_data, tgt_idx, adaptation_lr, exp_dir: Path):
    from Model.adaptive_trainer import AdaptivePINNTrainer
    from reproduce import build_adaptive_args, make_loaders_for_phase

    src_args = build_adaptive_args(src_data, batch_arg(src_data, src_idx))
    src_args.batch_size = 256
    src_loaders = make_loaders_for_phase(src_data, batch_arg(src_data, src_idx), src_args)
    tgt_args = build_adaptive_args(tgt_data, batch_arg(tgt_data, tgt_idx))
    tgt_args.batch_size = 128
    # 1 target battery for adaptation train/valid, full target test for scoring
    tgt_loaders = make_loaders_for_phase(tgt_data, batch_arg(tgt_data, tgt_idx), tgt_args, small_sample=1)

    src_args.save_folder = str(exp_dir)
    src_args.log_dir = "logging.txt"
    trainer = AdaptivePINNTrainer.from_args(src_args)
    trainer.Train(
        trainloader=src_loaders["train"],
        validloader=src_loaders["valid"],
        testloader=tgt_loaders["test"],
    )
    # Preserve source-only predictions before fine-tuning overwrites them
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


def run_one(pair, exp: int, model: str, out_root: Path) -> None:
    src_data, src_idx, tgt_data, tgt_idx, adaptation_lr = pair
    label = f"{batch_label(src_data, src_idx)}-to-{batch_label(tgt_data, tgt_idx)}"
    exp_dir = out_root / model / label / f"Experiment{exp}"
    if (exp_dir / "finetune_done.txt").exists():
        print(f"  skip {model} {label} Experiment{exp}")
        return
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [{model}] {label} Experiment{exp} (adaptation_lr={adaptation_lr})")
    if model == "orig":
        run_orig_finetune(src_data, src_idx, tgt_data, tgt_idx, adaptation_lr, exp_dir)
    else:
        run_adaptive_finetune(src_data, src_idx, tgt_data, tgt_idx, adaptation_lr, exp_dir)
    (exp_dir / "finetune_done.txt").write_text("done", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--model", choices=["orig", "adaptive", "both"], default="both")
    ap.add_argument("--out", default="results/transfer",
                    help="output root (aggregate with scripts/evaluate.py "
                         "--phase transfer --in results/transfer --out evaluation)")
    ap.add_argument("--seed_base", type=int, default=0,
                    help="seed of repeat k is seed_base + k (default 0: "
                         "seeded, deterministic reruns)")
    args = ap.parse_args()
    models = ["orig", "adaptive"] if args.model == "both" else [args.model]
    out_root = PROJECT_ROOT / args.out
    out_root.mkdir(parents=True, exist_ok=True)
    try:
        from provenance import record_provenance
    except ImportError:
        from scripts.provenance import record_provenance
    record_provenance(out_root, args)
    if "orig" in models:
        # Precheck so a missing public weight aborts before any GPU work starts.
        needed = []
        for pair in PAIRS:
            src_data, src_idx = pair[0], pair[1]
            name = (f"model_{src_data}_{src_idx}.pth" if src_data in ("XJTU", "TJU")
                    else f"model_{src_data}.pth")
            path = PROJECT_ROOT / "pretrained model" / name
            if not path.exists():
                needed.append(str(path))
        if needed:
            raise FileNotFoundError(
                "Missing public PINN4SOH pre-trained weights for the orig branch: "
                + ", ".join(needed)
                + ". Download them from https://github.com/wang-fujin/PINN4SOH "
                  "(folder 'pretrained model') or rerun with --model adaptive.")
    for model in models:
        for pair in PAIRS:
            for exp in range(1, args.repeats + 1):
                _set_seed(args.seed_base + exp)
                run_one(pair, exp, model, out_root)
    print("transfer done")


if __name__ == "__main__":
    main()

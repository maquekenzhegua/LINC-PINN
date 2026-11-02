# paper_data — the exact data behind every table and figure

This directory ships the per-run predictions and the aggregated summaries that
produce every number reported in the paper, so that all tables and figures can
be **recomputed exactly, without training anything** (see "Reproducing the
paper" in the top-level README).

## Provenance

- Produced 2026-09-15 to 2026-09-17 by the **seeded rerun campaign (2026-09)** — the
  complete experiment suite under the unchanged upstream PINN4SOH protocol
  (`scripts/reproduce.py`, `scripts/transfer_eval.py`, `scripts/baselines.py`,
  the five ablation variants, the gamma sweep, and the transfer operator
  ablation). Every suite root carries a `provenance.json` recording the sha256
  of every code file, the full argument namespace, and the environment.
- Seeding: every run uses `seed_base + repeat index` with cuDNN determinism
  flags on the transfer entries; rerunning on the same GPU class reproduces
  these numbers to float precision.
- Environment: Python 3.9.13, PyTorch 2.0.1+cu118, numpy 1.23.5,
  scikit-learn 1.6.1, single NVIDIA RTX 4060 (8 GB).
- Every condition is n = 10 repeats. Metrics are battery-level MAE/MAPE/RMSE
  with the per-dataset gap thresholds XJTU 0.05 / TJU 0.07 / HUST 0.10 /
  MIT 0.07, aggregated as mean +/- std over repeats by `scripts/evaluate.py`.
- Attribution controls beyond the main suite: `results/target_only/`
  (adaptive+orig from scratch on one target battery), source-only scoring of
  the archived pre-fine-tuning arrays, the gamma=0 direction-weight control,
  and four deterministic labeled-subset rotations (`results/small_combos/`).
- Cross-check: `evaluation/batteryid/` re-aggregates all 161 condition/model
  groups by true battery IDs (cycle-progress boundaries); only TJU-0, MIT and
  the XJTU-2C-to-TJU-0 direction shift (<= 0.008 MAPE), with no conclusion
  change — the label-gap numbers remain the paper convention.
- 2026-09-18 data repair: `results/baselines/small/GPR/small-XJTU-2C-n4/`
  Experiments 8-9 were degenerate zero-error runs from a superseded vintage
  (pred_label identical to true_label, test stream predating a battery-order
  fix). They were rerun under the same seeded protocol; the battery-ID
  snapshot (`evaluation/batteryid/`) and `figures/data/fig5_small.csv` were
  regenerated from the corrected predictions. No other suite was touched.

## Layout

```
evaluation/            the aggregated summaries behind every paper table:
                       reproduce/{summary,summary_transfer}.json (Tables 2-4),
                       baselines/{summary,summary_transfer}.json (tab5-7),
                       ablation/<v>/summary.json (tab8) incl.
                       no_operator/transfer (tab9),
                       gamma_sweep_summary.csv (tab10, fig7b),
                       source_only/summary.json (pre-fine-tuning scores),
                       target_only/summary_transfer.json (attribution control),
                       combos/combo{0..3}/summary.json (subset rotations),
                       batteryid/ true-cell-ID re-aggregation
                       (supplementary cross-check)
results/reproduce/     regular + small, orig & adaptive, seeded
                       {phase}/{model}/{condition}/Experiment{k}/true_label.npy
                                                            /pred_label.npy
results/transfer/      4 directed pairs, both models; each Experiment also has
                       src_only_*.npy = predictions before the 1-battery
                       fine-tuning
results/baselines/     LSTM/GRU/CNN/MLP/GPR, regular + small + transfer
results/ablation/      no_operator (HUST + XJTU small + transfer), no_rowmean,
                       no_anchored, src_scaler, latent3, gamma0..gamma10
results/target_only/   attribution control (both models, 4 pairs)
results/small_combos/  labeled-subset rotations combo0..combo3
```

Each suite directory also carries `provenance.json` (code sha256 + args);
the ablation variants carry per-variant files. Per-run training logs
(`logging.txt`) are shipped; the `model.pth` checkpoints are excluded from
the repository (they are regenerable by retraining and are not needed to
recompute any metric). Every metric in the paper is a function of the
shipped `.npy` predictions and `.json` summaries alone.

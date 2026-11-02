# AdaptiveInvariantPINN

Physics-informed neural network for lithium-ion battery **state-of-health (SOH)
estimation** that remains reliable in **small-sample** and **cross-dataset
regimes**. This repository is the official implementation and experiment suite
for our submission to *Journal of Energy Storage* (Elsevier).

The method builds on
[PINN4SOH](https://github.com/wang-fujin/PINN4SOH) (Wang et al.,
*Nature Communications*, 2024) and addresses its main failure mode under data
scarcity: with 1&ndash;4 training batteries, the original per-battery
min-max + learned encoder memorizes the source cells, the latent
representation collapses, and test batteries fall out of distribution
(constant predictions).

## Method in one paragraph

1. **Invariant feature map** (label-free, no fitted statistics): each row of
   the 16 charge features is mapped to (i) row-standardized coordinates,
   (ii) offsets from the first feature of the same row, (iii) the min-maxed
   cycle progress, and (iv) the row mean of the raw features &mdash; the last
   restores the common-mode aging level that channels (i)+(ii) are blind to,
   while staying bounded for every battery.
2. **No source-fitted scaler** (`identity` mode): a z-score fitted on a few
   source batteries distorts unseen cells and can amplify near-constant source
   dimensions by 1/&epsilon;.
3. **8-dim latent bottleneck** with a sine-activated encoder and a **shared
   degradation operator** `F(z_t, t) &asymp; z_{t+1} &minus; z_t` trained
   jointly (operator weight 0.5), which regularizes the latent dynamics to be
   battery-agnostic.
4. **Prediction-side monotonicity** penalty identical in form to the original
   physics loss, with per-dataset weights (XJTU 0.2 / TJU 0.05 / HUST 0.2 /
   MIT 0.02).

Result (10 repeats, battery-level MAPE, mean of ten runs): both mean MAPE and
RMSE improve over the original PINN4SOH in **16 of the 23 evaluated
conditions**; the remaining settings show mixed changes or consistent
regressions (e.g., regular HUST is marginally worse on both metrics).
One-standard-deviation overlap is not treated as a significance test.  The
largest gains occur where data is scarce (e.g., XJTU-2C n=1: 0.0130 vs
0.0186 MAPE; transfer XJTU-2C&rarr;HUST: 0.0237 vs 0.0333).

## Differences from upstream PINN4SOH

The dataloader, original-model code, and evaluation conventions are inherited
from the [PINN4SOH](https://github.com/wang-fujin/PINN4SOH) codebase
(Nature Communications 14, 2024).  The inherited protocol is kept unchanged
and is shared by all compared models, so model-to-model comparisons are
internally fair, but every absolute number below is a retrospective-protocol
result:

- **Full-trajectory min--max** (inherited): each battery's features are scaled
  by its own complete retained trajectory, including at test time.
- **Label-dependent row filtering** (inherited): rows outside any column's
  mean &plusmn; 3 standard deviations &mdash; including the capacity column &mdash;
  are dropped, so test labels affect which rows are retained.
- **Shared-cycle train/valid split** (inherited): adjacent cycle pairs are
  built first and split randomly afterwards, so one cycle can appear in both
  a training pair and a validation pair.
- **Label-gap segment evaluation** (inherited): battery boundaries are
  reconstructed from label jumps (thresholds 0.05/0.07/0.10/0.07 for
  XJTU/TJU/HUST/MIT).  In TJU-0 and MIT these segments do not exactly match
  physical battery IDs; archived scores are label-gap segment metrics.

Changes made in this fork relative to upstream:

- Battery file lists are sorted and one `nominal_capacities` indexing bug is
  fixed.
- `eval_metrix` is called as `(true_label, pred_label)`, matching its
  signature; the upstream code passes them swapped (affects MAPE/MSE
  direction).
- Per-experiment seeding (`--seed_base`), cuDNN determinism flags in
  `reproduce.py`, and `provenance.json` code snapshots were added.
- The adaptive model itself (row feature map, shared degradation operator,
  training procedure) is new; see the paper.

All PINN4SOH-orig baseline numbers reported in the paper come from running
this fork's unified pipeline on both models, not from the upstream
repository's own artifacts.  The paper's figure pipeline ships here:
`figures/data/make_figure_data.py` regenerates the nine tidy CSVs behind
fig2&ndash;fig7 from `paper_data/results` (fig2/fig3 additionally need the
downloaded datasets under `data/`; their shipped CSVs remain valid without),
and `figures/matlab/run_all.m` renders fig2&ndash;fig7 as vector PDFs.

## Repository layout

```
Model/                     AdaptiveInvariantPINN model, trainer, invariant features,
                           original PINN (Model.py) and comparison nets (Compare_Models.py)
dataloader/                dataset reading + per-battery min-max normalization
main_XJTU.py etc.          original per-dataset entry points (1:1 with the PINN4SOH paper;
                           their `--gamma`/loss defaults now match the final paper config, but the
                           campaign is driven by scripts/reproduce.py, which sets every hyperparameter
                           explicitly — prefer it for reproduction)
main_adaptive_transfer.py  LEGACY train-only reference (does NOT reproduce Table 4;
                           use scripts/transfer_eval.py for the published protocol)
main_adaptation_fine-tuning.py  paper fine-tune driver reused by scripts/transfer_eval.py
                           for the orig branch of Table 4
scripts/reproduce.py       Table 2/3 suite for orig + adaptive (regular / small / ablations)
scripts/baselines.py       external baselines: LSTM, GRU, CNN, MLP, GPR under the same protocol
scripts/evaluate.py        battery-level MAE/MAPE/RMSE aggregation (mean +/- std over repeats)
scripts/transfer_eval.py   Table 4 transfer suite (pretrain + freeze operator + 1-battery fine-tune;
                           scripts/run_transfer.py is a thin wrapper around it)
scripts/transfer_eval_ablation.py  Table 4 operator ablation (adaptive trained with alpha=0)
scripts/attribution/                  attribution controls (target_only, source_only_eval,
                           battery_reaggregate; see the docstrings)
tests/                     unit tests (11) incl. anti-regression checks for the physics losses;
                           the real-data test auto-skips without the downloaded datasets
figures/                   paper figure pipeline: data/ (nine tidy CSVs behind fig2-fig7),
                           matlab/ (run_all.m renders them as vector PDFs),
                           ppt/ (Fig 1 architecture sources)
paper_data/                seeded per-run predictions + aggregated summaries
                           behind every table/figure (see paper_data/README.md)
                           a4_campaign/ holds the runs the manuscript reports;
                           results/ and evaluation/ hold an earlier replicate
```

## Installation

```bash
conda create -n adaptive-pinn python=3.9
conda activate adaptive-pinn
pip install -r requirements.txt
# GPU (optional): install the CUDA 11.8 build of PyTorch 2.0.1
```

The paper results were produced with Python 3.9.13, PyTorch 2.0.1+cu118,
scikit-learn 1.6.1, numpy 1.23.5, pandas 2.0.3, scipy 1.10.1 on a single RTX
4060 Laptop (8 GB).

## Data

Download the four public datasets and place them under `data/` with these
exact folder names (one CSV per battery, as produced by the preprocessing
library below):

| Dataset | Folder | Source |
|---|---|---|
| XJTU | `data/XJTU data` | https://wang-fujin.github.io/ |
| TJU  | `data/TJU data`  | https://zenodo.org/record/6405084 |
| HUST | `data/HUST data` | https://data.mendeley.com/datasets/nsc7hnsg4s/2 |
| MIT  | `data/MIT data`  | https://data.matr.io/1/projects/5c48dd2bc625d700019f3204 |

Expected inner layout (the loaders hardcode these paths): XJTU — flat CSVs
`<batch>_battery-N.csv` (satellite files are `Sim_satellite_battery-N.csv`);
TJU — subfolders `Dataset_1_NCA_battery/`, `Dataset_2_NCM_battery/`,
`Dataset_3_NCM_NCA_battery/`; MIT — dated subfolders `2017-05-12/`,
`2017-06-30/`, `2018-04-12/`; HUST — flat CSVs `G-N.csv`. Run everything from
the repository root so the relative `data/...` paths resolve.

Protocol note: the released experiment suite ran the TJU/HUST/MIT baselines
and adaptive models at batch size 256 (GPU-memory cap on the reference card;
the original PINN4SOH entry points default to 512). `scripts/reproduce.py`
sets the cap itself, and the four `main_*.py` entry points now default to 256
to match.

Charging-feature preprocessing code:
https://github.com/wang-fujin/Battery-dataset-preprocessing-code-library

The transfer baselines (Table 4, `orig` branch) additionally expect the
source PINN checkpoints under `pretrained model/` (11 `.pth` files). Those
are the public pre-trained models of PINN4SOH -- they were not trained by
us and are therefore *not* redistributed here. Download that folder from
the [PINN4SOH repository](https://github.com/wang-fujin/PINN4SOH) and place
it at the repository root, or retrain the source models with the bundled
`main_XJTU.py` / `main_TJU.py` / `main_HUST.py` / `main_MIT.py` entry
points. The adaptive branch (`--model adaptive`) does not need them.

Please cite the corresponding dataset publications when you use this code.

## Reproducing the paper

All commands run from the repository root with the environment's Python
(`PY` below). There are two paths:

**Path A — verify the paper numbers exactly (no training, ~1 min).**
`paper_data/` ships the seeded per-run predictions and aggregated
summaries behind every table and figure, each suite accompanied by a
`provenance.json` code fingerprint. Recompute all metrics from that data
alone:

```bash
# Tables 2 and 3 (regular + small, orig & adaptive)
$PY scripts/evaluate.py --phase all --in paper_data/results/reproduce --out evaluation
# Table 4 (4 directed transfer pairs)
$PY scripts/evaluate.py --phase transfer --in paper_data/results/transfer --out evaluation
# External baselines (LSTM / GRU / CNN / MLP / GPR), regular + small
$PY scripts/evaluate.py --phase all --in paper_data/results/baselines --out evaluation --models LSTM GRU CNN MLP GPR
# External-baseline transfer under the same fine-tune protocol (tab7)
$PY scripts/summarize_baseline_transfer.py --baselines-in paper_data/results/baselines \
    --main-in paper_data/results/transfer --out evaluation/baselines
# Ablations (tab8): HUST + XJTU-2C small-sample extension
$PY scripts/evaluate.py --phase all --in paper_data/results/ablation/no_operator --out evaluation/ablation/no_operator
$PY scripts/evaluate.py --phase all --in paper_data/results/ablation/no_rowmean  --out evaluation/ablation/no_rowmean
$PY scripts/evaluate.py --phase all --in paper_data/results/ablation/no_anchored --out evaluation/ablation/no_anchored
$PY scripts/evaluate.py --phase all --in paper_data/results/ablation/src_scaler  --out evaluation/ablation/src_scaler
$PY scripts/evaluate.py --phase all --in paper_data/results/ablation/latent3     --out evaluation/ablation/latent3
# Transfer-phase ablation: adaptive with the operator loss removed (tab9)
$PY scripts/evaluate.py --phase transfer --in paper_data/results/ablation/no_operator/transfer \
    --out evaluation/ablation/no_operator/transfer
# Gamma sweep (tab10): four sweep variants + the collapse-curve CSV
$PY scripts/evaluate.py --phase all --in paper_data/results/ablation/gamma1  --out evaluation/ablation/gamma1
$PY scripts/evaluate.py --phase all --in paper_data/results/ablation/gamma2  --out evaluation/ablation/gamma2
$PY scripts/evaluate.py --phase all --in paper_data/results/ablation/gamma5  --out evaluation/ablation/gamma5
$PY scripts/evaluate.py --phase all --in paper_data/results/ablation/gamma10 --out evaluation/ablation/gamma10
$PY scripts/summarize_gamma_sweep.py --eval-dir evaluation --results-root paper_data/results \
    --main-subdir reproduce --sweep-subdir ablation --eval-ablation-subdir ablation

# Attribution controls (target-only / source-only / gamma=0 / battery-subset
# rotations) behind the paper's transfer-attribution and robustness text
$PY scripts/attribution/source_only_eval.py   # scores the shipped src_only arrays (paper_data/results/transfer)
# Battery-ID cross-check of the label-gap segmentation (supplementary)
$PY scripts/attribution/battery_reaggregate.py --with-baselines \
    --boundaries-from paper_data/evaluation/batteryid/boundaries

# Unit tests
$PY tests/test_adaptive_invariant_pinn.py

# Figure data (npy/paper_data -> the nine tidy CSVs; fig2/fig3 need data/)
$PY figures/data/make_figure_data.py
# Figures (MATLAB; vector PDFs into figures/)
cd figures/matlab && matlab -batch "run_all"
```

**Path B — retrain everything (days of GPU time).** Runs are resumable
(existing `Experiment*/true_label.npy` outputs are skipped) and **seeded by
default** (`seed_base + repeat index`, cuDNN deterministic), so a rerun on the
same hardware/PyTorch build is bit-exact; on other hardware expect results
within the reported std.

```bash
# Table 2: regular SOH estimation, both models, 10 repeats
$PY scripts/reproduce.py --phase regular --model both --repeats 10 --out results/reproduce

# Table 3: small-sample (1-4 training batteries), both models, 10 repeats
$PY scripts/reproduce.py --phase small --model both --repeats 10 --out results/reproduce

# Table 4: cross-dataset transfer, 10 repeats
# (the orig branch additionally needs the PINN4SOH pre-trained weights --
#  see "Data" above; use --model adaptive to skip them)
$PY scripts/transfer_eval.py --repeats 10 --out results/transfer

# External baselines (LSTM / GRU / CNN / MLP / GPR), same protocol
$PY scripts/baselines.py --phase all --models all --repeats 10 --out results/baselines

# Ablations (each n=10)
$PY scripts/reproduce.py --phase all --model adaptive --repeats 10 --out results/ablation/no_operator --operator_weight 0.0
$PY scripts/reproduce.py --phase small --model adaptive --datasets XJTU --repeats 10 --out results/ablation/no_rowmean --no-include_row_mean
$PY scripts/reproduce.py --phase small --model adaptive --datasets XJTU --repeats 10 --out results/ablation/no_anchored --no-include_anchored
$PY scripts/reproduce.py --phase small --model adaptive --datasets XJTU --repeats 10 --out results/ablation/src_scaler --feature_scaler source
$PY scripts/reproduce.py --phase small --model adaptive --datasets XJTU --repeats 10 --out results/ablation/latent3 --latent_dim 3

# Gamma sweep: XJTU small-sample, adaptive only, 10 repeats per gamma value
$PY scripts/reproduce.py --phase small --model adaptive --datasets XJTU --repeats 10 --out results/ablation/gamma1  --small_gamma 1
$PY scripts/reproduce.py --phase small --model adaptive --datasets XJTU --repeats 10 --out results/ablation/gamma2  --small_gamma 2
$PY scripts/reproduce.py --phase small --model adaptive --datasets XJTU --repeats 10 --out results/ablation/gamma5  --small_gamma 5
$PY scripts/reproduce.py --phase small --model adaptive --datasets XJTU --repeats 10 --out results/ablation/gamma10 --small_gamma 10

# Transfer-phase ablation: adaptive with the operator loss removed (alpha = 0)
# during source training; fine-tuning protocol unchanged
$PY scripts/transfer_eval_ablation.py --repeats 10 --out results/ablation/no_operator/transfer

# Aggregate into paper tables (battery-level metrics, mean +/- std;
# evaluation.py merges keys, so one shared --out is safe)
$PY scripts/evaluate.py --phase all --in results/reproduce --out evaluation
$PY scripts/evaluate.py --phase transfer --in results/transfer --out evaluation
$PY scripts/evaluate.py --phase all --in results/baselines --out evaluation --models LSTM GRU CNN MLP GPR
$PY scripts/evaluate.py --phase all  --in results/ablation/no_operator --out evaluation/ablation/no_operator
$PY scripts/evaluate.py --phase small --in results/ablation/no_rowmean  --out evaluation/ablation/no_rowmean
$PY scripts/evaluate.py --phase small --in results/ablation/no_anchored --out evaluation/ablation/no_anchored
$PY scripts/evaluate.py --phase small --in results/ablation/src_scaler  --out evaluation/ablation/src_scaler
$PY scripts/evaluate.py --phase small --in results/ablation/latent3     --out evaluation/ablation/latent3

# Transfer-phase ablation summary
$PY scripts/evaluate.py --phase transfer --in results/ablation/no_operator/transfer --out evaluation/ablation/no_operator/transfer

# Gamma-sweep summaries + collapse curve
$PY scripts/evaluate.py --phase all --in results/ablation/gamma1  --out evaluation/ablation/gamma1
$PY scripts/evaluate.py --phase all --in results/ablation/gamma2  --out evaluation/ablation/gamma2
$PY scripts/evaluate.py --phase all --in results/ablation/gamma5  --out evaluation/ablation/gamma5
$PY scripts/evaluate.py --phase all --in results/ablation/gamma10 --out evaluation/ablation/gamma10
$PY scripts/summarize_gamma_sweep.py --eval-dir evaluation --results-root results \
    --main-subdir reproduce --sweep-subdir ablation --eval-ablation-subdir ablation

# External-baseline transfer summary (pairs the external models with adaptive/orig)
$PY scripts/summarize_baseline_transfer.py --baselines-in results/baselines \
    --main-in results/transfer --out evaluation/baselines

# Unit tests + figures
$PY tests/test_adaptive_invariant_pinn.py
$PY figures/data/make_figure_data.py
cd figures/matlab && matlab -batch "run_all"
```

`figures/data/make_figure_data.py` reads your own rerun results when
present (Path B outputs) and otherwise the shipped `paper_data/` mirror, so
after Path B the figures reflect your runs, and on a fresh clone they
reproduce the paper exactly (fig2/fig3 additionally need `data/`). Ablation-study flags: `--latent_dim`, `--include_row_mean`,
`--include_anchored`, `--feature_scaler`, `--operator_weight`,
`--seed_base` (seeding is on by default, set `--seed_base` to shift it).

### Code fingerprint

Every shipped suite root in `paper_data/results/` carries a `provenance.json`
recording the SHA256 of every `.py` file at run time, the full CLI arguments,
and the environment (Python 3.9.13, torch 2.0.1+cu118, seeded `--seed_base 0`).
For the five main suites the recorded fingerprint is identical, and the
shipped code byte-matches it with the following documented exceptions:

- `main_adaptation - fine-tuning.py` was renamed to
  `main_adaptation_fine-tuning.py` at release (content byte-identical).
- `Model/adaptive_trainer.py` and `main_comparision.py` received post-campaign
  bug fixes (a missing exception import on a never-hit path; a syntax error in
  an upstream-inherited entry point).
- The evaluation helpers (`scripts/evaluate.py`, `scripts/baselines.py`,
  the summarize tools, the four `main_*.py` default values) differ from the
  recorded bytes only in docstrings, default paths, or annotations; the
  campaign always passed every hyperparameter explicitly (see `argv` inside
  each `provenance.json`).
- A handful of internal one-off tools named in the fingerprint are not
  redistributed. Ablation variants carry per-variant fingerprints recorded at
  the end of the campaign.

Directory and file names were standardized for the public release after the
campaign; paths recorded inside `provenance.json` (argv, args, and
`code_sha256` keys) keep the campaign-time names (e.g. `results/v4_reproduce`,
`scripts2/`) and are intentionally unmodified.

Local absolute paths were scrubbed from the shipped logs and metadata; the
code hashes inside `provenance.json` are untouched.

## License and attribution

This project is released under the MIT license (see `LICENSE`) for all code
written for this work. Parts of the data loading, the original PINN
implementation, and the evaluation protocol are reused from
[PINN4SOH](https://github.com/wang-fujin/PINN4SOH) (Wang et al.,
"Physics-informed neural network for lithium-ion battery degradation stable
modeling and prognosis", *Nature Communications* 15, 2024) for academic
reproducibility, with attribution; the original repository ships no license
file, so please cite the original paper if you use those components.

## Citation

A machine-readable `CITATION.cff` is included (fill in the author list upon
acceptance).

```bibtex
@article{adaptiveinvariantpinn2026,
  title  = {AdaptiveInvariantPINN: invariant physics-informed learning for
            battery state-of-health estimation under small-sample and
            cross-dataset regimes},
  author = {[authors]},
  journal= {Journal of Energy Storage},
  year   = {2026},
  note   = {[to be updated upon acceptance]}
}
```

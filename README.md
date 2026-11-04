# AdaptiveInvariantPINN

Reference implementation of the **latent-increment physics-informed network**
for lithium-ion battery **state-of-health (SOH) estimation** under **label
scarcity**. This repository is the official code and data release for our
submission to *Journal of Energy Storage* (Elsevier).

The constraint that physics-informed estimators usually write on the predicted
SOH field needs a derivative that a single record per cycle cannot supply. This
model moves it onto the internal state: a shared operator predicts the one-step
increment of the latent state and is penalised against the increment the
encoder produces. That increment is available on every retained cycle,
labelled or not, so the penalty stays informative at the budgets where a field
residual goes quiet.

## Results

Ten seed-matched runs per setting, battery-level MAPE, four public ageing
datasets (XJTU, TJU, MIT, HUST), 387 cells and 310,705 cycles.

| Regime | Proposed | Reference | Best external |
| --- | --- | --- | --- |
| Few labels, 8 settings (macro) | **1.141** | 2.092 | 1.665 (XGBoost) |
| Full labels, 11 conditions (macro) | 1.000 | **0.951** | 1.282 (LSTM) |
| Transfer, 4 shared directions (macro) | **1.684** | 1.986 | 3.140 (GRU) |

The proposed model is the lowest of the seven methods in all eight few-label
settings. Under full labels it is second to the reference estimator, so the
constraint is neutral where labels are plentiful. Every macro average above is
recomputable from `results/` alone.

## Method in one paragraph

Each cycle's 16 charge features are mapped to an invariant coordinate: row
standardisation, the min-maxed cycle progress, and the rate-normalised change
of every feature with respect to the previous cycle, giving a 33-dimensional
row. A sine-activated encoder `g_theta` maps that row to an 8-dimensional
latent state `z`, and a linear readout `h_omega` maps `z` to SOH. A shared
operator `F_psi` predicts the one-step latent increment `z_{i+1} - z_i` from
`(z_i, t_i)`; its residual is the operator loss, weighted by `alpha`. A
prediction-side monotonicity penalty weighted by `gamma` keeps SOH decreasing
along each trajectory. The operator is frozen at transfer time and only the
encoder and the readout are re-fitted on the target rows.

# 1. System requirements

The released results were produced with Python 3.9.13 and PyTorch 2.0.1+cu118
on a single RTX 4060 Laptop (8 GB). The code also runs on CPU and on the
versions pinned in `requirements.txt`.

| Package | Version used for the paper |
| --- | --- |
| python | 3.9.13 |
| torch | 2.0.1+cu118 |
| numpy | 1.23.5 |
| pandas | 2.0.3 |
| scikit-learn | 1.6.1 |
| scipy | 1.10.1 |
| matplotlib | >=3.5 |
| xgboost | 1.7.6 |
| pytest | 8.x (unit tests only) |

# 2. Installation guide

If you are not familiar with Python and the PyTorch framework, install Anaconda
first and use it to configure the environment.

## 2.1 Create environment

```bash
conda create -n adaptive-pinn python=3.9
```

## 2.2 Activate environment

```bash
conda activate adaptive-pinn
```

## 2.3 Install dependencies

```bash
pip install -r requirements.txt
# GPU (optional): install the CUDA 11.8 build of PyTorch 2.0.1
```

# 3. Demo

Every command runs from the repository root so the relative `data/...` paths
resolve. `$PY` below is the environment's Python. There are two paths.

## 3.1 Path A - verify the paper numbers without training (~1 min)

`results/` ships the seeded per-run predictions behind every table and figure:
`pred_label.npy` and `true_label.npy` per `Experiment1..10`, seed-matched
across models, so every paired comparison in the manuscript can be recomputed
by reading those files directly.

```bash
# Table 2 (full-label errors) and Table 5 (seven-method comparison)
$PY "results analysis/Regular results.py"

# Table 3 (few-label errors), Table 6 (few-label comparison), Table 8 (ablations)
$PY "results analysis/Small results.py"

# Table 4 (directed transfer), Table 7 (transfer comparison), target-only control
$PY "results analysis/Transfer results.py"

# Parameter budget: 11,257 for the proposed model, 13,662 for the reference
$PY "count parameters.py"

# Corpus size: 387 cells and 310,705 cycles
$PY "data analysis/sample count.py"

# Figure data: the nine tidy CSVs behind Fig. 2-7
$PY plotter/data/make_figure_data.py
```

The three `results analysis/*.py` scripts print the manuscript tables cell by
cell, including the macro averages. `count parameters.py` reproduces the
11,257 / 6,509 / 4,748 / 42% split quoted in Section 3.4, and
`data analysis/sample count.py` reproduces the 387 cells and 310,705 cycles
quoted in Section 3.1.

## 3.2 Path B - retrain everything (days of GPU time)

Runs are resumable (existing `Experiment*/true_label.npy` outputs are skipped)
and seeded (`seed_base + repeat index`, cuDNN deterministic), so a rerun on the
same hardware and PyTorch build is bit-exact; on other hardware expect results
within the reported standard deviation.

```bash
# Table 2: full-label SOH estimation, both models, 10 repeats
$PY scripts/reproduce.py --phase regular --model both --repeats 10 --out results/reproduce

# Table 3: few-label (1-4 labelled cells), both models, 10 repeats
$PY scripts/reproduce.py --phase small --model both --repeats 10 --out results/reproduce

# Table 4: directed transfer, six pairs, 10 repeats
$PY scripts/transfer_eval.py --repeats 10 --out results/transfer

# External baselines (MLP, CNN, XGBoost, GRU, LSTM) under the same protocol
$PY scripts/baselines.py --phase all --models all --repeats 10 --out results/baselines

# Table 8 ablations, each 10 repeats
$PY scripts/reproduce.py --phase small --model adaptive --repeats 10 --out results/ablation/no_operator --operator_weight 0.0
$PY scripts/reproduce.py --phase small --model adaptive --repeats 10 --out results/ablation/no_delta      --no-temporal_delta
$PY scripts/reproduce.py --phase small --model adaptive --repeats 10 --out results/ablation/no_z          --map_mode identity
$PY scripts/reproduce.py --phase small --model adaptive --repeats 10 --out results/ablation/latent3       --latent_dim 3

# Transfer-phase operator ablation (alpha = 0 during source training)
$PY scripts/transfer_eval_ablation.py --repeats 10 --out results/ablation/no_operator/transfer

# Unit tests (11 tests; the real-data test needs data/ and skips without it)
$PY -m pytest tests/test_adaptive_invariant_pinn.py -q
```

> As with any neural-network training, the runs are stochastic. The results
> obtained from the commands above are therefore not expected to be identical
> to those in our manuscript. The shipped `results/` tree is the record of the
> runs the manuscript actually reports, and a rerun seeded on the same
> hardware reproduces them bit-exactly.

## 3.3 Figures

`plotter/data/make_figure_data.py` reads `results/` and writes the nine tidy
CSVs that back Fig. 2-7. `plotter/matlab/run_all.m` renders them as vector
PDFs. `plotter/ppt/` holds the editable sources of the Fig. 1 architecture
drawing. Fig. 2 and Fig. 3 additionally need the raw cycles, which come from
the public datasets under `data/`.

# 4. Additional information

## 4.1 Data

The data in the `data` folder is preprocessed data: one CSV per cell, in the
form the loaders expect. The four original archives are public:

| Dataset | Folder | Source |
| --- | --- | --- |
| XJTU | `data/XJTU data` | https://wang-fujin.github.io/ |
| TJU | `data/TJU data` | https://zenodo.org/record/6405084 |
| HUST | `data/HUST data` | https://data.mendeley.com/datasets/nsc7hnsg4s/2 |
| MIT | `data/MIT data` | https://data.matr.io/1/projects/5c48dd2bc625d700019f3204 |

Expected inner layout (the loaders hardcode these paths): XJTU - flat CSVs
`<batch>_battery-N.csv`; TJU - subfolders `Dataset_1_NCA_battery/`,
`Dataset_2_NCM_battery/`, `Dataset_3_NCM_NCA_battery/`; MIT - dated subfolders
`2017-05-12/`, `2017-06-30/`, `2018-04-12/`; HUST - flat CSVs `G-N.csv`.

The code for **reading and preprocessing** the dataset is publicly available at
https://github.com/wang-fujin/Battery-dataset-preprocessing-code-library

Please cite the corresponding dataset publications when you use this data.

## 4.2 Repository layout

```
Model/                     proposed model, trainer, invariant feature map,
                           original PINN (Model.py) and comparison nets (Compare_Models.py)
dataloader/                dataset reading, per-battery min-max, split logic
main_XJTU.py etc.          per-dataset entry points (1:1 with the PINN4SOH paper;
                           their defaults now match the final paper configuration)
main_adaptive_transfer.py  transfer entry point (pretrain + freeze operator + fine-tune)
main_adaptation_fine-tuning.py  fine-tune driver for the reference estimator
scripts/reproduce.py       Table 2/3 suite for both models (regular / small)
scripts/baselines.py       external baselines under the same protocol
scripts/evaluate.py        battery-level MAE/MAPE/RMSE aggregation (mean +- std)
scripts/transfer_eval.py   Table 4 transfer suite
scripts/transfer_eval_ablation.py  transfer-phase operator ablation
scripts/attribution/       attribution controls (target-only, source-only, battery re-aggregation)
results analysis/          print Table 2-8 from the shipped predictions
data analysis/             corpus size (cells, cycles)
count parameters.py        parameter budget of both models
results/                   seeded per-run predictions behind every table and figure
data/                      preprocessed datasets (387 cells, 310,705 cycles)
plotter/                   figure pipeline: data/ CSVs, matlab/ scripts, ppt/ Fig. 1 sources
tests/                     11 unit tests incl. anti-regression checks for the physics losses
```

## 4.3 Protocol notes

The released suite ran the TJU/HUST/MIT baselines and the proposed model at
batch size 256 (a GPU-memory cap on the reference card; the upstream PINN4SOH
entry points default to 512). `scripts/reproduce.py` sets the cap itself, and
the four `main_*.py` entry points default to 256 to match.

The five external regressors use a per-battery chronological 80/20 split, while
the two physics-informed estimators use a pooled random split. This asymmetry
is disclosed in the Limitations of the manuscript.

## 4.4 Differences from upstream PINN4SOH

The data loading, the reference PINN implementation and the evaluation protocol
are reused from [PINN4SOH](https://github.com/wang-fujin/PINN4SOH) for academic
reproducibility, with attribution. The contributions of this work are the
invariant feature map (row standardisation plus the rate-normalised change
block), the shared latent-increment operator and its loss, the transfer
protocol that freezes the operator, and the few-label evaluation.

# 5. Citation

If you find it useful, please cite our paper (and the PINN4SOH paper whose
protocol and datasets this work builds on):

```bibtex
@article{du2026latentincrement,
  title   = {Latent-increment physics-informed network for lithium-ion battery
             state of health estimation under label scarcity},
  author  = {Du, Changrong and Shi, Yanchao and Guo, Jun and Lai, Jing},
  journal = {Journal of Energy Storage},
  year    = {2026},
  note    = {to be updated upon acceptance}
}

@article{wang2024physics,
  title   = {Physics-informed neural network for lithium-ion battery degradation
             stable modeling and prognosis},
  author  = {Wang, Fujin and others},
  journal = {Nature Communications},
  volume  = {15},
  number  = {1},
  pages   = {4332},
  year    = {2024}
}
```

## License

MIT for all code written for this work (see `LICENSE`). The upstream PINN4SOH
repository ships no license file; please cite the original paper if you use
those components.

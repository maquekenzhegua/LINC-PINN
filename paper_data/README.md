# paper_data

Seeded per-run predictions behind every number, table and figure of the
manuscript. One directory, four trees, nothing else.

```
reproduce/
  regular/adaptive   proposed model, full labels, 11 conditions
  regular/orig       reference estimator, full labels, 11 conditions
  small/adaptive     proposed model, 1-4 labelled cells, 8 settings
  small/orig         reference estimator, 1-4 labelled cells, 8 settings
transfer/
  adaptive           proposed model, six directed transfers
  orig               reference estimator, six directed transfers
ablation/
  no_operator        operator consistency loss set to alpha = 0
  no_delta           change block deleted from the input row
  no_z               within-row standardization deleted
  latent3            latent state narrowed from eight to three dimensions
baselines/
  regular|small|transfer / MLP, CNN, GRU, LSTM, XGBOOST
```

Every leaf is `<model>/<case>/Experiment<k>/` holding `pred_label.npy`,
`true_label.npy` and, where applicable, the per-run diagnostics. `k` runs
from 1 to 10 and is seed-matched across models and against the proposed
model, so any paired comparison in the manuscript can be recomputed by
reading these files directly.

## Recomputing the reported numbers

`figures/data/make_figure_data.py` reads this tree and rebuilds every
figure-data CSV. From the repository root:

```bash
python figures/data/make_figure_data.py
```

Battery-wise MAPE is computed exactly as in `scripts/evaluate.py`, so the
macro averages it prints are the ones in Sections 3.2 to 3.6. The two
raw-signal panels (Fig. 2 and Fig. 3) additionally need the public datasets
under `data/`, which are not redistributed here; their exported CSVs are
shipped in `figures/data/` so the figures can still be redrawn.

## Not in this tree

- Trained checkpoints. The models are re-trained by `scripts/reproduce.py`;
  the archival record is the per-run predictions above.
- The raw datasets. Download links are given in Section 3.1 of the
  manuscript.
- Runs outside the reported set: the gamma sweep, the ANCHORED/scaler
  variants and the four-way combination study were exploratory and are not
  referenced by any claim in the paper.

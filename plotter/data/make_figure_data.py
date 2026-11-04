# -*- coding: utf-8 -*-
"""Export tidy CSVs feeding every MATLAB data figure (Fig 2-7).

Reads the authoritative result trees (reproduce, baselines,
ablation) plus the raw 4-dataset CSVs, recomputes battery-wise
MAE/MAPE/RMSE per experiment exactly like scripts/evaluate.py, and writes
one flat CSV per figure into this directory.

Run:
    python make_figure_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent          # plotter/data/
PAPER = HERE.parents[1]                          # release repo root
RESULTS = PAPER / "results"       # shipped per-run predictions
RAWDATA = PAPER / "data"                         # public datasets (user downloads; not shipped)

GAPS = {"XJTU": 0.05, "TJU": 0.07, "HUST": 0.10, "MIT": 0.07}

FEATURES = [
    "v.mean", "v.std", "v.kurt", "v.skew",
    "CC Q", "CC t", "v.slope", "v.ent",
    "i.mean", "i.std", "i.kurt", "i.skew",
    "CV Q", "CV t", "i.slope", "i.ent",
]
NOMINAL = {"XJTU": 2.0, "TJU-1": 3.5, "TJU-2": 3.5,
           "TJU-3": 2.5, "MIT": 1.1, "HUST": 1.1}


def eval_metrix(t, p):
    from sklearn import metrics as skm
    return (float(skm.mean_absolute_error(t, p)),
            float(skm.mean_absolute_percentage_error(t, p)),
            float(np.sqrt(skm.mean_squared_error(t, p))))


def battery_metrics(true_label, pred_label, gap):
    true_label = np.asarray(true_label).reshape(-1)
    pred_label = np.asarray(pred_label).reshape(-1)
    diff = np.diff(true_label)
    splits = np.where(diff > gap)[0]
    starts = [0] + (splits + 1).tolist()
    ends = splits.tolist() + [len(true_label) - 1]
    maes, mapes, rmses = [], [], []
    for s, e in zip(starts, ends):
        t = true_label[s:e + 1]
        p = pred_label[s:e + 1]
        if len(t) < 2:
            continue
        mae, mape, rmse = eval_metrix(t, p)
        maes.append(mae); mapes.append(mape); rmses.append(rmse)
    if not mapes:
        return eval_metrix(true_label, pred_label)
    return float(np.mean(maes)), float(np.mean(mapes)), float(np.mean(rmses))


def gap_for_case(case: str) -> float:
    """Gap threshold from a case/pair label; transfer uses the target set."""
    target = case.split("-to-")[-1] if "-to-" in case else case
    for key, gap in GAPS.items():
        if key in target:
            return gap
    return 0.05


def walk_model_case(base: Path):
    """Yield (model, case, exp_dir) for trees shaped {model}/{case}/Experiment*."""
    if not base.exists():
        return
    for model_dir in sorted(base.iterdir()):
        if not model_dir.is_dir():
            continue
        for case_dir in sorted(model_dir.iterdir()):
            if not case_dir.is_dir():
                continue
            for exp in sorted(case_dir.glob("Experiment*")):
                yield model_dir.name, case_dir.name, exp


def metrics_frame(base: Path) -> pd.DataFrame:
    rows = []
    for model, case, exp in walk_model_case(base):
        t, p = exp / "true_label.npy", exp / "pred_label.npy"
        if not (t.exists() and p.exists()):
            continue
        mae, mape, rmse = battery_metrics(np.load(t), np.load(p), gap_for_case(case))
        rows.append({"model": model, "case": case, "exp": int(exp.name.replace("Experiment", "")),
                     "MAE": mae, "MAPE": mape, "RMSE": rmse})
    return pd.DataFrame(rows)


def export_metrics(base: Path, name: str, **extra):
    df = metrics_frame(base)
    for k, v in extra.items():
        df[k] = v
    df.to_csv(HERE / name, index=False)
    print(f"{name:24s} {len(df):5d} rows  ({base})")
    return df


def small_case_parts(case: str):
    # small-XJTU-2C-n3 -> ("XJTU-2C", 3); small-HUST-n1 -> ("HUST", 1)
    body, n = case.rsplit("-n", 1)
    return body.replace("small-", ""), int(n)


def main() -> None:
    # ---- Fig 4: regular violins + scatter -------------------------------
    export_metrics(RESULTS / "reproduce" / "regular", "fig4_violin.csv", phase="regular")

    scat = []
    base = RESULTS / "reproduce" / "regular" / "adaptive"
    for cond in ["XJTU-2C", "XJTU-3C", "XJTU-R2.5", "XJTU-R3", "XJTU-RW",
                 "XJTU-satellite", "TJU-0", "TJU-1", "TJU-2", "MIT", "HUST"]:
        t, p = base / cond / "Experiment1" / "true_label.npy", base / cond / "Experiment1" / "pred_label.npy"
        if not (t.exists() and p.exists()):
            print(f"!! missing scatter source {cond}")
            continue
        tv, pv = np.load(t).reshape(-1), np.load(p).reshape(-1)
        scat.append(pd.DataFrame({"condition": cond, "true": tv, "pred": pv}))
    pd.concat(scat).to_csv(HERE / "fig4_scatter.csv", index=False)
    print(f"fig4_scatter.csv         {sum(len(s) for s in scat):5d} rows")

    # ---- Fig 5: small sample (adaptive/orig + GPR/GRU/LSTM/CNN/MLP) -----
    frames = [export_metrics(RESULTS / "reproduce" / "small", "fig5_small.csv")]
    frames.append(metrics_frame(RESULTS / "baselines" / "small"))
    df = pd.concat(frames)
    df[["dataset", "n"]] = df["case"].map(small_case_parts).apply(pd.Series)
    df.to_csv(HERE / "fig5_small.csv", index=False)
    print(f"fig5_small.csv updated   {len(df):5d} rows")

    # ---- Fig 6: transfer (adaptive/orig + 5 baselines) ------------------
    frames = [export_metrics(RESULTS / "transfer", "fig6_transfer.csv")]
    frames.append(metrics_frame(RESULTS / "baselines" / "transfer"))
    df = pd.concat(frames)
    df.to_csv(HERE / "fig6_transfer.csv", index=False)
    print(f"fig6_transfer.csv updated {len(df):5d} rows")

    # ---- Fig 7a: ablation variants --------------------------------------
    frames = []
    for variant in ["no_operator", "latent3", "no_rowmean", "no_anchored", "src_scaler"]:
        root = RESULTS / "ablation" / variant
        for phase in ["regular", "small"]:
            frames.append(metrics_frame(root / phase).assign(variant=variant, phase=phase))
    full_reg = metrics_frame(RESULTS / "reproduce" / "regular")
    full_reg = full_reg[full_reg["model"].eq("adaptive") & full_reg["case"].isin(["XJTU-2C"])].assign(variant="full", phase="regular")
    full_sml = metrics_frame(RESULTS / "reproduce" / "small")
    full_sml = full_sml[full_sml["model"].eq("adaptive") & full_sml["case"].isin(["small-HUST-n1", "small-XJTU-2C-n1"])].assign(variant="full", phase="small")
    orig_reg = metrics_frame(RESULTS / "reproduce" / "regular")
    orig_reg = orig_reg[orig_reg["model"].eq("orig") & orig_reg["case"].isin(["XJTU-2C"])]
    orig_reg = orig_reg.assign(variant="orig", phase="regular")
    orig_sml = metrics_frame(RESULTS / "reproduce" / "small")
    orig_sml = orig_sml[orig_sml["model"].eq("orig") & orig_sml["case"].isin(["small-HUST-n1", "small-XJTU-2C-n1"])]
    orig_sml = orig_sml.assign(variant="orig", phase="small")
    df = pd.concat(frames + [full_reg, full_sml, orig_reg, orig_sml])
    df.to_csv(HERE / "fig7_ablation.csv", index=False)
    print(f"fig7_ablation.csv        {len(df):5d} rows")

    # ---- Fig 7b: gamma sweep on XJTU-2C small n=1..4 --------------------
    rows = []
    gamma_paths = {0.2: RESULTS / "reproduce" / "small" / "adaptive"}
    for g in (1, 2, 5, 10):
        gamma_paths[g] = RESULTS / "ablation" / f"gamma{g}" / "small" / "adaptive"
    for gamma, root in gamma_paths.items():
        for n in (1, 2, 3, 4):
            cell = root / f"small-XJTU-2C-n{n}"
            for exp in sorted(cell.glob("Experiment*")):
                t, p = exp / "true_label.npy", exp / "pred_label.npy"
                if not (t.exists() and p.exists()):
                    continue
                tv, pv = np.load(t).reshape(-1), np.load(p).reshape(-1)
                _mae, mape, _rmse = battery_metrics(tv, pv, GAPS["XJTU"])
                rows.append({"gamma": gamma, "n": n, "exp": int(exp.name.replace("Experiment", "")),
                             "MAPE": mape, "pred_range": float(pv.max() - pv.min()),
                             "true_range": float(tv.max() - tv.min())})
    pd.DataFrame(rows).to_csv(HERE / "fig7_gamma.csv", index=False)
    print(f"fig7_gamma.csv           {len(rows):5d} rows")

    # Fig 2/3 read the raw public datasets; without data/ on disk the
    # shipped CSVs stay untouched and only the npy-based figures regenerate.
    if RAWDATA.exists():
        # ---- Fig 2: degradation trajectories of all 387 batteries -----------
        rows = []

        def add_battery(df: pd.DataFrame, dataset: str, batch: str, battery: str, nominal: float, stride: int = 1):
            cap = df["capacity"].to_numpy(dtype=float)[::stride]
            rows.append(pd.DataFrame({
                "dataset": dataset, "batch": batch, "battery": battery,
                "cycle": np.arange(1, len(cap) + 1), "soh": cap / nominal,
            }))

        xjtu = RAWDATA / "XJTU data"
        for f in sorted(xjtu.glob("*.csv")):
            batch = f.stem.split("_")[0]
            add_battery(pd.read_csv(f), "XJTU", batch, f.stem, NOMINAL["XJTU"])
        tjname = {"Dataset_1_NCA_battery": ("TJU-1", "NCA"),
                  "Dataset_2_NCM_battery": ("TJU-2", "NCM"),
                  "Dataset_3_NCM_NCA_battery": ("TJU-3", "NCM+NCA")}
        for sub, (ds, batch) in tjname.items():
            for f in sorted((RAWDATA / "TJU data" / sub).glob("*.csv")):
                add_battery(pd.read_csv(f), "TJU", batch, f.stem, NOMINAL[ds])
        for date in sorted((RAWDATA / "MIT data").iterdir()):
            if not date.is_dir():
                continue
            for f in sorted(date.glob("*.csv")):
                add_battery(pd.read_csv(f), "MIT", date.name, f.stem, NOMINAL["MIT"])
        for f in sorted((RAWDATA / "HUST data").glob("*.csv")):
            add_battery(pd.read_csv(f), "HUST", "HUST", f.stem, NOMINAL["HUST"])
        pd.concat(rows).to_csv(HERE / "fig2_trajectories.csv", index=False)
        print(f"fig2_trajectories.csv    {len(rows):4d} batteries, {sum(len(r) for r in rows)} cycles")

        # ---- Fig 3a: feature-SOH Pearson correlation per dataset -------------
        def corr_rows(paths, dataset):
            feats, sohs = [], []
            for f in paths:
                df = pd.read_csv(f)
                feats.append(df.iloc[:, :16].to_numpy(dtype=float))
                nominal = (2.5 if "Dataset_3" in str(f) else 3.5) if dataset == "TJU" else NOMINAL[dataset]
                # $y_{b,i}=Q_{b,i}/Q_b^{\mathrm{nom}}$; TJU nominal capacities differ.
                sohs.append(df["capacity"].to_numpy(dtype=float) / nominal)
            x = np.concatenate(feats); y = np.concatenate(sohs)
            with np.errstate(invalid="ignore"):
                x = (x - x.mean(0)) / x.std(0)
                x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                y = (y - y.mean()) / y.std()
                r = np.nan_to_num((x * y[:, None]).mean(0), nan=0.0)
            return pd.DataFrame({"dataset": dataset,
                                 "feature": range(1, 17),
                                 "r": r})

        frames = [corr_rows(sorted(xjtu.glob("*.csv")), "XJTU"),
                  corr_rows([f for sub, _ in tjname.items()
                             for f in sorted((RAWDATA / "TJU data" / sub).glob("*.csv"))], "TJU"),
                  corr_rows([f for d in sorted((RAWDATA / "MIT data").iterdir()) if d.is_dir()
                             for f in sorted(d.glob("*.csv"))], "MIT"),
                  corr_rows(sorted((RAWDATA / "HUST data").glob("*.csv")), "HUST")]
        pd.concat(frames).to_csv(HERE / "fig3_corr.csv", index=False)
        print(f"fig3_corr.csv            {len(frames)} datasets x 16 features")

        # ---- Fig 3b: raw vs row-standardized feature distributions ----------
        inv_feat = [1, 5, 13]  # voltage mean, CC Q, CV Q (1-based, matches paper numbering)
        rows = []
        for dataset, paths in [("XJTU", sorted(xjtu.glob("*.csv"))),
                               ("TJU", [f for sub, _ in tjname.items()
                                        for f in sorted((RAWDATA / "TJU data" / sub).glob("*.csv"))]),
                               ("MIT", [f for d in sorted((RAWDATA / "MIT data").iterdir()) if d.is_dir()
                                        for f in sorted(d.glob("*.csv"))]),
                               ("HUST", sorted((RAWDATA / "HUST data").glob("*.csv")))]:
            raw_blocks, normalized_blocks = [], []
            for f in paths:
                df = pd.read_csv(f)
                df.insert(df.shape[1] - 1, "cycle index", np.arange(len(df)))
                df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
                outside = ((df < df.mean() - 3 * df.std()) | (df > df.mean() + 3 * df.std())).any(axis=1)
                df = df.loc[~outside]
                raw_features = df.iloc[:, :16].to_numpy(dtype=float)
                if raw_features.ndim != 2 or raw_features.shape[1] != 16:
                    raise ValueError(f"Expected 16 features in {f}")
                feature_range = np.ptp(raw_features, axis=0)
                if np.any(feature_range <= 0):
                    raise ValueError(f"Constant feature in {f}; cannot mirror loader scaling")
                # $x_{b,i,j}=2(r_{b,i,j}-m_{b,j})/(M_{b,j}-m_{b,j})-1$.
                scaled_features = 2 * (raw_features - raw_features.min(axis=0)) / feature_range - 1
                # $v_i=(x_i-\bar{x}_i)/\sqrt{s_i^2+\epsilon^2}$, after per-battery scaling.
                row_mean = scaled_features.mean(axis=1, keepdims=True)
                row_scale = np.sqrt(scaled_features.var(axis=1, keepdims=True) + 1e-6 ** 2)
                raw_blocks.append(raw_features[::7])
                normalized_blocks.append(((scaled_features - row_mean) / row_scale)[::7])
            x = np.concatenate(raw_blocks)
            z = np.concatenate(normalized_blocks)
            for j in inv_feat:
                rows.append(pd.DataFrame({"dataset": dataset, "feature": FEATURES[j - 1],
                                          "transform": "raw", "value": x[:, j - 1]}))
                rows.append(pd.DataFrame({"dataset": dataset, "feature": FEATURES[j - 1],
                                          "transform": "rowstd", "value": z[:, j - 1]}))
        pd.concat(rows).to_csv(HERE / "fig3_invariance.csv", index=False)
        print(f"fig3_invariance.csv      {sum(len(r) for r in rows)} rows")
    else:
        print("fig2/fig3 need the datasets under data/ - shipped CSVs kept")

    print("done.")


if __name__ == "__main__":
    main()

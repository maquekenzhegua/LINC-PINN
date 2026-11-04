# -*- coding: utf-8 -*-
'''
复现正文表 3（少标签误差）、表 6（少标签七方法对比）与表 8（消融）。

English:
Reproduce Table 3 (few-label errors of the two physics-informed estimators),
Table 6 (few-label seven-method comparison) and Table 8 (component ablations)
from ``results/reproduce/small``, ``results/baselines/small`` and
``results/ablation``.

Run from the repository root::

    python "results analysis/Small results.py"
'''

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common

ROOT = os.path.join("results", "reproduce", "small")
BASE = os.path.join("results", "baselines", "small")
ABLATION = os.path.join("results", "ablation")

DATASETS = ["HUST", "XJTU-2C"]
BUDGETS = [1, 2, 3, 4]
EXTERNALS = ["MLP", "CNN", "XGBOOST", "GRU", "LSTM"]

#: Ablation arms in the manuscript row order.
ARMS = [
    ("Full", None),
    ("w/o operator loss", "no_operator"),
    ("w/o change block", "no_delta"),
    ("w/o within-row standardization", "no_z"),
    ("latent 8 to 3", "latent3"),
]

#: The two datasets the ablations cover, in file order.
ABLATION_DATASETS = ["HUST", "XJTU-2C"]


def setting(dataset, budget):
    '''返回一个少标签设置的目录名。'''
    return "small-%s-n%d" % (dataset, budget)


def table3():
    '''打印表 3。'''
    print("Table 3  few-label errors, battery-level MAPE (%) and RMSE, mean +- std over 10 runs")
    print("%-10s %3s %18s %18s %9s %18s %18s" % (
        "Dataset", "n", "Reference MAPE", "Ours MAPE", "dMAPE", "Reference RMSE", "Ours RMSE"))
    for dataset in DATASETS:
        for budget in BUDGETS:
            name = setting(dataset, budget)
            ref = common.score_condition(os.path.join(ROOT, "orig"), name)
            ours = common.score_condition(os.path.join(ROOT, "adaptive"), name)
            print("%-10s %3d %8.2f +- %-6.2f %8.2f +- %-6.2f %9.2f %9.4f +- %-6.4f %9.4f +- %-6.4f" % (
                dataset if budget == 1 else "", budget,
                ref["mape"][0], ref["mape"][1],
                ours["mape"][0], ours["mape"][1],
                ours["mape"][0] - ref["mape"][0],
                ref["rmse"][0], ref["rmse"][1],
                ours["rmse"][0], ours["rmse"][1]))


def table6():
    '''打印表 6（少标签七方法对比）与 macro 平均值。'''
    print()
    print("Table 6  few-label seven-method comparison, battery-level MAPE (%), mean +- std over 10 runs")
    header = "%-16s %16s %16s" % ("Setting", "Proposed", "Reference")
    for name in EXTERNALS:
        header += " %16s" % name
    print(header)

    collected = {"Proposed": [], "Reference": []}
    for name in EXTERNALS:
        collected[name] = []

    for dataset in DATASETS:
        for budget in BUDGETS:
            name = setting(dataset, budget)
            ours = common.score_condition(os.path.join(ROOT, "adaptive"), name)
            ref = common.score_condition(os.path.join(ROOT, "orig"), name)
            collected["Proposed"].append(ours)
            collected["Reference"].append(ref)
            row = "%-16s %8.3f +- %-5.3f %8.3f +- %-5.3f" % (
                "%s n=%d" % (dataset, budget),
                ours["mape"][0], ours["mape"][1], ref["mape"][0], ref["mape"][1])
            for name_ext in EXTERNALS:
                folder = os.path.join(BASE, name_ext)
                if name not in common.conditions(folder):
                    row += " %16s" % "--"
                    continue
                score = common.score_condition(folder, name)
                collected[name_ext].append(score)
                row += " %8.3f +- %-5.3f" % (score["mape"][0], score["mape"][1])
            print(row)

    print("-" * len(header))
    row = "%-16s %8.3f %8s %8.3f %8s" % (
        "MACRO",
        sum(e["mape"][0] for e in collected["Proposed"]) / len(collected["Proposed"]), "",
        sum(e["mape"][0] for e in collected["Reference"]) / len(collected["Reference"]), "")
    for name_ext in EXTERNALS:
        values = collected[name_ext]
        row += " %8.3f %8s" % (sum(v["mape"][0] for v in values) / len(values), "")
    print(row)


def table8():
    '''打印表 8（四臂消融，两个数据集 × 四个预算）。'''
    print()
    print("Table 8  few-label component ablations, battery-level MAPE (%), mean +- std over 10 runs")
    for dataset in ABLATION_DATASETS:
        print("  %s" % dataset)
        print("  %-32s %16s %16s %16s %16s" % ("Configuration", "n=1", "n=2", "n=3", "n=4"))
        for label, arm in ARMS:
            cells = []
            for budget in BUDGETS:
                name = setting(dataset, budget)
                if arm is None:
                    folder = os.path.join(ROOT, "adaptive")
                else:
                    folder = os.path.join(ABLATION, arm, "small", "adaptive")
                score = common.score_condition(folder, name)
                cells.append("%7.2f +- %-5.2f" % (score["mape"][0], score["mape"][1]))
            print("  %-32s %s" % (label, " ".join(cells)))


if __name__ == "__main__":
    table3()
    table6()
    table8()

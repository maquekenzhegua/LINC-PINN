# -*- coding: utf-8 -*-
'''
复现正文表 4（有向迁移）、表 7（迁移七方法对比）、附录的 target-only 对照、
以及去掉算子损失的迁移消融。

English:
Reproduce Table 4 (directed transfer), Table 7 (transfer seven-method
comparison), the target-only control and the transfer-phase operator-loss
ablation from ``results/transfer``.

Run from the repository root::

    python "results analysis/Transfer results.py"
'''

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common

ROOT = os.path.join("results", "transfer")
BASE = os.path.join("results", "baselines", "transfer")

#: The six directed pairs, in the manuscript order.
DIRECTIONS = [
    "XJTU-2C-to-HUST",
    "XJTU-2C-to-TJU-0",
    "HUST-to-XJTU-2C",
    "TJU-0-to-XJTU-2C",
    "HUST-to-TJU-0",
    "TJU-0-to-HUST",
]

#: The four directions every method covers (the two HUST-destination pairs are
#: only available for the two physics-informed estimators).
SHARED = ["XJTU-2C-to-HUST", "XJTU-2C-to-TJU-0", "HUST-to-XJTU-2C", "TJU-0-to-XJTU-2C"]

EXTERNALS = ["MLP", "CNN", "XGBOOST", "GRU", "LSTM"]

#: The gap of the destination dataset decides how a run is split into batteries.
TARGET_GAP = {"HUST": 0.10, "TJU-0": 0.07, "XJTU-2C": 0.05}


def target_of(direction):
    '''从方向名取目标数据集。'''
    return direction.split("-to-")[-1]


def score_direction(root, direction, key="pred_label"):
    '''聚合一个迁移方向的 10 次重复。'''
    gap = TARGET_GAP[target_of(direction)]
    maes, mapes, rmses = [], [], []
    for experiment in range(1, common.REPEATS + 1):
        folder = os.path.join(root, direction, "Experiment%d" % experiment)
        true_label = np.load(os.path.join(folder, "true_label.npy")).reshape(-1)
        pred_label = np.load(os.path.join(folder, "%s.npy" % key)).reshape(-1)
        mae, mape, rmse = common.battery_metrics(true_label, pred_label, gap)
        maes.append(mae)
        mapes.append(mape)
        rmses.append(rmse)
    return float(np.mean(maes)), float(np.mean(mapes)) * 100.0, float(np.mean(rmses))


def table4():
    '''打印表 4。'''
    print("Table 4  directed-transfer MAPE (%), mean over 10 runs")
    print("%-22s %16s %16s %9s %9s" % ("Direction", "Reference", "Ours", "dMAPE", "rel."))
    for direction in DIRECTIONS:
        ours = score_direction(os.path.join(ROOT, "adaptive"), direction)
        if direction in common.conditions(os.path.join(ROOT, "orig")):
            ref = score_direction(os.path.join(ROOT, "orig"), direction)
            delta = ours[1] - ref[1]
            print("%-22s %8.2f %8s %8.2f %8s %9.2f %8.0f%%" % (
                direction.replace("-to-", " -> "), ref[1], "", ours[1], "", delta, 100.0 * delta / ref[1]))
        else:
            print("%-22s %8s %8s %8.2f %8s %9s %8s" % (
                direction.replace("-to-", " -> "), "---", "", ours[1], "", "---", "---"))


def table7():
    '''打印表 7 与四个共享方向的 macro。'''
    print()
    print("Table 7  transfer seven-method comparison, battery-level MAPE (%), mean over 10 runs")
    header = "%-22s %16s %16s" % ("Direction", "Proposed", "Reference")
    for name in EXTERNALS:
        header += " %16s" % name
    print(header)

    macro = {"Proposed": [], "Reference": []}
    for name in EXTERNALS:
        macro[name] = []

    for direction in DIRECTIONS:
        ours = score_direction(os.path.join(ROOT, "adaptive"), direction)
        row = "%-22s %16.3f" % (direction.replace("-to-", " -> "), ours[1])
        if direction in SHARED:
            macro["Proposed"].append(ours[1])
        if direction in common.conditions(os.path.join(ROOT, "orig")):
            ref = score_direction(os.path.join(ROOT, "orig"), direction)
            if direction in SHARED:
                macro["Reference"].append(ref[1])
            row += " %16.3f" % ref[1]
        else:
            row += " %16s" % "--"
        for name in EXTERNALS:
            folder = os.path.join(BASE, name)
            if direction not in common.conditions(folder):
                row += " %16s" % "--"
                continue
            score = score_direction(folder, direction)
            if direction in SHARED:
                macro[name].append(score[1])
            row += " %16.3f" % score[1]
        print(row)

    print("-" * len(header))
    row = "%-22s %16.3f %16.3f" % (
        "MACRO (4 shared)",
        sum(macro["Proposed"]) / len(macro["Proposed"]),
        sum(macro["Reference"]) / len(macro["Reference"]))
    for name in EXTERNALS:
        row += " %16.3f" % (sum(macro[name]) / len(macro[name]))
    print(row)


def target_only():
    '''打印 target-only 对照。

    训练集只有一颗目标电池、不带源域预训练，等价于该目标域上的少标签
    ``n = 1`` 设置，所以数值直接取自 ``results/reproduce/small/adaptive``；
    TJU-0 没有随仓的少标签臂，用 ``--`` 标注。
    '''
    print()
    print("Target-only control, battery-level MAPE (%), mean over 10 runs")
    print("%-22s %14s %14s %9s" % ("Direction", "Full transfer", "Target-only", "d"))
    small = os.path.join("results", "reproduce", "small", "adaptive")
    for direction in DIRECTIONS:
        target = target_of(direction)
        ours = score_direction(os.path.join(ROOT, "adaptive"), direction)
        name = "small-%s-n1" % target
        if name in common.conditions(small):
            score = common.score_condition(small, name)
            print("%-22s %14.2f %14.2f %+9.2f" % (
                direction.replace("-to-", " -> "), ours[1], score["mape"][0],
                score["mape"][0] - ours[1]))
        else:
            print("%-22s %14.2f %14s %9s" % (
                direction.replace("-to-", " -> "), ours[1], "--", "--"))


if __name__ == "__main__":
    table4()
    table7()
    target_only()

# -*- coding: utf-8 -*-
'''
共享的结果读取与聚合工具。

English:
Shared helpers that read the shipped per-run predictions and aggregate them the
way the manuscript reports them: one MAE/MAPE/RMSE per battery, then the mean
and the sample standard deviation over the ten seed-matched repeats.

Every number in the manuscript tables can be regenerated from
``results/`` with the scripts in this folder.
'''

import os
import numpy as np

from utils.util import battery_metrics

#: Label gap that separates two batteries inside one prediction array.
GAPS = {"XJTU": 0.05, "TJU": 0.07, "HUST": 0.10, "MIT": 0.07}

REPEATS = 10


def gap_of(condition):
    '''按条件名前缀取电池切分阈值。

    English:
    Pick the battery-splitting gap from the condition prefix.
    '''
    for key, value in GAPS.items():
        if condition.startswith(key):
            return value
    return 0.05


def run_folder(root, condition, experiment):
    '''拼一次实验的目录。

    English:
    Build the folder of one experiment. ``Experiment1`` to ``Experiment10``
    hold ``pred_label.npy`` and ``true_label.npy``.
    '''
    return os.path.join(root, condition, "Experiment%d" % experiment)


def score_condition(root, condition, repeats=REPEATS):
    '''把一个条件的 10 次重复聚合成 (mae, mape, rmse) 的均值与标准差。

    English:
    Aggregate the repeats of one condition into mean and sample standard
    deviation of the battery-averaged error.
    '''
    gap = gap_of(condition)
    maes, mapes, rmses = [], [], []
    for experiment in range(1, repeats + 1):
        folder = run_folder(root, condition, experiment)
        true_label = np.load(os.path.join(folder, "true_label.npy")).reshape(-1)
        pred_label = np.load(os.path.join(folder, "pred_label.npy")).reshape(-1)
        mae, mape, rmse = battery_metrics(true_label, pred_label, gap)
        maes.append(mae)
        mapes.append(mape)
        rmses.append(rmse)
    return {
        "mae": (float(np.mean(maes)), float(np.std(maes, ddof=1))),
        "mape": (float(np.mean(mapes)) * 100.0, float(np.std(mapes, ddof=1)) * 100.0),
        "rmse": (float(np.mean(rmses)), float(np.std(rmses, ddof=1))),
    }


def conditions(root):
    '''列出结果根下所有条件目录。

    English:
    List the condition folders under a results root, in the manuscript order.
    '''
    if not os.path.isdir(root):
        return []
    return sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))


def print_row(label, mean, std, width=15):
    '''按 "mean +- std" 打印一行。

    English:
    Print one "mean +- std" cell.
    '''
    print("%-*s %8.3f +- %-6.3f" % (width, label, mean, std))

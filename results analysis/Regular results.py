# -*- coding: utf-8 -*-
'''
复现正文表 2（两种模型的全标签误差）与表 5（七方法对比）。

English:
Reproduce Table 2 (full-label errors of the two physics-informed estimators)
and Table 5 (seven-method comparison) from ``results/reproduce/regular`` and
``results/baselines/regular``.

Run from the repository root::

    python "results analysis/Regular results.py"
'''

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common

ROOT = os.path.join("results", "reproduce", "regular")
BASE = os.path.join("results", "baselines", "regular")

#: Manuscript order of the eleven full-label conditions.
CONDITIONS = [
    "HUST", "MIT", "TJU-0", "TJU-1", "TJU-2",
    "XJTU-2C", "XJTU-3C", "XJTU-R2.5", "XJTU-R3", "XJTU-RW", "XJTU-satellite",
]

#: The five external regressors, in the manuscript column order.
EXTERNALS = ["MLP", "CNN", "XGBOOST", "GRU", "LSTM"]


def table2():
    '''打印表 2。'''
    print("Table 2  full-label errors, battery-level MAPE (%) and RMSE, mean +- std over 10 runs")
    print("%-15s %18s %18s %8s %18s %18s" % (
        "Condition", "Reference MAPE", "Ours MAPE", "dMAPE", "Reference RMSE", "Ours RMSE"))
    for condition in CONDITIONS:
        ref = common.score_condition(os.path.join(ROOT, "orig"), condition)
        ours = common.score_condition(os.path.join(ROOT, "adaptive"), condition)
        print("%-15s %8.2f +- %-6.2f %8.2f +- %-6.2f %8.2f %9.4f +- %-6.4f %9.4f +- %-6.4f" % (
            condition,
            ref["mape"][0], ref["mape"][1],
            ours["mape"][0], ours["mape"][1],
            ours["mape"][0] - ref["mape"][0],
            ref["rmse"][0], ref["rmse"][1],
            ours["rmse"][0], ours["rmse"][1]))


def macro(errors):
    '''对若干条件的 MAPE 取等权平均。'''
    values = [e["mape"][0] for e in errors]
    return sum(values) / len(values)


def table5():
    '''打印表 5（七方法对比）与 macro 平均值。'''
    print()
    print("Table 5  seven-method comparison, battery-level MAPE (%), mean +- std over 10 runs")
    header = "%-15s %16s %16s" % ("Setting", "Proposed", "Reference")
    for name in EXTERNALS:
        header += " %16s" % name
    print(header)

    columns = {"Proposed": [], "Reference": []}
    for name in EXTERNALS:
        columns[name] = []

    for condition in CONDITIONS:
        ours = common.score_condition(os.path.join(ROOT, "adaptive"), condition)
        ref = common.score_condition(os.path.join(ROOT, "orig"), condition)
        row = "%-15s %8.3f +- %-5.3f %8.3f +- %-5.3f" % (
            condition, ours["mape"][0], ours["mape"][1], ref["mape"][0], ref["mape"][1])
        columns["Proposed"].append(ours)
        columns["Reference"].append(ref)
        for name in EXTERNALS:
            folder = os.path.join(BASE, name)
            if condition not in common.conditions(folder):
                row += " %16s" % "--"
                continue
            score = common.score_condition(folder, condition)
            columns[name].append(score)
            row += " %8.3f +- %-5.3f" % (score["mape"][0], score["mape"][1])
        print(row)

    print("-" * len(header))
    row = "%-15s %8.3f %8s %8.3f %8s" % (
        "MACRO", macro(columns["Proposed"]), "", macro(columns["Reference"]), "")
    for name in EXTERNALS:
        row += " %8.3f %8s" % (macro(columns[name]), "")
    print(row)


if __name__ == "__main__":
    table2()
    table5()

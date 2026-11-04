# -*- coding: utf-8 -*-
'''
统计随仓发布的四个数据集：电池数与循环数。

English:
Count the cells and the cycles that the four shipped archives hold. The
manuscript quotes 387 cells and 310,705 cycles over 11 conditions before
filtering; the numbers printed here are the raw archive totals.

Run from the repository root::

    python "data analysis/sample count.py"
'''

import os

import pandas as pd

ROOTS = [
    os.path.join("data", "HUST data"),
    os.path.join("data", "MIT data", "2017-05-12"),
    os.path.join("data", "MIT data", "2017-06-30"),
    os.path.join("data", "MIT data", "2018-04-12"),
    os.path.join("data", "TJU data", "Dataset_1_NCA_battery"),
    os.path.join("data", "TJU data", "Dataset_2_NCM_battery"),
    os.path.join("data", "TJU data", "Dataset_3_NCM_NCA_battery"),
    os.path.join("data", "XJTU data"),
]

total_cycles = 0
total_cells = 0
for root in ROOTS:
    if not os.path.isdir(root):
        print("missing:", root)
        continue
    files = sorted(f for f in os.listdir(root) if f.endswith(".csv"))
    cycles = 0
    for name in files:
        frame = pd.read_csv(os.path.join(root, name))
        cycles += frame.shape[0]
    print("%-42s %4d cells %8d cycles" % (root, len(files), cycles))
    total_cycles += cycles
    total_cells += len(files)

print("-" * 62)
print("total cells  : %d" % total_cells)
print("total cycles : %d" % total_cycles)

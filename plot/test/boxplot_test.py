# -*- coding: utf-8 -*-
"""
    @author: 数模加油站
    @time  : 2025/8/23 18:10
    @file  : boxplot_test.py
"""


import pandas as pd, numpy as np
from ..distribution import *

rng = np.random.default_rng(0)
df = pd.DataFrame({
    "组别": np.repeat(list("ABCD"), 60),
    "类型": np.tile(np.repeat(["X","Y"], 30), 4),
    "数值": np.concatenate([
        rng.normal(10,2,60), rng.normal(12,2.5,60),
        rng.normal(9,1.8,60), rng.normal(14,3,60)
    ])
})

# 1) 与 seaborn.boxplot 完全一致的调用（默认 seaborn 引擎）
ax = boxplot(data=df, x="组别", y="数值", hue="类型", whis=(5,95), width=0.6, dodge=True)

# 2) 在不破坏 API 的前提下，开启增强功能（可选）
ax = boxplot(
    data=df, x="组别", y="数值", hue="类型",
    whis=(5,95), width=0.6, dodge=True,
    pro_points="strip", pro_mean=True,
    pro_sig_pairs=[(("A","X"),("A","Y")), (("C","X"),("D","Y"))],
    pro_sig_mode="p"
)

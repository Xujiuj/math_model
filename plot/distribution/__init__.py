# -*- coding: utf-8 -*-
"""
    @author: 数模加油站
    @time  : 2025/8/23 18:04
    @file  : __init__.py.py
"""

from .boxplot import *
from .lineplot import lineplot
from .kdeplot import kdeplot
from .corr import heatmap
from .settings import activate_cn_global

activate_cn_global()
__all__ = [
    'boxplot',
    'lineplot',
    'kdeplot',
    'heatmap'
]

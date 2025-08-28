# -*- coding: utf-8 -*-
"""
    @author: 数模加油站
    @time  : 2025/8/25 14:44
    @file  : settings.py
"""

from __future__ import annotations
import os
import sys
import glob
import warnings
import matplotlib as mpl
from matplotlib import font_manager
from typing import Iterable, List, Tuple, Optional

CJK_CANDIDATES: Tuple[str, ...] = (
    # Windows:
    "KaiTi", "SimHei", "KaiTi_GB2312", "FangSong",
    # macOS:
    "PingFang SC", "Hiragino Sans GB", "Heiti SC", "STHeiti",
    # Linux / 通用（建议安装）:
    "Noto Sans CJK SC", "Source Han Sans CN", "WenQuanYi Zen Hei"
)

def _registered_font_names() -> List[str]:
    return [f.name for f in font_manager.fontManager.ttflist]

def _choose_font(preferred: Iterable[str]) -> Optional[str]:
    avail = set(_registered_font_names())
    for name in preferred:
        if name in avail:
            return name
    return None

def _register_extra_fonts(extra_fonts: Optional[Iterable[str]]):
    """支持传入文件或目录；自动发现 .ttf/.otf 并注册到 Matplotlib。"""
    if not extra_fonts:
        return []
    added = []
    paths: List[str] = []
    for p in extra_fonts:
        if not p:
            continue
        if os.path.isdir(p):
            paths.extend(glob.glob(os.path.join(p, "**", "*.ttf"), recursive=True))
            paths.extend(glob.glob(os.path.join(p, "**", "*.otf"), recursive=True))
        elif os.path.isfile(p) and (p.lower().endswith(".ttf") or p.lower().endswith(".otf")):
            paths.append(p)
    for fp in paths:
        try:
            font_manager.fontManager.addfont(fp)
            added.append(fp)
        except Exception as e:
            warnings.warn(f"注册字体失败：{fp} ({e})")
    if added:
        # 重建字体缓存
        try:
            font_manager._rebuild()  # 私有接口；若未来失效，可提示用户删除缓存文件
        except Exception:
            pass
    return added

def activate_cn_global(
    preferred: Iterable[str] = CJK_CANDIDATES,
    *,
    extra_fonts: Optional[Iterable[str]] = None,  # 传入你随项目带的字体文件或目录
    use_seaborn: bool = True,                     # 若用 seaborn，建议保持 True
    seaborn_style: str = "whitegrid",
    seaborn_context: str = "notebook",
    embed_vector_text: bool = True,               # PDF/SVG 嵌入文字，防止方框
    verbose: bool = True
) -> Optional[str]:
    """
    全局启用中文字体与嵌字设置。
    返回：最终选中的字体名（若未找到可用字体，返回 None）
    """
    # 1) 可选注册项目自带字体
    added = _register_extra_fonts(extra_fonts)
    if verbose and added:
        print(f"[cn-fonts] 已注册 {len(added)} 个字体文件。")

    # 2) 选择可用字体
    chosen = _choose_font(preferred)
    if verbose:
        if chosen:
            print(f"[cn-fonts] 使用中文字体：{chosen}")
        else:
            print("[cn-fonts] 警告：未在系统中找到候选中文字体，可能仍会出现方框。"
                  "可通过 extra_fonts 指定 .ttf/.otf 或安装 Noto Sans CJK。")

    # 3) 统一 rc（全局）
    if embed_vector_text:
        mpl.rcParams.update({
            "pdf.fonttype": 42,  # 嵌入 TrueType
            "ps.fonttype": 42,
            "svg.fonttype": "none"  # SVG 保留文字（便于后期编辑）
        })
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["font.family"] = ["sans-serif"]
    # 将 chosen 放在首位，然后附上候选列表作为后备
    if chosen:
        mpl.rcParams["font.sans-serif"] = [chosen] + [f for f in preferred if f != chosen]
    else:
        mpl.rcParams["font.sans-serif"] = list(preferred)

    # 4) 处理 seaborn 主题（防止覆盖字体）
    if use_seaborn:
        try:
            import seaborn as sns
            # 直接把字体 rc 注入 set_theme，这样即使 set_theme 覆盖，也包含中文设置
            sns.set_theme(style=seaborn_style, context=seaborn_context, rc={
                "font.family": mpl.rcParams["font.family"],
                "font.sans-serif": mpl.rcParams["font.sans-serif"],
                "axes.unicode_minus": mpl.rcParams["axes.unicode_minus"],
            })
        except Exception as e:
            if verbose:
                print(f"[cn-fonts] seaborn 未安装或 set_theme 失败（{e}），已仅设置 Matplotlib 全局。")
    return chosen

# -*- coding: utf-8 -*-
"""
    @author: 数模加油站
    @time  : 2025/8/26 11:16
    @file  : corr.py
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Dict, Any, Union
from matplotlib.colors import Normalize, TwoSlopeNorm

try:
    import seaborn as sns
    _HAS_SNS = True
except Exception:
    _HAS_SNS = False

# Okabe–Ito（色盲友好，备用）
_OI = ["#0072B2","#E69F00","#009E73","#CC79A7",
       "#D55E00","#56B4E9","#F0E442","#000000"]

# ---------------- utils ----------------
def _ensure_2d_frame(data) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data
    arr = np.asarray(data)
    if arr.ndim == 1:
        return pd.DataFrame(arr.reshape(-1, 1))
    elif arr.ndim == 2:
        return pd.DataFrame(arr)
    else:
        raise ValueError("heatmap 的 data 需为二维（或一维将自动转为单列）。")

def _zscore(a, axis):
    a = np.asarray(a, float)
    mu = np.nanmean(a, axis=axis, keepdims=True)
    sd = np.nanstd(a, axis=axis, ddof=1, keepdims=True)
    sd = np.where(sd==0, 1.0, sd)
    return (a - mu) / sd

def _minmax(a, axis):
    a = np.asarray(a, float)
    mn = np.nanmin(a, axis=axis, keepdims=True)
    mx = np.nanmax(a, axis=axis, keepdims=True)
    denom = np.where((mx-mn)==0, 1.0, (mx-mn))
    return (a - mn) / denom

def _fmt_value(x, fmt: str):
    try:
        return format(x, fmt)
    except Exception:
        return f"{x}"

# ---------------- main ----------------
def heatmap(
    data=None, *,
    vmin=None, vmax=None, cmap=None, center=0, robust=False,
    annot: bool = False, fmt: str = ".2g", annot_kws: Optional[dict] = None,
    linewidths: float = 0, linecolor: str = "white",
    cbar: bool = True, cbar_kws: Optional[dict] = None, cbar_ax: Optional[plt.Axes] = None,
    square: bool = False, xticklabels="auto", yticklabels="auto",
    mask=None, ax: Optional[plt.Axes] = None,
    **kwargs: Any
) -> plt.Axes:
    """
    学术增强（均通过 **kwargs 的 pro_* 开关，不破坏 seaborn API）：
      - pro_corr: bool=False                # True 时对 DataFrame 数值列画相关矩阵
      - pro_corr_method: 'pearson'|'spearman'
      - pro_sigstars: bool=True             # 相关矩阵在标注中加显著性星号
      - pro_sig_levels: (1e-4,1e-3,1e-2,5e-2)
      - pro_norm: 'none'|'zscore_row'|'zscore_col'|'minmax_row'|'minmax_col'
      - pro_mask_tri: None|'upper'|'lower'
      - pro_annot: bool=True                # 用户未显式开 annot 时，默认开启标注
      - pro_saturation: float=0.9           # 降饱和
      - pro_no_grid: bool=True              # 去掉单元格分隔线
      - pro_kill_axes_grid: bool=True       # 关闭轴网格线（全局主题若开网格，这里强关）
      - pro_mask_diag: bool=True            # 遮罩对角线
      - pro_diag: 'none'|'mask'|'kde'       # 对角线显示方式（默认 kde）
      - pro_diag_pad: float=0.08            # 对角线 KDE 内边距（单元格比例）
      - pro_diag_color: str|dict|None       # 对角 KDE 颜色；None 时与热图 colormap 统一
      - pro_diag_cmap_from: 'diag'|'rowmean'|'colmean'|'percentile'|'fixed'
      - pro_diag_percentile: float=75.0
      - pro_diag_saturation: float=sat
      - pro_dtick: int|None
      - pro_xtick_rotation: float=0.0
      - pro_ytick_rotation: float=0.0
    """
    # ---- 隐藏参数（含若干别名）----
    is_corr: bool = bool(kwargs.pop("pro_corr", kwargs.pop("corr", False)))
    pro_corr_method: str = kwargs.pop("pro_corr_method", kwargs.pop("corr_method", "pearson"))
    pro_sigstars: bool = bool(kwargs.pop("pro_sigstars", kwargs.pop("sigstars", True)))
    pro_sig_levels = tuple(kwargs.pop("pro_sig_levels", (1e-4, 1e-3, 1e-2, 5e-2)))
    pro_norm: str = kwargs.pop("pro_norm", "none")
    pro_mask_tri: Optional[str] = kwargs.pop("pro_mask_tri", kwargs.pop("mask_tri", "lower"))
    pro_annot_default: bool = bool(kwargs.pop("pro_annot", kwargs.pop("annot_default", True)))
    pro_no_grid: bool = bool(kwargs.pop("pro_no_grid", True))
    pro_kill_axes_grid: bool = bool(kwargs.pop("pro_kill_axes_grid", True))
    pro_mask_diag: bool = bool(kwargs.pop("pro_mask_diag", True))
    pro_diag: str = kwargs.pop("pro_diag", "kde")
    pro_diag_pad: float = float(kwargs.pop("pro_diag_pad", 0.08))
    pro_diag_color = kwargs.pop("pro_diag_color", None)
    pro_diag_cmap_from: str = kwargs.pop("pro_diag_cmap_from", "rowmean")
    pro_diag_percentile: float = float(kwargs.pop("pro_diag_percentile", 75.0))
    sat: float = float(kwargs.pop("pro_saturation", kwargs.pop("saturation", 0.9)))
    pro_diag_saturation: float = float(kwargs.pop("pro_diag_saturation", sat))
    pro_dtick = kwargs.pop("pro_dtick", kwargs.pop("dtick", None))
    rot_x = float(kwargs.pop("pro_xtick_rotation", kwargs.pop("xtick_rotation", 0.0)))
    rot_y = float(kwargs.pop("pro_ytick_rotation", kwargs.pop("ytick_rotation", 0.0)))

    # 清理可能误透传给 seaborn 的键
    for k in list(kwargs.keys()):
        if k.startswith("pro_") or k in {
            "corr","corr_method","sigstars","mask_tri","annot_default",
            "saturation","dtick","xtick_rotation","ytick_rotation"
        }:
            kwargs.pop(k, None)

    # ---- 当前子图 ----
    if ax is None:
        ax = plt.gca()
    fig = ax.figure
    if pro_kill_axes_grid:
        ax.grid(False); ax.grid(False, which="minor")
        for gl in ax.get_xgridlines() + ax.get_ygridlines():
            gl.set_visible(False)

    # ---- 局部 rc（不污染全局）----
    rc_local = {"axes.spines.top": False, "axes.spines.right": False,
                "axes.grid": False, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"}

    # ---- 数据准备 ----
    df = _ensure_2d_frame(data)
    df_raw = df.copy()  # 对角线 KDE 用原始数据
    row_labels = list(df.index)
    col_labels = list(df.columns)

    # 相关矩阵与 p 值
    pvals = None
    source_df_for_diag = None

    if is_corr:
        num_df = df_raw.select_dtypes(include=[np.number])
        if num_df.shape[1] < 2:
            raise ValueError("pro_corr=True 需要至少两个数值列。")
        if pro_corr_method == "pearson":
            corr_mat = num_df.corr(method="pearson")
            # 计算 p 值
            try:
                from scipy import stats
                cols = list(num_df.columns)
                p = pd.DataFrame(np.ones((len(cols), len(cols))), index=cols, columns=cols, dtype=float)
                for i in range(len(cols)):
                    for j in range(i+1, len(cols)):
                        r, pp = stats.pearsonr(num_df[cols[i]].values, num_df[cols[j]].values)
                        p.iloc[i, j] = pp; p.iloc[j, i] = pp
                np.fill_diagonal(p.values, 0.0)
                pvals = p
            except Exception:
                pvals = None
            df = corr_mat
        else:  # spearman
            df = num_df.corr(method="spearman")
            pvals = None
        source_df_for_diag = num_df.copy()
        row_labels = list(df.index); col_labels = list(df.columns)
    else:
        # 非相关矩阵：对角线 KDE 的来源 = 原始数值列，按列顺序对齐
        source_df_for_diag = df_raw.select_dtypes(include=[np.number]).copy()
        if list(source_df_for_diag.columns) != list(df.columns):
            source_df_for_diag = source_df_for_diag.reindex(columns=df.columns, fill_value=np.nan)

    # 归一化到 A（绘制用矩阵）
    A = df.values.astype(float)
    if pro_norm == "zscore_row":
        A = _zscore(A, axis=1)
    elif pro_norm == "zscore_col":
        A = _zscore(A, axis=0)
    elif pro_norm == "minmax_row":
        A = _minmax(A, axis=1)
    elif pro_norm == "minmax_col":
        A = _minmax(A, axis=0)

    # 遮罩
    if mask is not None:
        M = np.array(mask, dtype=bool)
        if M.shape != A.shape:
            raise ValueError("mask 形状必须与 data 一致。")
    else:
        M = np.zeros_like(A, dtype=bool)
    if pro_mask_tri in ("upper", "lower"):
        tri = np.triu(np.ones_like(A, dtype=bool), 1) if pro_mask_tri == "upper" else np.tril(np.ones_like(A, dtype=bool), -1)
        M = np.logical_or(M, tri)
    if pro_mask_diag and A.shape[0] == A.shape[1]:
        M = np.logical_or(M, np.eye(A.shape[0], dtype=bool))

    # 默认 colormap
    if cmap is None:
        center_zero = (center is not None) and (float(center) == 0.0)
        cmap = "coolwarm" if (is_corr or center_zero) else "viridis"

    # 降饱和
    try:
        import matplotlib.colors as mcolors
        base = mpl.colormaps[cmap] if isinstance(cmap, str) else cmap
        lut = base(np.linspace(0, 1, 256))
        white = np.ones_like(lut[:, :3])
        lut[:, :3] = lut[:, :3] * sat + white * (1 - sat)
        cmap = mcolors.ListedColormap(lut)
    except Exception:
        pass

    # 标注
    if not annot and pro_annot_default:
        annot = True
    if annot_kws is None:
        annot_kws = dict(color="black", fontsize=9)

    star_levels = sorted(pro_sig_levels)
    def _stars(p):
        if p is None or not np.isfinite(p): return ""
        return "****" if p < star_levels[0] else ("***" if p < star_levels[1] else
               ("**" if p < star_levels[2] else ("*" if p < star_levels[3] else "")))

    annot_data = None
    if annot:
        val = df.values
        r, c = val.shape
        annot_str = np.empty((r, c), dtype=object)
        for i in range(r):
            for j in range(c):
                s = _fmt_value(val[i, j], fmt)
                if pro_sigstars and pvals is not None:
                    s = s + _stars(float(pvals.values[i, j]))
                annot_str[i, j] = s
        annot_data = annot_str

    # 刻度标签
    xtick = col_labels if (xticklabels in ("auto", True)) else (xticklabels if isinstance(xticklabels, (list, np.ndarray)) else False)
    ytick = row_labels if (yticklabels in ("auto", True)) else (yticklabels if isinstance(yticklabels, (list, np.ndarray)) else False)

    # ---------- 绘制 ----------
    with mpl.rc_context(rc=rc_local):
        # 线宽：用户未显式给时默认去掉
        _lw = (linewidths if (linewidths not in (None, 0)) else (0.0 if pro_no_grid else 0.4))

        if _HAS_SNS:
            sns.heatmap(
                A, vmin=vmin, vmax=vmax, cmap=cmap, center=center, robust=robust,
                annot=annot_data if annot else False,
                fmt="" if annot_data is not None else fmt,
                annot_kws=annot_kws, linewidths=_lw, linecolor=linecolor,
                cbar=cbar, cbar_kws=cbar_kws, cbar_ax=cbar_ax, square=square,
                xticklabels=xtick, yticklabels=ytick, mask=M, ax=ax, **kwargs
            )
        else:
            masked = np.ma.array(A, mask=M)
            im = ax.imshow(masked, vmin=vmin, vmax=vmax, cmap=cmap,
                           aspect='equal' if square else 'auto')
            if _lw > 0:
                for i in range(A.shape[0]+1):
                    ax.axhline(i-0.5, color=linecolor, lw=_lw)
                for j in range(A.shape[1]+1):
                    ax.axvline(j-0.5, color=linecolor, lw=_lw)
            if cbar:
                fig.colorbar(im, ax=ax if cbar_ax is None else cbar_ax, **(cbar_kws or {}))
            if xtick is not False:
                ax.set_xticks(np.arange(A.shape[1])); ax.set_xticklabels(col_labels if xtick is True else xtick)
            else:
                ax.set_xticks([])
            if ytick is not False:
                ax.set_yticks(np.arange(A.shape[0])); ax.set_yticklabels(row_labels if ytick is True else ytick)
            else:
                ax.set_yticks([])
            if annot and annot_data is not None:
                for i in range(A.shape[0]):
                    for j in range(A.shape[1]):
                        if M[i, j]: continue
                        ax.text(j, i, annot_data[i, j], ha="center", va="center",
                                fontsize=annot_kws.get("fontsize", 9),
                                color=annot_kws.get("color", "black"))

        # 刻度样式
        if xtick is not False:
            ax.tick_params(axis='x', labelrotation=rot_x)
            if isinstance(pro_dtick, int) and pro_dtick > 1:
                ax.set_xticks(np.arange(A.shape[1])[::pro_dtick])
        if ytick is not False:
            ax.tick_params(axis='y', labelrotation=rot_y)
            if isinstance(pro_dtick, int) and pro_dtick > 1:
                ax.set_yticks(np.arange(A.shape[0])[::pro_dtick])

        # ---- 统一 colormap/norm（供对角 KDE 取色）----
        cmap_obj = cmap if hasattr(cmap, "__call__") else mpl.colormaps[cmap]
        _data_vals = A[~M] if np.any(~M) else A.ravel()
        if robust:
            lo, hi = np.nanpercentile(_data_vals, 2), np.nanpercentile(_data_vals, 98)
        else:
            lo, hi = (np.nanmin(_data_vals), np.nanmax(_data_vals))
        vmin_eff = lo if vmin is None else float(vmin)
        vmax_eff = hi if vmax is None else float(vmax)
        norm_for_color = (TwoSlopeNorm(vcenter=float(center), vmin=vmin_eff, vmax=vmax_eff)
                          if center is not None else Normalize(vmin=vmin_eff, vmax=vmax_eff))

        # ---- 对角线 KDE（仅方阵）----
        if (pro_diag.lower() == "kde") and (A.shape[0] == A.shape[1]) and (source_df_for_diag is not None):
            from scipy.stats import gaussian_kde

            def _cell_value_for_color(i: int) -> float:
                if pro_diag_cmap_from == "diag" and np.isfinite(A[i, i]): return float(A[i, i])
                if pro_diag_cmap_from == "rowmean":
                    row, m = A[i, :], (~M[i, :] if M is not None else np.ones_like(A[i, :], bool))
                    return float(np.nanmean(row[m])) if np.any(m) else float(np.nanmean(row))
                if pro_diag_cmap_from == "colmean":
                    colv, m = A[:, i], (~M[:, i] if M is not None else np.ones_like(A[:, i], bool))
                    return float(np.nanmean(colv[m])) if np.any(m) else float(np.nanmean(colv))
                if pro_diag_cmap_from == "percentile":
                    return float(np.nanpercentile(_data_vals, pro_diag_percentile))
                return float(np.nanmean(_data_vals))

            for i, col in enumerate(col_labels):
                if col not in source_df_for_diag.columns:
                    continue
                s = source_df_for_diag[col].dropna()
                if s.size < 5 or float(np.std(s, ddof=1)) == 0.0:
                    continue  # 样本太少或无方差

                # 颜色（优先用户指定；否则与热图统一）
                if pro_diag_color is None:
                    val = _cell_value_for_color(i)
                    rgba = cmap_obj(norm_for_color(val))
                    rgb = np.array(rgba[:3]) * pro_diag_saturation + (1 - pro_diag_saturation)
                    kde_color = (float(rgb[0]), float(rgb[1]), float(rgb[2]))
                elif isinstance(pro_diag_color, dict):
                    kde_color = pro_diag_color.get(col, pro_diag_color.get(i, "dimgray"))
                else:
                    kde_color = pro_diag_color

                # KDE（先成功后建 inset 轴）
                try:
                    kde = gaussian_kde(s.values)
                    xx = np.linspace(s.min(), s.max(), 200)
                    yy = kde(xx)
                    if not np.all(np.isfinite(yy)) or yy.max() <= 0:
                        continue
                    yy = yy / yy.max()
                    gx = (xx - xx.min()) / max(xx.max() - xx.min(), 1e-12)
                except Exception:
                    continue

                # 计算该单元格在 Axes 坐标中的矩形（对齐 & 适配上下翻转）
                pad = float(pro_diag_pad)
                x0_d, y0_d = i + pad, i + pad
                x1_d, y1_d = i + 1 - pad, i + 1 - pad
                p0 = ax.transData.transform((x0_d, y0_d))
                p1 = ax.transData.transform((x1_d, y1_d))
                x0_a, y0_a = ax.transAxes.inverted().transform(p0)
                x1_a, y1_a = ax.transAxes.inverted().transform(p1)
                left, bottom = min(x0_a, x1_a), min(y0_a, y1_a)
                width, height = abs(x1_a - x0_a), abs(y1_a - y0_a)

                iax = ax.inset_axes([left, bottom, width, height], transform=ax.transAxes, zorder=5)
                iax.fill_between(gx, 0, yy, color=kde_color, alpha=0.18, linewidth=0)
                iax.plot(gx, yy, color=kde_color, lw=1.2)
                iax.set_xlim(0, 1); iax.set_ylim(0, 1)
                iax.axis("off")

        return ax

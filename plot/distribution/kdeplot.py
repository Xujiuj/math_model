# -*- coding: utf-8 -*-
"""
    @author: 数模加油站
    @time  : 2025/8/25 16:27
    @file  : kdeplot.py
"""

# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from typing import Optional, List, Tuple, Dict, Any, Union

try:
    from scipy.stats import gaussian_kde
except Exception as e:
    raise ImportError("需要 scipy 才能使用学术版 kdeplot（scipy.stats.gaussian_kde）。") from e

# 色盲友好 Okabe–Ito
_OI = ["#0072B2","#E69F00","#009E73","#CC79A7",
       "#D55E00","#56B4E9","#F0E442","#000000"]

def _is_num(s) -> bool:
    import pandas.api.types as ptypes
    return ptypes.is_numeric_dtype(s)

def _blend_to_white(color, saturation: float):
    import matplotlib.colors as mcolors
    r,g,b,_ = mcolors.to_rgba(color)
    s = float(np.clip(saturation, 0, 1))
    w = 1.0 - s
    return (r*s + 1*w, g*s + 1*w, b*s + 1*w)

def _bw_callable(method: Union[str, float, int, None], adjust: float):
    if method is None or isinstance(method, (float, int)):
        val = (1.0 if method is None else float(method)) * float(adjust)
        return val
    m = str(method).lower()
    if m == "scott":
        return lambda kde: kde.scotts_factor() * float(adjust)
    if m == "silverman":
        return lambda kde: kde.silverman_factor() * float(adjust)
    # 其他可调用：按 seaborn 兼容，直接套一层
    if callable(method):
        return lambda kde: float(method(kde)) * float(adjust)
    raise ValueError("bw_method 需为 'scott'/'silverman'/数值/可调用")

def _prep_series(data, key):
    if key is None: return None
    s = data[key]
    return s.dropna()

def kdeplot(
    data: Optional[Union[pd.DataFrame, pd.Series, np.ndarray, list]] = None, *,
    x: Optional[Union[str, pd.Series, np.ndarray, list]] = None,
    y: Optional[Union[str, pd.Series, np.ndarray, list]] = None,
    hue: Optional[Union[str, pd.Series, np.ndarray, list]] = None,
    weights: Optional[Union[str, pd.Series, np.ndarray, list]] = None,
    palette: Optional[Union[str, List, dict]] = None,
    hue_order: Optional[List] = None, hue_norm: Optional[Any] = None,  # 占位保持兼容
    bw_method: Union[str, float, int, None] = "scott", bw_adjust: float = 1.0,
    cut: float = 3.0, clip: Optional[Tuple[Tuple[float,float], Tuple[float,float]]] = None,
    gridsize: int = 256,
    levels: Union[int, List[float]] = 10, thresh: float = 0.05,
    fill: bool = True, alpha: Optional[float] = None,
    legend: bool = True, cbar: bool = False, cbar_ax: Optional[plt.Axes] = None, cbar_kws: Optional[dict] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs: Any
) -> plt.Axes:
    """
    兼容 seaborn.kdeplot 的灵活输入：
      - 单变量：kdeplot(data=一维数组/Series) 或 kdeplot(x=一维数组)
      - 双变量：kdeplot(data=二维数组[n,2]) 或 kdeplot(x=..., y=...)
      - DataFrame：若未显式给 x/y，则自动用其 1~2 个数值列
    同时保留学术风默认（HDR 50/80/95%、rug、均值/中位数线、纵向图例等）。
    """
    # ===== 学术增强的隐藏参数 =====
    sat = float(kwargs.pop("pro_saturation", 0.85))        # 降饱和
    pro_rug = bool(kwargs.pop("pro_rug", True))            # 单变量 rug
    pro_rug_height = float(kwargs.pop("pro_rug_height", 0.03))
    pro_show_loc = bool(kwargs.pop("pro_show_loc", True))  # 均值/中位数线
    pro_scatter2d = bool(kwargs.pop("pro_scatter2d", True))# 二维散点底图（n≤1000）
    pro_hdr = bool(kwargs.pop("pro_hdr", True))            # 二维按概率质量画等高线
    pro_mass_levels = tuple(kwargs.pop("pro_mass_levels", (0.5, 0.8, 0.95)))
    pro_linewidth = float(kwargs.pop("pro_linewidth", 2.0))
    pro_fill_alpha = 0.25 if alpha is None else float(alpha)

    # —— 统计信息开关/样式（隐藏参数，不破坏 seaborn API）——
    pro_show_statsbox = bool(kwargs.pop("pro_show_statsbox", True))  # 默认显示统计信息
    pro_stats_pos = tuple(kwargs.pop("pro_stats_pos", (0.02, 0.98)))  # 轴内坐标 (x,y)，左上
    pro_stats_fontsize = float(kwargs.pop("pro_stats_fontsize", 9.5))
    pro_stats_color = kwargs.pop("pro_stats_color", "dimgray")
    pro_stats_box_kws = kwargs.pop("pro_stats_box_kws",
                                   dict(facecolor="white", alpha=0.85,
                                        boxstyle="round,pad=0.3", edgecolor="0.5"))

    # 单变量与二维的文本格式（想自定义可传入 pro_stats_fmt / pro_stats2d_fmt）
    pro_stats_fmt = kwargs.pop("pro_stats_fmt",
                               "$n=${n:d}\n $\mu=${mean:.2f}±{sd:.2f}\n"
                               "$median=${median:.2f}\n[IQR {iqr:.2f}]")
    pro_stats2d_fmt = kwargs.pop("pro_stats2d_fmt",
                                 "n={n:d},\n $\mu_x$={mean_x:.2f}±{sd_x:.2f}, "
                                 "\n$\mu_y$={mean_y:.2f}±{sd_y:.2f}, $r$={r:.2f}")

    def _stats1d(arr: np.ndarray) -> dict:
        arr = np.asarray(arr, float)
        n = arr.size
        mean = float(np.mean(arr)) if n else np.nan
        sd = float(np.std(arr, ddof=1)) if n > 1 else np.nan
        median = float(np.median(arr)) if n else np.nan
        q1, q3 = (np.percentile(arr, [25, 75]) if n else (np.nan, np.nan))
        iqr = (q3 - q1) if n else np.nan
        return {"n": int(n), "mean": mean, "sd": sd, "median": median, "iqr": iqr}

    def _stats2d(xs: np.ndarray, ys: np.ndarray) -> dict:
        xs = np.asarray(xs, float);
        ys = np.asarray(ys, float)
        n = min(xs.size, ys.size)
        mean_x = float(np.mean(xs)) if n else np.nan
        sd_x = float(np.std(xs, ddof=1)) if n > 1 else np.nan
        mean_y = float(np.mean(ys)) if n else np.nan
        sd_y = float(np.std(ys, ddof=1)) if n > 1 else np.nan
        r = float(np.corrcoef(xs, ys)[0, 1]) if n > 1 else np.nan
        return {"n": int(n), "mean_x": mean_x, "sd_x": sd_x, "mean_y": mean_y, "sd_y": sd_y, "r": r}

    # ===== 将输入统一为 DataFrame: cols: x[, y][, hue][, weights] =====
    def _to_series(v, name):
        if v is None:
            return None
        if isinstance(v, pd.Series):
            s = v.copy()
            s.name = name
            return s
        if isinstance(v, (list, tuple, np.ndarray)):
            arr = np.asarray(v)
            if arr.ndim == 1:
                return pd.Series(arr, name=name)
            else:
                raise ValueError(f"{name} 需为一维数组/Series")
        # 字符串（列名）在此不处理，交给后面与 DataFrame 组合
        return v  # 可能是 str

    # 解析 x/y/hue/weights 的“数组直传”形式（data 可为 None）
    x_arr = _to_series(x, "x") if not isinstance(x, str) else None
    y_arr = _to_series(y, "y") if not isinstance(y, str) else None
    hue_arr = _to_series(hue, "hue") if (hue is not None and not isinstance(hue, str)) else None
    w_arr = _to_series(weights, "weights") if (weights is not None and not isinstance(weights, str)) else None

    # 构造基础 df
    df = None
    if isinstance(data, pd.DataFrame):
        # DataFrame + 可能的列名/数组混用
        if isinstance(x, str):
            x_ser = data[x]
        elif x_arr is not None:
            x_ser = x_arr
        else:
            x_ser = None

        if isinstance(y, str):
            y_ser = data[y]
        elif y_arr is not None:
            y_ser = y_arr
        else:
            y_ser = None

        # 若仍未给 x/y，则从数值列里猜
        if x_ser is None and y_ser is None:
            num_cols = [c for c in data.columns if _is_num(data[c])]
            if len(num_cols) == 0:
                raise ValueError("DataFrame 中没有可用的数值列，无法进行 KDE。")
            if len(num_cols) == 1:
                x_ser = data[num_cols[0]]
            else:
                x_ser = data[num_cols[0]]
                y_ser = data[num_cols[1]]

        # hue / weights
        if isinstance(hue, str):
            hue_ser = data[hue]
        else:
            hue_ser = hue_arr

        if isinstance(weights, str):
            w_ser = data[weights]
        else:
            w_ser = w_arr

        df = pd.DataFrame()
        if x_ser is not None: df["x"] = pd.Series(x_ser).reset_index(drop=True)
        if y_ser is not None: df["y"] = pd.Series(y_ser).reset_index(drop=True)
        if hue_ser is not None: df["hue"] = pd.Series(hue_ser).reset_index(drop=True)
        if w_ser is not None: df["weights"] = pd.Series(w_ser).reset_index(drop=True)

    elif data is None:
        # 纯数组直传：kdeplot(x=...), 或 kdeplot(data=None, y=...)；或 x/y 两个数组
        if x_arr is None and y_arr is None:
            raise ValueError("请提供 data 或 x/y 中的至少一个。")
        df = pd.DataFrame()
        if x_arr is not None: df["x"] = x_arr.reset_index(drop=True)
        if y_arr is not None: df["y"] = y_arr.reset_index(drop=True)
        if hue_arr is not None: df["hue"] = hue_arr.reset_index(drop=True)
        if w_arr is not None: df["weights"] = w_arr.reset_index(drop=True)

    else:
        # data 为数组/列表
        arr = np.asarray(data)
        if arr.ndim == 1:
            df = pd.DataFrame({"x": arr})
        elif arr.ndim == 2 and arr.shape[1] >= 2:
            df = pd.DataFrame({"x": arr[:, 0], "y": arr[:, 1]})
        else:
            raise ValueError("data 需为一维或二维数组（二维至少含两列）。")

        # 叠加外部给的 hue/weights（数组形式）
        if hue_arr is not None: df["hue"] = hue_arr.reset_index(drop=True)
        if w_arr is not None: df["weights"] = w_arr.reset_index(drop=True)

    # 清理：仅保留参与列，并按需要的列同时非空过滤
    needed = ["x"] + (["y"] if "y" in df.columns else []) + (["hue"] if "hue" in df.columns else []) + (["weights"] if "weights" in df.columns else [])
    df = df[needed].copy()
    df = df.dropna(subset=["x"] + (["y"] if "y" in df.columns else []))

    # —— hue 层级（如果存在 hue 列）——
    if "hue" in df.columns:
        h_levels = hue_order or list(pd.unique(df["hue"]))
    else:
        h_levels = None

    # —— 调色 ——
    if palette is None:
        pal = _OI if h_levels else ["#4C72B0"]
    elif isinstance(palette, str):
        pal = (mpl.colormaps[palette].colors if palette in mpl.colormaps else _OI)
    elif isinstance(palette, list):
        pal = palette
    elif isinstance(palette, dict) and h_levels is not None:
        pal = [palette.get(h, _OI[i % len(_OI)]) for i, h in enumerate(h_levels)]
    else:
        pal = _OI

    # 局部 rc
    rc_local = {
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.linestyle": "--", "grid.alpha": 0.35,
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    }
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
    else:
        fig = ax.figure

    # 单变量 or 双变量
    univariate = ("y" not in df.columns)

    with mpl.rc_context(rc=rc_local):
        if univariate:
            # —— 每个 hue 一条曲线；无 hue 则一条 ——
            groups = [("ALL", df)] if h_levels is None else [(hv, df[df["hue"]==hv]) for hv in h_levels]
            for i, (hv, sub) in enumerate(groups):
                xs = sub["x"].to_numpy()
                if xs.size == 0: continue
                col = _blend_to_white(pal[i % len(pal)], sat)

                # KDE
                bw = _bw_callable(bw_method, bw_adjust)
                kde = gaussian_kde(xs, bw_method=bw, weights=(sub["weights"].to_numpy() if "weights" in sub.columns else None))

                # 网格范围
                std = float(xs.std(ddof=1)) if xs.size>1 else 1.0
                bfac = float(kde.covariance_factor()) * std
                left = xs.min() - cut*bfac; right = xs.max() + cut*bfac
                if clip is not None and isinstance(clip, tuple) and len(clip)==2 and isinstance(clip[0], (int,float)):
                    left = max(left, clip[0]); right = min(right, clip[1])

                grid = np.linspace(left, right, int(gridsize))
                dens = kde(grid)

                if fill:
                    ax.fill_between(grid, 0, dens, color=col, alpha=pro_fill_alpha, linewidth=0)
                ax.plot(grid, dens, color=col, linewidth=pro_linewidth)

                # rug
                if pro_rug and xs.size <= 300:
                    ymin, ymax = ax.get_ylim(); span = ymax - ymin
                    y0 = ymin; y1 = ymin + span*pro_rug_height
                    ax.vlines(xs, y0, y1, color=col, linewidth=0.6, alpha=0.8)

                # 均值/中位数
                if pro_show_loc:
                    mu = float(xs.mean()); med = float(np.median(xs))
                    ax.axvline(med, color=col, linewidth=1.2, linestyle="--", alpha=0.9)
                    ax.axvline(mu,  color=col, linewidth=1.0, linestyle=":",  alpha=0.9)

            ax.set_xlabel(x if isinstance(x, str) else (x if x is not None else (getattr(df["x"], "name", "x") or "x")))
            ax.set_ylabel("密度")

            # === 统计信息：左上角文本框（单变量） ===
            if pro_show_statsbox:
                stats_lines = []
                stats_rows = []
                if h_levels is None:
                    xs = df["x"].to_numpy()
                    st = _stats1d(xs)
                    stats_lines.append(pro_stats_fmt.format(**st))
                    stats_rows.append(dict(hue=None, **st))
                else:
                    for i, hv in enumerate(h_levels):
                        sub = df[df["hue"] == hv]
                        xs = sub["x"].to_numpy()
                        st = _stats1d(xs)
                        # 每个分组一行，行首带分组名
                        stats_lines.append(f"{hv}: " + pro_stats_fmt.format(**st))
                        stats_rows.append(dict(hue=hv, **st))

                text = "\n".join(stats_lines)
                ax.text(pro_stats_pos[0], pro_stats_pos[1], text,
                        transform=ax.transAxes, ha="left", va="top",
                        fontsize=pro_stats_fontsize, color=pro_stats_color,
                        bbox=pro_stats_box_kws)

                # 可用于导出：把统计表挂到 Axes
                ax._kde_stats_table = pd.DataFrame(stats_rows)

            if legend and h_levels:
                handles = [Patch(facecolor=_blend_to_white(pal[i%len(pal)], sat), edgecolor="none", alpha=0.6, label=str(h))
                           for i,h in enumerate(h_levels)]
                leg = ax.legend(handles=handles, ncols=1, title=(hue if isinstance(hue, str) else "hue"),
                                loc="upper right", bbox_to_anchor=(0.99,0.99), bbox_transform=ax.transAxes,
                                frameon=False, handlelength=1.0, handletextpad=0.6, labelspacing=0.3)
                if leg and leg.get_title(): leg.get_title().set_fontweight("semibold")

        else:
            # —— 二维 KDE ——
            groups = [("ALL", df)] if h_levels is None else [(hv, df[df["hue"]==hv]) for hv in h_levels]
            last_Z = None  # 供可选色条使用
            for i,(hv, sub) in enumerate(groups):
                xs = sub["x"].to_numpy(); ys = sub["y"].to_numpy()
                if xs.size==0 or ys.size==0: continue
                col = _blend_to_white(pal[i % len(pal)], sat)

                bw = _bw_callable(bw_method, bw_adjust)
                xy = np.vstack([xs, ys])
                kde = gaussian_kde(xy, bw_method=bw, weights=(sub["weights"].to_numpy() if "weights" in sub.columns else None))
                cov = kde.covariance
                sx = np.sqrt(cov[0,0]); sy = np.sqrt(cov[1,1])
                x_min = xs.min() - cut*sx; x_max = xs.max() + cut*sx
                y_min = ys.min() - cut*sy; y_max = ys.max() + cut*sy
                if clip is not None and isinstance(clip, tuple) and len(clip)==2 and isinstance(clip[0], tuple):
                    (xa, xb), (ya, yb) = clip
                    x_min = max(x_min, xa); x_max = min(x_max, xb)
                    y_min = max(y_min, ya); y_max = min(y_max, yb)

                gx = np.linspace(x_min, x_max, int(gridsize))
                gy = np.linspace(y_min, y_max, int(gridsize))
                X, Y = np.meshgrid(gx, gy)
                coords = np.vstack([X.ravel(), Y.ravel()])
                Z = kde(coords).reshape(X.shape); last_Z = Z

                if pro_scatter2d and xs.size <= 1000:
                    ax.scatter(xs, ys, s=8, alpha=0.25, color=col, linewidths=0)

                if pro_hdr:
                    zflat = Z.ravel()
                    order = np.argsort(zflat)[::-1]
                    zsorted = zflat[order]
                    area = (gx[1]-gx[0]) * (gy[1]-gy[0])
                    mass_cum = np.cumsum(zsorted) * area
                    levels_mass = sorted(pro_mass_levels)
                    iso = []
                    for p in levels_mass:
                        idx = np.searchsorted(mass_cum, p, side="left")
                        thr = zsorted[min(idx, len(zsorted)-1)]
                        iso.append(thr)
                    use_levels = iso
                    Z_mask = np.where(Z >= (thresh*np.max(Z)), Z, np.nan)
                else:
                    if isinstance(levels, int):
                        use_levels = np.linspace(np.nanmin(Z), np.nanmax(Z), int(levels)+2)[1:-1]
                    else:
                        use_levels = list(levels)
                    Z_mask = np.where(Z >= (thresh*np.nanmax(Z)), Z, np.nan)

                cs = ax.contour(X, Y, Z_mask, levels=use_levels, colors=[col], linewidths=1.6)
                if pro_hdr and len(use_levels)>=2:
                    ax.contourf(X, Y, Z, levels=[use_levels[1], Z.max()], colors=[col], alpha=0.12)
                    ax.contourf(X, Y, Z, levels=[use_levels[0], Z.max()], colors=[col], alpha=0.18)
                try:
                    ax.clabel(cs, inline=True, fmt="%.3g", fontsize=8)
                except Exception:
                    pass
            # === 统计信息：左上角文本框（二维） ===
            if pro_show_statsbox:
                stats_lines = []
                stats_rows = []
                if h_levels is None:
                    xs = df["x"].to_numpy();
                    ys = df["y"].to_numpy()
                    st = _stats2d(xs, ys)
                    stats_lines.append(pro_stats2d_fmt.format(**st))
                    stats_rows.append(dict(hue=None, **st))
                else:
                    for hv in h_levels:
                        sub = df[df["hue"] == hv]
                        xs = sub["x"].to_numpy();
                        ys = sub["y"].to_numpy()
                        st = _stats2d(xs, ys)
                        stats_lines.append(f"{hv}: " + pro_stats2d_fmt.format(**st))
                        stats_rows.append(dict(hue=hv, **st))

                text = "\n".join(stats_lines)
                ax.text(pro_stats_pos[0], pro_stats_pos[1], text,
                        transform=ax.transAxes, ha="left", va="top",
                        fontsize=pro_stats_fontsize, color=pro_stats_color,
                        bbox=pro_stats_box_kws)

                ax._kde_stats_table = pd.DataFrame(stats_rows)

            ax.set_xlabel(x if isinstance(x, str) else (x if x is not None else "x"))
            ax.set_ylabel(y if isinstance(y, str) else (y if y is not None else "y"))

            if legend and h_levels:
                handles = [Patch(facecolor=_blend_to_white(pal[i%len(pal)], sat), edgecolor="none", alpha=0.6, label=str(h))
                           for i,h in enumerate(h_levels)]
                leg = ax.legend(handles=handles, ncols=1, title=(hue if isinstance(hue, str) else "hue"),
                                loc="upper right", bbox_to_anchor=(0.99,0.99), bbox_transform=ax.transAxes,
                                frameon=False, handlelength=1.0, handletextpad=0.6, labelspacing=0.3)
                if leg and leg.get_title(): leg.get_title().set_fontweight("semibold")

            if cbar and last_Z is not None:
                from matplotlib.cm import ScalarMappable
                norm = mpl.colors.Normalize(vmin=np.nanmin(last_Z), vmax=np.nanmax(last_Z))
                sm = ScalarMappable(norm=norm, cmap=mpl.cm.Blues); sm.set_array([])
                cbar_kws = ({} if cbar_kws is None else dict(cbar_kws))
                plt.colorbar(sm, ax=ax if cbar_ax is None else cbar_ax, **cbar_kws)

        return ax



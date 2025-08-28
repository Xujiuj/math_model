from __future__ import annotations
import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from typing import Optional, List, Tuple, Dict, Any, Union

_OI = ["#0072B2","#E69F00","#009E73","#CC79A7","#D55E00","#56B4E9","#F0E442","#000000"]

def _is_num(s) -> bool:
    import pandas.api.types as ptypes
    return ptypes.is_numeric_dtype(s)

def _blend_to_white(color, saturation: float):
    import matplotlib.colors as mcolors
    r,g,b,_ = mcolors.to_rgba(color); s=max(0.0,min(1.0,saturation)); w=1.0-s
    return (r*s+1*w,g*s+1*w,b*s+1*w)

def _z_from_ci(ci: float) -> float:
    return 1.96 if abs(ci-95.0)<1e-6 else 1.64485 if abs(ci-90.0)<1e-6 else 2.5758 if abs(ci-99.0)<1e-6 else 1.96

def lineplot(
    data: Optional[pd.DataFrame] = None, *,
    x: Optional[str] = None, y: Optional[str] = None,
    hue: Optional[str] = None, size: Optional[str] = None, style: Optional[str] = None,
    hue_order: Optional[List] = None, style_order: Optional[List] = None, size_order: Optional[List] = None,
    palette: Optional[Union[str, List, dict]] = None,
    hue_norm: Optional[Any] = None, size_norm: Optional[Any] = None,
    estimator: Union[str, Any] = "mean",
    errorbar: Optional[Union[str, Tuple[str, float]]] = ("ci", 95),
    n_boot: int = 1000, seed: Optional[int] = None,
    sort: bool = True, orient: Optional[str] = None,
    err_style: str = "band", err_kws: Optional[dict] = None,
    legend: Union[str, bool] = "auto",
    dashes: Union[bool, dict] = True, markers: Optional[Union[bool, List, dict]] = None,
    linewidth: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs: Any
) -> plt.Axes:
    if data is None: raise ValueError("需要传入 DataFrame：data=...")
    if "ci" in kwargs:
        ci = kwargs.pop("ci")
        errorbar = ("ci", ci) if ci is not None else None
    # 隐藏参数
    sat = float(kwargs.pop("pro_saturation", 0.85))
    pro_trend = bool(kwargs.pop("pro_trend", True))          # 是否画趋势线
    pro_trend_stats = bool(kwargs.pop("pro_trend_stats", True))# 是否计算并挂载回归统计

    if ax is None: fig, ax = plt.subplots(figsize=(8, 5))
    else: fig = ax.figure

    if orient is None and x is not None and y is not None:
        orient = "v" if (_is_num(data[y]) and not _is_num(data[x])) else ("h" if (_is_num(data[x]) and not _is_num(data[y])) else "v")
    if orient is None: orient = "v"
    vert = orient.lower().startswith("v")

    g_levels = hue_order or (list(pd.unique(data[hue])) if hue else None)
    s_levels = style_order or (list(pd.unique(data[style])) if style else None)

    rc_local = {
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.linestyle": "--", "grid.alpha": 0.35,
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    }

    lw = 2.2 if linewidth is None else linewidth
    msize = 4.5; band_alpha = 0.18; bar_capsize = 3.0; edge_color = "#404040"

    if palette is None:
        pal = (_OI[:max(1,len(g_levels))] if g_levels else ["#4C72B0"])
    elif isinstance(palette, str):
        pal = (mpl.colormaps[palette].colors if palette in mpl.colormaps else _OI)
    elif isinstance(palette, list):
        pal = palette
    elif isinstance(palette, dict) and g_levels is not None:
        pal = [palette.get(h, _OI[i % len(_OI)]) for i, h in enumerate(g_levels)]
    else:
        pal = _OI

    if isinstance(estimator, str):
        agg = np.mean if estimator=="mean" else (np.median if estimator=="median" else None)
        if agg is None: raise ValueError("estimator 仅支持 'mean'/'median' 或 callable")
    else:
        agg = estimator

    err_mode, ci = None, None
    if errorbar is None: err_mode = None
    elif isinstance(errorbar, str): err_mode = errorbar.lower()
    elif isinstance(errorbar, tuple) and len(errorbar)==2 and str(errorbar[0]).lower()=="ci":
        err_mode, ci = "ci", float(errorbar[1])
    else: raise ValueError("errorbar 需为 'sd'|'se'|('ci',level)|None")

    if g_levels is None: g_levels=[None]
    if s_levels is None: s_levels=[None]
    series_keys = [(g,s) for g in g_levels for s in s_levels]

    # 将聚合统计存起来
    agg_rows = []   # 每个系列每个 x 的 n/mean/sd/se/ci
    trend_rows = [] # 每个系列的回归统计

    with mpl.rc_context(rc=rc_local):
        for idx, (gh, gs) in enumerate(series_keys):
            # 子集
            if hue and style: sub = data[(data[hue]==gh) & (data[style]==gs)]
            elif hue:         sub = data[data[hue]==gh]
            elif style:       sub = data[data[style]==gs]
            else:             sub = data

            if vert:
                df = sub[[x, y]].dropna()
                grp = df.groupby(x, sort=sort)[y]
            else:
                df = sub[[y, x]].dropna()
                grp = df.groupby(y, sort=sort)[x]

            # 聚合
            mean = grp.apply(lambda v: float(agg(v.values)))
            arr  = grp.apply(lambda v: v.values)
            n    = arr.apply(len).astype(float)
            sd   = arr.apply(lambda v: np.std(v, ddof=1)).astype(float)
            se   = sd / np.sqrt(np.maximum(n,1))
            if err_mode is None:
                lo = hi = None
            elif err_mode == "sd":
                lo, hi = mean - sd, mean + sd
            elif err_mode == "se":
                lo, hi = mean - se, mean + se
            elif err_mode == "ci":
                z = _z_from_ci(ci if ci is not None else 95.0)
                lo, hi = mean - z*se, mean + z*se

            # 记录聚合统计
            frame = pd.DataFrame({
                "series_hue": gh, "series_style": gs,
                "x": mean.index.values, "n": n.values, "mean": mean.values,
                "sd": sd.values, "se": se.values,
                "ci_lo": (lo.values if lo is not None else np.full_like(mean.values, np.nan)),
                "ci_hi": (hi.values if hi is not None else np.full_like(mean.values, np.nan)),
            })
            agg_rows.append(frame)

            xs = mean.index.to_numpy(); ys = mean.to_numpy()
            if lo is not None:
                los = lo.reindex(mean.index).to_numpy(); his = hi.reindex(mean.index).to_numpy()

            color = _blend_to_white(pal[(g_levels.index(gh) if gh is not None else 0)%len(pal)], sat)
            # style → markers / dashes
            default_markers = ['o','s','^','D','P','X','v','*','<','>']
            default_dashes  = [(None,None),(4,2),(2,2),(6,2,2,2),(1,1),(8,2)]
            si = s_levels.index(gs) if gs is not None else 0
            mk = default_markers[si % len(default_markers)] if (markers is None or markers is True) else (markers if isinstance(markers,str) else None)
            ds = default_dashes[si % len(default_dashes)] if dashes is True else (None if dashes is False else None)

            # 误差
            if err_mode is not None and len(xs)>0 and lo is not None:
                if err_style=="band":
                    if vert: ax.fill_between(xs, los, his, color=color, alpha=0.18, linewidth=0)
                    else:    ax.fill_betweenx(xs, los, his, color=color, alpha=0.18, linewidth=0)
                elif err_style=="bars":
                    if vert: ax.errorbar(xs, ys, yerr=his-ys, fmt="none", ecolor=color, elinewidth=lw*0.6, capsize=3.0, alpha=0.9)
                    else:    ax.errorbar(ys, xs, xerr=his-ys, fmt="none", ecolor=color, elinewidth=lw*0.6, capsize=3.0, alpha=0.9)

            # 折线 + 点
            line_kwargs = dict(color=color, linewidth=lw)
            if ds is not None: line_kwargs["dashes"]=ds
            if vert: ln, = ax.plot(xs, ys, **line_kwargs)
            else:    ln, = ax.plot(ys, xs, **line_kwargs)
            if mk:
                if vert: ax.plot(xs, ys, linestyle="none", marker=mk, markersize=4.5, markerfacecolor="white", markeredgecolor=color, markeredgewidth=lw*0.5)
                else:    ax.plot(ys, xs, linestyle="none", marker=mk, markersize=4.5, markerfacecolor="white", markeredgecolor=color, markeredgewidth=lw*0.5)

            # 可选：回归统计（基于聚合均值）
            if pro_trend_stats or pro_trend:
                try:
                    from scipy import stats
                    xnum = xs.astype(float); ynum = ys.astype(float)
                    lr = stats.linregress(xnum, ynum)
                    trend_rows.append({
                        "series_hue": gh, "series_style": gs,
                        "slope": lr.slope, "intercept": lr.intercept,
                        "r": np.sign(lr.slope)*np.sqrt(max(lr.rvalue**2, 0.0)),  # 同向相关系数
                        "r2": lr.rvalue**2, "p": lr.pvalue, "stderr": lr.stderr, "n_points": len(xnum)
                    })
                    if pro_trend:
                        xfit = np.linspace(xnum.min(), xnum.max(), 200)
                        yfit = lr.slope*xfit + lr.intercept
                        if vert: ax.plot(xfit, yfit, color=color, linewidth=lw*0.6, alpha=0.5, linestyle="--")
                        else:    ax.plot(yfit, xfit, color=color, linewidth=lw*0.6, alpha=0.5, linestyle="--")
                except Exception:
                    pass

        if vert: ax.set_xlabel(x if x else ""); ax.set_ylabel(y if y else "")
        else:    ax.set_xlabel(x if x else ""); ax.set_ylabel(y if y else "")

        # 图例（纵向、轴内右上，防越界）
        if legend not in (False, "brief", None):
            handles=[]; labels=[]; title=None
            if hue and style:
                for i,hv in enumerate(hue_order or pd.unique(data[hue])):
                    c = _blend_to_white(pal[i%len(pal)], sat)
                    handles.append(Patch(facecolor=c, edgecolor="none", alpha=0.6, label=str(hv))); labels.append(str(hv))
                for j,st in enumerate(style_order or pd.unique(data[style])):
                    default_markers = ['o','s','^','D','P','X','v','*','<','>']
                    default_dashes  = [(None,None),(4,2),(2,2),(6,2,2,2),(1,1),(8,2)]
                    mk = default_markers[j%len(default_markers)]; ds = default_dashes[j%len(default_dashes)] if dashes is True else None
                    kw = dict(color=edge_color, lw=lw, marker=mk, markersize=4.5, markerfacecolor="white", label=str(st))
                    if ds is not None: kw["dashes"]=ds
                    handles.append(Line2D([0],[0], **kw)); labels.append(str(st))
                title=f"{hue} / {style}"
            elif hue:
                for i,hv in enumerate(hue_order or pd.unique(data[hue])):
                    c = _blend_to_white(pal[i%len(pal)], sat)
                    handles.append(Line2D([0],[0], color=c, lw=lw, label=str(hv))); labels.append(str(hv))
                title=hue
            elif style:
                default_markers = ['o','s','^','D','P','X','v','*','<','>']
                default_dashes  = [(None,None),(4,2),(2,2),(6,2,2,2),(1,1),(8,2)]
                for j,st in enumerate(style_order or pd.unique(data[style])):
                    mk = default_markers[j%len(default_markers)]; ds = default_dashes[j%len(default_dashes)] if dashes is True else None
                    kw = dict(color=edge_color, lw=lw, marker=mk, markersize=4.5, markerfacecolor="white", label=str(st))
                    if ds is not None: kw["dashes"]=ds
                    handles.append(Line2D([0],[0], **kw)); labels.append(str(st))
                title=style
            if handles:
                leg = ax.legend(handles=handles, labels=labels, title=title, ncols=1,
                                loc="upper right", bbox_to_anchor=(0.99,0.99), bbox_transform=ax.transAxes,
                                frameon=False, borderaxespad=0., handlelength=1.2, handletextpad=0.6, labelspacing=0.3)
                if leg and leg.get_title(): leg.get_title().set_fontweight("semibold")

        # —— 聚合统计表/趋势统计表：挂到 Axes ——
        ax._agg_table   = (pd.concat(agg_rows, ignore_index=True) if agg_rows else pd.DataFrame())
        ax._trend_table = (pd.DataFrame(trend_rows) if trend_rows else pd.DataFrame())
        return ax

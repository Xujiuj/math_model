# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from typing import Optional, List, Tuple, Dict, Any, Union, Iterable

# Okabe–Ito 色盲友好备选
_OI = ["#0072B2","#E69F00","#009E73","#CC79A7","#D55E00","#56B4E9","#F0E442","#000000"]

# ---------- utils ----------
def _is_num(s) -> bool:
    import pandas.api.types as ptypes
    return ptypes.is_numeric_dtype(s)

def _blend_to_white(color, saturation: float):
    import matplotlib.colors as mcolors
    r,g,b,_ = mcolors.to_rgba(color); s = max(0.0, min(1.0, saturation)); w = 1.0 - s
    return (r*s + 1*w, g*s + 1*w, b*s + 1*w)

def _all_pairs(lst: Iterable) -> List[Tuple]:
    lst = list(lst); return [(lst[i], lst[j]) for i in range(len(lst)) for j in range(i+1, len(lst))]

def _cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2: return np.nan
    sa2, sb2 = a.var(ddof=1), b.var(ddof=1)
    sp = np.sqrt(((na-1)*sa2 + (nb-1)*sb2) / max(na+nb-2, 1))
    if sp == 0: return 0.0
    return (a.mean() - b.mean()) / sp

def _hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    d = _cohen_d(a,b); na, nb = len(a), len(b)
    if not np.isfinite(d): return np.nan
    J = 1.0 - 3.0/(4*(na+nb)-9) if (na+nb) > 3 else 1.0
    return d * J

# ---------- main ----------
def boxplot(
    # 与 seaborn.boxplot 一致的公开参数
    data: Optional[pd.DataFrame] = None, *,
    x: Optional[str] = None, y: Optional[str] = None, hue: Optional[str] = None,
    order: Optional[List] = None, hue_order: Optional[List] = None,
    orient: Optional[str] = None, color: Optional[Union[str, Tuple[float,float,float]]] = None,
    palette: Optional[Union[str, List, dict]] = None,
    saturation: float = 0.75, width: float = 0.8, dodge: bool = True,
    fliersize: float = 5, linewidth: Optional[float] = None,
    whis: Union[float, Tuple[int,int]] = 1.5,
    ax: Optional[plt.Axes] = None,
    **kwargs: Any
) -> plt.Axes:
    """
    额外学术增强（隐藏参数，均为可选，不破坏 seaborn API）：
      - pro_points: 'auto'|'strip'|'none'（<=pro_points_nmax 时自动画抖点）
      - pro_points_nmax: 150
      - pro_points_alpha: 0.45
      - pro_points_size: 10
      - pro_show_fliers: False       # 更清爽，默认不画离群点
      - pro_n_where: 'group'|'combo' # hue 时 n= 只在每个组汇总标一次（默认）
      - pro_show_stats: True|False|'auto'（默认 'auto'，≤6 个箱才标 mean±SE）
      - pro_stats_fmt: "{mean:.2f}±{se:.2f}"
      - pro_stats_color: "dimgray"
      - pro_auto_sig: True           # 自动成对显著性（括线）
      - pro_auto_sig_scope: 'auto'|'adjacent'|'all'|'within'
      - pro_sig_test: 'ttest'|'mw'
      - pro_equal_var: False
      - pro_sig_mode: 'stars'|'p'
      - pro_sig_alpha: 0.05          # 只画显著的
      - pro_sig_correction: 'none'|'bonferroni'|'fdr_bh'
      - pro_sig_max_pairs: 25        # 最多画多少条括线
      - pro_effect: 'cohen_d'|'hedges_g'|None  # 在标签里附带效应量
      - fontsize: 9
    """
    if data is None:
        raise ValueError("需要传入 DataFrame：data=...")

    # —— 隐藏学术参数 ——（全部在函数内部使用）
    pro_auto_sig: bool = kwargs.pop("pro_auto_sig", True)
    pro_auto_sig_scope: str = kwargs.pop("pro_auto_sig_scope", "auto")
    pro_sig_test: str = kwargs.pop("pro_sig_test", "ttest")
    pro_equal_var: bool = kwargs.pop("pro_equal_var", False)
    pro_sig_mode: str = kwargs.pop("pro_sig_mode", "stars")
    pro_sig_correction: str = kwargs.pop("pro_sig_correction", "none")
    pro_sig_alpha: float = float(kwargs.pop("pro_sig_alpha", 0.05))
    pro_sig_max_pairs: Optional[int] = kwargs.pop("pro_sig_max_pairs", 25)
    pro_effect: Optional[str] = kwargs.pop("pro_effect", "cohen_d")

    pro_show_stats: Union[bool, str] = kwargs.pop("pro_show_stats", "auto")
    pro_stats_fmt: str = kwargs.pop("pro_stats_fmt", "{mean:.2f}±{se:.2f}")
    pro_stats_color: str = kwargs.pop("pro_stats_color", "dimgray")

    pro_points: str = kwargs.pop("pro_points", "auto")
    pro_points_nmax: int = int(kwargs.pop("pro_points_nmax", 150))
    pro_points_alpha: float = float(kwargs.pop("pro_points_alpha", 0.45))
    pro_points_size: float = float(kwargs.pop("pro_points_size", 10))

    pro_n_where: str = kwargs.pop("pro_n_where", "group")
    pro_show_fliers: bool = bool(kwargs.pop("pro_show_fliers", False))
    fontsize: int = kwargs.pop("fontsize", 12)

    # 轴与方向
    if ax is None: fig, ax = plt.subplots(figsize=(8, 5))
    else: fig = ax.figure

    if orient is None and x is not None and y is not None:
        orient = "v" if (_is_num(data[y]) and not _is_num(data[x])) else ("h" if (_is_num(data[x]) and not _is_num(data[y])) else "v")
    if orient is None: orient = "v"
    vert = orient.lower().startswith("v")

    # 分组水平
    g_levels = order or list(pd.unique(data[x] if vert else data[y]))
    h_levels = hue_order or (list(pd.unique(data[hue])) if hue else None)

    # 局部 rc（更克制的默认）
    rc_local = {
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False,  # 只在数值轴结尾再单独开细网格
        "grid.linestyle": "--", "grid.alpha": 0.18,
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    }
    edge_color = "#404040"; line_lw = 1.3 if linewidth is None else linewidth
    median_lw = 1.8; face_alpha = 0.34
    if whis == 1.5:  # 学术默认：5–95 分位触须
        whis = (5, 95)

    # 调色
    if palette is None:
        pal = (_OI if h_levels else ["#4C72B0"])
    elif isinstance(palette, str):
        pal = (mpl.colormaps[palette].colors if palette in mpl.colormaps else _OI)
    elif isinstance(palette, list):
        pal = palette
    elif isinstance(palette, dict) and h_levels is not None:
        pal = [palette.get(h, _OI[i % len(_OI)]) for i, h in enumerate(h_levels)]
    else:
        pal = _OI

    flier_kw  = dict(markersize=fliersize, markerfacecolor=edge_color, markeredgecolor=edge_color, alpha=0.6)
    median_kw = dict(lw=median_lw, color=edge_color)
    mean_kw   = dict(lw=line_lw, ls="--", color=edge_color)
    whisk_kw  = dict(lw=line_lw, color=edge_color)
    cap_kw    = dict(lw=line_lw, color=edge_color)
    box_kw    = dict(lw=line_lw, edgecolor=edge_color)

    pos_map: Dict[Tuple, float] = {}
    stats_rows = []  # 描述性统计表

    with mpl.rc_context(rc=rc_local):
        # ========== 绘制箱线 ==========
        if h_levels is not None:
            n_g, n_h = len(g_levels), len(h_levels)
            width_total = width; gap = width_total / max(n_h, 1)
            centers = np.arange(n_g)

            for hj, hval in enumerate(h_levels):
                offset = (hj - (n_h-1)/2) * gap * (0.9 if dodge else 0.0)
                pos = centers + offset
                dat = []
                for g in g_levels:
                    sub = data[(data[x]==g) & (data[hue]==hval)] if vert else data[(data[y]==g) & (data[hue]==hval)]
                    vals = (sub[y].to_numpy() if vert else sub[x].to_numpy()); vals = vals[~pd.isna(vals)]
                    dat.append(vals)
                    if len(vals)>0:
                        q1, med, q3 = np.percentile(vals, [25,50,75])
                        iqr = q3-q1; lo, hi = q1-1.5*iqr, q3+1.5*iqr
                        out_n = int(((vals<lo)|(vals>hi)).sum())
                        n = len(vals); sd = vals.std(ddof=1) if n>1 else 0.0; se = sd/np.sqrt(max(n,1))
                        z = 1.96; ci_lo, ci_hi = (vals.mean()-z*se, vals.mean()+z*se)
                        stats_rows.append({
                            "group":g, "hue":hval, "n":n, "mean":vals.mean(), "sd":sd, "se":se,
                            "ci95_lo":ci_lo, "ci95_hi":ci_hi, "median":med, "q1":q1, "q3":q3, "iqr":iqr,
                            "min":float(vals.min()), "max":float(vals.max()), "outliers":out_n
                        })

                face = _blend_to_white(pal[hj % len(pal)], saturation)
                bp = ax.boxplot(
                    dat, positions=pos, widths=gap*0.7 if dodge else width_total*0.7,
                    patch_artist=True, showmeans=True, meanline=True, notch=True,
                    whis=whis, vert=vert, showfliers=pro_show_fliers,
                    flierprops=flier_kw, medianprops=median_kw, meanprops=mean_kw,
                    whiskerprops=whisk_kw, capprops=cap_kw, boxprops=box_kw
                )
                for b in bp["boxes"]:
                    b.set_facecolor(face); b.set_alpha(face_alpha); b.set_edgecolor(edge_color)
                for gi, g in enumerate(g_levels):
                    pos_map[(g, hval)] = float(pos[gi])

            if vert:
                ax.set_xticks(np.arange(len(g_levels))); ax.set_xticklabels(g_levels)
            else:
                ax.set_yticks(np.arange(len(g_levels))); ax.set_yticklabels(g_levels)

            # 图例：竖排放到图外
            handles = [Patch(facecolor=_blend_to_white(pal[j%len(pal)], saturation),
                             alpha=0.6, edgecolor="none", label=str(h_levels[j]))
                       for j in range(len(h_levels))]
            leg = ax.legend(handles=handles, title=hue, ncols=1,
                            loc="upper left", bbox_to_anchor=(1.02, 1.0),
                            borderaxespad=0.0, frameon=False,
                            handlelength=1.0, handletextpad=0.6, labelspacing=0.3,
                            fontsize=fontsize, title_fontsize=fontsize+1)
            if leg and leg.get_title(): leg.get_title().set_fontweight("semibold")

        else:
            centers = np.arange(len(g_levels))
            dat = []
            for g in g_levels:
                sub = data[data[x]==g] if vert else data[data[y]==g]
                vals = (sub[y].to_numpy() if vert else sub[x].to_numpy()); vals = vals[~pd.isna(vals)]
                dat.append(vals)
                if len(vals)>0:
                    q1, med, q3 = np.percentile(vals, [25,50,75])
                    iqr = q3-q1; lo, hi = q1-1.5*iqr, q3+1.5*iqr
                    out_n = int(((vals<lo)|(vals>hi)).sum())
                    n = len(vals); sd = vals.std(ddof=1) if n>1 else 0.0; se = sd/np.sqrt(max(n,1))
                    z = 1.96; ci_lo, ci_hi = (vals.mean()-z*se, vals.mean()+z*se)
                    stats_rows.append({
                        "group":g, "hue":None, "n":n, "mean":vals.mean(), "sd":sd, "se":se,
                        "ci95_lo":ci_lo, "ci95_hi":ci_hi, "median":med, "q1":q1, "q3":q3, "iqr":iqr,
                        "min":float(vals.min()), "max":float(vals.max()), "outliers":out_n
                    })
            face = _blend_to_white((_OI[0] if palette is None else pal[0]), saturation)
            bp = ax.boxplot(
                dat, positions=centers, widths=width*0.7, patch_artist=True,
                showmeans=True, meanline=True, notch=True, whis=whis, vert=vert,
                showfliers=pro_show_fliers,
                flierprops=flier_kw, medianprops=median_kw, meanprops=mean_kw,
                whiskerprops=whisk_kw, capprops=cap_kw, boxprops=box_kw
            )
            for b in bp["boxes"]:
                b.set_facecolor(face); b.set_alpha(face_alpha); b.set_edgecolor(edge_color)
            for gi, g in enumerate(g_levels):
                pos_map[g] = float(centers[gi])
            if vert:
                ax.set_xticks(centers); ax.set_xticklabels(g_levels)
            else:
                ax.set_yticks(centers); ax.set_yticklabels(g_levels)

        # ========== 均值点 + 抖点（自适应）==========
        rng = np.random.default_rng(0)
        for key, x0 in pos_map.items():
            if h_levels is not None:
                g,hv = key
                sub = data[(data[x]==g) & (data[hue]==hv)] if vert else data[(data[y]==g) & (data[hue]==hv)]
            else:
                g = key
                sub = data[data[x]==g] if vert else data[data[y]==g]
            vals = (sub[y] if vert else sub[x]).dropna().to_numpy()
            if len(vals)==0: continue
            m = float(np.mean(vals))
            if vert: ax.scatter([x0],[m], s=24, zorder=4, edgecolors="#202020", linewidths=0.6)
            else:    ax.scatter([m],[x0], s=24, zorder=4, edgecolors="#202020", linewidths=0.6)

            # 抖点：少量时才画，或用户强制 'strip'
            if pro_points != "none" and (pro_points == "strip" or len(vals) <= pro_points_nmax):
                jtr = 0.04 if (h_levels is not None and dodge) else 0.03
                if vert:
                    ax.scatter(rng.normal(x0, jtr, size=len(vals)), vals,
                               s=pro_points_size, alpha=pro_points_alpha, linewidths=0)
                else:
                    ax.scatter(vals, rng.normal(x0, jtr, size=len(vals)),
                               s=pro_points_size, alpha=pro_points_alpha, linewidths=0)

        # ========== n 标注（更克制）==========
        lo_lim, hi_lim = (ax.get_ylim() if vert else ax.get_xlim())
        base = lo_lim
        if h_levels is not None and pro_n_where == "group":
            # 每个组只标一次
            for gi, g in enumerate(g_levels):
                if vert:
                    xs = [pos_map[(g,hv)] for hv in h_levels if (g,hv) in pos_map]
                    if not xs: continue
                    x_center = float(np.mean(xs))
                    n = int(data[data[x]==g].shape[0])
                    ax.text(x_center, base, f"n={n}", ha="center", va="bottom", fontsize=fontsize, color="dimgray")
                else:
                    ys = [pos_map[(g,hv)] for hv in h_levels if (g,hv) in pos_map]
                    if not ys: continue
                    y_center = float(np.mean(ys))
                    n = int(data[data[y]==g].shape[0])
                    ax.text(base, y_center, f"n={n}", va="center", ha="left", fontsize=fontsize, color="dimgray")
        else:
            for key, x0 in pos_map.items():
                if h_levels is not None:
                    g,hv = key; sub = data[(data[x]==g) & (data[hue]==hv)] if vert else data[(data[y]==g) & (data[hue]==hv)]
                else:
                    g = key; sub = data[data[x]==g] if vert else data[data[y]==g]
                n = sub.shape[0]
                if vert: ax.text(x0, base, f"n={n}", ha="center", va="bottom", fontsize=fontsize, color="dimgray")
                else:    ax.text(base, x0, f"n={n}", va="center", ha="left", fontsize=fontsize, color="dimgray")

        # ========== mean±SE 注释（自适应显示）==========
        _num_boxes = len(g_levels) * (len(h_levels) if h_levels else 1)
        _show_stats = (pro_show_stats is True) or (pro_show_stats=="auto" and _num_boxes <= 6)
        if _show_stats and len(stats_rows)>0:
            df_stats = pd.DataFrame(stats_rows)
            lo, hi = (ax.get_ylim() if vert else ax.get_xlim()); span = hi - lo
            for row in df_stats.itertuples(index=False):
                key = (row.group, row.hue) if h_levels is not None else row.group
                if key not in pos_map: continue
                x0 = pos_map[key]
                txt = pro_stats_fmt.format(**row._asdict())
                if vert:
                    y_txt = row.q3 + 0.02*span
                    ax.text(x0, y_txt, txt, ha="center", va="bottom", fontsize=fontsize, color=pro_stats_color)
                else:
                    x_txt = row.q3 + 0.02*span
                    ax.text(x_txt, x0, txt, va="center", ha="left", fontsize=fontsize, color=pro_stats_color)

        # 轴标签
        if vert: ax.set_xlabel(x if x else "", fontsize=fontsize); ax.set_ylabel(y if y else "", fontsize=fontsize)
        else:    ax.set_xlabel(x if x else "", fontsize=fontsize); ax.set_ylabel(y if y else "", fontsize=fontsize)

        # 只在数值轴画细网格（更清爽）
        if vert:
            ax.yaxis.grid(True, linestyle="--", alpha=0.18)
        else:
            ax.xaxis.grid(True, linestyle="--", alpha=0.18)

        # 保存描述性统计表到 Axes（给调用者取用）
        ax._stats_table = pd.DataFrame(stats_rows) if stats_rows else pd.DataFrame()

        # ========== 自动显著性（只画显著 & 限制条数）==========
        ax._pairwise_table = pd.DataFrame()
        if pro_auto_sig:
            # 选择比较对
            if h_levels is None:
                scope = "adjacent" if pro_auto_sig_scope=="auto" else pro_auto_sig_scope
                pairs = ([(g_levels[i], g_levels[i+1]) for i in range(len(g_levels)-1)]
                         if scope=="adjacent" else _all_pairs(g_levels))
            else:
                scope = "within" if pro_auto_sig_scope in ("auto","within") else pro_auto_sig_scope
                if scope=="within":
                    hpairs = _all_pairs(h_levels) if len(h_levels)>2 else [(h_levels[0], h_levels[1])]
                    pairs = [((g,h1),(g,h2)) for g in g_levels for (h1,h2) in hpairs]
                else:
                    allk = [(g,h) for g in g_levels for h in h_levels]; pairs = _all_pairs(allk)

            from scipy import stats
            rows, p_raw = [], []

            def _vals(key):
                if h_levels is None:
                    sub = data[data[x]==key] if vert else data[data[y]==key]
                else:
                    g,hv = key
                    sub = data[(data[x]==g) & (data[hue]==hv)] if vert else data[(data[y]==g) & (data[hue]==hv)]
                return (sub[y] if vert else sub[x]).dropna().to_numpy()

            for (k1,k2) in pairs:
                v1, v2 = _vals(k1), _vals(k2)
                if len(v1)==0 or len(v2)==0:
                    stat, p = np.nan, np.nan
                else:
                    if pro_sig_test=="ttest":
                        stat, p = stats.ttest_ind(v1, v2, equal_var=pro_equal_var)
                    elif pro_sig_test=="mw":
                        stat, p = stats.mannwhitneyu(v1, v2, alternative="two-sided"); stat=float(stat)
                    else:
                        raise ValueError("pro_sig_test 仅支持 'ttest' 或 'mw'")
                p_raw.append(p)
                eff = None
                if np.isfinite(p) and pro_effect:
                    eff = _hedges_g(v1,v2) if pro_effect=="hedges_g" else _cohen_d(v1,v2)
                rows.append({"k1":k1,"k2":k2,"stat":stat,"p_raw":p,"effect":eff,"effect_type":pro_effect})

            dfp = pd.DataFrame(rows)
            # 多重校正
            if pro_sig_correction=="bonferroni":
                m = dfp["p_raw"].notna().sum()
                dfp["p_adj"] = np.minimum(dfp["p_raw"] * max(m,1), 1.0)
            elif pro_sig_correction=="fdr_bh":
                ps = dfp["p_raw"].values.copy()
                valid = np.isfinite(ps)
                q = np.full_like(ps, np.nan, dtype=float)
                if valid.any():
                    idx = np.where(valid)[0]
                    pv = ps[valid]
                    order = np.argsort(pv, kind="mergesort")
                    ranks = np.empty_like(order); ranks[order] = np.arange(len(pv)) + 1
                    qv = pv * len(pv) / np.maximum(ranks, 1)
                    # 保持单调
                    for i in range(len(qv)-2, -1, -1):
                        qv[order[i]] = min(qv[order[i]], qv[order[i+1]])
                    q[valid] = np.minimum(qv, 1.0)
                dfp["p_adj"] = q
            else:
                dfp["p_adj"] = dfp["p_raw"]

            # 过滤只保留显著 & 限制数量
            dfp = dfp[np.isfinite(dfp["p_adj"])]
            dfp = dfp.sort_values("p_adj")
            dfp = dfp[dfp["p_adj"] <= pro_sig_alpha]
            if pro_sig_max_pairs:
                dfp = dfp.head(int(pro_sig_max_pairs))
            ax._pairwise_table = dfp.reset_index(drop=True)

            # 画括线（跨度长的先画，减少遮挡）
            def _p_text(p):
                return (f"p={p:.3g}" if pro_sig_mode=="p"
                        else ("****" if p<1e-4 else ("***" if p<1e-3 else ("**" if p<1e-2 else ("*" if p<5e-2 else "ns")))))

            if not dfp.empty:
                y_min, y_max = ax.get_ylim(); span = y_max - y_min; base_top = y_max
                dfp["_span"] = dfp[["k1","k2"]].apply(lambda s: abs(pos_map.get(s["k2"],0) - pos_map.get(s["k1"],0)), axis=1)
                dfp = dfp.sort_values(["_span","p_adj"], ascending=[False, True])

                used = []
                for _, r in dfp.iterrows():
                    k1, k2 = r["k1"], r["k2"]; p = float(r["p_adj"])
                    if (k1 not in pos_map) or (k2 not in pos_map): continue
                    x1, x2 = pos_map[k1], pos_map[k2]
                    # 层叠避让
                    layer = 0
                    while any((min(x1,x2)<=hi and max(x1,x2)>=lo) for (lo,hi,lv) in used if lv==layer):
                        layer += 1
                    used.append((min(x1,x2), max(x1,x2), layer))
                    level = base_top + (layer+1) * span * 0.035
                    h = span * 0.022
                    ax.plot([x1,x1,x2,x2], [level,level+h,level+h,level], c="k", lw=1.1)
                    label = _p_text(p)
                    if (r.get("effect_type") in ("cohen_d","hedges_g")) and np.isfinite(r.get("effect")):
                        label += f", d={float(r['effect']):.2f}"
                    ax.text((x1+x2)/2, level+h, label, ha="center", va="bottom")
                if used:
                    ax.set_ylim(y_min, base_top + (max(l for _,_,l in used)+3)*span*0.04)

        return ax

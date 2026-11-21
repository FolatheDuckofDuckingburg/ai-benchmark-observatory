#!/usr/bin/env python3
"""Independent audit of AI Benchmark Observatory (chat vs pretrained).

Usage: python audit_analysis.py path/to/open_llm_leaderboard_snapshot.tsv [outdir]

The raw TSV is read only. All outputs go to outdir (default: audit_out).
STATUS: written and smoke-tested on SYNTHETIC data only. It has not been run
on the real snapshot in the session that produced it.
"""
import sys, hashlib, platform, re
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CHAT = "💬 chat models (RLHF, DPO, IFT, ...)"
BASE = "🟢 pretrained"
# Column names exactly as in the TSV header (note: MUSR, not MuSR).
BENCH = ["MMLU-PRO", "GPQA", "MATH Lvl 5", "BBH", "MUSR", "IFEval"]


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def hedges_g(a, b):
    n1, n2 = len(a), len(b)
    sp = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    d = (a.mean() - b.mean()) / sp
    j = 1 - 3 / (4 * (n1 + n2) - 9)
    g = d * j
    se = np.sqrt((n1 + n2) / (n1 * n2) + g**2 / (2 * (n1 + n2)))
    return g, g - 1.96 * se, g + 1.96 * se


def welch(a, b):
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
    diff = a.mean() - b.mean()
    t = diff / se
    p = 2 * stats.t.sf(abs(t), df)
    crit = stats.t.ppf(0.975, df)
    return diff, t, df, p, diff - crit * se, diff + crit * se


def family_key(model_id):
    """HEURISTIC family key: strip common chat/instruct suffixes. The TSV's
    'Base Model' column is NOT reliable for pairing (it can equal the model's own id)."""
    s = model_id.lower()
    s = re.sub(r"[-_.](chat|instruct|it|dpo|sft|rlhf|v\d+(\.\d+)*)(?=$|[-_.])", "", s)
    return s


def main(tsv, out="audit_out"):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    raw_bytes = Path(tsv).read_bytes()
    log = [f"python {platform.python_version()} pandas {pd.__version__} numpy {np.__version__} scipy {scipy.__version__}",
           f"sha256 {hashlib.sha256(raw_bytes).hexdigest()}"]
    raw = pd.read_csv(tsv, sep="\t", dtype=str, low_memory=False)
    log.append(f"source rows {len(raw)}, columns {raw.shape[1]}")
    need = {"Type", "fullname", "Flagged", "#Params (B)", *BENCH}
    miss = need - set(raw.columns)
    if miss:
        raise ValueError(f"Missing columns: {sorted(miss)}")

    d = raw.copy()
    d["Type"] = d["Type"].str.strip(); d["fullname"] = d["fullname"].str.strip()
    log.append("Type counts (all rows):\n" + d["Type"].value_counts(dropna=False).to_string())
    d = d[d["Type"].isin([CHAT, BASE])]
    d["flag"] = d["Flagged"].str.strip().str.lower().map({"true": True, "false": False})
    log.append(f"rows with unparseable/missing Flagged (excluded): {int(d['flag'].isna().sum())}")
    log.append(f"rows flagged TRUE (excluded): {int((d['flag'] == True).sum())}")
    d = d[d["flag"] == False]
    d = d[d["fullname"].notna() & (d["fullname"] != "")]
    for b in BENCH + ["#Params (B)"]:
        d[b] = pd.to_numeric(d[b], errors="coerce")
    d = d[d[BENCH].notna().any(axis=1)]

    # Classification ambiguity: same repo labelled with both types
    both = d.groupby("fullname")["Type"].nunique()
    amb = both[both > 1].index.tolist()
    pd.Series(amb, name="fullname").to_csv(out / "ambiguous_classification.csv", index=False)
    log.append(f"repos labelled both chat and pretrained: {len(amb)}")

    clean = (d.groupby(["fullname", "Type"], as_index=False)
               .agg({**{b: "mean" for b in BENCH}, "#Params (B)": "first"}))
    clean["rows"] = d.groupby(["fullname", "Type"]).size().values
    clean["family"] = clean["fullname"].map(family_key)
    clean.to_csv(out / "cleaned_models.csv", index=False)
    log.append("consolidated models by Type:\n" + clean["Type"].value_counts().to_string())

    # Primary: per-benchmark Welch, g, MWU, Holm
    rows = []
    for b in BENCH:
        c = clean.loc[clean.Type == CHAT, b].dropna().to_numpy()
        p = clean.loc[clean.Type == BASE, b].dropna().to_numpy()
        r = {"benchmark": b, "n_chat": len(c), "n_pre": len(p)}
        if len(c) > 2 and len(p) > 2:
            diff, t, df, pv, lo, hi = welch(c, p)
            g, glo, ghi = hedges_g(c, p)
            u = stats.mannwhitneyu(c, p, alternative="two-sided")
            r.update(mean_chat=c.mean(), mean_pre=p.mean(), median_chat=np.median(c), median_pre=np.median(p),
                     sd_chat=c.std(ddof=1), sd_pre=p.std(ddof=1), diff=diff, ci_lo=lo, ci_hi=hi,
                     welch_t=t, welch_df=df, p=pv, hedges_g=g, g_lo=glo, g_hi=ghi, mwu_p=u.pvalue,
                     cles=u.statistic / (len(c) * len(p)),
                     shapiro_p_chat=stats.shapiro(c[:5000]).pvalue, shapiro_p_pre=stats.shapiro(p[:5000]).pvalue,
                     levene_p=stats.levene(c, p).pvalue)
        rows.append(r)
    res = pd.DataFrame(rows)
    ok = res["p"].notna()
    res.loc[ok, "p_holm"] = holm(res.loc[ok, "p"])
    res.loc[ok, "mwu_p_holm"] = holm(res.loc[ok, "mwu_p"])
    res.to_csv(out / "benchmark_results.csv", index=False)

    # Confounding: parameter-size strata (exploratory)
    srows = []
    clean["size_bin"] = pd.cut(clean["#Params (B)"], [0, 3, 8, 15, 40, 1e9], labels=["<=3B", "3-8B", "8-15B", "15-40B", ">40B"])
    for b in BENCH:
        for sb, g_ in clean.groupby("size_bin", observed=True):
            c = g_.loc[g_.Type == CHAT, b].dropna(); p = g_.loc[g_.Type == BASE, b].dropna()
            srows.append({"benchmark": b, "size_bin": sb, "n_chat": len(c), "n_pre": len(p),
                          "diff": c.mean() - p.mean() if len(c) and len(p) else np.nan})
    pd.DataFrame(srows).to_csv(out / "size_stratified.csv", index=False)

    # Matched family pairs (heuristic; exploratory). Paired t and Wilcoxon on within-family differences.
    fam = clean.groupby(["family", "Type"])[BENCH].mean().unstack("Type")
    prows = []
    for b in BENCH:
        pair = pd.concat([fam[(b, CHAT)], fam[(b, BASE)]], axis=1, keys=["chat", "pre"]).dropna()
        r = {"benchmark": b, "n_pairs": len(pair)}
        if len(pair) > 2:
            dd = pair.chat - pair.pre
            r.update(mean_diff=dd.mean(), median_diff=dd.median(),
                     paired_t_p=stats.ttest_rel(pair.chat, pair.pre).pvalue,
                     wilcoxon_p=stats.wilcoxon(dd).pvalue if (dd != 0).any() else np.nan,
                     frac_chat_higher=(dd > 0).mean())
        prows.append(r)
    pr = pd.DataFrame(prows); ok = pr.get("paired_t_p", pd.Series(dtype=float)).notna()
    if ok.any():
        pr.loc[ok, "paired_t_p_holm"] = holm(pr.loc[ok, "paired_t_p"])
    pr.to_csv(out / "matched_pairs.csv", index=False)

    # Cross-benchmark Spearman correlations
    clean[BENCH].corr(method="spearman").to_csv(out / "benchmark_spearman.csv")

    # Plots
    fig, ax = plt.subplots(2, 3, figsize=(13, 7))
    for a, b in zip(ax.ravel(), BENCH):
        a.hist([clean.loc[clean.Type == CHAT, b].dropna(), clean.loc[clean.Type == BASE, b].dropna()],
               bins=30, label=["chat", "pretrained"], alpha=0.7); a.set_title(b); a.legend()
    fig.tight_layout(); fig.savefig(out / "distributions.png", dpi=130); plt.close(fig)
    r2 = res[res.hedges_g.notna()] if "hedges_g" in res else res.iloc[0:0]
    if len(r2):
        fig, a = plt.subplots(figsize=(6, 4))
        a.errorbar(r2.hedges_g, range(len(r2)), xerr=[r2.hedges_g - r2.g_lo, r2.g_hi - r2.hedges_g], fmt="o")
        a.set_yticks(range(len(r2))); a.set_yticklabels(r2.benchmark); a.axvline(0, color="k", lw=.5)
        a.set_xlabel("Hedges' g (chat - pretrained), 95% CI"); fig.tight_layout(); fig.savefig(out / "effect_sizes.png", dpi=130)
        plt.close(fig)

    (out / "run_log.txt").write_text("\n".join(log))
    print("\n".join(log)); print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "audit_out")

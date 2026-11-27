"""Belebele analysis (v5): analysis.py's primary tests on one benchmark with many more languages.
Reads {RESULTS_DIR}/{model}.json from belebele.py and metrics.csv / proportions.csv from
metrics_belebele.py. Prints the report (save it as analysis.txt) and writes belebele_zdata.csv,
one row per model and language with a within-model z-score, for pooling with results_v2/zdata.csv.
Unit of analysis: one seen, non-English language. English is the premium's reference and the
harness check, as English COPA is in the main study."""
import json, os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from belebele_langs import family, seen
from stats_tools import boot_ci, min_detectable, perm_p

SAVE = os.environ.get("RESULTS_DIR", "results_belebele")
TAGS = ["xglm-564M", "xglm-1.7B", "xglm-2.9B", "xglm-7.5B", "bloom-560m", "bloom-1b7", "bloom-3b", "bloom-7b1"]
FMT = os.environ.get("BELEBELE_FORMAT") or (open(f"{SAVE}/format.txt").read().strip()
                                            if os.path.exists(f"{SAVE}/format.txt") else "cloze")
KEY = f"{FMT}_0shot_s0"
ACC = "acc_mean" if FMT == "cloze" else "acc"
CHANCE_BAND = 25 + 1.96 * np.sqrt(0.25 * 0.75 / 900) * 100     # 95% upper edge of chance for 900 items
METRICS = ["premium", "fertility", "strr"]

rows = []
for tag in TAGS:
    path = f"{SAVE}/{tag}.json"
    if not os.path.exists(path): continue
    for code, st in json.load(open(path)).get(KEY, {}).items():
        rows.append(dict(model=tag, family=family(tag), lang=code, acc=st[ACC],
                         acc_sum=st.get("acc_sum"), seen=seen(tag, code)))
df = pd.DataFrame(rows)
metrics = pd.read_csv(f"{SAVE}/metrics.csv")
props = pd.read_csv(f"{SAVE}/proportions.csv")
df = df.merge(metrics, on=["family", "lang"], how="left").merge(props, on=["family", "lang"], how="left")
work = df[(df.lang != "eng_Latn") & df.seen].copy()

print(f"Format: {FMT} ({KEY}); accuracy column: {ACC}; chance band upper edge {CHANCE_BAND:.1f}%")
print("\n======== HARNESS CHECK: English accuracy, and languages above the chance band ========")
for m in TAGS:
    d = df[df.model == m]
    if d.empty: continue
    en = d[d.lang == "eng_Latn"].acc
    s, u = d[d.seen & (d.lang != "eng_Latn")], d[~d.seen]
    print(f"{m}: English {en.iloc[0] if len(en) else float('nan'):.1f} | seen above chance "
          f"{(s.acc > CHANCE_BAND).sum()}/{len(s)} | unseen above chance {(u.acc > CHANCE_BAND).sum()}/{len(u)}")

print("\n======== PRIMARY: 7B models, 0-shot, one point per seen non-English language ========")
pvals = []
for m in ["xglm-7.5B", "bloom-7b1"]:
    d = work[work.model == m]
    if d.empty: continue
    print(f"\n--- {m} --- data share alone: rho={spearmanr(d.share, d.acc)[0]:.3f} (n={len(d)})")
    for metric in METRICS:
        dd = d.dropna(subset=[metric])
        if len(dd) < 5: continue
        r, p = spearmanr(dd[metric], dd.acc); pvals.append(p)
        lo, hi = boot_ci(dd[metric].values, dd.acc.values)
        pr, pp = perm_p(dd[metric].values, dd.acc.values, dd.share.values)
        extra = ""
        alt = "share_sampled" if family(m) == "xglm" else "share_bp"
        if dd[alt].notna().all():
            ar, ap = perm_p(dd[metric].values, dd.acc.values, dd[alt].values)
            extra = f" | partial|{alt}={ar:.3f} (perm p={ap:.3f})"
        print(f"{metric}: rho={r:.3f} (p={p:.4f}, CI[{lo},{hi}], n={len(dd)}) | partial={pr:.3f} (perm p={pp:.3f}){extra}")
if pvals:
    alpha = 0.05 / len(pvals)
    print(f"\nBonferroni across {len(pvals)} Belebele tests: alpha={alpha:.4f}; "
          f"{sum(p <= alpha for p in pvals)} survive (min p={min(pvals):.4f})")

print("\n======== REVERSE PARTIALS: does data share survive a control for each metric? (7B) ========")
for m in ["xglm-7.5B", "bloom-7b1"]:
    d = work[work.model == m]
    if d.empty: continue
    for metric in METRICS:
        dd = d.dropna(subset=[metric])
        if len(dd) < 5: continue
        pr, pp = perm_p(dd.share.values, dd.acc.values, dd[metric].values)
        print(f"{m} | data share | {metric}: partial={pr:+.3f} (perm p={pp:.3f}, n={len(dd)})")

if FMT == "cloze":
    print("\n======== SUMMED SCORING (Appendix B analogue), 7B ========")
    for m in ["xglm-7.5B", "bloom-7b1"]:
        d = work[work.model == m]
        for metric in METRICS:
            dd = d.dropna(subset=[metric, "acc_sum"])
            if len(dd) < 5: continue
            r, p = spearmanr(dd[metric], dd.acc_sum)
            print(f"{m} | {metric}: rho={r:.3f} (p={p:.4f}, n={len(dd)})")

print("\n======== SCALE: rho with accuracy by model size ========")
for m in TAGS:
    d = work[work.model == m]
    if d.empty: continue
    parts = []
    for metric in METRICS:
        dd = d.dropna(subset=[metric])
        parts.append(f"{metric} {spearmanr(dd[metric], dd.acc)[0]:+.3f} (n={len(dd)})")
    print(f"{m}: " + " | ".join(parts))

print("\n======== SEEN vs UNSEEN (mean accuracy) ========")
for m in TAGS:
    d = df[(df.model == m) & (df.lang != "eng_Latn")]
    if d.empty: continue
    print(f"{m}: seen {d[d.seen].acc.mean():.1f} (n={d.seen.sum()}) | unseen {d[~d.seen].acc.mean():.1f} (n={(~d.seen).sum()})")

print("\n======== POWER at the realized n (80% power, two-sided alpha .05) ========")
for n in sorted({len(work[work.model == m].dropna(subset=[x])) for m in ["xglm-7.5B", "bloom-7b1"]
                 for x in METRICS} - {0}, reverse=True):
    print(f"n={n:2d}: zero-order {min_detectable(n):.2f} | partial, one covariate {min_detectable(n, 1):.2f}")

z = []
for m, g in work.groupby("model"):
    if len(g) < 3 or g.acc.std() == 0: continue
    g = g.assign(z=(g.acc - g.acc.mean()) / g.acc.std(), benchmark="belebele")
    z.append(g[["model", "benchmark", "lang", "acc", "z", "premium", "fertility", "strr", "vocab_use", "share"]])
if z:
    pd.concat(z).to_csv(f"{SAVE}/belebele_zdata.csv", index=False)
    print(f"\nbelebele_zdata.csv written to {SAVE}")

"""Full analysis: seed averaging, within-benchmark z-scoring, Spearman + 10k bootstrap CI,
partial Spearman controlling training-data share + 5k permutation p, Bonferroni note,
scale table, seen/unseen table, per-benchmark check, metric intercorrelations."""
import json, os
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata

SAVE = "results"
TAGS = ["xglm-564M","xglm-1.7B","xglm-2.9B","xglm-7.5B",
        "bloom-560m","bloom-1b7","bloom-3b","bloom-7b1"]
SEEN_X = {"et","ht","id","it","sw","ta","th","tr","vi","zh","qu","ru","es","ar","hi","te","eu","my"}
SEEN_B = {"id","ta","vi","zh","sw","es","ar","hi","te","eu"}
def seen(tag, lang): return lang in (SEEN_B if tag.startswith("bloom") else SEEN_X)

rows = []
for tag in TAGS:
    path = f"{SAVE}/{tag}.json"
    if not os.path.exists(path): continue
    for key, val in json.load(open(path)).items():
        parts = key.split("_")
        if not parts[-1].startswith("s"): continue
        seed, shot = int(parts[-1][1:]), int(parts[-2][0])
        bench, lang = ("encopa", "en") if key.startswith("en_copa") else (parts[0], parts[1])
        rows.append(dict(model=tag, benchmark=bench, lang=lang, shot=shot, seed=seed, acc=val))
df = pd.DataFrame(rows)
avg = df.groupby(["model","benchmark","lang","shot"], as_index=False).agg(
    accuracy=("acc","mean"), sd=("acc","std"), runs=("acc","size"))
avg.to_csv(f"{SAVE}/master_seeds.csv", index=False)

metrics = pd.read_csv(f"{SAVE}/metrics.csv")
props = pd.read_csv(f"{SAVE}/proportions.csv")

work = avg[avg.benchmark != "encopa"].copy()
work["family"] = work.model.str.split("-").str[0]
work = work.merge(metrics.drop(columns=["model"]), on=["family","lang"], how="left") \
           .merge(props.drop(columns=["model"]), on=["family","lang"], how="left")
wseen = work[[seen(m, l) for m, l in zip(work.model, work.lang)]]

zrows = []
for (m, b, s), g in wseen.groupby(["model","benchmark","shot"]):
    if len(g) < 3 or g.accuracy.std() == 0: continue
    z = (g.accuracy - g.accuracy.mean()) / g.accuracy.std()
    for lang, zv in zip(g.lang, z): zrows.append(dict(model=m, lang=lang, shot=s, z=zv))
zdf = pd.DataFrame(zrows).groupby(["model","lang","shot"], as_index=False).z.mean()
data = zdf.merge(wseen.drop_duplicates(["model","lang"])[["model","lang","premium","fertility","strr","vocab_use","proportion"]],
                 on=["model","lang"])

def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    rxy, rxz, ryz = spearmanr(rx,ry)[0], spearmanr(rx,rz)[0], spearmanr(ry,rz)[0]
    return (rxy - rxz*ryz) / np.sqrt((1-rxz**2)*(1-ryz**2))

def boot_ci(x, y, n=10000):
    rng = np.random.default_rng(0); idx = np.arange(len(x)); cs = []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        cs.append(spearmanr(np.array(x)[s], np.array(y)[s])[0])
    return np.nanpercentile(cs, [2.5, 97.5]).round(3)

def perm_p(x, y, z, n=5000):
    obs = partial_spearman(x, y, z)
    rng = np.random.default_rng(0); yv = np.array(y); cnt = 0
    for _ in range(n):
        cnt += abs(partial_spearman(x, rng.permutation(yv), z)) >= abs(obs)
    return obs, (cnt + 1) / (n + 1)

print("======== PRIMARY: 7B models, seed-averaged, one point per seen language ========")
pvals = []
for m in ["xglm-7.5B","bloom-7b1"]:
    for s in [0, 4]:
        d = data[(data.model == m) & (data.shot == s)]
        print(f"\n--- {m} | {s}-shot ---")
        for metric in ["premium","fertility","strr"]:
            dd = d.dropna(subset=[metric])
            if len(dd) < 5: continue
            r, p = spearmanr(dd[metric], dd.z); pvals.append(p)
            lo, hi = boot_ci(dd[metric].values, dd.z.values)
            pr, pp = perm_p(dd[metric].values, dd.z.values, dd.proportion.values)
            print(f"{metric}: rho={r:.3f} (p={p:.3f}, CI[{lo},{hi}], n={len(dd)}) | partial={pr:.3f} (perm p={pp:.3f})")
alpha = 0.05/len(pvals)
print(f"\nBonferroni across {len(pvals)} primary tests: alpha={alpha:.4f}; "
      f"{sum(p <= alpha for p in pvals)} survive (min p={min(pvals):.3f})")

print("\n======== SCALE: 4-shot rho by model size ========")
en = avg[(avg.benchmark == "encopa") & (avg.shot == 4)].set_index("model").accuracy
for m in TAGS:
    d = data[(data.model == m) & (data.shot == 4)]
    line = f"{m:<12} en-copa={en.get(m, float('nan')):.1f}"
    for metric in ["premium","fertility","strr"]:
        dd = d.dropna(subset=[metric])
        if len(dd) >= 5: line += f" | {metric} rho={spearmanr(dd[metric], dd.z)[0]:+.3f}"
    print(line)

print("\n======== SEEN vs UNSEEN (4-shot XCOPA) ========")
xc = avg[(avg.benchmark == "xcopa") & (avg.shot == 4)]
for m in TAGS:
    d = xc[xc.model == m]
    sn = d[[seen(m, l) for l in d.lang]].accuracy
    un = d[[not seen(m, l) for l in d.lang]].accuracy
    if len(d): print(f"{m:<12} seen mean={sn.mean():.1f} (n={len(sn)}) | unseen mean={un.mean():.1f} (n={len(un)})")

print("\n======== PER-BENCHMARK: 7B, 4-shot ========")
for m in ["xglm-7.5B","bloom-7b1"]:
    for b in ["xcopa","xsc"]:
        g = wseen[(wseen.model==m)&(wseen.benchmark==b)&(wseen.shot==4)]
        for metric in ["premium","fertility","strr"]:
            gg = g.dropna(subset=[metric])
            if len(gg) >= 5:
                r, p = spearmanr(gg[metric], gg.accuracy)
                print(f"{m} | {b} | {metric}: rho={r:+.3f} (p={p:.3f}, n={len(gg)})")

print("\n======== METRIC INTERCORRELATIONS ========")
for fam in ["xglm","bloom"]:
    mm = metrics[metrics.family==fam].merge(props[["family","lang","proportion"]], on=["family","lang"], how="left")
    print(f"\n--- {fam} ---")
    print(mm[["premium","fertility","strr","vocab_use","proportion"]].corr(method="spearman").round(2).to_string())

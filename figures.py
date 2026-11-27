"""The paper's two figures, computed from the result files. Run analysis.py first (it writes zdata.csv).
v2 (Nov 2026): v1 hard-coded the plotted values; they are now read from {RESULTS_DIR} (default results_v2)."""
import os, sys
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

SAVE = os.environ.get("RESULTS_DIR", "results_v2")
for f in ["master_seeds.csv", "zdata.csv"]:
    if not os.path.exists(f"{SAVE}/{f}"):
        sys.exit(f"{SAVE}/{f} not found: run analysis.py first (same RESULTS_DIR).")
ms = pd.read_csv(f"{SAVE}/master_seeds.csv")
zd = pd.read_csv(f"{SAVE}/zdata.csv")
SIZES = {"xglm-564M": 0.56, "xglm-1.7B": 1.7, "xglm-2.9B": 2.9, "xglm-7.5B": 7.5,
         "bloom-560m": 0.56, "bloom-1b7": 1.7, "bloom-3b": 3.0, "bloom-7b1": 7.1}
SEEN_B = {"id","ta","vi","zh","sw","es","ar","hi","te","eu"}

def strr_rho(tag):
    d = zd[(zd.model == tag) & (zd.shot == 4)].dropna(subset=["strr"])
    return spearmanr(d.strr, d.z)[0] if len(d) >= 5 else float("nan")

plt.figure(figsize=(6,4))
for fam, label in [("xglm", "XGLM"), ("bloom", "BLOOM")]:
    tags = [t for t in SIZES if t.startswith(fam)]
    rhos = [float(strr_rho(t)) for t in tags]
    print(f"Figure 1 | {label} STRR rho by size:", [round(r, 3) for r in rhos])
    plt.plot([SIZES[t] for t in tags], rhos, marker="o", label=label)
plt.axhline(0, color="gray", lw=.5)
plt.xlabel("Model size (billions of parameters)"); plt.ylabel("STRR vs accuracy (Spearman rho)")
plt.title("STRR-accuracy correlation by model size (4-shot)")
plt.legend(); plt.tight_layout(); plt.savefig(f"{SAVE}/fig1_scale.png", dpi=200)

btags = [t for t in SIZES if t.startswith("bloom")]
xc = ms[(ms.benchmark == "xcopa") & (ms.shot == 4)]
seen = [float(xc[(xc.model == t) & xc.lang.isin(SEEN_B)].accuracy.mean()) for t in btags]
unseen = [float(xc[(xc.model == t) & ~xc.lang.isin(SEEN_B)].accuracy.mean()) for t in btags]
n_seen = xc[xc.model == btags[-1]].lang.isin(SEEN_B).sum()
n_unseen = (~xc[xc.model == btags[-1]].lang.isin(SEEN_B)).sum()
print("Figure 2 | BLOOM XCOPA 4-shot seen:", [round(v, 1) for v in seen], "unseen:", [round(v, 1) for v in unseen])
plt.figure(figsize=(6,4))
plt.plot([SIZES[t] for t in btags], seen, marker="o", label=f"Seen languages (n={n_seen})")
plt.plot([SIZES[t] for t in btags], unseen, marker="s", label=f"Unseen languages (n={n_unseen})")
plt.axhline(50, color="gray", ls="--", lw=.8, label="Chance")
plt.xlabel("BLOOM size (billions of parameters)"); plt.ylabel("XCOPA accuracy (%)")
plt.title("Scale helps seen languages only")
plt.legend(); plt.tight_layout(); plt.savefig(f"{SAVE}/fig2_seen_unseen.png", dpi=200)
print(f"figures written to {SAVE}/")

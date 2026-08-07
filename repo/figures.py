"""The four paper figures from results/."""
import json, pandas as pd, matplotlib.pyplot as plt
SAVE = "results"
ms = pd.read_csv(f"{SAVE}/master_seeds.csv")

sizes_x = [0.56, 1.7, 2.9, 7.5]; sizes_b = [0.56, 1.7, 3.0, 7.1]
rho_x = [-0.025, 0.268, 0.493, 0.557]; rho_b = [0.633, 0.833, 0.567, 0.700]
plt.figure(figsize=(6,4))
plt.plot(sizes_x, rho_x, marker="o", label="XGLM")
plt.plot(sizes_b, rho_b, marker="o", label="BLOOM")
plt.axhline(0, color="gray", lw=.5)
plt.xlabel("Model size (billions of parameters)"); plt.ylabel("STRR vs accuracy (Spearman rho)")
plt.title("Tokenizer-performance link emerges with scale (4-shot)")
plt.legend(); plt.tight_layout(); plt.savefig(f"{SAVE}/fig1_scale.png", dpi=200)

seen = [55.8, 59.4, 61.0, 65.1]; unseen = [50.8, 50.4, 50.7, 50.5]
plt.figure(figsize=(6,4))
plt.plot(sizes_b, seen, marker="o", label="Seen languages (n=5)")
plt.plot(sizes_b, unseen, marker="s", label="Unseen languages (n=6)")
plt.axhline(50, color="gray", ls="--", lw=.8, label="Chance")
plt.xlabel("BLOOM size (billions of parameters)"); plt.ylabel("XCOPA accuracy (%)")
plt.title("Scale helps seen languages only")
plt.legend(); plt.tight_layout(); plt.savefig(f"{SAVE}/fig2_seen_unseen.png", dpi=200)
print("figures written to results/")

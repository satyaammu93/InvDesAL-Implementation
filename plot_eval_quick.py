"""Plot side-by-side eval_quick results across checkpoints."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# (label, path-to-eval_quick.json, short note)
RUNS = [
    ("gen_1k",         "eval_quick.json",            "Entry 6 baseline"),
    ("gen_10k",        "eval_quick_10k.json",        "Entry 7"),
    ("gen_10k_r1",     "eval_quick_10k_r1.json",     "+1h resume"),
    ("gen_10k_r2",     "eval_quick_10k_r2.json",     "+2h resume"),
    ("gen_10k_ax0",    "eval_quick_10k_ax0.json",    "A x0 (Entry 8)"),
    ("generator",      "eval_quick_overnight.json",  "Entry 3 (pre-fix)"),
]

data = []
for label, path, note in RUNS:
    d = json.loads(Path(path).read_text())
    data.append({
        "label": label, "note": note,
        "val": d["ckpt_val_loss"],
        "loss_total": d["heldout_loss"]["total"],
        "loss_A": d["heldout_loss"]["type"],
        "loss_L": d["heldout_loss"]["lattice"],
        "loss_F": d["heldout_loss"]["coord"],
        "unique": d["sample"]["unique_rate"],
        "vpa_med": d["sample"]["vpa_median"],
        "vpa_max": d["sample"]["vpa_max"],
        "sane": d["sample"]["sane_fraction"],
    })

labels = [r["label"] for r in data]
x = np.arange(len(labels))
colors = ["#4c72b0", "#dd8452", "#dd8452", "#dd8452", "#55a467", "#c44e52"]

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("eval_quick: side-by-side checkpoint comparison "
             "(N=64 sample, N=256 held-out loss)", fontsize=13)

# --- (a) val_loss + held-out total loss ---
ax = axes[0, 0]
w = 0.38
ax.bar(x - w/2, [r["val"] for r in data], w, color=colors, label="ckpt val_loss",
       edgecolor="black", linewidth=0.5)
ax.bar(x + w/2, [r["loss_total"] for r in data], w,
       color=colors, alpha=0.5, label="held-out total",
       edgecolor="black", linewidth=0.5, hatch="//")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=20, ha="right")
ax.set_ylabel("loss")
ax.set_title("(a) val_loss vs held-out total loss")
ax.set_yscale("log")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)

# --- (b) per-channel held-out loss (gen.ckpt excluded as pre-fix parametrization) ---
ax = axes[0, 1]
keep = [i for i, r in enumerate(data) if r["label"] != "generator"]
xk = np.arange(len(keep))
w = 0.27
ax.bar(xk - w, [data[i]["loss_A"] for i in keep], w,
       label="A (type)", color="#4c72b0", edgecolor="black", linewidth=0.5)
ax.bar(xk,     [data[i]["loss_L"] for i in keep], w,
       label="L (lattice)", color="#55a467", edgecolor="black", linewidth=0.5)
ax.bar(xk + w, [data[i]["loss_F"] for i in keep], w,
       label="F (coord)", color="#dd8452", edgecolor="black", linewidth=0.5)
ax.set_xticks(xk)
ax.set_xticklabels([labels[i] for i in keep], rotation=20, ha="right")
ax.set_ylabel("MSE")
ax.set_title("(b) per-channel held-out loss\n(generator.ckpt excluded — pre-fix L param)")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)

# --- (c) unique rate @ N=64 ---
ax = axes[1, 0]
bars = ax.bar(x, [r["unique"] for r in data], color=colors,
              edgecolor="black", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=20, ha="right")
ax.set_ylabel("chemical-formula unique rate")
ax.set_title("(c) unique rate @ N=64 (paper Fig S.4 @ N=1000: 0.992)")
ax.set_ylim(0.85, 1.02)
ax.axhline(0.992, color="black", linestyle="--", linewidth=1,
           label="paper @ N=1000 (0.992)")
for b, r in zip(bars, data):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.003,
            f"{r['unique']:.3f}", ha="center", fontsize=8)
ax.legend(fontsize=9, loc="lower right")
ax.grid(axis="y", alpha=0.3)

# --- (d) vpa median (with data reference band) ---
ax = axes[1, 1]
bars = ax.bar(x, [r["vpa_med"] for r in data], color=colors,
              edgecolor="black", linewidth=0.5)
ax.axhspan(15, 25, color="green", alpha=0.12, label="data ~21 Å³ (±5)")
ax.axhline(21, color="green", linestyle="--", linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=20, ha="right")
ax.set_ylabel("volume/atom median (Å³)")
ax.set_title("(d) lattice sanity: vpa median (sampled N=64)")
for b, r in zip(bars, data):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
            f"{r['vpa_med']:.1f}", ha="center", fontsize=8)
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
out = "eval_quick_compare.png"
plt.savefig(out, dpi=140, bbox_inches="tight")
print(f"wrote {out}")

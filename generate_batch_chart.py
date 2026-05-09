import pandas as pd
import matplotlib.pyplot as plt
import os

# ─────────────────────────────────────────────
# CONFIG — change paths if needed
# ─────────────────────────────────────────────
CSV_BS2 = "training_log_exp1_bs2.csv"
CSV_BS4 = "training_log.csv"
CSV_BS8 = "training_log_exp1_bs8.csv"
OUTPUT  = "chart_batch_comparison.png"

# ─────────────────────────────────────────────
# COLORS
# ─────────────────────────────────────────────
BG   = "#F5F0EB"
C_B2 = "#C0504D"   # red   — BS=2
C_B4 = "#3D6B9E"   # blue  — BS=4
C_B8 = "#7B68A8"   # purple— BS=8
DARK = "#2B2B2B"

# ─────────────────────────────────────────────
# LOAD & CLEAN
# ─────────────────────────────────────────────
def load(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    df = df.sort_values("epoch").drop_duplicates("epoch").reset_index(drop=True)
    return df

bs2 = load(CSV_BS2)
bs4 = load(CSV_BS4)
bs8 = load(CSV_BS8)

# Limit all to 50 epochs for fair comparison
bs2 = bs2[bs2["epoch"] <= 50]
bs4 = bs4[bs4["epoch"] <= 50]
bs8 = bs8[bs8["epoch"] <= 50]

# ─────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

ax.plot(bs4["epoch"], bs4["avg_loss"], color=C_B4, linewidth=2.5, label="BS=4 (baseline)")
ax.plot(bs2["epoch"], bs2["avg_loss"], color=C_B2, linewidth=2.5, label="BS=2 (small batch)")
ax.plot(bs8["epoch"], bs8["avg_loss"], color=C_B8, linewidth=2.5, label="BS=8 (large batch)")

# Annotate final values
for df, c, tag in [(bs4, C_B4, "BS=4"), (bs2, C_B2, "BS=2"), (bs8, C_B8, "BS=8")]:
    last = df.iloc[-1]
    ax.annotate(
        f"{tag}: {last['avg_loss']:.3f}",
        xy=(last["epoch"], last["avg_loss"]),
        xytext=(last["epoch"] + 0.5, last["avg_loss"]),
        fontsize=10, color=c, va="center"
    )

ax.set_title("Experiment 3 — Batch Size Comparison (Last 2 Layers, LR=1e-5, 50 epochs)",
             fontsize=13, fontweight="bold", color=DARK, pad=12)
ax.set_xlabel("Epoch", fontsize=12, color=DARK)
ax.set_ylabel("Avg Loss", fontsize=12, color=DARK)
ax.legend(fontsize=11, framealpha=0.7)
ax.grid(True, alpha=0.3, linestyle="--")
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(colors=DARK, labelsize=11)

plt.tight_layout()
plt.savefig(OUTPUT, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"Saved: {OUTPUT}")

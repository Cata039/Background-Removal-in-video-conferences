import pandas as pd
import matplotlib.pyplot as plt
import os

# ─────────────────────────────────────────────
# CONFIG — change these paths if needed
# ─────────────────────────────────────────────
CSV_E1  = "training_log.csv"
CSV_E2A = "training_log_attempt2.csv"
CSV_E2B = "training_log_attempt2_lr1e-5.csv"
OUTPUT_DIR = "charts"

# ─────────────────────────────────────────────
# COLORS — matching presentation palette
# ─────────────────────────────────────────────
BG     = "#F5F0EB"
C1     = "#3D6B9E"   # blue   — E1
C2A    = "#C0504D"   # red    — E2a
C2B    = "#7B68A8"   # purple — E2b
DARK   = "#2B2B2B"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# LOAD & CLEAN
# ─────────────────────────────────────────────
def load(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    df = df.sort_values("epoch").drop_duplicates("epoch").reset_index(drop=True)
    return df

e1  = load(CSV_E1)
e2a = load(CSV_E2A)
e2b = load(CSV_E2B)

# ─────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────
def style(ax, title):
    ax.set_facecolor(BG)
    ax.set_title(title, fontsize=14, fontweight="bold", color=DARK, pad=12)
    ax.set_xlabel("Epoch", fontsize=12, color=DARK)
    ax.set_ylabel("Loss", fontsize=12, color=DARK)
    ax.tick_params(colors=DARK, labelsize=11)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=10, framealpha=0.7)

def plot_single(df, color, title, filename, best_range=None):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor(BG)

    ax.fill_between(df["epoch"], df["min_loss"], df["max_loss"],
                    alpha=0.15, color=color, label="Min-Max range")
    ax.plot(df["epoch"], df["avg_loss"],
            color=color, linewidth=2.5, label="Avg loss")
    ax.plot(df["epoch"], df["min_loss"],
            color="green", linewidth=1, linestyle="--", alpha=0.6, label="Min loss")
    ax.plot(df["epoch"], df["max_loss"],
            color="orange", linewidth=1, linestyle="--", alpha=0.6, label="Max loss")

    # Optional green shading for best visual range
    if best_range:
        ax.axvspan(best_range[0], best_range[1], alpha=0.1,
                   color="green", label=f"Best visual range (E{best_range[0]}–{best_range[1]})")

    # Annotate best avg loss epoch
    best_idx   = df["avg_loss"].idxmin()
    best_epoch = int(df.loc[best_idx, "epoch"])
    best_loss  = df.loc[best_idx, "avg_loss"]
    ax.annotate(
        f"Best avg: E{best_epoch} → {best_loss:.4f}",
        xy=(best_epoch, best_loss),
        xytext=(best_epoch - max(5, len(df) // 6), best_loss + (df["avg_loss"].max() * 0.05)),
        fontsize=9, color=color,
        arrowprops=dict(arrowstyle="->", color=color, lw=1.2)
    )

    style(ax, title)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved: {path}")

# ─────────────────────────────────────────────
# CHART 1 — E1 (100 epochs)
# ─────────────────────────────────────────────
plot_single(
    e1, C1,
    "E1 — Last 2 Layers, LR=1e-5 (100 epochs)",
    "chart_e1.png",
    best_range=(10, 20)
)

# ─────────────────────────────────────────────
# CHART 2 — E2a (50 epochs, LR=1e-4)
# ─────────────────────────────────────────────
plot_single(
    e2a, C2A,
    "E2a — Decoder+Head, LR=1e-4 (50 epochs)",
    "chart_e2a.png"
)

# ─────────────────────────────────────────────
# CHART 3 — E2b (50 epochs, LR=1e-5)
# ─────────────────────────────────────────────
plot_single(
    e2b, C2B,
    "E2b — Decoder+Head, LR=1e-5 (50 epochs)",
    "chart_e2b.png"
)

# ─────────────────────────────────────────────
# CHART 4 — All 3 together (50 epochs fair comparison)
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5.5))
fig.patch.set_facecolor(BG)

e1_50 = e1[e1["epoch"] <= 50]

ax.plot(e1_50["epoch"], e1_50["avg_loss"],
        color=C1,  linewidth=2.5, label="E1  — Last 2 layers, LR=1e-5")
ax.plot(e2a["epoch"], e2a["avg_loss"],
        color=C2A, linewidth=2.5, label="E2a — Decoder+Head, LR=1e-4")
ax.plot(e2b["epoch"], e2b["avg_loss"],
        color=C2B, linewidth=2.5, label="E2b — Decoder+Head, LR=1e-5")

# Annotate final values
for df, c, tag in [(e1_50, C1, "E1"), (e2a, C2A, "E2a"), (e2b, C2B, "E2b")]:
    last = df.iloc[-1]
    ax.annotate(
        f"{tag}: {last['avg_loss']:.3f}",
        xy=(last["epoch"], last["avg_loss"]),
        xytext=(last["epoch"] + 0.5, last["avg_loss"]),
        fontsize=9, color=c, va="center"
    )

style(ax, "All Experiments — Avg Loss Comparison (50 epochs)")
plt.tight_layout()
path = os.path.join(OUTPUT_DIR, "chart_all.png")
plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"Saved: {path}")

print("\nAll 4 charts generated in:", OUTPUT_DIR)

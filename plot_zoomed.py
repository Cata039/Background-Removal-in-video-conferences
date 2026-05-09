import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import csv
import os

# ── Read CSV ──
csv_path = os.path.join(os.path.dirname(__file__), 'training_log.csv')

epochs, avg, mn, mx = [], [], [], []
with open(csv_path, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        epochs.append(int(row['epoch']))
        avg.append(float(row['avg_loss']))
        mn.append(float(row['min_loss']))
        mx.append(float(row['max_loss']))

# ── Plot ──
fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor('#F0EDE8')
ax.set_facecolor('#FFFFFF')

# Shaded area under avg
ax.fill_between(epochs, min(avg)*0.995, avg, alpha=0.15, color='#2D4B8E')
ax.plot(epochs, avg, color='#2D4B8E', linewidth=2.5, label='Avg loss', zorder=5)

# Milestone markers every 10 epochs
milestone_colors = {
    10:  '#6D28D9',
    20:  '#0D9488',
    30:  '#D97706',
    40:  '#DC2626',
    50:  '#22C55E',
    60:  '#0EA5E9',
    70:  '#F59E0B',
    80:  '#8B5CF6',
    90:  '#EC4899',
    100: '#2D4B8E',
}

for ep, col in milestone_colors.items():
    if ep <= max(epochs):
        val = avg[ep - 1]
        ax.axvline(x=ep, color=col, linewidth=0.6, linestyle=':', alpha=0.5)
        ax.scatter(ep, val, color=col, zorder=6, s=55)
        offset_y = 0.0005 if ep % 20 == 0 else -0.0012
        ax.annotate(f'E{ep}\n{val:.4f}',
            xy=(ep, val),
            xytext=(ep + 0.8, val + offset_y),
            fontsize=7.5, color=col,
            arrowprops=dict(arrowstyle='->', color=col, lw=0.7))

# Best epoch star
best_ep  = epochs[avg.index(min(avg))]
best_val = min(avg)
ax.scatter(best_ep, best_val, color='#065F46', zorder=7, s=120, marker='*')
ax.annotate(f'Best: E{best_ep}  →  {best_val:.4f}',
    xy=(best_ep, best_val),
    xytext=(best_ep - 22, best_val - 0.0018),
    fontsize=9, color='#065F46', fontweight='bold',
    arrowprops=dict(arrowstyle='->', color='#065F46', lw=1.3),
    bbox=dict(boxstyle='round,pad=0.35', facecolor='#ECFDF5', edgecolor='#6EE7B7'))

# Start annotation
ax.annotate(f'Start: {avg[0]:.4f}',
    xy=(1, avg[0]),
    xytext=(6, avg[0] + 0.001),
    fontsize=9, color='#991B1B', fontweight='bold',
    arrowprops=dict(arrowstyle='->', color='#991B1B', lw=1.0))

# Plateau region shading
ax.axvspan(50, 100, alpha=0.04, color='#22C55E')
ax.text(75, min(avg) + 0.0005, 'Plateau region\n(ep. 50–100)',
        ha='center', fontsize=8, color='#065F46', alpha=0.7,
        bbox=dict(boxstyle='round,pad=0.2', facecolor='#ECFDF5', edgecolor='none', alpha=0.6))

# Styling
ax.set_title('Training Avg Loss — Zoomed View (RVM Fine-Tuning, Last 2 Layers, LR=1e-5)',
             fontsize=13, color='#2D4B8E', pad=14, fontweight='bold')
ax.set_xlabel('Epoch', fontsize=11, color='#1A1A1A', labelpad=8)
ax.set_ylabel('Avg Loss', fontsize=11, color='#1A1A1A', labelpad=8)

pad = 0.003
ax.set_ylim(min(avg) - pad, max(avg) + pad)
ax.set_xlim(1, max(epochs))
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.002))
ax.tick_params(colors='#555555', labelsize=9)
ax.spines[['top', 'right']].set_visible(False)
ax.spines[['left', 'bottom']].set_color('#CCCCCC')
ax.grid(axis='y', color='#EEEEEE', linewidth=0.8)
ax.grid(axis='x', color='#EEEEEE', linewidth=0.4)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), 'training_loss_zoomed.png')
plt.savefig(out_path, dpi=180, bbox_inches='tight', facecolor='#F0EDE8')
print(f"Salvat: {out_path}")
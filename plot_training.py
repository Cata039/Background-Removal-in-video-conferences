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

# Fill between min and max
ax.fill_between(epochs, mn, mx, alpha=0.12, color='#3266AD', label='Min–Max range')

# Lines
ax.plot(epochs, avg, color='#2D4B8E', linewidth=2.5, label='Avg loss', zorder=5)
ax.plot(epochs, mn,  color='#22C55E', linewidth=1.2, linestyle='--', label='Min loss', zorder=4)
ax.plot(epochs, mx,  color='#F59E0B', linewidth=1.2, linestyle='--', label='Max loss', zorder=4)

# Milestone vertical lines every 10 epochs
for m in range(10, max(epochs)+1, 10):
    ax.axvline(x=m, color='#2D4B8E', linewidth=0.5, linestyle=':', alpha=0.4)
    ax.text(m, ax.get_ylim()[1]*0.97, f'E{m}', ha='center', va='top',
            fontsize=7, color='#2D4B8E', alpha=0.7)

# Best epoch annotation
best_epoch = epochs[avg.index(min(avg))]
best_val   = min(avg)
ax.annotate(f'Best avg\nEpoch {best_epoch}\n{best_val:.4f}',
    xy=(best_epoch, best_val),
    xytext=(best_epoch - 15, best_val - 0.015),
    fontsize=8, color='#065F46',
    arrowprops=dict(arrowstyle='->', color='#065F46', lw=1.2),
    bbox=dict(boxstyle='round,pad=0.3', facecolor='#ECFDF5', edgecolor='#A7F3D0'))

# Styling
ax.set_xlabel('Epoch', fontsize=11, color='#1A1A1A', labelpad=8)
ax.set_ylabel('Loss',  fontsize=11, color='#1A1A1A', labelpad=8)
ax.set_title('Training Loss — RVM Fine-Tuning (Last 2 Layers, LR=1e-5)',
             fontsize=13, color='#2D4B8E', pad=14, fontweight='bold')
ax.set_xlim(1, max(epochs))
ax.set_ylim(0, max(mx) * 1.08)
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
ax.tick_params(colors='#555555', labelsize=9)
ax.spines[['top', 'right']].set_visible(False)
ax.spines[['left', 'bottom']].set_color('#CCCCCC')
ax.grid(axis='y', color='#E5E5E5', linewidth=0.8)
ax.legend(loc='upper right', fontsize=9, framealpha=0.9,
          facecolor='#F8F8F8', edgecolor='#DDDDDD')

# Save
out_path = os.path.join(os.path.dirname(__file__), 'training_loss_chart.png')
plt.tight_layout()
plt.savefig(out_path, dpi=180, bbox_inches='tight', facecolor='#F0EDE8')
print(f"Salvat: {out_path}")
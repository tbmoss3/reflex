import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

data = {"skillgate_first_tool": (60.9, 59.0, 28.0, 6.4),
        "skillgate_noul": (83.6, 93.6, 6.68, 6.4),
        "stuck": (99.63, 98.9, 0.0, 0.5),
        "compact": (80.74, 84.6, 5.24, 4.0)}
names = list(data.keys())
acc = [d[0] for d in data.values()]; bar_acc = [d[1] for d in data.values()]
ece = [d[2] for d in data.values()]; bar_ece = [d[3] for d in data.values()]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
x = range(len(names))
axes[0].bar([i-0.18 for i in x], acc, 0.36, label='v0.5mt final', color='#4f8ef7')
axes[0].bar([i+0.18 for i in x], bar_acc, 0.36, label='bar', color='#999', alpha=0.6)
for i,(a,b) in enumerate(zip(acc,bar_acc)):
    axes[0].text(i-0.18, a+1, f'{a:.1f}', ha='center', fontsize=9)
    axes[0].text(i+0.18, b+1, f'{b:.1f}', ha='center', fontsize=9)
axes[0].set_xticks(list(x)); axes[0].set_xticklabels(names, rotation=15, fontsize=8)
axes[0].set_title('Accuracy (%) vs bar'); axes[0].legend(); axes[0].set_ylim(50,105)
axes[1].bar([i-0.18 for i in x], ece, 0.36, label='v0.5mt final', color='#f76f4f')
axes[1].bar([i+0.18 for i in x], bar_ece, 0.36, label='bar (max)', color='#999', alpha=0.6)
for i,(a,b) in enumerate(zip(ece,bar_ece)):
    axes[1].text(i-0.18, a+0.5, f'{a:.2f}', ha='center', fontsize=9)
    axes[1].text(i+0.18, b+0.5, f'{b:.2f}', ha='center', fontsize=9)
axes[1].set_xticks(list(x)); axes[1].set_xticklabels(names, rotation=15, fontsize=8)
axes[1].set_title('ECE (%) vs bar'); axes[1].legend()
fig.suptitle('Reflex v0.5mt final - frozen-set eval (2026-09-21)')
plt.tight_layout()
os.makedirs('/home/hermes/diagrams', exist_ok=True)
p='/home/hermes/diagrams/reflex_stats_v05mt_final.png'
plt.savefig(p, dpi=120)
print(p)

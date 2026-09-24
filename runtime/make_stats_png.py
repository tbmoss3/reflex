#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, os

bars = {'first_tool': (59, 60.9), 'needs_full_agent': (93.6, 83.6), 'stuck': (97.4, 99.6), 'compact': (82, 80.7)}
ece = {'needs_full_agent': (0.064, 0.0668), 'stuck': (0.005, 0.0), 'compact': (0.040, 0.0524)}

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
names = list(bars); x = list(range(len(names)))
a1.bar([i-0.18 for i in x], [bars[n][0] for n in names], 0.36, label='Bar/incumbent', color='#888888')
a1.bar([i+0.18 for i in x], [bars[n][1] for n in names], 0.36, label='v0.5mt final',
       color=['#2ca02c' if bars[n][1]>=bars[n][0] else '#d62728' for n in names])
for i,n in enumerate(names): a1.text(i+0.18, bars[n][1]+0.5, f"{bars[n][1]:.1f}", ha='center', fontsize=9)
a1.set_xticks(x); a1.set_xticklabels(names, fontsize=8); a1.set_ylabel('Accuracy %'); a1.set_ylim(40, 102)
a1.set_title('v0.5mt multi-task - accuracy vs bars'); a1.legend(fontsize=8)

names2 = list(ece); x2 = list(range(len(names2)))
a2.bar([i-0.18 for i in x2], [ece[n][0] for n in names2], 0.36, label='Bar', color='#888888')
a2.bar([i+0.18 for i in x2], [ece[n][1] for n in names2], 0.36, label='v0.5mt final',
       color=['#2ca02c' if ece[n][1]<=ece[n][0] else '#d62728' for n in names2])
for i,n in enumerate(names2): a2.text(i+0.18, ece[n][1]+0.001, f"{ece[n][1]:.4f}", ha='center', fontsize=8)
a2.set_xticks(x2); a2.set_xticklabels(names2, fontsize=8); a2.set_ylabel('ECE (lower better)')
a2.set_title('Calibration vs bars'); a2.legend(fontsize=8)

fig.suptitle('Reflex v0.5mt final (2 ep, frozen evals) - 2026-09-21', fontsize=11)
fig.tight_layout()
p = '/home/hermes/diagrams/reflex_stats_v05mt_final.png'
os.makedirs('/home/hermes/diagrams', exist_ok=True)
fig.savefig(p, dpi=120)
print(p)

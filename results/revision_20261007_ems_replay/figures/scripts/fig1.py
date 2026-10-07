"""Archive evidence and frozen completion widths. No numerical re-estimation."""
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import StrMethodFormatter
import figstyle_nature as F

plt=F.style()
widths=pd.read_csv(F.DATA/'archive_width_decomposition.csv')
w=widths.set_index('pair').loc[F.PAIR_KEYS]
fig=plt.figure(figsize=(F.mm(183),F.mm(80)))
F.header(fig,'a','Archive evidence',.025)
F.header(fig,'b','Completion-width decomposition',.525)

# Three row strips: labels and exact numbers occupy white space above bars.
ax=fig.add_axes([.060,.23,.395,.56])
states=['Known outcome','Ambiguous reference','No qualified observation']
counts=np.array([1468,2,2451])
colors=[F.C['main'],F.C['fifth'],F.C['light']]
y=np.array([2.,1.,0.])
bars=ax.barh(y,counts,height=.23,color=colors,edgecolor=F.C['edge'],lw=.50)
for bar in bars: F.tagged(bar,'candidate count')
for yi,label,n in zip(y,states,counts):
    pct=100*n/counts.sum()
    percent=f'{pct:.1f}%' if pct>=.1 else '<0.1%'
    ax.text(0,yi+.28,label,ha='left',va='center',fontsize=7.8)
    ax.text(2800,yi+.28,f'{n:,}  ({percent})',ha='right',va='center',fontsize=7.8)
ax.set_xlim(0,2800)
ax.set_ylim(-.53,2.60)
ax.set_yticks([])
ax.set_xticks([0,1000,2000])
ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.set_xlabel('Candidate records')
F.clean(ax,left=False)

# Bar widths are not inflated: the tiny amber segment is numerically explicit.
ax2=fig.add_axes([.560,.23,.410,.56])
y2=np.array([2.8,1.4,0.])
for yi,pair,label in zip(y2,F.PAIR_KEYS,F.PAIR_LABELS):
    row=w.loc[pair]
    a=float(row.absent_width); m=float(row.ambiguous_width)
    bar=ax2.barh(yi,a,height=.24,color=F.C['light'],edgecolor=F.C['edge'],lw=.5)[0]
    F.tagged(bar,'no qualified observation width')
    bar=ax2.barh(yi,m,left=a,height=.24,color=F.C['fifth'],edgecolor=F.C['fifth'],lw=.30)[0]
    F.tagged(bar,'ambiguous reference width')
    # The target guide traverses bar bands only, never row labels or numbers.
    guide=ax2.vlines(.10,yi-.20,yi+.20,color=F.C['ink'],lw=.8,linestyle=(0,(3,2)),zorder=4)
    F.tagged(guide,'width target 0.10')
    ax2.text(0,yi+.38,label,ha='left',va='center',fontsize=8.)
    ax2.text(0,yi-.40,f'Floor  {a:.5f}',ha='left',va='center',fontsize=7.8)
    ax2.text(.50,yi-.40,f'Resolvable  {m:.5f}',ha='left',va='center',fontsize=7.8)
ax2.set_xlim(0,1.01)
ax2.set_ylim(-.80,3.65)
ax2.set_xticks([0,.25,.5,.75,1.])
ax2.set_xticklabels(['0','0.25','0.50','0.75','1.00'])
ax2.set_yticks([])
ax2.set_xlabel('Paired completion width')
F.clean(ax2,left=False)

handles=[Patch(facecolor=F.C['light'],edgecolor=F.C['edge'],lw=.5,label='No qualified observation'),
         Patch(facecolor=F.C['fifth'],edgecolor=F.C['fifth'],lw=.3,label='Ambiguous reference'),
         Line2D([],[],color=F.C['ink'],lw=.8,ls=(0,(3,2)),label='Width target 0.10')]
F.legend(fig,handles,x=.51,y=.055,ncol=3,columnspacing=1.9)

source=pd.concat([pd.DataFrame({'record_type':'candidate state','state':states,'candidate_count':counts,
                               'candidate_share':counts/counts.sum()}),
                  widths.assign(record_type='paired completion width')],ignore_index=True,sort=False)
F.finish(fig,1,source)

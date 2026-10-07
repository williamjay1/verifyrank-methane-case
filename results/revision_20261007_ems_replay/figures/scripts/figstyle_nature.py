"""Fixed-size vector artwork, SciencePlots foundation, explicit print-size QA."""
from pathlib import Path
import json
import os
import numpy as np

if 'VERIFYRANK_FIGURE_WORK' not in os.environ:
    raise RuntimeError('Set VERIFYRANK_FIGURE_WORK to a writable scratch directory before drawing figures.')
ROOT = Path(os.environ['VERIFYRANK_FIGURE_WORK']).expanduser().resolve()
DATA = Path(os.environ.get('VERIFYRANK_ANALYSIS', str(Path(__file__).resolve().parents[1] / 'inputs'))).expanduser().resolve()
OUT = ROOT / 'outputs'
QA = ROOT / 'qa'
for folder in (OUT, QA, ROOT / 'cache'):
    folder.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'cache' / 'matplotlib'))

def apply_house_style(fontsize=8.0, use_tex=False):
    import matplotlib.pyplot as plt
    import scienceplots
    plt.style.use(['science', 'nature'])
    plt.rcParams.update({'font.size': fontsize, 'text.usetex': use_tex,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    return plt

from matplotlib.text import Text
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
from matplotlib.transforms import Bbox

C = {'main':'#0072B2', 'alt':'#D55E00', 'third':'#009E73',
     'fourth':'#AA6B91', 'fifth':'#E69F00', 'light':'#C9CDD0',
     'ink':'#202124', 'muted':'#5B6065', 'edge':'#787E83'}
PAIR_KEYS = ['logistic_vs_beta_rate','hist_gradient_boosting_vs_beta_rate','beta_rate_vs_uniform_random']
PAIR_LABELS = ['Logistic vs history','Boosting vs history','History vs random']
PAIR_COLORS = [C['main'], C['alt'], C['third']]
PAIR_MARKERS = ['o', 's', '^']
MM_WIDTH = 183.0
PRINT_WIDTH_MM = 164.6
SCALE = PRINT_WIDTH_MM / MM_WIDTH

def mm(v):
    return v / 25.4

def style():
    plt = apply_house_style(fontsize=8.0, use_tex=False)
    plt.rcParams.update({
        'font.family':'sans-serif', 'font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':8.0, 'axes.labelsize':8.0, 'xtick.labelsize':7.8,
        'ytick.labelsize':7.8, 'legend.fontsize':7.8, 'axes.titlesize':8.6,
        'axes.linewidth':0.65, 'axes.edgecolor':C['ink'], 'text.color':C['ink'],
        'axes.labelcolor':C['ink'], 'xtick.color':C['ink'], 'ytick.color':C['ink'],
        'axes.grid':False, 'xtick.direction':'out', 'ytick.direction':'out',
        'xtick.top':False, 'ytick.right':False, 'xtick.minor.visible':False,
        'ytick.minor.visible':False, 'xtick.major.size':2.6, 'ytick.major.size':2.6,
        'xtick.major.width':0.6, 'ytick.major.width':0.6,
        'xtick.major.pad':3.0, 'ytick.major.pad':3.0,
        'lines.linewidth':1.35, 'lines.markersize':3.4,
        'legend.frameon':False, 'legend.handlelength':1.8,
        'legend.handletextpad':0.55, 'legend.columnspacing':1.8,
        'legend.borderaxespad':0, 'savefig.bbox':None,
        'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none',
        'svg.hashsalt':'verifyrank-figure-redesign-20261007',
        'figure.facecolor':'white', 'axes.facecolor':'white', 'savefig.facecolor':'white',
        'hatch.linewidth':0.35,
    })
    return plt

def clean(ax, left=True, bottom=True):
    ax.grid(False)
    for name in ('top', 'right'):
        ax.spines[name].set_visible(False)
    ax.spines['left'].set_visible(left)
    ax.spines['bottom'].set_visible(bottom)
    ax.tick_params(which='minor', bottom=False, left=False, right=False, top=False)
    ax.tick_params(which='major', direction='out')
    ax.xaxis.labelpad = 5
    ax.yaxis.labelpad = 5

def header(fig, letter, title, x, y=0.95):
    fig.text(x, y, letter, fontsize=9.2, weight='bold', va='top', ha='left')
    fig.text(x+0.031, y-0.001, title, fontsize=8.6, weight='normal', va='top', ha='left')

def legend(fig, handles, labels=None, x=.5, y=.85, ncol=3, **kwargs):
    return fig.legend(handles=handles, labels=labels, loc='center', bbox_to_anchor=(x,y),
                      ncol=ncol, fontsize=7.8, **kwargs)

def pair_handles():
    return [Line2D([], [], color=c, marker=m, markerfacecolor='white',
                   markeredgewidth=.85, lw=1.35, label=l)
            for c,m,l in zip(PAIR_COLORS,PAIR_MARKERS,PAIR_LABELS)]

def tagged(artist, name='data'):
    artist.set_gid(name)
    return artist

def _segment_hits_box(a, b, box):
    """Slab intersection, including crossings with both vertices outside."""
    d = b-a
    lo, hi = 0., 1.
    for j, (mn,mx) in enumerate(((box.x0,box.x1),(box.y0,box.y1))):
        if abs(d[j]) < 1e-12:
            if a[j] < mn or a[j] > mx: return False
        else:
            t0,t1 = (mn-a[j])/d[j], (mx-a[j])/d[j]
            lo=max(lo,min(t0,t1)); hi=min(hi,max(t0,t1))
            if lo > hi: return False
    return True

def audit(fig, number):
    """Include all titles, figure text and both legends; check data paths."""
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    canvas=fig.bbox
    texts=[]
    seen=set()
    # findobj includes figure-level labels, left/right axis titles, old legends.
    for t in fig.findobj(match=Text):
        if id(t) in seen or not t.get_visible() or not t.get_text().strip(): continue
        seen.add(id(t))
        bb=t.get_window_extent(renderer)
        if not np.isfinite(bb.extents).all() or bb.width<=0 or bb.height<=0: continue
        # Invisible (non-rendered) tick-label counterpart on top/right axes.
        if t.get_alpha() == 0: continue
        texts.append((t,bb))
    clip=[]; overlaps=[]; line_hits=[]; mark_hits=[]
    for t,b in texts:
        if b.x0<canvas.x0-.25 or b.y0<canvas.y0-.25 or b.x1>canvas.x1+.25 or b.y1>canvas.y1+.25:
            clip.append(t.get_text())
    for i,(ta,ba) in enumerate(texts):
        for tb,bb in texts[i+1:]:
            # Require at least a 0.4-pixel intersection to ignore anti-aliasing.
            if min(ba.x1,bb.x1)-max(ba.x0,bb.x0)>.4 and min(ba.y1,bb.y1)-max(ba.y0,bb.y0)>.4:
                overlaps.append([ta.get_text(),tb.get_text()])
    data_paths=[]; patches=[]; markers=[]
    for ax in fig.axes:
        for l in ax.lines:
            if not l.get_visible() or l.get_gid()=='qa-exempt': continue
            vs=l.get_transform().transform(l.get_path().vertices)
            data_paths.append((l.get_gid() or l.get_label(),vs,ax.bbox))
            if str(l.get_marker()).lower() not in ('none','',' ','null'):
                selected=np.arange(len(vs))
                every=l.get_markevery()
                if isinstance(every,(list,np.ndarray)): selected=selected[every]
                elif isinstance(every,tuple) and len(every)==2 and all(isinstance(n,int) for n in every):
                    selected=selected[every[0]::every[1]]
                elif isinstance(every,int): selected=selected[::every]
                radius=(l.get_markersize()/2+l.get_markeredgewidth()/2)*fig.dpi/72
                markers.append((l.get_gid() or l.get_label(),vs[selected],radius,ax.bbox))
        for coll in ax.collections:
            if isinstance(coll,LineCollection) and coll.get_visible():
                for seg in coll.get_segments():
                    data_paths.append((coll.get_gid() or 'line collection',coll.get_transform().transform(seg),ax.bbox))
        for p in ax.patches:
            if isinstance(p,Rectangle) and p.get_visible() and p.get_gid()!='qa-exempt':
                patches.append((p.get_gid() or 'bar',p.get_window_extent(renderer),ax.bbox))
    pad=.45*fig.dpi/72
    for t,bb in texts:
        expanded=Bbox.from_extents(bb.x0-pad,bb.y0-pad,bb.x1+pad,bb.y1+pad)
        for name,vs,axbb in data_paths:
            # Only plotted portions inside axes can collide.
            inter=Bbox.intersection(expanded,axbb)
            if inter is None: continue
            for a,b in zip(vs[:-1],vs[1:]):
                if np.isfinite(a).all() and np.isfinite(b).all() and _segment_hits_box(a,b,inter):
                    line_hits.append([t.get_text(),name]); break
        for name,pb,axbb in patches:
            inter=Bbox.intersection(pb,axbb)
            if inter is not None and expanded.overlaps(inter) and inter.width>.2 and inter.height>.2:
                mark_hits.append([t.get_text(),name])
        for name,vs,radius,axbb in markers:
            inter=Bbox.intersection(expanded,axbb)
            if inter is None: continue
            for p in vs:
                if not np.isfinite(p).all(): continue
                markerbox=Bbox.from_extents(p[0]-radius,p[1]-radius,p[0]+radius,p[1]+radius)
                clipped=Bbox.intersection(markerbox,axbb)
                if clipped is not None and inter.overlaps(clipped):
                    mark_hits.append([t.get_text(),name+' marker']); break
    minimum=min(t.get_fontsize() for t,_ in texts)
    report={
        'figure':number, 'width_mm':float(fig.get_figwidth()*25.4),
        'height_mm':float(fig.get_figheight()*25.4), 'fixed_canvas':True,
        'embedding_width_mm':PRINT_WIDTH_MM,
        'minimum_source_font_pt':minimum, 'minimum_embedded_font_pt':minimum*SCALE,
        'text_count':len(texts), 'clipped_text':clip, 'text_text_overlap':overlaps,
        'text_data_line_overlap':line_hits, 'text_bar_overlap':mark_hits,
        'pdf_fonttype':42, 'svg_text_editable':True,
    }
    report['ok']=not (clip or overlaps or line_hits or mark_hits) and minimum*SCALE>=6.99
    (QA/f'Figure_{number}_layout_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report

def finish(fig, number, source=None):
    # Preview first. No crop, resize or interpolation is used for raster export.
    fig.savefig(QA/f'Figure_{number}_preview.png',dpi=220,bbox_inches=None)
    report=audit(fig,number)
    print(json.dumps(report,indent=2))
    if not report['ok']:
        raise RuntimeError(f'Figure {number}: layout audit failed; inspect preview and repair.')
    meta={'Creator':'Matplotlib / SciencePlots; VerifyRank artwork',
          'Title':f'VerifyRank Figure {number}', 'CreationDate':None, 'ModDate':None}
    fig.savefig(OUT/f'Figure_{number}.pdf',bbox_inches=None,metadata=meta)
    fig.savefig(OUT/f'Figure_{number}.svg',bbox_inches=None,metadata={'Date':None})
    fig.savefig(OUT/f'Figure_{number}_1200dpi.png',dpi=1200,bbox_inches=None)
    fig.savefig(OUT/f'Figure_{number}_900dpi.png',dpi=900,bbox_inches=None)
    if source is not None:
        source.to_csv(OUT/f'Figure_{number}_source_data.csv',index=False,float_format='%.17g')
    return report

"""Nature-style native vector figures with direct 1200-dpi companion exports."""
from __future__ import annotations
import argparse, hashlib, json, math, sys, re, textwrap
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.text import Text
from matplotlib.transforms import Bbox
from PIL import Image
import pymupdf
import scienceplots
import evaluate_campaigns as ev

P=Path(__file__).resolve().parents[1]
R=P/'results/revision_20261006'
OUT=R/'figures';OUT.mkdir(parents=True,exist_ok=True)
DATA=OUT/'source_data';DATA.mkdir(exist_ok=True)
ORIG=P/'results/campaign_validation_20261005T152339_897315Z'
BLUE='#0072B2';ORANGE='#D55E00';GREEN='#009E73';PURPLE='#8064A2';GREY='#72777D';BLACK='#20252B'
PAIR_NAMES={'logistic_vs_beta_rate':'Logistic − history','hist_gradient_boosting_vs_beta_rate':'Boosting − history','beta_rate_vs_uniform_random':'History − random'}
MODELS=['beta_rate','last_label','max_emission','uniform_random','logistic','hist_gradient_boosting']
MODEL_NAMES=['Historical rate','Last state','Maximum flux','Random','Logistic','Boosting']
COLOURS=[BLUE,GREY,PURPLE,BLACK,GREEN,ORANGE]
plt.style.use(['science','nature','no-latex'])
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
    'font.size':6.5,'axes.labelsize':6.5,'axes.titlesize':7,'xtick.labelsize':6,'ytick.labelsize':6,
    'legend.fontsize':6,'axes.linewidth':.55,'lines.linewidth':1,'lines.markersize':3,
    'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white',
    'figure.facecolor':'white','savefig.bbox':None,'savefig.pad_inches':0,
    'axes.spines.top':False,'axes.spines.right':False,
    'xtick.direction':'out','ytick.direction':'out','xtick.minor.visible':False,'ytick.minor.visible':False,
    'xtick.top':False,'ytick.right':False,
    'text.color':BLACK,'axes.labelcolor':BLACK,'axes.edgecolor':BLACK})
reports=[]

def panel(ax,letter,title):
    ax.set_title(title,loc='left',pad=11,fontweight='normal')
    offset=(12/72)/(ax.figure.get_figwidth()*ax.get_position().width)
    ax.text(-offset,1.075,letter,transform=ax.transAxes,fontsize=8,fontweight='bold',ha='left',va='bottom')
    ax.tick_params(length=2.5,width=.55,pad=2)

def dataset(name,frame):
    frame.to_csv(DATA/f'{name}.csv',index=False)

def audit(fig,name):
    fig.canvas.draw();renderer=fig.canvas.get_renderer();canvas=fig.bbox
    records=[]
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text().strip():continue
        box=t.get_window_extent(renderer)
        if box.width==0 or box.height==0:continue
        records.append((t,box))
    clipped=[t.get_text() for t,b in records if b.x0<-.5 or b.y0<-.5 or b.x1>canvas.x1+.5 or b.y1>canvas.y1+.5]
    collisions=[]
    for i,(a,ba) in enumerate(records):
        for b,bb in records[i+1:]:
            inter=Bbox.intersection(ba,bb)
            if inter is not None and inter.width>.6 and inter.height>.6:
                collisions.append([a.get_text(),b.get_text()])
    crossing=[]
    def intersects(p,q,b):
        # Liang–Barsky clipping against an inset text box.
        lo,hi=0.,1.;dx,dy=q-p
        for z,r in [(-dx,p[0]-b.x0),(dx,b.x1-p[0]),(-dy,p[1]-b.y0),(dy,b.y1-p[1])]:
            if abs(z)<1e-12:
                if r<0:return False
            elif z<0:lo=max(lo,r/z)
            else:hi=min(hi,r/z)
        return lo<=hi
    for ax in fig.axes:
        for line in ax.lines:
            if not line.get_visible() or line.get_linestyle() in ['None','none','']:continue
            points=line.get_transform().transform(line.get_xydata())
            for t,b in records:
                if line.get_clip_on() and Bbox.intersection(b,ax.bbox) is None:continue
                inset=Bbox.from_extents(b.x0+1,b.y0+1,b.x1-1,b.y1-1)
                if any(np.isfinite(p).all() and np.isfinite(q).all() and intersects(p,q,inset) for p,q in zip(points[:-1],points[1:])):
                    crossing.append(t.get_text())
    return {'visible_text_count':len(records),'clipped_text':clipped,'text_bbox_collisions':collisions,'text_data_line_crossings':crossing,
            'minimum_font_pt':min(t.get_fontsize() for t,b in records),'maximum_font_pt':max(t.get_fontsize() for t,b in records)}

def save(fig,name):
    layout=audit(fig,name)
    for ext in ['pdf','svg']:
        fig.savefig(OUT/f'{name}.{ext}',bbox_inches=None,facecolor='white')
    fig.savefig(OUT/f'{name}.png',dpi=300,bbox_inches=None,facecolor='white')
    # Render directly from the live vector artists; never upscale a preview.
    fig.savefig(OUT/f'{name}.tiff',dpi=1200,bbox_inches=None,facecolor='white',
                pil_kwargs={'compression':'tiff_lzw'})
    path=OUT/f'{name}.tiff'
    with Image.open(path) as image:
        rgb=image.convert('RGB');rgb.save(path,compression='tiff_lzw',dpi=(1200,1200))
    with Image.open(path) as image: raster={'pixels':list(image.size),'dpi':[float(v) for v in image.info['dpi']],'mode':image.mode}
    with pymupdf.open(OUT/f'{name}.pdf') as pdf:
        vector={'page_width_mm':pdf[0].rect.width/72*25.4,'page_height_mm':pdf[0].rect.height/72*25.4,
                'embedded_images':len(pdf[0].get_images()),'embedded_fonts':len(pdf[0].get_fonts())}
    assert vector['embedded_images']==0
    assert '<image' not in (OUT/f'{name}.svg').read_text(encoding='utf-8')
    reports.append({'figure':name,'layout':layout,'raster':raster,'vector':vector})
    plt.close(fig)

def primary():
    x=pd.read_csv(ORIG/'predictions.csv',dtype={'campaign':str})
    return x.loc[x.scenario.eq(ev.PRIMARY_SCENARIO)&x.evaluation_mode.eq('external_2023_to_2024')].copy()

def figure1():
    fig,axes=plt.subplots(2,1,figsize=(183/25.4,144/25.4),gridspec_kw={'height_ratios':[1,1.15]})
    fig.subplots_adjust(left=.075,right=.975,top=.94,bottom=.08,hspace=.37)
    ax=axes[0];panel(ax,'a','Evaluation architecture');ax.axis('off');ax.set_xlim(-.025,1.01);ax.set_ylim(0,1)
    boxes=[(.00,.62,'Archive adapter','Geometry · quality\nVersioned observations'),(.36,.62,'Temporal builder','Fixed source anchors\nAvailable history only'),(.72,.62,'Policy interface','Six historical features\nFrozen scores and weights'),
           (.00,.06,'Reference protocol','Qualified first scene\nFour evidence states'),(.36,.06,'Comparison module','Conditional estimates\nPaired / joint completions'),(.72,.06,'Verification demand','Unique reference outcomes\nWidth and direction targets')]
    for x,y,title,body in boxes:
        patch=FancyBboxPatch((x,y),.25,.29,boxstyle='round,pad=.012,rounding_size=.013',fc='#F4F7F9',ec='#CBD3DA',lw=.6)
        ax.add_patch(patch);ax.text(x+.125,y+.215,title,ha='center',va='center',fontsize=6.8,fontweight='bold')
        ax.text(x+.125,y+.092,body,ha='center',va='center',fontsize=6.1,linespacing=1.6)
    for start,end in [((.267,.765),(.341,.765)),((.627,.765),(.701,.765)),((.267,.205),(.341,.205)),((.627,.205),(.701,.205)),((.125,.605),(.125,.367))]:
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=7,lw=.8,color=GREY))
    # Policy outputs enter evaluation through a separate connector corridor.
    ax.plot([.845,.845,.485,.485],[.605,.49,.49,.367],color=GREEN,lw=.8)
    ax.add_patch(FancyArrowPatch((.485,.40),(.485,.367),arrowstyle='-|>',mutation_scale=7,lw=.8,color=GREEN))
    ax.text(.53,.52,'Scores frozen before reference join',ha='center',va='bottom',fontsize=6,color=GREEN)
    ax=axes[1];panel(ax,'b','Decision time and reference evidence');ax.axis('off');ax.set_xlim(-58,88);ax.set_ylim(-.15,1.15)
    ax.axvspan(0,72,ymin=.62,ymax=.98,color='#EAF1F7',zorder=0)
    ax.plot([-52,77],[.82,.82],color=GREY,lw=.8)
    event=[(-48,'Historical\nacquisition',BLUE),(-24,'Scenario\navailability',BLUE),(0,'Decision and\npolicy freeze',BLACK),(36,'First qualified\nreference',GREEN),(72,'Window\nendpoint',GREY)]
    for x,label,col in event:
        ax.plot([x,x],[.775,.865],color=col,lw=1);ax.plot(x,.82,'o',color=col,ms=3)
        ax.text(x,1.035,label,ha='center',va='bottom',fontsize=6.1,linespacing=1.3)
        ax.text(x,.725,f'{x:+d} h' if x else '0 h',ha='center',va='top',fontsize=6)
    states=[('Accepted\ndetection',GREEN),('Qualified\nnondetection',BLUE),('Unresolved\nqualified scene',ORANGE),('No qualified\nscene',GREY)]
    for j,(label,col) in enumerate(states):
        x=-52+j*34
        ax.add_patch(FancyBboxPatch((x,.1),29,.29,boxstyle='round,pad=.012,rounding_size=.035',fc='white',ec=col,lw=.8))
        ax.text(x+14.5,.245,label,ha='center',va='center',fontsize=6.1,linespacing=1.4)
    ax.text(-37,.025,'Known 1',ha='center',va='top',fontsize=6,color=GREEN)
    ax.text(-3,.025,'Known 0',ha='center',va='top',fontsize=6,color=BLUE)
    ax.text(31,.025,'Actual reference unresolved',ha='center',va='top',fontsize=5.6,color=ORANGE)
    ax.text(65,.025,'Hypothetical score slot',ha='center',va='top',fontsize=5.6,color=GREY)
    save(fig,'Figure_1_protocol')

def figure2():
    pred=primary();dm=pd.read_csv(ORIG/'daily_metrics.csv');dm=dm.loc[dm.scenario.eq(ev.PRIMARY_SCENARIO)&dm.evaluation_mode.eq('external_2023_to_2024')&dm.budget_kind.eq('fraction')&dm.budget_value.eq(.25)]
    days=pred.groupby('decision_at').agg(candidate_count=('source_id','size'),known_count=('outcome_label','count')).reset_index()
    days['known_fraction']=days.known_count/days.candidate_count;dataset('Figure_2a_coverage',days)
    mass=dm.assign(unknown_fraction=dm.expected_unknown/dm.effective_budget).groupby('model').unknown_fraction.mean().reindex(MODELS)
    dataset('Figure_2b_unknown_selection',mass.rename('unknown_fraction').reset_index())
    pairs=[('beta_rate','uniform_random'),('logistic','beta_rate'),('hist_gradient_boosting','beta_rate')]
    pairlabels=['History − random','Logistic − history','Boosting − history']
    bi=pd.read_csv(ORIG/'bootstrap_intervals.csv');ms=pd.read_csv(ORIG/'macro_paired_summary.csv')
    summary=[];reason=[];overlap=[]
    for a,b in pairs:
        row=ms.loc[ms.scenario.eq(ev.PRIMARY_SCENARIO)&ms.evaluation_mode.eq('external_2023_to_2024')&ms.budget_kind.eq('fraction')&ms.budget_value.eq(.25)&ms.model.eq(a)&ms.reference.eq(b)].iloc[0]
        ci=bi.loc[bi.model.eq(a)&bi.reference.eq(b)&bi.metric.eq('conditional_precision_difference')].iloc[0]
        summary.append({'model':a,'reference':b,'conditional':row.conditional_precision_difference,'ci_lower':ci.ci_lower,'ci_upper':ci.ci_upper,'lower':row.completion_difference_lower,'upper':row.completion_difference_upper})
        values=[]
        for date,day in pred.groupby('decision_at'):
            y=day.outcome_label.to_numpy(float);unknown=~np.isfinite(y)
            for kind,val in ev.BUDGETS:
                m=ev.effective_budget(kind,val,len(day));wa=ev.tie_selection_weights(day[f'score_{a}'].to_numpy(),m);wb=ev.tie_selection_weights(day[f'score_{b}'].to_numpy(),m)
                c=(wa-wb)/m;paired=np.abs(c[unknown]).sum();independent=((wa+wb)[unknown]/m).sum()
                overlap.append({'model':a,'reference':b,'date':date,'budget_kind':kind,'budget_value':val,
                    'overall_selection_overlap':np.minimum(wa,wb).sum()/m,'paired_width':paired,'independent_width':independent,
                    'relative_width_reduction':1-paired/independent if independent else np.nan})
                if kind=='fraction' and val==.25:
                    no_scene=day.first_future_scene.isna().to_numpy()|day.first_future_scene.fillna('').eq('').to_numpy()
                    values.append({'no_scene':np.abs(c[unknown&no_scene]).sum(),'ambiguous':np.abs(c[unknown&~no_scene]).sum()})
        reason.append({'model':a,'reference':b,**pd.DataFrame(values).mean().to_dict()})
    summary=pd.DataFrame(summary);reason=pd.DataFrame(reason);overlap=pd.DataFrame(overlap)
    dataset('Figure_2cd_comparisons',summary);dataset('Figure_2e_width_reasons',reason);dataset('Figure_S1_overlap',overlap)
    fig,ax=plt.subplots(3,2,figsize=(183/25.4,173/25.4));fig.subplots_adjust(left=.14,right=.98,top=.955,bottom=.15,wspace=.62,hspace=.78)
    a=ax[0,0];panel(a,'a','Daily reference coverage');x=np.arange(len(days));a.plot(x,days.known_fraction*100,'o-',color=BLUE)
    a.set_ylim(0,100);a.set_yticks([0,50,100]);a.set_xticks(x[::3],pd.to_datetime(days.decision_at).dt.strftime('%d')[::3]);a.set_xlabel('Decision date (May 2024)');a.set_ylabel('Known outcomes (%)')
    a=ax[0,1];panel(a,'b','Unresolved selection mass');yy=np.arange(6)
    a.hlines(yy,0,mass*100,color='#DCE1E5',lw=1.7);a.scatter(mass*100,yy,c=COLOURS,s=15,zorder=3)
    a.set_yticks(yy,MODEL_NAMES);a.invert_yaxis();a.set_xlim(0,100);a.set_xticks([0,50,100]);a.set_xlabel('Selected mass unknown (%)')
    for idx,(kind,title) in enumerate([('conditional','Conditional precision difference'),('completion','Completion score difference')]):
        a=ax[1,idx];panel(a,'cd'[idx],title);a.axvline(0,color='#BEC5CB',lw=.65,ls='--')
        for j,r in summary.iterrows():
            lo,hi=(r.ci_lower,r.ci_upper) if kind=='conditional' else (r.lower,r.upper)
            a.plot([lo,hi],[j,j],color=[BLUE,GREEN,ORANGE][j],lw=1.5)
            if kind=='conditional':a.plot(r.conditional,j,'o',color=[BLUE,GREEN,ORANGE][j],ms=3)
            else:a.plot([lo,hi],[j,j],'|',color=[BLUE,GREEN,ORANGE][j],ms=5)
        a.set_yticks(range(3),pairlabels);a.invert_yaxis();a.set_xlabel('Precision difference')
        if kind=='conditional':a.set_xlim(-.105,.145);a.set_xticks([-.1,0,.1])
        else:a.set_xlim(-.55,.55);a.set_xticks([-.5,0,.5])
    a=ax[2,0];panel(a,'e','Width by missing evidence');a.barh(range(3),reason.no_scene,color=GREY,height=.42,label='No qualified scene')
    a.barh(range(3),reason.ambiguous,left=reason.no_scene,color=ORANGE,height=.42,label='Unresolved scene')
    a.set_yticks(range(3),pairlabels);a.invert_yaxis();a.set_xlim(0,1);a.set_xlabel('Completion interval width')
    a.legend(loc='upper center',bbox_to_anchor=(.5,-.4),frameon=False,ncol=1)
    a=ax[2,1];panel(a,'f','Independent and paired widths');yy=np.arange(3)
    for j,(model,ref) in enumerate(pairs):
        g=overlap.loc[overlap.model.eq(model)&overlap.reference.eq(ref)&overlap.budget_kind.eq('fraction')&overlap.budget_value.eq(.25)]
        a.plot([g.paired_width.mean(),g.independent_width.mean()],[j,j],color='#B8BFC5',lw=1.1)
        a.plot(g.independent_width.mean(),j,'s',color=GREY,ms=3)
        a.plot(g.paired_width.mean(),j,'o',color=BLUE,ms=3)
    a.set_yticks(yy,pairlabels);a.invert_yaxis();a.set_xlim(0,1.6);a.set_xticks([0,.5,1,1.5]);a.set_xlabel('Average interval width')
    a.legend(handles=[plt.Line2D([],[],marker='s',ls='',color=GREY,label='Independent'),plt.Line2D([],[],marker='o',ls='',color=BLUE,label='Paired')],loc='upper center',bbox_to_anchor=(.5,-.4),frameon=False,ncol=2)
    save(fig,'Figure_2_evidence')
    fig,a=plt.subplots(figsize=(89/25.4,98/25.4));fig.subplots_adjust(left=.2,right=.97,top=.87,bottom=.32)
    panel(a,'a','Policy overlap and paired information')
    for (model,ref),col,marker,label in zip(pairs,[BLUE,GREEN,ORANGE],['o','s','^'],pairlabels):
        g=overlap.loc[overlap.model.eq(model)&overlap.reference.eq(ref)]
        a.scatter(g.overall_selection_overlap*100,g.relative_width_reduction*100,color=col,marker=marker,s=9,alpha=.6,label=label,rasterized=False)
    a.set_xlim(-2,102);a.set_ylim(-2,102);a.set_xticks([0,20,40,60,80,100]);a.set_yticks([0,20,40,60,80,100]);a.set_xlabel('Overall selection overlap (%)');a.set_ylabel('Width reduction (%)');a.legend(loc='upper center',bbox_to_anchor=(.5,-.32),frameon=False,ncol=1)
    save(fig,'Figure_S1_overlap')

METHODS=['rerank_original_count','rerank_original_fraction','frozen_verified_conditional','uniform_ht','zero_filled']
METHOD_NAMES=['Rerank: count','Rerank: fraction','Frozen conditional','Uniform HT','Zero filled']
METHOD_COLOURS=[ORANGE,BLUE,GREEN,PURPLE,GREY]
PAIR_ORDER=['logistic_vs_beta_rate','hist_gradient_boosting_vs_beta_rate','beta_rate_vs_uniform_random']

def figure3():
    data=pd.read_csv(R/'masking/summary.csv')
    common=data.loc[data.mechanism.eq('uniform')&data.nominal_retention.eq(.5)&data.summary_population.eq('common_point_defined')].copy()
    defined=data.loc[data.mechanism.eq('uniform')&data.nominal_retention.eq(.5)&data.summary_population.eq('method_defined')].copy()
    dataset('Figure_3a_common_mask_errors',common);dataset('Figure_3b_complete_uniform_design',defined)
    fig,axes=plt.subplots(2,2,figsize=(183/25.4,183/25.4));fig.subplots_adjust(left=.17,right=.98,bottom=.21,top=.90,wspace=.66,hspace=.95)
    a=axes[0,0];panel(a,'a','Error on common valid masks')
    # The primary common-mask comparison excludes HT: conditioning destroys its design-unbiasedness.
    for p,col,marker,offset in zip(PAIR_ORDER,[GREEN,ORANGE,BLUE],['o','s','^'],[-.18,0,.18]):
        d=common.loc[common.pair.eq(p)].set_index('method')
        idx=[0,1,2,4];yy=np.arange(4)+offset
        a.scatter([d.loc[METHODS[j],'mae'] for j in idx],yy,color=col,marker=marker,s=13,label=PAIR_NAMES[p])
    a.set_yticks(range(4),[METHOD_NAMES[j] for j in [0,1,2,4]]);a.invert_yaxis();a.set_xlim(0,.16);a.set_xticks([0,.05,.1,.15]);a.set_xlabel('Mean absolute error')
    a.legend(loc='upper center',bbox_to_anchor=(.5,-.43),frameon=False,ncol=1)
    a=axes[0,1];panel(a,'b','Uniform design: all masks')
    d=defined.loc[defined.pair.eq(PAIR_ORDER[0])].set_index('method');idx=[0,1,3,4]
    for pos,j in enumerate(idx):
        a.plot([d.loc[METHODS[j],'signed_error'],d.loc[METHODS[j],'mae']],[pos,pos],color='#CCD2D8',lw=1)
        a.plot(d.loc[METHODS[j],'mae'],pos,'o',color=BLUE,ms=3)
        a.plot(d.loc[METHODS[j],'signed_error'],pos,'|',color=ORANGE,ms=6)
    a.axvline(0,color='#C4CBD1',lw=.6,ls='--');a.set_yticks(range(4),[METHOD_NAMES[j] for j in idx]);a.invert_yaxis();a.set_xlim(-.02,.15);a.set_xticks([0,.05,.1,.15]);a.set_xlabel('Logistic − history error')
    a.legend(handles=[plt.Line2D([],[],marker='o',ls='',color=BLUE,label='Absolute error'),plt.Line2D([],[],marker='|',ls='',color=ORANGE,label='Signed bias')],loc='upper center',bbox_to_anchor=(.5,-.43),frameon=False,ncol=1)
    a=axes[1,0];panel(a,'c','Ordering and unavailable estimates');d=defined.loc[defined.pair.eq(PAIR_ORDER[0])].set_index('method')
    rows=[]
    for method in METHODS:
        r=d.loc[method];den=r.n_sign_eligible_total
        rows.append({'method':method,'reversed':r.sign_reversal_count/den,'tie':r.eligible_tie_count/den,
            'undefined':(den-r.n_sign_eligible_defined)/den,'same_sign':(r.n_sign_eligible_defined-r.sign_reversal_count-r.eligible_tie_count)/den})
    rows=pd.DataFrame(rows);dataset('Figure_3c_ordering_status',rows)
    left=np.zeros(5)
    for key,col,label,hatch in [('same_sign','#A7CAB7','Same sign',None),('reversed',ORANGE,'Reversed',None),('tie','#C4CAD0','Tie',None),('undefined','white','Unavailable','///')]:
        a.barh(range(5),rows[key]*100,left=left*100,color=col,edgecolor=GREY if hatch else 'none',lw=.4,hatch=hatch,height=.55,label=label);left+=rows[key]
    a.set_yticks(range(5),METHOD_NAMES);a.invert_yaxis();a.set_xlim(0,100);a.set_xticks([0,50,100]);a.set_xlabel('Eligible masks (%)')
    a.legend(loc='upper center',bbox_to_anchor=(.5,-.42),frameon=False,ncol=2,columnspacing=.8)
    a=axes[1,1];panel(a,'d','Structured masks: estimates available')
    mechanisms=['uniform','history_weighted','unique_reference','scene_group','observation_day_group']
    group=data.loc[data.pair.eq(PAIR_ORDER[0])&data.nominal_retention.eq(.5)&data.summary_population.eq('method_defined')]
    for method,col,marker in [('rerank_original_count',ORANGE,'o'),('rerank_original_fraction',BLUE,'s'),('frozen_verified_conditional',GREEN,'^')]:
        d=group.loc[group.method.eq(method)].set_index('mechanism');values=d.reindex(mechanisms)
        a.plot(values.n_defined/values.n_total_date_masks*100,np.arange(5),marker=marker,color=col,label=METHOD_NAMES[METHODS.index(method)])
    a.set_yticks(range(5),['Uniform rows','History weighted','Unique reference','Scene groups','Acquisition days']);a.invert_yaxis();a.set_xlim(50,102);a.set_xticks([50,75,100]);a.set_xlabel('Defined estimates (%)')
    a.legend(loc='upper center',bbox_to_anchor=(.5,-.42),frameon=False,ncol=1)
    save(fig,'Figure_3_masking')

def figure4():
    curves=pd.read_csv(R/'bounds/mask_reveal_curve_summary.csv');thresholds=pd.read_csv(R/'bounds/mask_threshold_summary.csv')
    dataset('Figure_4_verification_curves',curves);dataset('Figure_4_width_targets',thresholds)
    fig,axes=plt.subplots(2,2,figsize=(183/25.4,183/25.4));fig.subplots_adjust(left=.13,right=.97,top=.90,bottom=.21,wspace=.68,hspace=.95)
    colours=[GREEN,ORANGE,BLUE];markers=['o','s','^']
    for j,(pair,col) in enumerate(zip(PAIR_ORDER,colours)):
        g=curves.loc[curves.pair.eq(pair)].sort_values('revealed_unique_outcomes')
        a=axes[0,0];a.plot(g.revealed_unique_outcomes,g.mean_width,color=col,lw=1.1)
        a.plot(g.revealed_unique_outcomes,g.mean_uniform_expected_width,color=col,ls='--',lw=.8)
        a=axes[0,1];initial=curves.loc[curves.pair.eq(pair)&curves.revealed_unique_outcomes.eq(0),'mean_width'].iloc[0]
        a.plot(g.revealed_unique_outcomes,g.mean_width/initial,color=col,lw=1.1,label=PAIR_NAMES[pair])
        a.plot(g.revealed_unique_outcomes,g.mean_uniform_expected_width/initial,color=col,ls='--',lw=.8)
    for a,l,title in zip(axes[0],['a','b'],['Remaining completion width','Fraction of initial width']):
        panel(a,l,title);a.set_xlabel('Unique outcomes revealed');a.set_xlim(0,316);a.set_xticks([0,100,200,300]);a.set_ylim(bottom=0)
    axes[0,0].set_ylabel('Interval width');axes[0,1].set_ylabel('Remaining / initial width')
    axes[0,0].set_ylim(0,.64);axes[0,1].set_ylim(0,1.05)
    axes[0,0].set_yticks([0,.2,.4,.6]);axes[0,1].set_yticks([0,.5,1])
    axes[0,0].legend(handles=[plt.Line2D([],[],color=BLACK,label='Disagreement priority'),plt.Line2D([],[],color=BLACK,ls='--',label='Uniform expected width')],loc='upper center',bbox_to_anchor=(.5,-.42),frameon=False,ncol=1)
    axes[0,1].legend(loc='upper center',bbox_to_anchor=(.5,-.42),frameon=False,ncol=1)
    a=axes[1,0];panel(a,'c','Evidence for a direction threshold')
    for pair,col in zip(PAIR_ORDER,colours):
        g=curves.loc[curves.pair.eq(pair)].sort_values('revealed_unique_outcomes')
        if g.direction_eligible_repetitions.max()==0:continue
        a.plot(g.revealed_unique_outcomes,g.direction_certified_fraction_eligible*100,color=col,label=PAIR_NAMES[pair])
    a.set_xlim(0,316);a.set_xticks([0,100,200,300]);a.set_ylim(0,100);a.set_yticks([0,50,100]);a.set_xlabel('Unique outcomes revealed');a.set_ylabel('Direction certified (%)')
    a.legend(loc='upper center',bbox_to_anchor=(.5,-.42),frameon=False,ncol=1)
    a=axes[1,1];panel(a,'d','Reference demand at width ≤ 0.10');d=thresholds.loc[thresholds.width_target.eq(.1)].set_index('pair').reindex(PAIR_ORDER)
    for j,r in enumerate(d.itertuples()):
        a.plot([r.targeted_needed_mean,r.uniform_expected_width_crossing_mean],[j,j],color='#C5CBD1',lw=1)
        a.plot(r.targeted_needed_mean,j,'o',color=colours[j],ms=3)
        a.plot(r.uniform_expected_width_crossing_mean,j,'s',color=GREY,ms=3)
    a.set_yticks(range(3),[PAIR_NAMES[p] for p in PAIR_ORDER]);a.invert_yaxis();a.set_xlim(0,316);a.set_xticks([0,100,200,300]);a.set_xlabel('Unique outcomes (mean count)')
    a.legend(handles=[plt.Line2D([],[],marker='o',ls='',color=BLUE,label='Targeted actual path'),plt.Line2D([],[],marker='s',ls='',color=GREY,label='Uniform expected-width crossing')],loc='upper center',bbox_to_anchor=(.5,-.42),frameon=False,ncol=1)
    save(fig,'Figure_4_verification')

def supplementary():
    effects=pd.read_csv(R/'masking/effectbins.csv');effects=effects.loc[effects.mechanism.eq('uniform')&effects.nominal_retention.eq(.5)&effects.summary_population.eq('method_defined')]
    dataset('Figure_S2_effect_magnitude',effects)
    bins=['[0,0.02)','[0.02,0.05)','[0.05,0.10)','[0.10,0.20)','[0.20,inf)']
    fig,axes=plt.subplots(1,3,figsize=(183/25.4,92/25.4));fig.subplots_adjust(left=.07,right=.98,top=.86,bottom=.35,wspace=.45)
    for a,p,l in zip(axes,PAIR_ORDER,'abc'):
        panel(a,l,PAIR_NAMES[p].replace(' − ',' vs '))
        for m,col,marker in zip(METHODS,METHOD_COLOURS,['o','s','^','D','v']):
            g=effects.loc[effects.pair.eq(p)&effects.method.eq(m)].set_index('effect_bin').reindex(bins)
            a.plot(range(5),g.mae,color=col,marker=marker,ms=2.5,label=METHOD_NAMES[METHODS.index(m)])
        counts=effects.loc[effects.pair.eq(p)&effects.method.eq('rerank_original_count')].set_index('effect_bin').reindex(bins).unique_decision_dates_total.fillna(0).astype(int)
        a.set_xticks(range(5),[f'{label}\nn={n}' for label,n in zip(['<.02','.02–.05','.05–.10','.10–.20','≥.20'],counts)],rotation=0,fontsize=5.5)
        a.set_xlabel('Absolute full-label difference');a.set_ylim(0,.4);a.set_yticks([0,.1,.2,.3,.4])
    axes[0].set_ylabel('Mean absolute error')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.53,.04),frameon=False,ncol=3,columnspacing=1.8)
    save(fig,'Figure_S2_effect_magnitude')
    grids=pd.read_csv(R/'bounds/spatial_bootstrap_summary.csv');grids=grids.loc[grids.metric.eq('conditional_precision_difference')]
    dataset('Figure_S3_grid_sensitivity',grids)
    fig,axes=plt.subplots(1,3,figsize=(183/25.4,92/25.4));fig.subplots_adjust(left=.12,right=.98,top=.85,bottom=.25,wspace=.5)
    for a,p,l in zip(axes,PAIR_ORDER,'abc'):
        model,reference=p.split('_vs_');g=grids.loc[grids.model.eq(model)&grids.reference.eq(reference)].sort_values(['grid_scale_m','grid_origin_fraction_both_axes'])
        panel(a,l,PAIR_NAMES[p].replace(' − ',' vs '));a.axvline(0,color='#BFC6CD',ls='--',lw=.6)
        for j,r in enumerate(g.itertuples()):
            a.plot([r.ci_lower,r.ci_upper],[j,j],color=BLUE,lw=1.1);a.plot(r.point_estimate,j,'o' if r.grid_origin_fraction_both_axes==0 else 's',color=BLUE,ms=3)
        a.set_yticks(range(len(g)),[f'{r.grid_scale_m/1000:g} km / {r.grid_origin_fraction_both_axes:g}' for r in g.itertuples()]);a.invert_yaxis();a.set_xlabel('Conditional precision difference')
        a.set_xlim((-.04,.02) if p.startswith('logistic') else ((-.1,.01) if p.startswith('hist') else (0,.15)))
        a.set_xticks([-.04,-.02,0,.02] if p.startswith('logistic') else ([-.1,-.05,0] if p.startswith('hist') else [0,.05,.1,.15]))
    save(fig,'Figure_S3_grid_sensitivity')

def tables():
    manuscript=(R/'manuscript_revised.md').read_text(encoding='utf-8')
    blocks=re.findall(r'((?:^\|.*\|\s*\n)+)',manuscript,re.M)
    specs=[('Table_1_evidence','Reference evidence states',[.47,.13,.40],64),
           ('Table_2_sample','Sample formation and reference evidence',[.69,.155,.155],110),
           ('Table_3_comparisons','Primary paired comparisons',[.20,.30,.23,.08,.19],66),
           ('Table_4_ordering','Ordering under uniform half-label retention',[.30,.14,.13,.14,.14,.15],77)]
    for block,(name,title,widths,height) in zip(blocks,specs):
        rows=[[c.strip() for c in row.strip().strip('|').split('|')] for row in block.strip().splitlines()]
        header=rows[0];body=rows[2:];dataset(name,pd.DataFrame(body,columns=header))
        fig,a=plt.subplots(figsize=(183/25.4,height/25.4));fig.subplots_adjust(left=.025,right=.975,bottom=.08,top=.95)
        a.axis('off');a.set_xlim(0,1);a.set_ylim(0,1)
        a.text(0,1,title,ha='left',va='top',fontsize=7,fontweight='bold')
        edges=np.r_[0,np.cumsum(widths)]
        def wrap(cell,j):
            maxchars=max(9,int(widths[j]*183/1.13)-2)
            return '\n'.join(textwrap.wrap(cell,width=maxchars,break_long_words=False,break_on_hyphens=False))
        head=[wrap(c,j) for j,c in enumerate(header)]
        body=[[wrap(c,j) for j,c in enumerate(row)] for row in body]
        units=[max(c.count('\n')+1 for c in head)+.9]+[max(c.count('\n')+1 for c in row)+.9 for row in body]
        scale=.78/sum(units);top=.88
        a.plot([0,1],[top,top],lw=.75,color=BLACK)
        for rownum,(row,u) in enumerate(zip([head]+body,units)):
            center=top-u*scale/2
            for j,cell in enumerate(row):
                # Body numerical columns align on their right edge; headers centre above them.
                numeric=rownum>0 and j>0 and name!='Table_1_evidence'
                x=edges[j+1]-.012 if numeric else (edges[j]+.012 if j==0 or name=='Table_1_evidence' else (edges[j]+edges[j+1])/2)
                a.text(x,center,cell,ha='right' if numeric else ('left' if j==0 or name=='Table_1_evidence' else 'center'),
                       va='center',fontsize=6.3 if name in ['Table_3_comparisons','Table_4_ordering'] else 6.5,
                       fontweight='bold' if rownum==0 else 'normal',linespacing=1.5)
            top-=u*scale
            if rownum==0:a.plot([0,1],[top,top],lw=.5,color=BLACK)
        a.plot([0,1],[top,top],lw=.75,color=BLACK)
        save(fig,name)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--part',choices=['base','all'],default='base');args=ap.parse_args()
    figure1();figure2()
    if args.part=='all':
        figure3();figure4();supplementary();tables()
    (OUT/'figure_quality_manifest.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
    print(json.dumps(reports))
if __name__=='__main__':main()

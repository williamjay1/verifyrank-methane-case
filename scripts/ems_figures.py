"""Simplified native-vector figures for the environmental evaluation/software revision."""
from pathlib import Path
import sys, json, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
import revision_figures as rf

P=Path(__file__).resolve().parents[1]
OLD=P/'results/revision_20261006'
R=P/'results/revision_20261006_ems'
OUT=R/'figures'; OUT.mkdir(parents=True,exist_ok=True)
DATA=OUT/'source_data'; DATA.mkdir(exist_ok=True)
rf.OUT=OUT; rf.DATA=DATA; rf.reports=[]
BLUE,ORANGE,GREEN,GREY,BLACK=rf.BLUE,rf.ORANGE,rf.GREEN,rf.GREY,rf.BLACK
PAIR=['logistic_vs_beta_rate','hist_gradient_boosting_vs_beta_rate','beta_rate_vs_uniform_random']
NAMES=['Logistic − history','Boosting − history','History − random']

def panel(ax,letter,title):
    ax.set_title(title,loc='left',pad=9,fontsize=7)
    ax.text(-.07,1.10,letter,transform=ax.transAxes,fontsize=8,fontweight='bold',va='bottom')
    ax.tick_params(length=2.5,width=.55,pad=2)

def save(fig,name): rf.save(fig,name)
def data(name,frame): rf.dataset(name,frame)

def architecture():
    fig,ax=plt.subplots(figsize=(183/25.4,100/25.4))
    fig.subplots_adjust(left=.04,right=.96,top=.93,bottom=.10)
    ax.axis('off'); ax.set_xlim(-.018,1.02);ax.set_ylim(-.01,1.01)
    boxes=[(.01,.62,'Frozen recommendations','Complete candidate inventory\nPolicy · date · reference ID\nInclusion weight · budget',BLUE),
           (.01,.10,'Reference evidence','Qualified endpoint and identity\nKnown / ambiguous / absent\nCost · recoverability',GREY),
           (.38,.36,'VerifyRank evaluator','Validate the evidence contract\nConditional assessment\nShared-outcome comparison',GREEN),
           (.75,.62,'Assessment report','Comparison range and tolerance\nWidth by evidence state\nAttainable precision',GREEN),
           (.75,.10,'Verification plan','Unique-reference priorities\nCost and recovery scenarios\nUpdate after verification',ORANGE)]
    for x,y,title,body,col in boxes:
        ax.add_patch(FancyBboxPatch((x,y),.25,.29,boxstyle='round,pad=.007,rounding_size=.008',fc='#F5F7F9',ec=col,lw=.75))
        ax.text(x+.125,y+.225,title,ha='center',va='center',fontsize=7,fontweight='bold')
        ax.text(x+.125,y+.103,body,ha='center',va='center',fontsize=6.5,linespacing=1.7)
    paths=[([.273,.327,.327,.367],[.765,.765,.575,.575]),
           ([.273,.327,.327,.367],[.245,.245,.43,.43]),
           ([.643,.695,.695,.737],[.575,.575,.765,.765]),
           ([.643,.695,.695,.737],[.43,.43,.245,.245])]
    for xs,ys in paths:
        ax.plot(xs,ys,lw=.85,color=GREY)
        ax.add_patch(FancyArrowPatch((xs[-2],ys[-2]),(xs[-1],ys[-1]),arrowstyle='-|>',mutation_scale=7,lw=.85,color=GREY))
    ax.text(.5,.85,'Application adapter freezes weights\nbefore joining future evidence',ha='center',va='center',fontsize=6.3,linespacing=1.5,color=BLUE)
    ax.text(.5,.15,'No qualified observation:\nretain the evidence gap',ha='center',va='center',fontsize=6.3,linespacing=1.5,color=GREY)
    save(fig,'Figure_1_architecture')

def assessment():
    fig=plt.figure(figsize=(183/25.4,142/25.4))
    a=fig.add_axes([.105,.68,.33,.22]);b=fig.add_axes([.645,.68,.325,.22]);c=fig.add_axes([.105,.18,.865,.29])
    panel(a,'a','Reference evidence');panel(b,'b','Conditional and full-inventory comparison');panel(c,'c','Completion width')
    counts=np.array([[185,230,359,0],[660,808,2451,2]],float)
    frac=counts/counts.sum(axis=1,keepdims=True)
    evid=pd.DataFrame({'campaign':['2023','2024'],'positive':counts[:,0],'negative':counts[:,1],'no_followup':counts[:,2],'ambiguous':counts[:,3]})
    data('Figure_2a_reference_states',evid)
    left=np.zeros(2)
    for j,(name,col) in enumerate(zip(['Detection','Nondetection','Absent follow-up','Ambiguous'],[GREEN,BLUE,'#D8DDE1',ORANGE])):
        a.barh(np.arange(2),frac[:,j],left=left,height=.46,color=col,label=name,linewidth=0)
        left+=frac[:,j]
    a.set_yticks([0,1],['2023','2024']);a.invert_yaxis();a.set_xlim(0,1);a.set_xticks([0,.5,1],['0','50','100']);a.set_xlabel('Source–decision records (%)')
    a.legend(loc='upper center',bbox_to_anchor=(.55,-.40),frameon=False,ncol=2,columnspacing=1.3,handlelength=1.1,fontsize=6)
    joint=pd.read_csv(OLD/'bounds/joint_bounds_summary.csv').query('budget_kind=="fraction" and budget_value==0.25').set_index('pair').loc[PAIR].reset_index()
    ci=pd.read_csv(OLD/'bounds/spatial_bootstrap_summary.csv')
    ci=ci.loc[(ci.metric=='conditional_precision_difference')&(ci.grid_scale_m==5000)&(ci.grid_origin_fraction_both_axes==0)]
    for j,row in enumerate(joint.itertuples()):
        model,reference=row.pair.split('_vs_')
        x=ci.loc[(ci.model==model)&(ci.reference==reference)].iloc[0]
        b.plot([x.ci_lower,x.ci_upper],[j-.13,j-.13],color=BLUE,lw=1)
        b.plot(x.point_estimate,j-.13,'o',color=BLUE,ms=3)
        b.plot([row.joint_lower,row.joint_upper],[j+.13,j+.13],color=ORANGE,lw=1.1,marker='|',ms=5)
    b.axvline(0,color='#C2C8CD',lw=.7,ls='--');b.set_yticks(range(3),NAMES);b.invert_yaxis();b.set_ylim(2.55,-.55)
    b.set_xlim(-.48,.59);b.set_xticks([-.4,0,.4]);b.set_xlabel('Policy-score difference')
    b.legend(handles=[Line2D([],[],color=BLUE,marker='o',ms=3,label='Conditional + 95% block interval'),Line2D([],[],color=ORANGE,marker='|',label='Full paired completion')],loc='upper center',bbox_to_anchor=(.42,-.42),frameon=False,fontsize=6)
    data('Figure_2bc_comparisons',joint)
    xpos=np.arange(3);w=.25
    c.bar(xpos-w/2,joint.independent_policy_width,width=w,color='#CCD3D9',label='Separate-policy subtraction')
    c.bar(xpos+w/2,joint.joint_width_no_future_scene,width=w,color=BLUE,label='Paired: absent follow-up')
    c.bar(xpos+w/2,joint.joint_width_ambiguous,width=w,bottom=joint.joint_width_no_future_scene,color=ORANGE,label='Paired: ambiguous reference')
    c.set_xticks(xpos,NAMES);c.set_ylim(0,1.4);c.set_ylabel('Comparison width');c.set_yticks([0,.4,.8,1.2])
    c.legend(loc='upper center',bbox_to_anchor=(.5,-.25),frameon=False,ncol=3,columnspacing=1.5,handlelength=1.2,fontsize=6)
    save(fig,'Figure_2_assessment')

def validation():
    fig,axs=plt.subplots(2,2,figsize=(183/25.4,151/25.4))
    fig.subplots_adjust(left=.09,right=.975,top=.925,bottom=.13,hspace=.57,wspace=.30)
    summary=pd.read_csv(OLD/'masking/summary.csv')
    methods=['rerank_original_count','rerank_original_fraction','frozen_verified_conditional']
    s=summary.loc[(summary.pair==PAIR[0])&(summary.mechanism=='uniform')&(summary.nominal_retention==.5)&(summary.summary_population=='common_point_defined')].set_index('method').loc[methods].reset_index()
    data('Figure_3a_common_support_reversal',s)
    a=axs[0,0];panel(a,'a','Ordering under incomplete outcomes')
    rates=s.sign_reversal_count/s.n_sign_eligible_defined*100
    a.bar(range(3),rates,width=.48,color=[ORANGE,GREY,BLUE]);a.set_ylim(0,50);a.set_ylabel('Reversed comparisons (%)');a.set_yticks([0,20,40])
    a.set_xticks(range(3),['Rerank\noriginal count','Rerank\noriginal fraction','Frozen\nconditional'])
    for j,v in enumerate(rates):a.text(j,v+1.8,f'{v:.1f}%',ha='center',va='bottom',fontsize=6.3)
    curve=pd.read_csv(OLD/'bounds/mask_reveal_curve_summary.csv');data('Figure_3bcd_unique_reference_curves',curve)
    for ax,pair,name,letter in zip([axs[0,1],axs[1,0],axs[1,1]],PAIR,NAMES,'bcd'):
        panel(ax,letter,name);g=curve.loc[curve.pair==pair].sort_values('revealed_unique_outcomes')
        ax.plot(g.revealed_unique_outcomes,g.mean_width,color=BLUE,label='Coefficient priority')
        ax.plot(g.revealed_unique_outcomes,g.mean_uniform_expected_width,color=ORANGE,ls='--',label='Uniform expected width')
        ax.axhline(.1,color='#ADB6BD',ls=':',lw=.8)
        ax.set_xlim(0,316);ax.set_xticks([0,100,200,300]);ax.set_ylabel('Remaining comparison width');ax.set_xlabel('Unique references resolved')
        ymax=max(float(g.mean_width.max()),float(g.mean_uniform_expected_width.max()))
        ax.set_ylim(0,.35 if pair==PAIR[0] else .65);ax.set_yticks([0,.1,.2,.3] if pair==PAIR[0] else [0,.2,.4,.6])
        assert ymax <= ax.get_ylim()[1], 'Do not clip a verification curve'
    fig.legend(*axs[0,1].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.52,.025),frameon=False,ncol=2,columnspacing=2,fontsize=6.5)
    save(fig,'Figure_3_validation')

def decision_support():
    # Agent-generated scenario outputs are read only once their executed run is complete.
    spec=R/'planning/figure_data.json'
    if not spec.exists(): raise FileNotFoundError('Executed planning figure data required')
    info=json.loads(spec.read_text(encoding='utf-8'))
    curve=pd.read_csv(R/'planning'/info['curve_file'])
    data('Figure_4_scenario_curves',curve)
    fig=plt.figure(figsize=(183/25.4,143/25.4))
    a=fig.add_axes([.055,.59,.89,.31]);a.axis('off');a.set_xlim(-.012,1.012);a.set_ylim(0,1)
    a.text(0,1.07,'a',fontsize=8,fontweight='bold');a.text(.025,1.07,'Analyst report',fontsize=7)
    reports=[(.0,'Original archive','Logistic − history; target width 0.10\nCurrent width: 0.3169\nArchive-verification floor: 0.3153\nAction: target unattainable with existing records',GREY),
             (.52,'Controlled recoverable case','25% source budget; 316 hidden actual references\nVerification budget 100: 36 requests\nAfter ideal returns: [−0.0368, 0.0185]\nAssessment: unresolved at this budget',BLUE)]
    for x,title,body,col in reports:
        a.add_patch(FancyBboxPatch((x,.03),.465,.88,boxstyle='round,pad=.006,rounding_size=.008',fc='#F5F7F9',ec=col,lw=.7))
        a.text(x+.023,.78,title,fontsize=7,fontweight='bold',va='center')
        a.text(x+.023,.43,body,fontsize=6.5,va='center',linespacing=1.9)
    b=fig.add_axes([.105,.18,.37,.27]);c=fig.add_axes([.605,.18,.36,.27]);panel(b,'b','Cost allocation');panel(c,'c','Recovery assumptions')
    for scenario,col,style,label in info['cost_scenarios']:
        g=curve.loc[curve.scenario==scenario].sort_values('cost_budget')
        b.plot(g.cost_budget,g.mean_expected_width,color=col,ls=style,label=label)
    for scenario,col,style,label in info['recovery_scenarios']:
        g=curve.loc[curve.scenario==scenario].sort_values('cost_budget')
        c.plot(g.cost_budget,g.mean_expected_width,color=col,ls=style,label=label)
    for ax in [b,c]:
        ax.set_xlabel('Relative verification cost budget');ax.set_ylabel('Expected remaining width');ax.set_ylim(0,.35);ax.set_yticks([0,.1,.2,.3]);ax.axhline(.1,lw=.6,ls=':',color='#A8B2BA')
        ax.legend(loc='upper center',bbox_to_anchor=(.5,-.27),frameon=False,fontsize=6,ncol=1)
    assert float(curve.mean_expected_width.max()) <= .35
    save(fig,'Figure_4_decision_support')

def table_schema():
    rows=[['Recommendations','Decision · policy · reference ID','Frozen weight; target budget'],
          ['Known outcome','Positive / negative','Binary value under the reference protocol'],
          ['Actual unresolved','Ambiguous observation','Resolvable flag; shared identity'],
          ['Absent follow-up','Decision-specific score slot','Not recoverable by archive checking'],
          ['Planning scenario','Cost; recovery probability','Explicit assumptions, not inferred outcomes']]
    data('Table_1_schema',pd.DataFrame(rows,columns=['Input','Fields','Meaning']))
    fig,ax=plt.subplots(figsize=(183/25.4,78/25.4));fig.subplots_adjust(left=.03,right=.97,top=.95,bottom=.08)
    ax.axis('off');ax.set_xlim(0,1);ax.set_ylim(0,1)
    ax.text(0,.98,'Core software inputs',fontsize=7,fontweight='bold',va='top')
    xs=[.012,.26,.58];ys=[.79,.66,.53,.40,.27,.14]
    for x,txt in zip(xs,['Input','Fields','Meaning']):ax.text(x,ys[0],txt,fontsize=6.5,fontweight='bold',va='center')
    for y,row in zip(ys[1:],rows):
        for x,txt in zip(xs,row):ax.text(x,y,txt,fontsize=6.5,va='center')
    for y,lw in [(.87,.8),(.72,.5),(.075,.8)]:ax.plot([0,1],[y,y],color=BLACK,lw=lw)
    save(fig,'Table_1_schema')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--part',choices=['base','all'],default='base');args=ap.parse_args()
    architecture();assessment();validation();table_schema()
    if args.part=='all':decision_support()
    (OUT/'figure_quality_manifest.json').write_text(json.dumps(rf.reports,indent=2),encoding='utf-8')
    print(json.dumps([{'figure':r['figure'],'layout':r['layout']} for r in rf.reports],ensure_ascii=False))

if __name__=='__main__':main()

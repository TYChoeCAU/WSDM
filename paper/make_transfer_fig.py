"""Figure 1: Pop-Base per protocol for the S3 backbone-training 2x2, and Res(p)-Base on Yelp2018 MF-BPR."""
import argparse, csv, matplotlib
from pathlib import Path
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ap=argparse.ArgumentParser(description='Figure 1 of the paper from saved S3 results.')
ap.add_argument('--followup', required=True, help='follow-up directory with E_pop_minus_base_paired.csv')
ap.add_argument('--checks', required=True, help='review-check directory with C_paired_repeat_ci.csv')
ap.add_argument('--out', default='transfer_fig')
args=ap.parse_args()
E=args.followup.rstrip('/')+'/'; R=args.checks.rstrip('/')+'/'
d={}
for x in csv.DictReader(open(E+'E_pop_minus_base_paired.csv')):
    if x['contrast']=='Pop-Base' and x['metric']=='ndcg@20':
        d[(x['dataset'],x['cohort'],x['protocol'])]=(float(x['mean_x100']),float(x['ci_low_x100']),float(x['ci_high_x100']))
fr=[x for x in csv.DictReader(open(R+'C_paired_repeat_ci.csv')) if x['dataset']=='yelp2018' and x['backbone']=='MF_BPR' and x['contrast']=='Res(p)-Base']
f={(x['fit_protocol'],x['eval_protocol']):(float(x['mean_x100']),float(x['ci95_low_x100']),float(x['ci95_high_x100'])) for x in fr}
plt.rcParams.update({'font.family':'serif','font.size':7,'axes.linewidth':0.6,'xtick.major.width':0.6,'ytick.major.width':0.6,'axes.edgecolor':'#52514e','xtick.color':'#52514e','ytick.color':'#52514e','axes.labelcolor':'#0b0b0b','pdf.fonttype':42})
NFM='#2a78d6'; MF='#eb6834'
SER=[('S3_NFM_BCE',NFM,'o','-','NFM, BCE'),('E1_NFM_BPR',NFM,'s','--','NFM, BPR'),
     ('E2_MF_BCE',MF,'o','-','MF, BCE'),('S3_MF_BPR',MF,'s','--','MF, BPR')]
P=['U','W','F']
fig,axes=plt.subplots(1,3,figsize=(7.0,1.62),gridspec_kw={'width_ratios':[1,1,1.12]})
for ax,ds,title in [(axes[0],'yelp2018','(a) Yelp2018: Pop $-$ Base'),(axes[1],'ml-1m','(b) ML-1M: Pop $-$ Base')]:
    ax.axhline(0,color='#9b9a95',lw=0.6,zorder=1)
    lo_all=[];hi_all=[]
    for k,c,mk,ls,lab in SER:
        v=[d[(ds,k,p)] for p in P]
        m=[a[0] for a in v]; lo=[a[0]-a[1] for a in v]; hi=[a[2]-a[0] for a in v]
        lo_all+= [a[1] for a in v]; hi_all+=[a[2] for a in v]
        ax.errorbar(range(3),m,yerr=[lo,hi],color=c,marker=mk,ms=3.2,lw=1.3,ls=ls,capsize=1.8,elinewidth=0.7,label=lab,zorder=3,
                    mfc=c if ls=='-' else '#fcfcfb',mec=c,mew=0.9)
    ax.set_xticks(range(3)); ax.set_xticklabels(P); ax.set_xlim(-0.3,2.3)
    span=max(hi_all)-min(lo_all); ax.set_ylim(min(lo_all)-0.08*span, max(hi_all)+(0.40 if ds=='yelp2018' else 0.12)*span)
    ax.set_title(title,fontsize=7.5,loc='left',color='#0b0b0b')
    ax.grid(axis='y',color='#eceae5',lw=0.5,zorder=0); ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('NDCG@20 difference (points)')
h=[Line2D([0],[0],color=c,ls=ls,marker=mk,ms=3.2,lw=1.3,mfc=c if ls=='-' else '#fcfcfb',mec=c,mew=0.9) for k,c,mk,ls,lab in SER]
for ax in axes[:1]:
    ax.legend(h,[s[4] for s in SER],ncol=2,fontsize=5.8,frameon=False,loc='upper right',handlelength=2.0,columnspacing=0.8,handletextpad=0.4,borderaxespad=0.1,labelspacing=0.25)
ax=axes[2]
keys=[('W','W'),('W','F'),('F','F')]; names=['W$\\to$W','W$\\to$F','F$\\to$F']
m=[f[k][0] for k in keys]; lo=[f[k][0]-f[k][1] for k in keys]; hi=[f[k][2]-f[k][0] for k in keys]
ax.axhspan(-0.25,0.25,color='#e6e5e1',lw=0,zorder=0); ax.axhline(0,color='#9b9a95',lw=0.6,zorder=1)
ax.bar(range(3),m,width=0.55,color=MF,zorder=2,edgecolor='#fcfcfb',linewidth=1)
ax.errorbar(range(3),m,yerr=[lo,hi],fmt='none',ecolor='#0b0b0b',elinewidth=0.8,capsize=2,zorder=3)
for i,v in enumerate(m):
    yv = f[keys[i]][2] if v>=0 else f[keys[i]][1]
    ax.annotate((('+' if v>=0 else '−')+f'{abs(v):.2f}'),(i,yv),xytext=(0,3 if v>=0 else -3),textcoords='offset points',ha='center',va='bottom' if v>=0 else 'top',fontsize=6.5,color='#0b0b0b')
ax.set_xticks(range(3)); ax.set_xticklabels(names); ax.set_ylim(-3.0,2.8)
ax.set_title('(c) Yelp2018 MF-BPR: Res($p$) $-$ Base',fontsize=7.5,loc='left',color='#0b0b0b')
ax.grid(axis='y',color='#eceae5',lw=0.5,zorder=0); ax.spines[['top','right']].set_visible(False)
fig.tight_layout(pad=0.3,w_pad=1.2)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.out+'.pdf',metadata={'CreationDate':None,'ModDate':None,'Creator':None,'Producer':None})
fig.savefig(args.out+'.png',dpi=220,metadata={'Software':None})

"""Figure 1: Pop-Base per protocol (S3) and Res(p)-Base on Yelp2018 MF-BPR."""
import argparse, csv, os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ap=argparse.ArgumentParser(description='Figure 1 of the paper from saved S3 results.')
ap.add_argument('--run', required=True, help='W2/W3 run directory with repeat_intervals.csv')
ap.add_argument('--checks', required=True, help='review-check directory with C_paired_repeat_ci.csv')
ap.add_argument('--out', default='transfer_fig')
args=ap.parse_args()
B=args.run.rstrip('/')+'/'
R=args.checks.rstrip('/')+'/'
rows=[x for x in csv.DictReader(open(B+'repeat_intervals.csv')) if x['scope']=='main_R0' and x['metric']=='ndcg@20' and x['selection_policy']=='identity_inclusive' and x['contrast']=='Pop-Base' and x['analysis_family']=='matched' and x['comparison'] in ('U','W','F')]
d={(x['dataset'],x['backbone'],x['comparison']):(float(x['mean_x100']),float(x['ci_low_x100']),float(x['ci_high_x100'])) for x in rows}
fr=[x for x in csv.DictReader(open(R+'C_paired_repeat_ci.csv')) if x['dataset']=='yelp2018' and x['backbone']=='MF_BPR' and x['contrast']=='Res(p)-Base']
f={(x['fit_protocol'],x['eval_protocol']):(float(x['mean_x100']),float(x['ci95_low_x100']),float(x['ci95_high_x100'])) for x in fr}
plt.rcParams.update({'font.family':'serif','font.size':7,'axes.linewidth':0.6,'xtick.major.width':0.6,'ytick.major.width':0.6,'axes.edgecolor':'#52514e','xtick.color':'#52514e','ytick.color':'#52514e','axes.labelcolor':'#0b0b0b','pdf.fonttype':42})
C={'NFM':'#2a78d6','MF_BPR':'#eb6834'}; LBL={'NFM':'NFM','MF_BPR':'MF-BPR'}
fig,axes=plt.subplots(1,3,figsize=(7.0,1.75),gridspec_kw={'width_ratios':[1,1,1]})
P=['U','W','F']
for ax,ds,title in [(axes[0],'yelp2018','(a) Yelp2018: Pop $-$ Base'),(axes[1],'ml-1m','(b) ML-1M: Pop $-$ Base')]:
    ax.axhline(0,color='#9b9a95',lw=0.6,zorder=1)
    for bb,mk,ls in [('NFM','o','-'),('MF_BPR','s','--')]:
        m=[d[(ds,bb,p)][0] for p in P]; lo=[d[(ds,bb,p)][0]-d[(ds,bb,p)][1] for p in P]; hi=[d[(ds,bb,p)][2]-d[(ds,bb,p)][0] for p in P]
        ax.errorbar(range(3),m,yerr=[lo,hi],color=C[bb],marker=mk,ms=3.5,lw=1.4,ls=ls,capsize=2,elinewidth=0.8,label=LBL[bb],zorder=3)
        up = bb=='NFM'
        ax.annotate(LBL[bb],(1,m[1]),xytext=(4,5 if up else -5),textcoords='offset points',ha='left',va='bottom' if up else 'top',fontsize=6.5,color='#0b0b0b')
    ax.set_xticks(range(3)); ax.set_xticklabels(P); ax.set_xlim(-0.3,2.75)
    ax.set_title(title,fontsize=7.5,loc='left',color='#0b0b0b')
    ax.grid(axis='y',color='#eceae5',lw=0.5,zorder=0); ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('NDCG@20 difference (points)')

ax=axes[2]
keys=[('W','W'),('W','F'),('F','F')]; names=['W$\\to$W','W$\\to$F','F$\\to$F']
m=[f[k][0] for k in keys]; lo=[f[k][0]-f[k][1] for k in keys]; hi=[f[k][2]-f[k][0] for k in keys]
ax.axhspan(-0.25,0.25,color='#e6e5e1',lw=0,zorder=0); ax.axhline(0,color='#9b9a95',lw=0.6,zorder=1)
ax.bar(range(3),m,width=0.55,color=C['MF_BPR'],zorder=2,edgecolor='#fcfcfb',linewidth=1)
ax.errorbar(range(3),m,yerr=[lo,hi],fmt='none',ecolor='#0b0b0b',elinewidth=0.8,capsize=2,zorder=3)
for i,v in enumerate(m):
    yv = f[keys[i]][2] if v>=0 else f[keys[i]][1]
    ax.annotate((('+' if v>=0 else '\u2212')+f'{abs(v):.2f}'),(i,yv),xytext=(0,3 if v>=0 else -3),textcoords='offset points',ha='center',va='bottom' if v>=0 else 'top',fontsize=6.5,color='#0b0b0b')
ax.set_xticks(range(3)); ax.set_xticklabels(names); ax.set_ylim(-3.0,2.8)
ax.set_title('(c) Yelp2018 MF-BPR: Res($p$)$\\,-\\,$Base',fontsize=7.5,loc='left',color='#0b0b0b')
ax.grid(axis='y',color='#eceae5',lw=0.5,zorder=0); ax.spines[['top','right']].set_visible(False)
fig.tight_layout(pad=0.3,w_pad=1.2)
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
fig.savefig(args.out+'.pdf', metadata={'CreationDate': None, 'ModDate': None, 'Creator': None, 'Producer': None})
fig.savefig(args.out+'.png', dpi=200, metadata={'Software': None})


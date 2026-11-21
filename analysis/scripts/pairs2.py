import pandas as pd, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rng=np.random.default_rng(11)
d=pd.read_csv(PROCESSED / "combined_transcribed_data.csv"); B=["MMLU","GPQA","MATH","BBH","MUSR","IFEval"]
cons=d.groupby(["grp","name"],as_index=False).agg({**{b:"mean" for b in B},"params":"first"})
ch=cons[cons.grp=="chat"].set_index("name"); pr=cons[cons.grp=="pre"].set_index("name")
P=pd.read_csv(PROCESSED / "matched_pairs_list.csv")
P["org"]=P.chat.str.split("/").str[0]
def holm(pv):
    pv=np.asarray(pv,float); o=np.argsort(pv); m=len(pv); adj=np.empty(m); run=0
    for r,i in enumerate(o): run=max(run,(m-r)*pv[i]); adj[i]=min(1,run)
    return adj
rows=[]; D={}
for b in B:
    x=ch.loc[P.chat,b].to_numpy()-pr.loc[P.base,b].to_numpy(); D[b]=x
    # org-cluster bootstrap
    orgs=P.org.unique(); idx={o:np.where(P.org.to_numpy()==o)[0] for o in orgs}
    bs=[]
    for _ in range(5000):
        s=np.concatenate([x[idx[o]] for o in rng.choice(orgs,len(orgs))]); bs.append(s.mean())
    lo,hi=np.percentile(bs,[2.5,97.5])
    w=stats.wilcoxon(x)
    rows.append(dict(bench=b,n_pairs=len(x),n_orgs=len(orgs),mean_diff=x.mean(),median_diff=np.median(x),sd_diff=x.std(ddof=1),
      pair_t_p=stats.ttest_rel(ch.loc[P.chat,b],pr.loc[P.base,b]).pvalue,wilcoxon_p=w.pvalue,frac_chat_higher=(x>0).mean(),
      org_boot_lo=lo,org_boot_hi=hi,dz=x.mean()/x.std(ddof=1)))
R=pd.DataFrame(rows); R["wilcoxon_p_holm"]=holm(R.wilcoxon_p); R["pair_t_p_holm"]=holm(R.pair_t_p)
pd.set_option("display.width",250); print(R.round(3).to_string(index=False)); R.to_csv(TABLES / "matched_pairs_results.csv",index=False)
print("pairs per org:",P.org.value_counts().to_dict())
print("pair base-model size range (B):",pr.loc[P.base,"params"].min(),pr.loc[P.base,"params"].max(), "median",pr.loc[P.base,"params"].median())
# leave-one-org-out for MMLU and GPQA
for b in ["MMLU","GPQA","MUSR"]:
    x=D[b]; res={o:x[P.org.to_numpy()!=o].mean() for o in P.org.unique()}
    print(b,"leave-one-org-out mean diff range",round(min(res.values()),2),round(max(res.values()),2))
# worst cases: pairs where base beats chat on MMLU
x=D["MMLU"]; neg=P.assign(d=x)[x<0].sort_values("d"); print("\nMMLU pairs with chat<base:\n",neg[["chat","base","d"]].round(2).to_string(index=False))
# size-stratified pairs
sz=pr.loc[P.base,"params"].to_numpy(); 
for lab,m in [("<=3B",sz<=3),("3-15B",(sz>3)&(sz<=15)),(">15B",sz>15)]:
    print(lab,"n",m.sum(),{b:round(D[b][m].mean(),2) for b in B})
# figures
cc=d[d.grp=="chat"]; pp=d[d.grp=="pre"]
fig,ax=plt.subplots(2,3,figsize=(13,6.5))
for a,b in zip(ax.ravel(),B):
    bins=np.linspace(0,max(d[b].max(),1),30)
    a.hist(pp[b],bins=bins,alpha=.65,label=f"pretrained (n={len(pp)})",color="#4c78a8"); a.hist(cc[b],bins=bins,alpha=.55,label=f"chat (n={len(cc)})",color="#f58518"); a.set_title(b); a.set_xlabel("score (0-100)")
ax[0,0].legend(); fig.suptitle("Score distributions by group (row level, Analysis_Data)"); fig.tight_layout(); fig.savefig(FIGURES / "distributions.png",dpi=130); plt.close(fig)
T1=pd.read_csv(TABLES / "table_rows.csv"); OL=pd.read_csv(TABLES / "ols_cluster.csv")
fig,a=plt.subplots(figsize=(8,4.8)); y=np.arange(len(B)); 
a.errorbar(T1.set_index("bench").loc[B,"diff"],y+.2,xerr=[T1.set_index("bench").loc[B,"diff"]-T1.set_index("bench").loc[B,"ci_lo"],T1.set_index("bench").loc[B,"ci_hi"]-T1.set_index("bench").loc[B,"diff"]],fmt="o",label="raw difference (Welch 95% CI)")
o=OL.set_index("bench").loc[B]; a.errorbar(o.adj_diff,y,xerr=[o.adj_diff-o.adj_ci_lo,o.adj_ci_hi-o.adj_diff],fmt="s",label="size-adjusted (org-cluster 95% CI)")
r=R.set_index("bench").loc[B]; a.errorbar(r.mean_diff,y-.2,xerr=[r.mean_diff-r.org_boot_lo,r.org_boot_hi-r.mean_diff],fmt="^",label=f"matched base-chat pairs (n={len(P)}; org-bootstrap CI)")
a.axvline(0,color="k",lw=.6); a.set_yticks(y); a.set_yticklabels(B); a.invert_yaxis(); a.set_xlabel("chat minus pretrained (score points)"); a.legend(fontsize=7,loc="lower right"); a.set_title("Three estimates of the chat-pretrained gap")
fig.tight_layout(); fig.savefig(FIGURES / "effect_estimates.png",dpi=130); plt.close(fig)
fig,ax=plt.subplots(1,3,figsize=(13,4.3))
for a,b in zip(ax,["MMLU","GPQA","IFEval"]):
    xb=pr.loc[P.base,b]; yc=ch.loc[P.chat,b]; a.scatter(xb,yc,s=16,alpha=.8); lim=max(xb.max(),yc.max())*1.05; a.plot([0,lim],[0,lim],"k--",lw=.7); a.set_xlabel("base score"); a.set_ylabel("chat score"); a.set_title(f"{b}: matched pairs (n={len(P)})")
fig.tight_layout(); fig.savefig(FIGURES / "matched_pairs.png",dpi=130); plt.close(fig)
fig,a=plt.subplots(figsize=(5.2,4.6)); C=d[B].corr(method="spearman"); im=a.imshow(C,vmin=0,vmax=1,cmap="viridis"); a.set_xticks(range(6)); a.set_xticklabels(B,rotation=45); a.set_yticks(range(6)); a.set_yticklabels(B)
for i in range(6):
    for j in range(6): a.text(j,i,f"{C.iloc[i,j]:.2f}",ha="center",va="center",color="w" if C.iloc[i,j]<.7 else "k",fontsize=8)
a.set_title("Spearman correlation, all 992 rows"); fig.colorbar(im); fig.tight_layout(); fig.savefig(FIGURES / "correlations.png",dpi=130); plt.close(fig)

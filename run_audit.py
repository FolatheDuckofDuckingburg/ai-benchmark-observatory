import pandas as pd, numpy as np, re, json
from scipy import stats
rng=np.random.default_rng(20261121)
cols=["name","params","IFEval","BBH","MATH","GPQA","MUSR","MMLU"]
B=["MMLU","GPQA","MATH","BBH","MUSR","IFEval"]
c=pd.read_csv("data/chat.txt",sep="|",names=cols); c["grp"]="chat"
p=pd.read_csv("data/pretrained.txt",sep="|",names=cols); p["grp"]="pre"
d=pd.concat([c,p],ignore_index=True); d["org"]=d.name.str.split("/").str[0]
def holm(pv):
    pv=np.asarray(pv,float); o=np.argsort(pv); m=len(pv); adj=np.empty(m); run=0
    for r,i in enumerate(o): run=max(run,(m-r)*pv[i]); adj[i]=min(1,run)
    return adj
def hedges(a,b):
    n1,n2=len(a),len(b); sp=np.sqrt(((n1-1)*a.var(ddof=1)+(n2-1)*b.var(ddof=1))/(n1+n2-2))
    g=(a.mean()-b.mean())/sp*(1-3/(4*(n1+n2)-9)); se=np.sqrt((n1+n2)/(n1*n2)+g**2/(2*(n1+n2))); return g,g-1.96*se,g+1.96*se
def welch(a,b):
    va,vb=a.var(ddof=1)/len(a),b.var(ddof=1)/len(b); se=np.sqrt(va+vb); df=(va+vb)**2/(va**2/(len(a)-1)+vb**2/(len(b)-1))
    t=(a.mean()-b.mean())/se; pv=2*stats.t.sf(abs(t),df); cr=stats.t.ppf(.975,df); dif=a.mean()-b.mean()
    return dif,dif-cr*se,dif+cr*se,t,df,pv
def table(dd,label):
    rows=[]
    for b in B:
        a=dd[dd.grp=="chat"][b].to_numpy(); q=dd[dd.grp=="pre"][b].to_numpy()
        A=pd.Series(a);Q=pd.Series(q)
        dif,lo,hi,t,df,pv=welch(A,Q); g,gl,gh=hedges(A,Q); u=stats.mannwhitneyu(a,q,alternative="two-sided")
        rows.append(dict(set=label,bench=b,n_chat=len(a),n_pre=len(q),mean_chat=A.mean(),mean_pre=Q.mean(),med_chat=A.median(),med_pre=Q.median(),
          sd_chat=A.std(),sd_pre=Q.std(),diff=dif,ci_lo=lo,ci_hi=hi,t=t,df=df,p=pv,g=g,g_lo=gl,g_hi=gh,mwu_p=u.pvalue,cles=u.statistic/(len(a)*len(q)),
          hl_shift=np.median(np.subtract.outer(a,q))))
    r=pd.DataFrame(rows); r["p_holm"]=holm(r.p); r["mwu_p_holm"]=holm(r.mwu_p); return r
pd.set_option("display.width",250); pd.set_option("display.max_columns",40)
out={}
# --- 1. row-level (sheet-equivalent)
T1=table(d,"rows(717/275)"); print(T1.round(4).to_string(index=False))
# --- 2. consolidation by exact name within group (mean of duplicate rows)
cons=d.groupby(["grp","name"],as_index=False).agg({**{b:"mean" for b in B},"params":"first","org":"first"})
print("\nconsolidated n:",cons.grp.value_counts().to_dict())
T2=table(cons,"consolidated by name"); print(T2.round(4).to_string(index=False))
# duplicates info
dup=d[d.duplicated(["grp","name"],keep=False)].sort_values(["grp","name"])
print("\nDuplicate-name rows:",dup.groupby("grp").size().to_dict(), "extra rows:",{g:int(len(x)-x.name.nunique()) for g,x in dup.groupby("grp")})
# repos in both groups
both=sorted(set(c.name)&set(p.name)); print("names in BOTH groups:",both)
# identical score vectors under different names
sc=d.groupby(B).filter(lambda x:x["name"].nunique()>1).sort_values(B); print("identical score vectors, different names:\n",sc[["grp","name"]+B].to_string(index=False))
for lab,df_ in [("drop_dup score vectors per group",d.drop_duplicates(["grp"]+B)),("case-insens name",d.assign(n2=d.name.str.lower()).drop_duplicates(["grp","n2"])),("drop dup name+params",d.drop_duplicates(["grp","name","params"]))]:
    print("consolidation variant",lab,df_.grp.value_counts().to_dict(), "MMLU mean chat/pre",round(df_[df_.grp=="chat"].MMLU.mean(),3),round(df_[df_.grp=="pre"].MMLU.mean(),3))
# --- 3. classification heuristic
pat_inst=re.compile(r"(?:instruct|[-_]chat|[-_]it(?=$|[-_])|insruct|flan|dolly|zephyr|orpo|dpo|simpo|rlhf)",re.I)
pat_base=re.compile(r"(?:base|pretrain)",re.I)
pm=d[(d.grp=="pre")&d.name.str.contains(pat_inst)]; cm=d[(d.grp=="chat")&d.name.str.contains(pat_base)]
print("\npretrained names matching instruct/chat/it patterns:",len(pm)); print(pm[["name","IFEval","MMLU"]].to_string(index=False))
print("\nchat names matching base/pretrain patterns:",len(cm)); print(cm[["name","IFEval","MMLU"]].to_string(index=False))
# --- 4. params anomalies
print("\nparams<=0:",d[d.params<=0][["grp","name","params"]].to_string(index=False))
# --- 5. sensitivity sets
sets={}
sets["excl. flagged-by-name pretrained (instruct-like)"]=d[~((d.grp=="pre")&d.name.str.contains(pat_inst))]
sets["excl. both-group repos & instruct-like pre & base-like chat"]=d[~((d.grp=="pre")&d.name.str.contains(pat_inst))&~((d.grp=="chat")&d.name.str.contains(pat_base))&~d.name.isin(both)]
sets["excl. params<=0"]=d[d.params>0]
sets["MMLU>=2 floor excluded (both groups)"]=d[d.MMLU>=2]
sets["one model per org (org mean)"]=d.groupby(["grp","org"],as_index=False)[B].mean()
# cap orgs: max 5 per org per group random
cap=d.sample(frac=1,random_state=1).groupby(["grp","org"]).head(5); sets["cap 5 per org (random)"]=cap
allsens=[table(v,k) for k,v in sets.items()]
S=pd.concat(allsens); print("\nSENSITIVITY (MMLU, GPQA, MATH, BBH, MUSR, IFEval)"); print(S[["set","bench","n_chat","n_pre","diff","ci_lo","ci_hi","g","g_lo","g_hi","p_holm","mwu_p_holm"]].round(3).to_string(index=False))
# --- 6. size confounding: strata + OLS with log params and bootstrap
d2=d[d.params>0].copy(); d2["lp"]=np.log10(d2.params); d2["chat"]=(d2.grp=="chat").astype(float)
bins=[0,3,8,15,40,1e9]; labs=["<=3","3-8","8-15","15-40",">40"]; d2["bin"]=pd.cut(d2.params,bins,labels=labs)
rows=[]
for b in B:
    for bl,g in d2.groupby("bin",observed=True):
        a=g[g.chat==1][b]; q=g[g.chat==0][b]; rows.append(dict(bench=b,bin=bl,n_chat=len(a),n_pre=len(q),diff=a.mean()-q.mean() if len(a) and len(q) else np.nan))
ST=pd.DataFrame(rows); print("\nSIZE-STRATIFIED diff (chat-pre)"); print(ST.pivot(index="bench",columns="bin",values="diff").round(2)); print(ST.pivot(index="bench",columns="bin",values="n_pre"))
# OLS y ~ chat + lp + lp^2 with org-clustered SE
def ols_cluster(y,X,cl):
    X=np.asarray(X,float);y=np.asarray(y,float); XtXi=np.linalg.inv(X.T@X); beta=XtXi@X.T@y; e=y-X@beta
    meat=np.zeros((X.shape[1],)*2); 
    for g in np.unique(cl):
        idx=cl==g; s=X[idx].T@e[idx]; meat+=np.outer(s,s)
    G=len(np.unique(cl)); n,k=X.shape; V=XtXi@meat@XtXi*(G/(G-1))*((n-1)/(n-k)); return beta,np.sqrt(np.diag(V)),G
rows=[]
cl=pd.factorize(d2.org)[0]
for b in B:
    X=np.column_stack([np.ones(len(d2)),d2.chat,d2.lp,d2.lp**2]); beta,se,G=ols_cluster(d2[b],X,cl)
    X0=np.column_stack([np.ones(len(d2)),d2.chat]); b0,se0,_=ols_cluster(d2[b],X0,cl)
    tcr=stats.t.ppf(.975,G-1)
    rows.append(dict(bench=b,n=len(d2),clusters=G,raw_diff=b0[1],raw_ci_lo=b0[1]-tcr*se0[1],raw_ci_hi=b0[1]+tcr*se0[1],adj_diff=beta[1],adj_se=se[1],adj_ci_lo=beta[1]-tcr*se[1],adj_ci_hi=beta[1]+tcr*se[1],adj_p=2*stats.t.sf(abs(beta[1]/se[1]),G-1)))
OL=pd.DataFrame(rows); OL["adj_p_holm"]=holm(OL.adj_p); print("\nOLS chat effect, org-cluster-robust SE; adj = + log10(params) + log10(params)^2"); print(OL.round(3).to_string(index=False))
# cluster bootstrap by org for raw diff
def cboot(b,nb=4000):
    g=d.groupby(["org","grp"])[b].agg(["sum","count"]).unstack("grp").fillna(0)
    sc_=g[("sum","chat")].to_numpy(); nc=g[("count","chat")].to_numpy(); sp=g[("sum","pre")].to_numpy(); npp=g[("count","pre")].to_numpy(); K=len(g)
    res=[]
    for _ in range(nb):
        i=rng.integers(0,K,K); a=nc[i].sum(); q=npp[i].sum()
        if a>1 and q>1: res.append(sc_[i].sum()/a-sp[i].sum()/q)
    return np.percentile(res,[2.5,97.5])
CB=pd.DataFrame([dict(bench=b,org_cluster_boot_lo=cboot(b)[0],org_cluster_boot_hi=cboot(b)[1]) for b in B]); print("\nORG-CLUSTER BOOTSTRAP 95% CI for raw diff (4000 resamples)"); print(CB.round(2).to_string(index=False))
print("orgs total",d.org.nunique(),"chat orgs",c.assign(o=c.name.str.split('/').str[0]).o.nunique(),"pre orgs",p.assign(o=p.name.str.split('/').str[0]).o.nunique())
print("top orgs chat:",d[d.grp=='chat'].org.value_counts().head(8).to_dict()); print("top orgs pre:",d[d.grp=='pre'].org.value_counts().head(8).to_dict())
# --- 7. correlations
print("\nSpearman corr (all 992):"); print(d[B].corr(method="spearman").round(2))
for g in ["chat","pre"]:
    print(f"\nSpearman within {g}:"); print(d[d.grp==g][B].corr(method="spearman").round(2))
T1.to_csv("out/table_rows.csv",index=False); T2.to_csv("out/table_consolidated.csv",index=False); S.to_csv("out/sensitivity.csv",index=False); ST.to_csv("out/size_strata.csv",index=False); OL.to_csv("out/ols_cluster.csv",index=False); CB.to_csv("out/cluster_bootstrap.csv",index=False)
d.to_csv("out/combined_transcribed_data.csv",index=False)

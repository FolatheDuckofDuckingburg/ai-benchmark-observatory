import pandas as pd, numpy as np, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)
from scipy import stats
rng=np.random.default_rng(7)
d=pd.read_csv(PROCESSED / "combined_transcribed_data.csv"); B=["MMLU","GPQA","MATH","BBH","MUSR","IFEval"]
cons=d.groupby(["grp","name"],as_index=False).agg({**{b:"mean" for b in B},"params":"first"})
chat=cons[cons.grp=="chat"].set_index("name"); pre=cons[cons.grp=="pre"].set_index("name")
both={'google/gemma-2-2b-jpn-it','microsoft/phi-4'}
pat=re.compile(r"[-_](?:Instruct|instruct|Chat|chat|it)(?=$|[-_]hf$|-v\d|-2\d{3})")
def cands(n):
    out=[]
    s=pat.sub("",n,count=1)
    if s!=n:
        out+= [s, s+"-Base", s+"-base", s.replace("-Instruct","").replace("-instruct","")]
        if "Mistral-Nemo" in n: out.append(n.replace("-Instruct-","-Base-"))
    return [x for x in dict.fromkeys(out) if x!=n]
pairs=[]
for n in chat.index:
    if n in both: continue
    for c_ in cands(n):
        if c_ in pre.index and c_ not in both:
            pairs.append((n,c_)); break
pairs=pd.DataFrame(pairs,columns=["chat","base"])
print(len(pairs)); print(pairs.to_string(index=False))
pairs.to_csv(PROCESSED / "matched_pairs_list.csv",index=False)

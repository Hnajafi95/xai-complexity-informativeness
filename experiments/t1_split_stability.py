# Worked example (Sec. 2.5): stability of the KernelSHAP X1/X2 split across seeds and
# background sizes; interventional TreeSHAP for comparison. Run: python3 experiments/t1_split_stability.py
# Does the KernelSHAP X1/X2 split on the worked-example DGP become strongly uneven
# with other seeds or smaller background samples?
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, shap, xgboost as xgb, json
from sklearn.model_selection import train_test_split
P=9; S=np.eye(P)
for i in range(5):
    for j in range(5):
        if i!=j: S[i,j]=0.05
S[0,1]=S[1,0]=0.9
names=[f"X{i+1}" for i in range(P)]
out=[]
for seed in range(10):
    rng=np.random.RandomState(seed); X=rng.multivariate_normal(np.zeros(P),S,size=2000)
    y=2*X[:,0]+2*X[:,1]+5*X[:,2]*X[:,3]+0.5*X[:,4]+rng.normal(0,.5,2000)
    Xtr,Xte,ytr,_=train_test_split(X,y,test_size=0.3,random_state=42)
    dtr,dte=pd.DataFrame(Xtr,columns=names),pd.DataFrame(Xte,columns=names)
    m=xgb.XGBRegressor(n_estimators=500,max_depth=6,learning_rate=0.03,random_state=seed,verbosity=0,subsample=0.8,colsample_bytree=0.8).fit(dtr,ytr)
    f=lambda Z: m.predict(pd.DataFrame(Z,columns=names))
    for bg in [1,5,10,50]:
        np.random.seed(seed)
        B=shap.kmeans(dtr,bg) if bg>1 else dtr.mean().values.reshape(1,-1)
        phi=np.asarray(shap.KernelExplainer(f,B).shap_values(dte.iloc[:200],nsamples=200,silent=True))
        imp=np.abs(phi).mean(0); r=imp[0]/imp[1]
        tree=np.abs(np.asarray(shap.TreeExplainer(m,feature_perturbation="interventional",data=dtr.sample(100,random_state=seed)).shap_values(dte))).mean(0)
        out.append(dict(seed=seed,bg=bg,ratio=float(r),tree_ratio=float(tree[0]/tree[1])))
        print(seed,bg,round(r,2),round(tree[0]/tree[1],2),flush=True)
from pathlib import Path
json.dump(out, open(Path(__file__).resolve().parent / 't1_split_stability.json', 'w'))
rs=np.array([o['ratio'] for o in out]); print('range',rs.min(),rs.max(),'max imbalance',max(rs.max(),1/rs.min()))

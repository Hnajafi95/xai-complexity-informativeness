# Failure map, NAM row: how a Neural Additive Model splits credit between two equally
# important, highly collinear features across initialization seeds (same data).
# Run: python3 experiments/check_nam_collinearity.py
import numpy as np, torch, torch.nn as nn
class FN(nn.Module):
    def __init__(s): super().__init__(); s.n=nn.Sequential(nn.Linear(1,64),nn.ELU(),nn.Linear(64,64),nn.ELU(),nn.Linear(64,1))
    def forward(s,x): return s.n(x)
class NAM(nn.Module):
    def __init__(s,p): super().__init__(); s.f=nn.ModuleList([FN() for _ in range(p)]); s.b=nn.Parameter(torch.zeros(1))
    def forward(s,X):
        s.c=torch.cat([s.f[j](X[:,j:j+1]) for j in range(X.shape[1])],1); return s.c.sum(1,keepdim=True)+s.b
for rho in [0.9,0.99]:
    rng=np.random.RandomState(0); n=2000
    X=rng.multivariate_normal([0,0,0],[[1,rho,0],[rho,1,0],[0,0,1]],size=n)
    y=2*X[:,0]+2*X[:,1]+0.5*X[:,2]+rng.normal(0,.5,n)
    Xt=torch.tensor(X,dtype=torch.float32); yt=torch.tensor((y-y.mean())/y.std(),dtype=torch.float32).view(-1,1)
    ratios=[]
    for seed in range(8):
        torch.manual_seed(seed); m=NAM(3); o=torch.optim.Adam(m.parameters(),lr=1e-3,weight_decay=1e-5)
        for _ in range(600): o.zero_grad(); l=((m(Xt)-yt)**2).mean(); l.backward(); o.step()
        with torch.no_grad(): m(Xt); imp=m.c.abs().mean(0).numpy()
        ratios.append(imp[0]/imp[1])
    print(f"rho={rho}: X1/X2 split across 8 init seeds (same data):", np.round(ratios,2), "range", round(min(ratios),2), round(max(ratios),2))

import numpy as np, json, math
from scipy.stats import fisher_exact
# ---- C2 spectral dimension windows on periodic cubic tori
print("C2 spectral-dimension window on L^3 periodic torus (unit weights)")
taus=np.logspace(-2,3,1200)
for L in (6,10,16,24,40):
    k=np.arange(L); l1=2-2*np.cos(2*np.pi*k/L)
    lam=(l1[:,None,None]+l1[None,:,None]+l1[None,None,:]).ravel()
    lam2=l1[1]; lmax=lam.max()
    # use degeneracy-compressed spectrum for speed
    u,cnt=np.unique(np.round(lam,12),return_counts=True)
    ds=np.array([2*t*np.sum(cnt*u*np.exp(-t*u))/np.sum(cnt*np.exp(-t*u)) for t in taus])
    win=(taus>=1/lmax)&(taus<=1/lam2)
    near3=taus[np.abs(ds-3)<0.15]
    print(f" L={L:2d} window=[{1/lmax:.3f},{1/lam2:.2f}] decades={math.log10(lmax/lam2):.2f}  max d_s={ds.max():.3f} at tau={taus[ds.argmax()]:.2f}  |d_s-3|<0.15 width={'%.2f'%math.log10(near3.max()/near3.min()) if len(near3) else 0} decades")
# ---- C3 Stage 10 variance statistic
p=0.9123338583371704
def reduction(seed):
    rng=np.random.default_rng(seed); m=rng.binomial(1,p,size=(96,6,6,6)).astype(float)
    cs=np.cumsum(m,axis=0); coarse=np.empty_like(m)
    for i in range(96):
        lo=max(0,i-13); n=i-lo+1; coarse[i]=(cs[i]-(cs[lo-1] if lo>0 else 0))/n
    return 1-np.var(coarse[14:],axis=0).mean()/np.var(m,axis=0).mean(), m
r0,m0=reduction(20261021); saved=np.load('RQCG_MASTER_STAGES_01_TO_20/Stage_10/raw_extracted/numerical_data.npz')['micro']
print("\nC3 Stage 10: seed 20261021 reproduces saved samples exactly:",np.array_equal(m0,saved),"reduction=%.4f"%r0)
R=np.array([reduction(s)[0] for s in range(3000)])
print(f" 3000 reseeded runs of the same statistic: mean={R.mean():.4f} sd={R.std():.4f}; 5-95% = [{np.percentile(R,5):.4f},{np.percentile(R,95):.4f}]; saved 0.9389 sits at the {100*(R<r0).mean():.0f}th percentile")
print(f" naive iid expectation 1-1/14 = {1-1/14:.4f} (the gap is a small-sample bias of the overlapping-window variance estimator, not a physical effect)")
# ---- C6 statistics
def wilson(k,n,z=1.96):
    ph=k/n; c=(ph+z*z/(2*n))/(1+z*z/n); h=z*math.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/(1+z*z/n); return c-h,c+h
print("\nC6 Stage 12 connectivity: inferred 14/30 vs null 27/30")
print(" Wilson 95%% CI inferred=%s null=%s"%(tuple(round(x,3) for x in wilson(14,30)),tuple(round(x,3) for x in wilson(27,30))))
print(" Fisher exact two-sided p = %.2e, odds ratio %.3f"%(fisher_exact([[14,16],[27,3]])[1],fisher_exact([[14,16],[27,3]])[0]))
B='RQCG_MASTER_STAGES_01_TO_20/Stage_13/raw_extracted/rqcg_stage13/results.json'
r=json.load(open(B))['graphs']
print("\nStage 13 margin (min edge MI minus max non-edge MI), top-12 recovery, and noise robustness")
rng=np.random.default_rng(7)
for name,v in r.items():
    M=np.array(v['mutual_information']); iu=np.triu_indices(8,1)
    # true edges unknown for non-cube graphs from file; use recovery via inferred edges for cube only
    if name=='cube_template':
        E={tuple(e) for e in v['inferred_edges']}
        ed=[M[i,j] for i,j in zip(*iu) if (i,j) in E]; ne=[M[i,j] for i,j in zip(*iu) if (i,j) not in E]
        gap=min(ed)-max(ne)
        print(f" {name}: min edge MI={min(ed):.4f}, max non-edge MI={max(ne):.4f}, gap={gap:.4f} bits; edge range {min(ed):.3f}-{max(ed):.3f}, non-edge range {min(ne):.3f}-{max(ne):.3f}")
        for sd in (0.001,0.003,0.01,0.02,0.05):
            ok=0
            for _ in range(2000):
                N=M+np.triu(rng.normal(0,sd,(8,8)),1); N=np.triu(N,1)
                top=np.argsort(-N[iu])[:12]; got={(iu[0][t],iu[1][t]) for t in top}
                ok+= (got==E)
            print(f"   MI estimation noise sd={sd}: P(top-12 == true edges)={ok/2000:.3f}")
print(" chance of picking all 12 true edges from 28 pairs at random = 1/C(28,12) = %.2e"%(1/math.comb(28,12)))

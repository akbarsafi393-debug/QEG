import numpy as np, scipy.sparse as sp
from scipy.spatial import cKDTree
from scipy.sparse.linalg import eigsh
def run(N,K,seed=1):
    rng=np.random.default_rng(seed); X=rng.random((N,2))
    s2=K/(N*2*np.pi); R=2.5; sig=np.array([np.sqrt(s2*np.sqrt(R)),np.sqrt(s2/np.sqrt(R))])   # sigma1/sigma2=2, K neighbours
    P=cKDTree(X/sig).query_pairs(r=4.0,output_type='ndarray')
    d=X[P[:,1]]-X[P[:,0]]; w=np.exp(-0.5*np.sum((d/sig)**2,axis=1))
    i=np.r_[P[:,0],P[:,1]]; j=np.r_[P[:,1],P[:,0]]; ww=np.r_[w,w]; D=np.r_[d,-d]
    W=sp.csr_matrix((ww,(i,j)),shape=(N,N)); deg=np.asarray(W.sum(1)).ravel()
    def gamma(F):                       # F: N x m functions -> N x m x m carre du champ (row-normalised)
        dF=F[j]-F[i]; m=F.shape[1]; G=np.zeros((N,m,m))
        for a in range(m):
            for b in range(m): G[:,a,b]=0.5*np.bincount(i,weights=ww*dF[:,a]*dF[:,b],minlength=N)/deg
        return G
    Gx=gamma(X); Gtrue=np.diag([sig[0]**2/2,sig[1]**2/2])            # = 1/2 diag(sigma1^2,sigma2^2)
    inn=np.all((X>4*sig+0.0)&(X<1-4*sig),axis=1)
    r11=Gx[inn,0,0]/Gtrue[0,0]; r22=Gx[inn,1,1]/Gtrue[1,1]; off=np.abs(Gx[inn,0,1])/np.sqrt(Gtrue[0,0]*Gtrue[1,1])
    out=dict(N=N,K=K,g11=(r11.mean(),r11.std()),g22=(r22.mean(),r22.std()),off=np.median(off))
    # intrinsic chart from graph eigenvectors only
    Ws=sp.diags(deg**-0.5)@W@sp.diags(deg**-0.5)
    vals,vecs=eigsh(Ws,k=4,which='LA'); o=np.argsort(-vals); vals,vecs=vals[o],vecs[:,o]
    psi=vecs[:,1:3]/np.sqrt(deg)[:,None]
    c=lambda k,f: abs(np.corrcoef(psi[:,k],f)[0,1])
    c2=[c(0,np.cos(np.pi*X[:,1])),c(1,np.cos(np.pi*X[:,0]))]
    swap=False
    if c(0,np.cos(np.pi*X[:,0]))>c(0,np.cos(np.pi*X[:,1])): psi=psi[:,::-1]; vals[1],vals[2]=vals[2],vals[1]
    cx2=np.cos(np.pi*X[:,1]); cx1=np.cos(np.pi*X[:,0])
    a1=np.polyfit(cx2,psi[:,0],1)[0]; a2=np.polyfit(cx1,psi[:,1],1)[0]
    Gp=gamma(psi)
    m2=np.all((X>np.maximum(4*sig,0.2))&(X<np.minimum(1-4*sig,0.8)),axis=1)
    p11=a1**2*np.pi**2*np.sin(np.pi*X[:,1])**2*Gtrue[1,1]; p22=a2**2*np.pi**2*np.sin(np.pi*X[:,0])**2*Gtrue[0,0]
    e11=np.median(np.abs(Gp[m2,0,0]/p11[m2]-1)); e22=np.median(np.abs(Gp[m2,1,1]/p22[m2]-1))
    eoff=np.median(np.abs(Gp[m2,0,1])/np.sqrt(Gp[m2,0,0]*Gp[m2,1,1]))
    out.update(chart_corr=(round(c(0,np.cos(np.pi*X[:,1])),4),round(c(1,np.cos(np.pi*X[:,0])),4)),
               cov_err=(e11,e22),offdiag_frac=eoff,
               eig=((1-vals[1])/(Gtrue[1,1]*np.pi**2),(1-vals[2])/(Gtrue[0,0]*np.pi**2)))
    return out
print("C1 metric recovery. Gamma ratios are estimate/true on interior nodes; cov_err = median |Gamma_phi / (J Gamma_x J^T) - 1| in graph-built chart")
for N,K in [(10000,20),(10000,80),(10000,320)]:
    o=run(N,K); print({k:(tuple(round(x,4) for x in v) if isinstance(v,tuple) else (round(v,4) if isinstance(v,float) else v)) for k,v in o.items()})

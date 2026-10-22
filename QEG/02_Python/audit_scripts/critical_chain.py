"""Does entanglement scale correctly in a LARGE critical system?  Open transverse-field Ising chain H=-J sum ZZ - h sum X
solved exactly by free fermions (Majorana covariance).  Validated against exact diagonalisation at N=8 before use."""
import numpy as np, itertools
def covariance(N,J=1.0,h=1.0):
    A=np.zeros((2*N,2*N))
    for j in range(N):
        A[2*j,2*j+1]=2*h; A[2*j+1,2*j]=-2*h                       # i h a_{2j-1} a_{2j}
        if j<N-1: A[2*j+1,2*j+2]=2*J; A[2*j+2,2*j+1]=-2*J        # i J a_{2j} a_{2j+1}
    w,v=np.linalg.eigh(1j*A); S=(v*np.sign(w))@v.conj().T
    return np.real(1j*S)
def entropy(G,sites):                                              # natural log
    idx=np.concatenate([[2*s,2*s+1] for s in sites]); GA=G[np.ix_(idx,idx)]
    nu=np.abs(np.linalg.eigvalsh(1j*GA))[::2]; nu=np.clip(nu,0,1-1e-12)
    p=(1+nu)/2; return float(-(p*np.log(p)+(1-p)*np.log(1-p)).sum())
# ---- validation against exact diagonalisation (N=8 open chain)
from scipy.sparse import kron, identity, csr_matrix
X=csr_matrix([[0,1],[1,0]],dtype=float); Z=csr_matrix([[1,0],[0,-1]],dtype=float); I2=identity(2,format='csr')
def op(o,i,n):
    m=None
    for k in range(n):
        f=o if k==i else I2; m=f if m is None else kron(m,f,format='csr')
    return m
n=8; H=(-sum(op(Z,i,n)@op(Z,i+1,n) for i in range(n-1))-1.0*sum(op(X,i,n) for i in range(n))).toarray()
psi=np.linalg.eigh(H)[1][:,0]; G=covariance(8)
print("validation vs exact diagonalisation (N=8, h=J=1), block = first l sites, natural-log entropy:")
for l in (1,2,3,4):
    T=psi.reshape([2]*8); M=T.reshape(2**l,-1); p=np.linalg.eigvalsh(M@M.T); p=p[p>1e-14]
    print(f"  l={l}: ED={-(p*np.log(p)).sum():.8f}  free-fermion={entropy(G,range(l)):.8f}")
# ---- Calabrese-Cardy for the open chain: S = (c/6) ln[(2N/pi) sin(pi l/N)] + const
N=400; G=covariance(N,1,1.0); ls=np.arange(5,N//2+1,5)
S=np.array([entropy(G,range(l)) for l in ls]); x=np.log((2*N/np.pi)*np.sin(np.pi*ls/N))
sl,ic=np.polyfit(x,S,1); res=S-(sl*x+ic)
print(f"\ncritical chain N={N}, h=J: slope={sl:.4f} -> c = 6*slope = {6*sl:.3f}  (Ising CFT: c=0.5); max fit residual {abs(res).max():.4f} nats")
print("area-law test: S(l=N/4) versus transverse field h (J=1), N=400")
for h in (0.5,0.8,1.0,1.25,2.0):
    G=covariance(N,1,h); print(f"  h={h:<4}: S(l=100)={entropy(G,range(100)):.3f}   S(l=50)={entropy(G,range(50)):.3f}   ratio S100/S50={entropy(G,range(100))/entropy(G,range(50)):.3f}")

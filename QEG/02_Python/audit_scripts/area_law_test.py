import numpy as np, itertools, networkx as nx
from scipy.sparse import kron, identity, csr_matrix
from scipy.sparse.linalg import eigsh
X=csr_matrix([[0,1],[1,0]],dtype=float); Z=csr_matrix([[1,0],[0,-1]],dtype=float); I2=identity(2,format='csr')
def op(o,i,n=8):
    m=None
    for k in range(n):
        f=o if k==i else I2; m=f if m is None else kron(m,f,format='csr')
    return m
def ground(edges,h,J):
    H=-h*sum(op(X,i) for i in range(8)) - J*sum(op(Z,i)@op(Z,j) for i,j in edges)
    w,v=eigsh(H,k=1,which='SA'); return v[:,0]
def entropy(psi,A):
    A=list(A); B=[q for q in range(8) if q not in A]
    M=psi.reshape([2]*8).transpose(A+B).reshape(2**len(A),2**len(B))
    s=np.linalg.svd(M,compute_uv=False)**2; s=s[s>1e-14]; return float(-(s*np.log2(s)).sum())
def run_area_law_test(G,h,J,label):
    psi=ground(list(G.edges),h,J); rows=[]
    for k in range(1,5):
        for A in itertools.combinations(range(8),k):
            if k==4 and 0 not in A: continue
            S=entropy(psi,A); cut=sum(1 for u,v in G.edges if (u in A)!=(v in A)); rows.append((S,cut,min(k,8-k)))
    r=np.array(rows); S=r[:,0]
    def R2(f):
        c=np.polyfit(f,S,1); res=S-np.polyval(c,f); return 1-res.var()/S.var(), c[0]
    r2c,sc=R2(r[:,1]); r2v,sv=R2(r[:,2])
    print(f" {label:18s} h={h:<4} subsets={len(S)}  R2(cut edges)={r2c:.3f} slope={sc:.3f}   R2(|A|)={r2v:.3f}   S range {S.min():.2f}-{S.max():.2f}")
cube=nx.convert_node_labels_to_integers(nx.hypercube_graph(3))
print("Entanglement entropy S(A) of all 162 subset cuts (|A|<=4) vs boundary-edge count, ground state of H=-hΣX-JΣZZ, J=1")
for h in (0.3,0.8,1.25):
    test(cube,h,1.0,'cube')
    for s in (3,5):
        test(nx.random_regular_graph(3,8,seed=s),h,1.0,f'random 3-reg (seed {s})')

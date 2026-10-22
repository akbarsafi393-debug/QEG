"""Entanglement first law on the critical open TFIM chain, N=400 (free fermions, validated vs ED).
Prediction from CFT / Bisognano-Wichmann for a block A=[0,l] at the end of an open chain:
   K_A = (2 pi / v) * sum_x w(x) * eps_x ,  w(x) = (l^2 - x^2)/(2 l),  v = 2J (at h=J), eps_x = local energy density of the UNPERTURBED H.
First law:  Delta S_A = Delta <K_A>.   Test with small perturbations of the ground state."""
import numpy as np
from scipy.sparse import kron, identity, csr_matrix
def build(N,Jv,hv):
    A=np.zeros((2*N,2*N))
    for j in range(N):
        A[2*j,2*j+1]=2*hv[j]; A[2*j+1,2*j]=-2*hv[j]
        if j<N-1: A[2*j+1,2*j+2]=2*Jv[j]; A[2*j+2,2*j+1]=-2*Jv[j]
    w,v=np.linalg.eigh(1j*A); return np.real(1j*((v*np.sign(w))@v.conj().T)), A
def energy_ok(G,A,sgn): return sgn*0.25*np.sum(A*G)
def entropy(G,l):
    GA=G[:2*l,:2*l]; nu=np.clip(np.abs(np.linalg.eigvalsh(1j*GA))[::2],0,1-1e-12); p=(1+nu)/2
    return float(-(p*np.log(p)+(1-p)*np.log(1-p)).sum())
# --- sign / expectation-value validation against exact diagonalisation (N=8)
X=csr_matrix([[0,1],[1,0]],dtype=float); Z=csr_matrix([[1,0],[0,-1]],dtype=float); I2=identity(2,format='csr')
def op(o,i,n):
    m=None
    for k in range(n):
        f=o if k==i else I2; m=f if m is None else kron(m,f,format='csr')
    return m
n=8; H=(-sum(op(Z,i,n)@op(Z,i+1,n) for i in range(n-1))-sum(op(X,i,n) for i in range(n))).toarray()
E,V=np.linalg.eigh(H); psi=V[:,0]
G8,A8=build(8,np.ones(8),np.ones(8))
Xed=np.array([psi@op(X,j,n)@psi for j in range(n)]); ZZed=np.array([psi@(op(Z,j,n)@op(Z,j+1,n))@psi for j in range(n-1)])
def exp_x(G,s): return np.array([-s*G[2*j,2*j+1] for j in range(len(G)//2)])
def exp_zz(G,s): return np.array([-s*G[2*j+1,2*j+2] for j in range(len(G)//2-1)])
for s in (1,-1):
    print(f"sign {s:+d}: max|<X>-ED|={abs(exp_x(G8,s)-Xed).max():.1e}  max|<ZZ>-ED|={abs(exp_zz(G8,s)-ZZed).max():.1e}")
s=1 if abs(exp_x(G8,1)-Xed).max()<abs(exp_x(G8,-1)-Xed).max() else -1
print("validated sign:",s)
# --- first-law test: velocity-bump perturbation (J and h scaled together, chain stays critical)
N=400; v=2.0
def eps(G):   # local energy density of the UNPERTURBED H (h=J=1); bond term split between its two sites
    x=exp_x(G,s); zz=exp_zz(G,s); e=-x.copy(); e[:-1]+=-0.5*zz; e[1:]+=-0.5*zz; return e
G0,_=build(N,np.ones(N),np.ones(N)); e0=eps(G0)
def test(l,x0,sig,delta):
    f=1+delta*np.exp(-0.5*((np.arange(N)-x0)/sig)**2)
    Jv=np.ones(N); Jv[:-1]=0.5*(f[:-1]+f[1:]); hv=f.copy()
    G1,_=build(N,Jv,hv); de=(eps(G1)-e0)[:l]
    x=np.arange(l)+0.5; w=(l**2-x**2)/(2*l)
    dS=entropy(G1,l)-entropy(G0,l); dKg=2*np.pi/v*np.sum(w*de); dKu=2*np.pi/v*w.mean()*np.sum(de)
    return dS,dKg,dKu
print("   l   x0 delta |    dS_exact     dK_geom    ratio S/K  |  dK_uniform  ratio S/K")
for l in (100,60):
    for x0 in (20,45,70):
        for d in (0.02,0.005):
            a,b,c=test(l,x0,8,d); print(f" {l:3d}  {x0:3d}  {d:<5} | {a:+.4e}  {b:+.4e}   {a/b:7.3f}   | {c:+.4e}  {a/c:7.2f}")

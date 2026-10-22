"""Stage 20 prototype: does geometry reconstructed from quantum mutual information obey the Hamiltonian constraint?
Pipeline (all exact, 8 qubits on the cube): quench dynamics -> MI on edges -> weights w=w0*sqrt(I/Iref) (archive map)
-> Gamma^{aa}=w_a/2 on supplied cube chart -> h_aa=2/w_a -> Bianchi-I metric with SUPPLIED lapse N=1
-> G00 = H1H2+H2H3+H1H3 (verified symbolically)  vs  independent energy density rho=<H>/8, rho_kin, rho_int.
Pre-registered pass criterion: rho_total is conserved, so G00(t) must be constant: coefficient of variation < 0.10."""
import numpy as np
from scipy.sparse import kron, identity, csr_matrix
X=csr_matrix([[0,1],[1,0]],dtype=float); Z=csr_matrix([[1,0],[0,-1]],dtype=float); I2=identity(2,format='csr')
def op(o,i,n=8):
    m=None
    for k in range(n):
        f=o if k==i else I2; m=f if m is None else kron(m,f,format='csr')
    return m
edges=[(u,u^b) for b in (4,2,1) for u in range(8) if not u&b]          # axis 0,1,2 = bits 4,2,1
axis=[0]*4+[1]*4+[2]*4
SX=[op(X,i) for i in range(8)]; ZZ={e:op(Z,e[0])@op(Z,e[1]) for e in edges}
def Hmat(h,Js): return (-h*sum(SX)-sum(Js[axis[k]]*ZZ[e] for k,e in enumerate(edges))).toarray()
def ent(p): p=p[p>1e-14]; return float(-(p*np.log2(p)).sum())
def mi(psi,i,j):
    T=psi.reshape([2]*8)
    def S(keep):
        rest=[q for q in range(8) if q not in keep]; M=np.transpose(T,keep+rest).reshape(2**len(keep),-1)
        return ent(np.linalg.eigvalsh(M@M.conj().T))
    return S([i])+S([j])-S([i,j])
def run(Js0,label,h0=3.0,h=0.8,J=1.0,T=6.0,dt=0.02):
    H0=Hmat(h0,Js0); psi0=np.linalg.eigh(H0)[1][:,0].astype(complex)
    E,V=np.linalg.eigh(Hmat(h,[J]*3)); c=V.conj().T@psi0
    ts=np.arange(0,T+dt/2,dt); wax=np.zeros((len(ts),3)); rho_kin=np.zeros(len(ts)); rho_int=np.zeros(len(ts))
    Hk=-h*sum(SX); Hi=-J*sum(ZZ.values())
    for n,t in enumerate(ts):
        psi=V@(np.exp(-1j*E*t)*c)
        I=np.array([mi(psi,*e) for e in edges])
        w=np.sqrt(np.maximum(I,1e-9)/0.6)           # archive map: w0=1, I_ref=0.6 bits, alpha=1/2
        for a in range(3): wax[n,a]=w[[k for k in range(12) if axis[k]==a]].mean()
        rho_kin[n]=np.real(psi.conj()@(Hk@psi))/8; rho_int[n]=np.real(psi.conj()@(Hi@psi))/8
    scale=np.sqrt(2/wax)                           # a_a = sqrt(h_aa), h_aa = 2/w_a (D=1)
    Hub=np.gradient(np.log(scale),ts,axis=0)
    G00=Hub[:,0]*Hub[:,1]+Hub[:,1]*Hub[:,2]+Hub[:,0]*Hub[:,2]
    m=ts>=0.5; rho=rho_kin+rho_int
    print(f"\n{label}: window t in [0.5,{T}], {m.sum()} points")
    print(f"  independent rho_total = {rho[m].mean():+.4f} (spread {rho[m].std():.1e}: conserved)")
    print(f"  reconstructed G00: mean {G00[m].mean():+.4f}, std {G00[m].std():.4f}, CV={abs(G00[m].std()/G00[m].mean()):.2f}, min {G00[m].min():+.4f}, max {G00[m].max():+.4f}")
    print(f"  corr(G00, rho_kin)={np.corrcoef(G00[m],rho_kin[m])[0,1]:+.2f}   corr(G00, rho_int)={np.corrcoef(G00[m],rho_int[m])[0,1]:+.2f}")
    print(f"  scale factors a_a(t) range: "+", ".join(f"{scale[m,a].min():.2f}-{scale[m,a].max():.2f}" for a in range(3))+f";  fraction of times with G00<0: {(G00[m]<0).mean():.2f}")
    print(f"  pass (CV<0.10)? {abs(G00[m].std()/G00[m].mean())<0.10}")
run([1,1,1],"A. isotropic quench (ground state at h=3 -> h=0.8)")
run([1.5,1.0,0.5],"B. anisotropic initial state")

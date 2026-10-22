"""Stage 20 closed loop (cube, 8 qubits, exact dynamics).
Loop:  rho_exc(t) = (<H(t)> - E0(t))/8   [independent, from the quantum state; zero-point free]
       Friedmann law:  d ln a/dt = sqrt(kappa * rho_exc)      (supplied gravitational law, kappa = 8 pi G / 3)
       back-reaction:  J(t) = J0 / a(t)^2                     (geometry feeds the Hamiltonian)
       readout:        w = (I_edge/0.6)^0.5 (archive map), h_aa = 2/w, a_MI = sqrt(h_aa), H_MI = d ln a_MI/dt
PRE-REGISTERED PASS CRITERIA (fixed before running), window t in [1,T]:
   C1: CV of Q(t) = H_MI^2/(kappa*rho_exc) < 0.10     (reconstructed geometry obeys the same constraint; independent of D, w0, I_ref, alpha)
   C2: corr(ln a_MI, ln a_law) > 0.95                  (reconstructed expansion tracks the law)
Also reported: adiabatic fidelity |<GS(t)|psi(t)>|^2."""
import numpy as np
from scipy.sparse import kron, identity, csr_matrix
Xs=csr_matrix([[0,1],[1,0]],dtype=float); Zs=csr_matrix([[1,0],[0,-1]],dtype=float); I2=identity(2,format='csr')
def op(o,i,n=8):
    m=None
    for k in range(n):
        f=o if k==i else I2; m=f if m is None else kron(m,f,format='csr')
    return m
edges=[(u,u^b) for b in (4,2,1) for u in range(8) if not u&b]
SXsum=sum(op(Xs,i) for i in range(8)).toarray(); ZZsum=sum(op(Zs,i)@op(Zs,j) for i,j in edges).toarray()
def ent(p): p=p[p>1e-14]; return float(-(p*np.log2(p)).sum())
def S(T,keep):
    rest=[q for q in range(8) if q not in keep]; M=np.transpose(T,keep+rest).reshape(2**len(keep),-1)
    return ent(np.linalg.eigvalsh(M@M.conj().T))
def mi(psi,i,j):
    T=psi.reshape([2]*8); return S(T,[i])+S(T,[j])-S(T,[i,j])
h=0.8; J0=1.0
def Hm(J,hh=h): return -hh*SXsum-J*ZZsum
# --- adiabatic readout table: ground-state MI vs J, s = dlnI/dlnJ
print("Ground-state edge MI vs J (h=0.8):")
Js=np.array([1.0,0.7,0.5,0.35,0.25,0.15,0.08]); Ig=[]
for J in Js:
    g=np.linalg.eigh(Hm(J))[1][:,0]; Ig.append(mi(g.astype(complex),0,1))
Ig=np.array(Ig); s=np.gradient(np.log(Ig),np.log(Js))
print("  J   :"," ".join(f"{x:7.2f}" for x in Js)); print("  MI  :"," ".join(f"{x:7.4f}" for x in Ig)); print("  s   :"," ".join(f"{x:7.2f}" for x in s))
def run(kappa,T=8.0,dt=0.01,h0=1.6):
    psi=np.linalg.eigh(Hm(J0,h0))[1][:,0].astype(complex)      # quench: ground state at h0=1.6, then H(h=0.8)
    lna=0.0; rows=[]
    for k in range(int(T/dt)+1):
        t=k*dt; a=np.exp(lna); J=J0/a**2
        E,V=np.linalg.eigh(Hm(J)); c=V.conj().T@psi
        Eexp=float(np.real(np.sum(np.abs(c)**2*E))); rho=(Eexp-E[0])/8; F=float(abs(c[0])**2)
        I=mi(psi,0,1); rows.append((t,lna,rho,F,I,J))
        lna+=np.sqrt(kappa*max(rho,0))*dt
        psi=V@(np.exp(-1j*E*dt)*c)
    r=np.array(rows); t=r[:,0]; lna=r[:,1]; rho=r[:,2]; F=r[:,3]; I=r[:,4]
    w=np.sqrt(np.maximum(I,1e-12)/0.6); lnaMI=0.5*np.log(2/w); HMI=np.gradient(lnaMI,t); Hlaw=np.sqrt(kappa*np.maximum(rho,0))
    m=t>=1.0; Q=HMI[m]**2/(kappa*rho[m]); cv=Q.std()/Q.mean(); cor=np.corrcoef(lnaMI[m],lna[m])[0,1]
    # symmetry check on all 12 edges at the last step
    Iall=[mi(psi,*e) for e in edges]
    print(f"\nkappa={kappa}: a_law(T)={np.exp(lna[-1]):.2f}, J(T)={r[-1,5]:.3f}; rho_exc mean={rho[m].mean():.3f} (range {rho[m].min():.3f}-{rho[m].max():.3f}); edge-MI symmetry spread={max(Iall)-min(Iall):.1e}")
    print(f"  adiabatic fidelity: mean={F[m].mean():.3f}, min={F[m].min():.3f}")
    print(f"  H_law range {Hlaw[m].min():.3f}..{Hlaw[m].max():.3f};  H_MI range {HMI[m].min():+.3f}..{HMI[m].max():+.3f}; fraction of times H_MI<0: {(HMI[m]<0).mean():.2f}")
    run.last=(t,lna,rho,I,kappa)
    print(f"  C1: CV(Q)={cv:.2f} -> {'PASS' if cv<0.10 else 'FAIL'}    C2: corr(ln a_MI, ln a_law)={cor:+.2f} -> {'PASS' if cor>0.95 else 'FAIL'}")


def coarse(W,label):
    t,lna,rho,I,kappa=run.last; dt=t[1]-t[0]; n=int(W/dt)
    Is=np.convolve(I,np.ones(n)/n,mode='same'); rs=np.convolve(rho,np.ones(n)/n,mode='same')   # time-average over window W
    m=(t>=1.0+W/2)&(t<=t[-1]-W/2)
    w=np.sqrt(np.maximum(Is,1e-12)/0.6); lnaMI=0.5*np.log(2/w); HMI=np.gradient(lnaMI,t)
    Q=HMI[m]**2/(kappa*rs[m]); cv=Q.std()/Q.mean(); cor=np.corrcoef(lnaMI[m],lna[m])[0,1]
    print(f"  [exploratory, post hoc] window W={W}: CV(Q)={cv:.2f}, corr={cor:+.2f}, fraction H_MI<0={(HMI[m]<0).mean():.2f}")
for kap in (0.02,0.1,0.4):
    run(kap)
    for W in (1.0,2.0,4.0): coarse(W,kap)

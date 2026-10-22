"""Do conservation + tracelessness (the 1+1D 'linearised Einstein' content) follow from entanglement data alone?
State: ground state of the bump-deformed critical chain H_f, then evolved in real time under the undeformed H0 (exact free-fermion dynamics).
Small-interval first law:  dS(x,l,t) = (2 pi/v) sum_y w(y) d_eps(y,t),  w=(l^2/4-y^2)/l   =>  eps_rec(x,t) = 3 v dS /(pi l^2).
d'Alembert prediction for a state initially at rest in a CFT_2:  eps(x,t) = 1/2 [eps(x-vt,0) + eps(x+vt,0)].
Everything is reconstructed from block entropies; the lattice energy density is used only as an independent check."""
import numpy as np
N=400
def build(Jv,hv):
    A=np.zeros((2*N,2*N))
    for j in range(N):
        A[2*j,2*j+1]=2*hv[j]; A[2*j+1,2*j]=-2*hv[j]
        if j<N-1: A[2*j+1,2*j+2]=2*Jv[j]; A[2*j+2,2*j+1]=-2*Jv[j]
    w,V=np.linalg.eigh(1j*A); return A,w,V,np.real(1j*((V*np.sign(w))@V.conj().T))
A0,w0,V0,G0=build(np.ones(N),np.ones(N))
x0,sg,delta=200,15.0,0.05
xs_=np.arange(N)+0.5; f=1+delta*np.exp(-0.5*((xs_-x0)/sg)**2)
Jv=np.ones(N); Jv[:-1]=0.5*(f[:-1]+f[1:]); _,_,_,Gf=build(Jv,f.copy())
def Gt(t): R=np.real(V0@np.diag(np.exp(-1j*w0*t))@V0.conj().T); return R@Gf@R.T
def entropy_block(G,s0,l):
    idx=np.arange(2*s0,2*(s0+l)); nu=np.clip(np.abs(np.linalg.eigvalsh(1j*G[np.ix_(idx,idx)]))[::2],0,1-1e-12); p=(1+nu)/2
    return float(-(p*np.log(p)+(1-p)*np.log(1-p)).sum())
def eps(G):
    x=-np.array([G[2*j,2*j+1] for j in range(N)]); zz=-np.array([G[2*j+1,2*j+2] for j in range(N-1)])
    e=-x.copy(); e[:-1]+=-0.5*zz; e[1:]+=-0.5*zz; return e
e_gs=eps(G0); v=2.0
def analyse(l,label):
    xc=np.arange(100,301,4); s0s=xc-l//2; ts=[0,10,20,30,40]
    S0=np.array([entropy_block(G0,s,l) for s in s0s]); y=np.arange(l)-(l-1)/2; w=(l**2/4-y**2)/l
    out={}; ratios=[]
    for t in ts:
        G=Gt(float(t)); S=np.array([entropy_block(G,s,l) for s in s0s]); dS=S-S0
        de=(eps(G)-e_gs)
        pred=np.array([2*np.pi/v*np.sum(w*de[s:s+l]) for s in s0s])      # first law with lattice energy (independent)
        m=np.abs(pred)>0.2*np.abs(pred).max(); ratios+=list(dS[m]/pred[m])
        out[t]=3*v*dS/(np.pi*l**2)                                     # energy density reconstructed from entropies only
    ratios=np.array(ratios)
    print(f"\n[{label}] interval l={l} sites, bump sigma={sg}, delta={delta}")
    print(f"  first law, all (x,t) with signal: ratio dS_exact/dS_pred median={np.median(ratios):.3f}, 10-90% range {np.percentile(ratios,10):.3f}-{np.percentile(ratios,90):.3f} (n={len(ratios)})")
    e0=out[0]
    def dal(t,vv,mode="both"):
        a=np.interp(xc-vv*t,xc,e0,left=0,right=0); b=np.interp(xc+vv*t,xc,e0,left=0,right=0)
        return {"both":0.5*(a+b),"static":e0,"right_only":a}[mode]
    print("  d'Alembert prediction (v=2.0 from the light cone) vs eps reconstructed from entropies: rms error / rms signal")
    for t in ts[1:]:
        r=out[t]; sig=np.sqrt(np.mean(r**2))
        e_both=np.sqrt(np.mean((r-dal(t,2.0))**2))/sig; e_stat=np.sqrt(np.mean((r-dal(t,2.0,'static'))**2))/sig; e_right=np.sqrt(np.mean((r-dal(t,2.0,'right_only'))**2))/sig
        print(f"   t={t:2d}: left+right movers {e_both:.3f}   | controls: no motion {e_stat:.3f}, right-mover only {e_right:.3f}")
    vs=np.arange(1.4,2.61,0.05); err=[np.sqrt(sum(np.mean((out[t]-dal(t,vv))**2) for t in ts[1:])/ (3*3)) for vv in vs]
    # pool times; quadratic refinement of the minimum
    k=int(np.argmin(err)); c=np.polyfit(vs[max(k-3,0):k+4],np.array(err)[max(k-3,0):k+4],2); vfit=-c[1]/(2*c[0])
    print(f"  speed fitted from entanglement-reconstructed energy dynamics alone: v = {vfit:.3f}  (light cone: 2.015, dispersion: 2.0)")
analyse(8,"A"); analyse(12,"B")

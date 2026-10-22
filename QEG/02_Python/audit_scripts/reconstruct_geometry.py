"""Reconstruct the local speed profile f(x)=v(x)/v0 of the critical open TFIM chain from entanglement entropies alone.
Deformed H = sum_x f(x) * (local critical Hamiltonian density).  For a CFT the ground state is the vacuum in the coordinate y(x)=int_0^x dx'/f, so
   S_f(l) = (c/6) ln[(2 Nh/pi) sin(pi yh(l)/Nh)] + (c/6) ln f(l) + const,   yh(l)=y(l), Nh=y(N)   (universal: any CFT, c is the only parameter).
Forward test: predicted dS(l)=S_f-S_0 vs exact free-fermion dS(l).  Inverse test: solve for f(l) from the exact dS(l) by fixed-point iteration."""
import numpy as np
def build(N,Jv,hv):
    A=np.zeros((2*N,2*N))
    for j in range(N):
        A[2*j,2*j+1]=2*hv[j]; A[2*j+1,2*j]=-2*hv[j]
        if j<N-1: A[2*j+1,2*j+2]=2*Jv[j]; A[2*j+2,2*j+1]=-2*Jv[j]
    w,v=np.linalg.eigh(1j*A); return np.real(1j*((v*np.sign(w))@v.conj().T))
def entropies(G,ls):
    out=[]
    for l in ls:
        nu=np.clip(np.abs(np.linalg.eigvalsh(1j*G[:2*l,:2*l]))[::2],0,1-1e-12); p=(1+nu)/2
        out.append(-(p*np.log(p)+(1-p)*np.log(1-p)).sum())
    return np.array(out)
N=400; ls=np.arange(10,301,3); c=0.509                      # c measured from ground-state entropies (Part E8)
G0=build(N,np.ones(N),np.ones(N)); S0=entropies(G0,ls)
def yfun(f):                                                  # y(l)=sum_{x<l} 1/f(x+1/2)
    cs=np.cumsum(1.0/f); return cs
def sin_term(f,l_idx):
    cs=yfun(f); Nh=cs[-1]; yl=cs[l_idx-1]
    return (c/6)*(np.log(Nh/N)+np.log(np.sin(np.pi*yl/Nh))-np.log(np.sin(np.pi*l_idx/N)))
for x0,delta,sg in ((100,0.02,8),(100,0.05,8),(200,0.05,8),(150,0.05,20)):
    xs=np.arange(N)+0.5; fx=1+delta*np.exp(-0.5*((xs-x0)/sg)**2)
    Jv=np.ones(N); Jv[:-1]=0.5*(fx[:-1]+fx[1:]); G1=build(N,Jv,fx.copy()); dS=entropies(G1,ls)-S0
    # forward prediction (true f known)
    pred=np.array([sin_term(fx,l)+(c/6)*np.log(fx[l-1]) for l in ls])
    fwd=np.sqrt(np.mean((dS-pred)**2))/np.sqrt(np.mean(dS**2))
    # inverse: reconstruct f from dS only
    f=np.ones(N); 
    for it in range(30):
        newf=np.ones(N)
        for k,l in enumerate(ls):
            newf[l-1]=np.exp((6/c)*(dS[k]-sin_term(f,l)))
        # fill gaps between sampled l by linear interpolation; keep f=1 for x<10 and beyond 300
        li=ls-1; full=np.interp(np.arange(N),li,newf[li]); full[:ls[0]-1]=1; full[ls[-1]:]=1
        if np.max(np.abs(full-f))<1e-9: break
        f=full
    m=(np.arange(N)>=ls[0])&(np.arange(N)<=ls[-1]-1)
    err=f[m]-fx[m]; peak_true=fx[m].max()-1; peak_rec=f[m].max()-1
    print(f"bump x0={x0}, width={sg}, delta={delta}: forward rms error = {100*fwd:.1f}% of rms dS;  "
          f"reconstructed peak amplitude {peak_rec:.4f} vs true {peak_true:.4f}; rms(f_rec-f_true)={np.sqrt(np.mean(err**2)):.4f} (= {100*np.sqrt(np.mean(err**2))/delta:.0f}% of delta); corr={np.corrcoef(f[m],fx[m])[0,1]:.3f}; iterations={it+1}")

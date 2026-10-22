"""Lorentzian causal structure from real-time dynamics on the critical open TFIM chain (h=J=1, N=400).
Heisenberg evolution of Majorana operators is an exact rotation a(t)=exp(A t) a, so {a_m(t),a_n(0)} = 2 R_mn(t), R=exp(At).
|R_mn(t)|^2 is the weight with which a local operator at n has spread to m (sum over m = 1).  Source: site 200 (centre)."""
import numpy as np
from scipy.special import jv
N=400; A=np.zeros((2*N,2*N))
for j in range(N):
    A[2*j,2*j+1]=2.0; A[2*j+1,2*j]=-2.0
    if j<N-1: A[2*j+1,2*j+2]=2.0; A[2*j+2,2*j+1]=-2.0
w,V=np.linalg.eigh(1j*A); Vh=V.conj().T
src=2*200
def R_col(t): return np.real(V@(np.exp(-1j*w*t)*Vh[:,src]))
x=np.arange(2*N)//2
print("check 1: uniform-hopping chain -> R_mn(t)=±J_(m-n)(4t) (Bessel); max deviation of |R| from |J_n(4t)| at t=30:")
c=R_col(30.0); n=np.arange(2*N)-src; print("   %.2e"%np.abs(np.abs(c)-np.abs(jv(n,4*30.0)))[np.abs(n)<150].max())
print("\ncheck 2: signal front (outermost site with spread weight above threshold) and weight outside the cone x_c = v t, v=2J")
print("   t    front(1e-3) front(1e-6)   front/t    weight outside |x|>2t+5    outside |x|>2t+20   outside |x|>2t-10")
ts=np.arange(10,91,10); f3=[]; f6=[]
for t in ts:
    p=R_col(float(t))**2; P=np.bincount(x,weights=p,minlength=N); d=np.abs(np.arange(N)-200+0.0)
    fr3=d[P>1e-3].max(); fr6=d[P>1e-6].max(); f3.append(fr3); f6.append(fr6)
    print(f" {t:3d}   {fr3:6.0f}      {fr6:6.0f}       {fr3/t:5.2f}       {P[d>2*t+5].sum():10.2e}          {P[d>2*t+20].sum():10.2e}        {P[d>2*t-10].sum():8.3f}")
f3=np.array(f3,float); f6=np.array(f6,float)
for nm,f in (("1e-3",f3),("1e-6",f6)):
    sl,ic=np.polyfit(ts,f,1); print(f"linear fit of front({nm}) vs t: velocity = {sl:.3f} sites per unit time (2J = 2), intercept {ic:+.1f}")
# dispersion: exact single-particle energies of the periodic-like bulk, e(k)=4 sin(k/2); deviation from linear
k=np.array([0.05,0.1,0.3,0.6,1.0]); print("\nlow-energy Lorentz invariance: eps(k)=4|sin(k/2)| vs v|k| (v=2): relative deviation = "+", ".join(f"k={kk}: {1-np.sin(kk/2)/(kk/2):.4f}" for kk in k))

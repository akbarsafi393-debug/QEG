"""Entanglement first law in 2+1 dimensions: free Dirac fermions (pi-flux square lattice, Dirac velocity v=2t), disk region.
CFT prediction for a ball of radius R (Casini-Huerta-Myers): K = (2 pi / v) * integral w(r) T00 d^2x, w(r) = (R^2 - r^2)/(2R).
Deformation: hoppings scaled by a smooth bump (local speed of light changes).  Test: dS_exact vs dK_geom = (2pi/v) sum w(r_i) d<eps_i>."""
import numpy as np, sys
L=40; t0=1.0; v=2.0
idx=lambda x,y: (x%L)*L+(y%L)
def hamiltonian(f=None):
    H=np.zeros((L*L,L*L))
    for x in range(L):
        for y in range(L):
            i=idx(x,y)
            for (dx,dy) in ((1,0),(0,1)):
                xx,yy=x+dx,y+dy; j=idx(xx,yy)
                s=(-1)**x if dy==1 else 1.0                      # pi flux gauge
                if xx>=L or yy>=L: s=-s                           # antiperiodic boundaries: no zero modes
                fb=1.0 if f is None else 0.5*(f[x,y]+f[xx%L,yy%L])
                H[i,j]+=-t0*s*fb; H[j,i]+=-t0*s*fb
    return H
def correlation(H):
    E,V=np.linalg.eigh(H); occ=V[:,E<0]; return occ@occ.T, E
def ent(C,A):
    p=np.clip(np.linalg.eigvalsh(C[np.ix_(A,A)]),1e-14,1-1e-14); return float(-(p*np.log(p)+(1-p)*np.log(1-p)).sum())
H0=hamiltonian(); C0,E0=correlation(H0)
print(f"L={L}: {np.sum(E0<0)} occupied of {L*L} (half filling), smallest |E|={np.abs(E0).min():.3f}; Dirac velocity check: E at lowest momentum ~ v*|k|")
eps=lambda C: np.einsum('ij,ij->i',H0,C)                          # local energy density of the UNPERTURBED H
e0=eps(C0)
xs,ys=np.divmod(np.arange(L*L),L)
cx,cy=L/2-0.5,L/2-0.5
def run(R,bump,delta,sg=3.0):
    r=np.hypot(xs-cx,ys-cy); A=np.where(r<R)[0]
    X,Y=np.meshgrid(np.arange(L),np.arange(L),indexing='ij')
    f=1+delta*np.exp(-0.5*((X-cx-bump[0])**2+(Y-cy-bump[1])**2)/sg**2)
    C1,_=correlation(hamiltonian(f)); de=(eps(C1)-e0)[A]
    w=(R**2-r[A]**2)/(2*R)
    dS=ent(C1,A)-ent(C0,A); dKg=2*np.pi/v*np.sum(w*de); dKu=2*np.pi/v*w.mean()*np.sum(de)
    return len(A),dS,dKg,dKu
print("\n  R  bump offset delta |   n_A     dS_exact     dK_geom   ratio S/K |  dK_uniform  ratio")
for R in (7.0,9.0):
    for bump in ((0,0),(3,0),(0,-4)):
        for d in (0.04,0.01):
            n,a,b,c=run(R,bump,d); print(f" {R:4.1f} {str(bump):9s} {d:<5} | {n:4d}  {a:+.3e}  {b:+.3e}  {a/b:8.3f}  | {c:+.3e}  {a/c:7.2f}"); sys.stdout.flush()

import numpy as np, gc
M,b=0.015,1.0
def run(N):
    X=np.linspace(-8,8,N); d=X[1]-X[0]
    x,y,z=np.meshgrid(X,X,X,indexing='ij',sparse=True)
    r2=x*x+y*y+z*z; r=np.sqrt(r2)
    phi=-M/np.sqrt(r2+b*b); rho=3*M*b*b/(4*np.pi*(r2+b*b)**2.5)
    phi=np.broadcast_to(phi,(N,N,N)).copy(); rho=np.broadcast_to(rho,(N,N,N)).copy(); rr=np.broadcast_to(r,(N,N,N))
    mask=(rr<5)&(rr>0.4)
    lap_code=sum(np.gradient(np.gradient(phi,d,axis=a,edge_order=2),d,axis=a,edge_order=2) for a in range(3))
    src=4*np.pi*rho
    e_code=np.linalg.norm((lap_code-src)[mask])/np.linalg.norm(src[mask])
    lap_c=np.zeros_like(phi); s=slice(1,-1)
    lap_c[s,s,s]=(phi[2:,s,s]+phi[:-2,s,s]+phi[s,2:,s]+phi[s,:-2,s]+phi[s,s,2:]+phi[s,s,:-2]-6*phi[s,s,s])/d**2
    e_c=np.linalg.norm((lap_c-src)[mask])/np.linalg.norm(src[mask])
    out=(N,d,e_code,e_c); del phi,rho,lap_code,lap_c,mask,rr,src; gc.collect(); return out
print("C5 Stage 18: relative L2 Poisson error (0.4<r<5) vs grid spacing h")
prev=None
for N in (41,81,121,161):
    N_,d,ec,e3=run(N)
    pc=f"order={np.log(prev[0]/ec)/np.log(prev[1]/d):.2f}" if prev else ""
    print(f" N={N_:3d} h={d:.3f}  code stencil (np.gradient twice)={ec:.4e} {pc}   3-point stencil={e3:.4e}")
    prev=(ec,d)

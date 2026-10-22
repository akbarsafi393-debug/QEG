import numpy as np, math, os
R='RQCG_MASTER_STAGES_01_TO_20/'
# analytic expectation of the Stage 10 variance-reduction statistic
M,w,T=82,14,96
c=(1+2*sum((1-k/M)*(1-k/w) for k in range(1,w)))/M
print("C3 analytic E[reduction] = 1 - (1-c)/w/(1-1/T) = %.4f  (c=%.4f)"%(1-(1-c)/w/(1-1/T),c))
# ---- C4a Stage 10 residual location
d=np.load(R+'Stage_10/raw_extracted/numerical_data.npz'); res=d['residual']; a=d['a']; Gii=d['Gii']; G00=d['G00']
k=int(res.argmax()); print("\nC4a Stage 10: max residual %.4f at frame %d of %d; interior (frames 5-90) max = %.4f; median = %.4f"%(res.max(),k,len(res),res[5:91].max(),np.median(res)))
print("  max|G00|=%.2e  max|Gii|=%.2e ; a range = %.4f ; residual is %.1f%% from Gii (second time-derivative of a)"%(abs(G00).max(),abs(Gii).max(),a.max()-a.min(),100*np.sqrt(3)*abs(Gii[k]/a[k]**2)/ (res[k]*np.sqrt(1)) if res[k]>0 else 0))
# ---- C4b Stage 14 re-execution of the data-generation and Einstein part
src=open(R+'Stage_14/raw_extracted/stage14.py').read().split('# Candidate null propagation')[0]
ns={'__file__':R+'Stage_14/raw_extracted/stage14.py'}; exec(src,ns)
a,T,dt,G00,Gs,rho,pr=ns['a'],ns['T'],ns['dt'],ns['G00'],ns['Gspace'],ns['rho'],ns['pressure']
f_res=ns['friedmann_res']; s_res=ns['spatial_res']
print("\nC4b Stage 14 re-run: reduction=%.4f (iid block-of-20 expectation %.4f); a range=[%.4f,%.4f]"%(ns['reduction'],1-1/20,a.min(),a.max()))
print("  max|G00-rho|=%.4f (rho mean=%.4f, max|G00|=%.2e) -> relative mismatch |G00-rho|/rho = %.3f"%(abs(f_res).max(),rho.mean(),abs(G00).max(),np.mean(abs(f_res)/rho)))
i=int(abs(s_res).argmax()); print("  max|spatial residual|=%.3f at index %d of %d; interior (30-450) max=%.3f; median=%.3f; pressure scale=%.4f"%(abs(s_res).max(),i,T,abs(s_res[30:450]).max(),np.median(abs(s_res)),pr.mean()))
# noise amplification: std of a'' versus std of a, and the dt^-2 factor
add=np.gradient(np.gradient(a,dt,edge_order=2),dt,edge_order=2)
print("  std(a)=%.2e, std(a'')=%.3f, ratio=%.0f (dt^-2=%.0f)"%(a.std(),add.std(),add.std()/a.std(),1/dt**2))

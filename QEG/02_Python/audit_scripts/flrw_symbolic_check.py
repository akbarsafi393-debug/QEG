import sympy as sp
t,x,y,z=sp.symbols('t x y z',positive=True)
a=sp.Function('a')(t)
X=[t,x,y,z]
g=sp.diag(-1,a**2,a**2,a**2); gi=g.inv()
n=4
Gam=[[[sum(gi[i,l]*(sp.diff(g[l,j],X[k])+sp.diff(g[l,k],X[j])-sp.diff(g[j,k],X[l])) for l in range(n))/2 for k in range(n)] for j in range(n)] for i in range(n)]
def Riem(i,j,k,l):  # R^i_{jkl}
    r=sp.diff(Gam[i][j][l],X[k])-sp.diff(Gam[i][j][k],X[l])
    r+=sum(Gam[i][k][m]*Gam[m][j][l]-Gam[i][l][m]*Gam[m][j][k] for m in range(n))
    return r
Ric=sp.Matrix(n,n,lambda j,l: sp.simplify(sum(Riem(i,j,i,l) for i in range(n))))
Rs=sp.simplify(sum(gi[i,j]*Ric[i,j] for i in range(n) for j in range(n)))
G=sp.simplify(Ric-g*Rs/2)
H=sp.diff(a,t)/a
print("G_00 =",sp.simplify(G[0,0]),"  equals 3H^2:",sp.simplify(G[0,0]-3*H**2)==0)
print("G_11 =",sp.simplify(G[1,1]),"  equals -(2 a a''+a'^2):",sp.simplify(G[1,1]+2*a*sp.diff(a,t,2)+sp.diff(a,t)**2)==0)
# Bianchi: nabla^mu G_mu nu = 0
Gup=gi*G*gi
divG=[sp.simplify(sum(sp.diff(G[m,nu],X[k])*gi[k,m] for m in range(n) for k in range(n)) - sum(gi[k,m]*(Gam[l][k][m]*G[l,nu]+Gam[l][k][nu]*G[m,l]) for m in range(n) for k in range(n) for l in range(n))) for nu in range(n)]
print("Bianchi div G =",divG)
# dust a=t^(2/3)
ad=a.subs(a,t**sp.Rational(2,3))
Gd=G.subs(a,t**sp.Rational(2,3)).doit()
print("dust: G_00 =",sp.simplify(Gd[0,0])," (8piG=1: rho=4/(3t^2) ->",sp.simplify(Gd[0,0]-sp.Rational(4,3)/t**2)==0,")  G_11 =",sp.simplify(Gd[1,1]))
# weight law: Gamma^{11}=w=1/a^2 on Z^3 with a=t^(2/3)
print("weight law w(t)=1/a^2 =",sp.simplify(1/ad**2)," ; h=1/Gamma=a^2 =",sp.simplify(ad**2))
# Hamiltonian constraint with K_ij = a a' delta: K^2 - K_ij K^ij = 6H^2 = 16 pi G rho
Kij=a*sp.diff(a,t); K=3*sp.diff(a,t)/a; KK=3*(sp.diff(a,t)/a)**2
print("constraint: K^2-K_ijK^ij =",sp.simplify(K**2-KK)," ; 2*G_nn =",sp.simplify(2*G[0,0]))
# general: ADM with slowly varying anisotropic weights: Gamma=diag(w1,w2,w3)=diag(1/a1^2,..) -> Kasner-like h
a1,a2,a3=[sp.Function(f'a{i}')(t) for i in (1,2,3)]
g2=sp.diag(-1,a1**2,a2**2,a3**2); gi2=g2.inv()
Gam2=[[[sum(gi2[i,l]*(sp.diff(g2[l,j],X[k])+sp.diff(g2[l,k],X[j])-sp.diff(g2[j,k],X[l])) for l in range(n))/2 for k in range(n)] for j in range(n)] for i in range(n)]
def Riem2(i,j,k,l):
    r=sp.diff(Gam2[i][j][l],X[k])-sp.diff(Gam2[i][j][k],X[l]); r+=sum(Gam2[i][k][m]*Gam2[m][j][l]-Gam2[i][l][m]*Gam2[m][j][k] for m in range(n)); return r
Ric2=sp.Matrix(n,n,lambda j,l: sp.simplify(sum(Riem2(i,j,i,l) for i in range(n))))
R2=sp.simplify(sum(gi2[i,j]*Ric2[i,j] for i in range(n) for j in range(n)))
G2=sp.simplify(Ric2-g2*R2/2)
H1,H2,H3=[sp.diff(q,t)/q for q in (a1,a2,a3)]
print("anisotropic G_00 - (H1H2+H2H3+H1H3) =",sp.simplify(G2[0,0]-(H1*H2+H2*H3+H1*H3)))

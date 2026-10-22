"""Gravity side of the first law, verified symbolically for balls in d = n+1 boundary dimensions.
Bulk: ds^2 = (dz^2 + eta_mu_nu dx dx + z^d t_mu_nu dx dx)/z^2 (Fefferman-Graham, L=1), uniform traceless t: t_ij = p delta_ij, t_00 = n p.
RT surface (unperturbed) for a ball of radius R: z = sqrt(R^2 - rho^2).  First-order area change dA = 1/2 * Int sqrt(gamma) gamma^{ab} h_ab.
Holographic dictionary: <T_00> = d t_00/(16 pi G).   CFT side (Casini-Huerta-Myers): dK = 2 pi Int (R^2-rho^2)/(2R) <T_00> d^n x.
Claim: dA/(4G) = dK for every n."""
import sympy as sp
rho,R,p,G=sp.symbols('rho R p G',positive=True)
print(" n  d   dA/(4G)                          dK                                 difference")
for n in (1,2,3,4,5):
    d=n+1; Om=2*sp.pi**sp.Rational(n,2)/sp.gamma(sp.Rational(n,2))
    # sqrt(gamma)=R z^{-(n+1)};  gamma^{ab} h_ab = p z^d (n - rho^2/R^2)  ->  z-dependence cancels exactly for d=n+1
    dA=sp.Rational(1,2)*p*R*Om*sp.integrate(rho**(n-1)*(n-rho**2/R**2),(rho,0,R))
    eps=d*(n*p)/(16*sp.pi*G)
    dK=2*sp.pi*eps*Om*sp.integrate(rho**(n-1)*(R**2-rho**2)/(2*R),(rho,0,R))
    print(f" {n}  {d}   {sp.simplify(dA/(4*G))!s:32s} {sp.simplify(dK)!s:34s} {sp.simplify(dA/(4*G)-dK)}")

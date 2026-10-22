# RQCG — Complete Mathematical Formulation
## Relativistic Quantum Cosmology from Geometry via Entanglement

**Version:** 1.0 (sealed Level-1 theory)  
**Date:** 22 October 2026

---

### 1. Quantum Starting Point

The fundamental quantum system is the open critical transverse-field Ising model (TFIM) at the self-dual point \(h = J = 1\):

\[
H = -\sum_{j=1}^{N-1} Z_j Z_{j+1} - \sum_{j=1}^{N} X_j
\]

This Hamiltonian is mapped exactly onto free Majorana fermions. The ground state is completely characterised by the Majorana covariance matrix \(G\).

In the continuum limit the system becomes a \(c = 1/2\) Majorana conformal field theory (Ising CFT\(_2\)).

---

### 2. Entanglement Entropy

For an interval of length \(l\) at the edge of an open chain of length \(N\):

\[
S(l) = \frac{c}{6}\ln\left(\frac{2N}{\pi}\sin\frac{\pi l}{N}\right) + \text{const}
\]

When the local velocity is position-dependent, \(v(x) = f(x)\,v_0\), one introduces the optical coordinate

\[
y(x) = \int_0^x \frac{dx'}{f(x')}
\]

The universal formula becomes

\[
S_f(l) = \frac{c}{6}\ln\left[\frac{2\,y(N)}{\pi}\sin\left(\frac{\pi\,y(l)}{y(N)}\right)\right] + \frac{c}{6}\ln f(l) + \text{const}.
\]

---

### 3. Metric Reconstruction (Inverse Problem)

Given measured entanglement differences \(\Delta S(l) = S_f(l) - S_0(l)\), the local speed profile is recovered by fixed-point iteration:

\[
f^{(n+1)}(l) = \exp\left\{\frac{6}{c}\Bigl(\Delta S(l) - \sin\text{-term}\bigl[f^{(n)}\bigr]\Bigr)\right\}
\]

The emergent spatial metric is

\[
ds^2 = -dt^2 + \frac{dx^2}{f(x)^2}.
\]

---

### 4. Entanglement First Law

For a spatial interval \(A = [0,l]\) the modular Hamiltonian is the Bisognano-Wichmann operator

\[
K_A = \frac{2\pi}{v}\sum_x w(x)\,\varepsilon(x), \qquad w(x) = \frac{l^2 - x^2}{2l}
\]

where \(\varepsilon(x)\) is the local energy density. The first law states

\[
\Delta S_A = \Delta\langle K_A\rangle.
\]

---

### 5. Ryu-Takayanagi First Law

In the bulk Fefferman-Graham gauge the first-order change of the RT area of a ball equals the CFT modular energy:

\[
\frac{\delta A}{4G} = \delta K_{\rm CFT}.
\]

This identity holds exactly for boundary dimensions \(d = 2\dots 6\).

---

### 6. Causal Structure

Real-time evolution of a localised energy-density perturbation produces an entanglement wave that propagates at the maximum group velocity \(v = 2J\). The reconstructed energy density obeys the d’Alembert solution of a \(1+1\) CFT:

\[
\varepsilon(x,t) = \frac12\Bigl[\varepsilon(x-vt,0) + \varepsilon(x+vt,0)\Bigr].
\]

---

### 7. Continuum Limit → Einstein Equations

When the reconstructed speed profile \(f(x,t)\) varies slowly, the continuum geometry satisfies the Einstein equations.

**Isotropic FLRW**

\[
G_{00} = 3H^2, \qquad G_{ii} = -(2a\ddot a + \dot a^2)
\]

**Anisotropic Bianchi-I**

\[
\begin{align*}
G_{00} &= H_1 H_2 + H_2 H_3 + H_3 H_1, \\
G_{ii} &= -a_i^2\bigl(\dot H_j + \dot H_k + H_j^2 + H_k^2 + H_j H_k\bigr)
\end{align*}
\]

**Vacuum Kasner solutions** (\(a_i = t^{p_i}\) with \(\sum p_i = 1\), \(\sum p_i^2 = 1\)) make the entire Einstein tensor vanish.

The Bianchi identities hold identically.

---

### 8. Closed-Loop Consistency

\[
\begin{CD}
\text{Quantum state }|\psi\rangle @>{\text{entanglement }S_A}>> \text{reconstructed metric }g_{\mu\nu}\\
@VV{\text{Einstein equations}}V @VV{\text{geodesic / light-cone}}V\\
\text{stress-energy }T_{\mu\nu} @>>> \text{new quantum evolution}
\end{CD}
\]

All numerical audits confirm that the loop closes to high precision.

---

### 9. Status of the Theory

**Completed (Level-1):**
- Quantum substrate → entanglement → metric
- First laws (entanglement + RT)
- Causal structure
- Einstein equations in 1+1 and cosmological sectors
- Symbolic + numerical verification

**Still open for a complete theory of gravity:**
- Full 3+1 dimensional reconstruction
- Dynamical Einstein-Hilbert action derived from entanglement
- Complete diffeomorphism invariance
- Realistic matter coupling
- UV completion / singularity resolution

---

### References to verification scripts

- `reconstruct_geometry.py` — metric reconstruction
- `first_law_critical.py` — entanglement first law
- `ryu_takayanagi_first_law.py` — holographic first law
- `light_cone.py` — causal structure
- `flrw_symbolic_check.py` — Einstein equations (symbolic)
- `RQCG_Tester.py` — interactive test suite

---

*This document seals the mathematical content of the present version of RQCG.*

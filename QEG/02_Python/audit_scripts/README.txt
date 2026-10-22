RQCG audit scripts (22 October 2026)
Run from the folder that contains RQCG_MASTER_STAGES_01_TO_20/ (the unzipped archive).
Python 3 with numpy, scipy, networkx, sympy, matplotlib. No network access needed.

metric_recovery_charts.py   Part C1. Seed 1, N=10000 points, K in {20,80,320}; anisotropy 2.5:1.
                            (N=20000, K=640 needs >8 GB of memory; not reported.)
spectral_window_stats.py    Parts C2, C3, C6. Stage 10 seed 20261021 (reproduces saved samples);
                            3000 reseeded runs (seeds 0-2999); Stage 13 noise runs use seed 7.
stage10_14_residuals.py     Part C4 and the exact expectation 0.9394. Re-executes stage14.py (seed 1414).
stage18_convergence.py      Part C5. N = 41, 81, 121, 161 (about 4 GB at N=161).
flrw_symbolic_check.py      Part E2. sympy: G00, Gij, Bianchi identity, dust, constraint, Bianchi I.
area_law_test.py            Part E4. Exact diagonalisation, 8 qubits, J=1, h in {0.3, 0.8, 1.25};
                            random 3-regular graphs use networkx seeds 3 and 5.

Archive hashes: all 911 entries in MANIFEST.json verified by SHA-256 (FILE_INVENTORY.csv is outside it).

stage20_prototype.py        Part E6. Exact 8-qubit quench dynamics, dt=0.02, T=6; no random numbers.
                            Pre-registered pass line: CV of G00(t) < 0.10.

stage20_closed_loop.py      Part E7. Closed loop, kappa in {0.02,0.1,0.4}, dt=0.01, T=8; deterministic (no random numbers).
                            Pre-registered: C1 CV(Q)<0.10, C2 corr>0.95; the windowed readout is post hoc.
                            Runtime about 1-2 minutes.

critical_chain.py           Part E8. Free-fermion TFIM chain, validated against exact diagonalisation (N=8); N=400; deterministic.

first_law_critical.py       Part E9. Entanglement first law on the critical chain, N=400, validated vs ED; deterministic.

light_cone.py              Part E10. Exact Majorana operator spreading, N=400, h=J=1; deterministic.

reconstruct_geometry.py     Part E11. Speed-profile reconstruction from entropies, N=400, c=0.509; deterministic.

first_law_2d.py            Part E12. 2+1D pi-flux Dirac lattice, L=40 (L=24, 56 for the trend), disk region; deterministic. About 1 minute per L=40 run set.
ryu_takayanagi_first_law.py Part E13. sympy: RT area variation = CFT first law for balls, d=2..6.

entanglement_wave.py        Part E14. Exact free-fermion quench dynamics, N=400, intervals l=8 and 12; deterministic. About 1-2 minutes.

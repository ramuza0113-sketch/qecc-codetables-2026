# 1,654 Improved Bounds and 18 Exact-Optimal Closures on Quantum Stabilizer Codes (`CodeTables.de`, `v1.2.0`)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24513985.svg)](https://doi.org/10.5281/zenodo.24513985)

**Author**: Katsutoshi Hashimoto (Independent Researcher, Japan — `ramuza0113@gmail.com` · [ORCID: `0009-0009-8292-5085`](https://orcid.org/0009-0009-8292-5085))  
**Date**: November 28, 2026 (`v1.2.0`)  
**Zenodo Concept DOI (all versions)**: [`10.5281/zenodo.24513985`](https://doi.org/10.5281/zenodo.24513985)  
**Paper (`v1.2.0` PDF)**: [`Hashimoto_2026_Quantum_Codes_v1_2.pdf`](Hashimoto_2026_Quantum_Codes_v1_2.pdf) · [`paper/main.pdf`](paper/main.pdf) ([`paper/main.tex`](paper/main.tex))  
**Machine-Readable Impact Report**: [`CODETABLES_V1_2_IMPACT_REPORT.json`](CODETABLES_V1_2_IMPACT_REPORT.json)  
**Magma Verification Scripts**: [`magma/verify_all_18_exact_optimal.m`](magma/verify_all_18_exact_optimal.m) · [`magma/verify_all_397_parents.m`](magma/verify_all_397_parents.m)

---

## 1. Overview & Two-Tier Purity Accounting (`v1.2.0`)

This release (`v1.2.0`) provides **397 explicit pure $q$-ary quantum stabilizer code certificates** across prime and prime-power alphabets $q \in \{2, 3, 4, 5, 7, 8\}$, spanning block lengths $11 \le n \le 100$ and pure minimum distances $4 \le d \le 16$, constructed via four complementary algebraic-geometric mechanisms:

1. **Comb-Free Multi-Eigenmode Vandermonde Bordered $\lambda$-Constacyclic Hermitian Codes & Spectral Fourier Diagonalization** (covering every length residue $b_p \in \{0, \dots, m-1\}$).
2. **Equivariant $\mathbb{F}_q$-Linear Symplectic Stabilizer Codes** accommodating both **odd and even** stabilizer ranks $r_S = n - k$ (including 20 odd-$r_S$ parent codes up to $d = 13$).
3. **Symmetry-Free Navigation of the Hermitian Isotropic Grassmannian $\mathrm{IGr}_H(m, n; \mathbb{F}_{q^2})$** via Dieudonné–Witt rank-$1$ unitary Householder reflections $I_n - \tau \mathbf{u}\mathbf{u}^\dagger$ ($\tau + \tau^q = \tau^{q+1}(\mathbf{u}^\dagger \mathbf{u})$), unlocking short-length non-GRS Quantum MDS codes.
4. **Frobenius-Conjugated 2-Block (`MP2`) and 3-Block NSC (`MP3`) Quasi-Unitary Matrix-Product Codes** over projective generalized Reed–Solomon (PGRS) components, reaching pure $d = 16$ at $n = 100$ and 116 direct $\Delta d = +2$ leaps.

Compared against Markus Grassl's reference database ([`CodeTables.de`](http://www.codetables.de/QECC/), November 2026 snapshot):
- **Tier 1 — Strictly Pure Closure** (`PureSubcode`, `Puncture`, `Shorten` only; zero `Extend`): **1,450 strictly improved pure entries** (**16 strictly pure Exact-Optimal closures**).
- **Tier 2 — Full Purity-Respecting Closure** (`Tier 1` + terminal weight-$1$ ancilla `Extend`, $+204$ cells): **1,654 strictly improved entries** (**18 Exact-Optimal closures** = 16 pure + 2 terminal `Extend`).
- **Single-Operator Propagation** (1-step `Extend`, `PureSubcode`, `Puncture`, `Shorten` from the 397 parents): **1,575 strictly improved entries**.

| $q$ | Parents | Herm. | Symp. | MP2/MP3 | $d_{\mathrm{new}}$ | $\max \Delta d$ | 1-Op | Pure Closure | **Full Closure** | **Exact Optimal** |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | --: | --: | --: | :-: |
| $2$ | **8** | 8 | 0 | 0 | `4..8` | `+1` | 10 | 9 | **10** | **2** (1P) |
| $3$ | **5** | 5 | 0 | 0 | `5..9` | `+1` | 7 | 5 | **7** | **2** |
| $4$ | **29** | 25 | 4 | 0 | `4..9` | `+1` | 52 | 43 | **52** | **11** (10P) |
| $5$ | **42** | 19 | 11 | 12 | `6..12` | `+1` | 90 | 73 | **93** | **2** |
| $7$ | **98** | 8 | 1 | 89 | `7..16` | `+2` | 421 | 379 | **449** | **0** |
| $8$ | **215** | 14 | 15 | 186 | `6..16` | `+2` | 995 | 941 | **1,043** | **1** |
| **Total** | **397** | **79** | **31** | **287** | **`4..16`** | **`+2`** | **1,575** | **1,450** | **1,654** | **18 (16P)** |

---

## 2. All 18 Exact-Optimal Closures ($d_{\mathrm{low}} = d_{\mathrm{up}}$)

| # | $q$ | Parameters $[[n, k, d^\star]]_q$ | Prev. | **New Exact Bound** | Purity | Certificate / Derivation | SHA-256 (12-hex) |
| :-: | :-: | :--- | :-: | :-: | :-: | :--- | :--- |
| 1 | $2$ | **`[[36, 26, 4]]_2`** | `3..4` | **`4` (Exact)** | Pure | [`certificates/QECC_q2_n36_k26_d4.json`](certificates/QECC_q2_n36_k26_d4.json) | `579c5890fd4c` |
| 2 | $2$ | **`[[37, 26, 4]]_2`** | `3..4` | **`4` (Exact)** | Ext. | `Extend([[36,26,4]]_2, +1)` | — |
| 3 | $3$ | **`[[13, 1, 6]]_3`** | `5..6` | **`6` (Exact)** | Pure | [`certificates/QECC_q3_n13_k1_d6.json`](certificates/QECC_q3_n13_k1_d6.json) | `b6ddad35b344` |
| 4 | $3$ | **`[[20, 10, 5]]_3`** | `4..5` | **`5` (Exact)** | Pure | [`certificates/QECC_q3_n20_k10_d5.json`](certificates/QECC_q3_n20_k10_d5.json) | `4be77100499b` |
| 5 | $4$ | **`[[11, 2, 5]]_4`** | `4..5` | **`5` (Exact)** | Pure | `PureSubcode([[11,3,5]]_4, -1k)` | — |
| 6 | $4$ | **`[[11, 3, 5]]_4`** | `4..5` | **`5` (Exact)** | Pure | [`certificates/QECC_q4_n11_k3_d5.json`](certificates/QECC_q4_n11_k3_d5.json) | `440b445eb310` |
| 7 | $4$ | **`[[12, 3, 5]]_4`** | `4..5` | **`5` (Exact)** | Ext. | `Extend([[11,3,5]]_4, +1)` | — |
| 8 | $4$ | **`[[21, 14, 4]]_4`** | `3..4` | **`4` (Exact)** | Pure | [`certificates/QECC_q4_n21_k14_d4.json`](certificates/QECC_q4_n21_k14_d4.json) | `1e1f62fb8225` |
| 9 | $4$ | **`[[23, 13, 5]]_4`** | `4..5` | **`5` (Exact)** | Pure | [`certificates/QECC_q4_n23_k13_d5.json`](certificates/QECC_q4_n23_k13_d5.json) | `5d6d9eaab2db` |
| 10 | $4$ | **`[[24, 14, 5]]_4`** | `4..5` | **`5` (Exact)** | Pure | [`certificates/QECC_q4_n24_k14_d5.json`](certificates/QECC_q4_n24_k14_d5.json) | `f1b5bddf1b09` |
| 11 | $4$ | **`[[25, 15, 5]]_4`** | `4..5` | **`5` (Exact)** | Pure | [`certificates/QECC_q4_n25_k15_d5.json`](certificates/QECC_q4_n25_k15_d5.json) | `6198f35bf566` |
| 12 | $4$ | **`[[26, 16, 5]]_4`** | `4..5` | **`5` (Exact)** | Pure | [`certificates/QECC_q4_n26_k16_d5.json`](certificates/QECC_q4_n26_k16_d5.json) | `739c1d90d608` |
| 13 | $4$ | **`[[27, 17, 5]]_4`** | `4..5` | **`5` (Exact)** | Pure | [`certificates/QECC_q4_n27_k17_d5.json`](certificates/QECC_q4_n27_k17_d5.json) | `0ca33cb9420e` |
| 14 | $4$ | **`[[28, 18, 5]]_4`** | `4..5` | **`5` (Exact)** | Pure | [`certificates/QECC_q4_n28_k18_d5.json`](certificates/QECC_q4_n28_k18_d5.json) | `a6f6142593e9` |
| 15 | $4$ | **`[[29, 19, 5]]_4`** | `4..5` | **`5` (Exact)** | Pure | [`certificates/QECC_q4_n29_k19_d5.json`](certificates/QECC_q4_n29_k19_d5.json) | `8d9f9d3a56ea` |
| 16 | $5$ | **`[[11, 1, 6]]_5`** | `5..6` | **`6` (Exact)** | Pure | [`certificates/QECC_q5_n11_k1_d6.json`](certificates/QECC_q5_n11_k1_d6.json) | `40b181c84577` |
| 17 | $5$ | **`[[12, 0, 7]]_5`** | `6..7` | **`7` (Exact)** | Pure | [`certificates/QECC_q5_n12_k0_d7.json`](certificates/QECC_q5_n12_k0_d7.json) | `349002c6433a` |
| 18 | $8$ | **`[[15, 5, 6]]_8`** | `5..6` | **`6` (Exact)** | Pure | [`certificates/QECC_q8_n15_k5_d6.json`](certificates/QECC_q8_n15_k5_d6.json) | `bf5e6f2be19e` |

---

## 3. Standalone Zero-Symmetry Clean-Room Verification

Every certificate in `certificates/QECC_q{q}_n{n}_k{k}_d{d}.json` is verified from raw generator matrix entries with **zero** cyclic, constacyclic, Vandermonde, or group-symmetry assumptions:

### A. One-Command Python + Multi-Threaded Rust Suite

```bash
# Fast full algebraic audit (all 397 certificates) + Rust distance sweep (all 15 Exact Optimal, all 287 MP2/MP3, and C(n, d-1) <= 5e6 direct codes) + two-tier propagation audit:
python verify_all_qecc.py

# Exhaustive 20-thread Rust subset sweep across all 397 parent certificates (including multi-hundred-million-subset d=11..13 direct codes):
python verify_all_qecc.py --full
```

### B. Direct Execution of Standalone Rust Clean-Room Verifiers (`verifiers/`)

```bash
rustc -C opt-level=3 verifiers/verify_hermitian_universal.rs -o verify_hermitian_universal
rustc -C opt-level=3 verifiers/verify_symplectic_universal.rs -o verify_symplectic_universal
rustc -C opt-level=3 verifiers/verify_mp2_cleanroom.rs -o verify_mp2_cleanroom
rustc -C opt-level=3 verifiers/verify_mp3_cleanroom.rs -o verify_mp3_cleanroom

./verify_hermitian_universal certificates/QECC_q5_n12_k0_d7.json
./verify_symplectic_universal certificates/QECC_q4_n21_k14_d4.json
./verify_mp2_cleanroom certificates/QECC_q8_n100_k56_d16.json
./verify_mp3_cleanroom certificates/QECC_q4_n48_k30_d6.json
```

### C. Native Magma Verification (`QuantumCode` / `HermitianDual` / `MinimumWeight`)

```bash
magma magma/verify_all_18_exact_optimal.m
magma magma/verify_all_397_parents.m
```

---

## 4. Repository Layout

- [`Hashimoto_2026_Quantum_Codes_v1_2.pdf`](Hashimoto_2026_Quantum_Codes_v1_2.pdf) / [`paper/main.tex`](paper/main.tex) — Full `v1.2.0` preprint with OpenType Unicode math, complete mathematical proofs, all 15 direct Exact-Optimal generator matrices (with 64-hex SHA-256), and the 397-code catalog.
- [`certificates/`](certificates/) — All 397 canonical JSON certificates (`QECC_q{q}_n{n}_k{k}_d{d}.json`) with explicit finite-field encodings and SHA-256 digests.
- [`CODETABLES_V1_2_IMPACT_REPORT.json`](CODETABLES_V1_2_IMPACT_REPORT.json) — Machine-readable breakdown of all 397 parent certificates, 1,450 strictly pure closure cells, 1,575 single-operator cells, 1,654 full closure cells, and 18 Exact-Optimal closures.
- [`verifiers/`](verifiers/) — Four zero-shared-code multi-threaded Rust clean-room verifiers (`verify_hermitian_universal.rs`, `verify_symplectic_universal.rs`, `verify_mp2_cleanroom.rs`, `verify_mp3_cleanroom.rs`).
- [`verify_all_qecc.py`](verify_all_qecc.py) — Unified Python + NumPy + Rust verification driver.
- [`magma/`](magma/) — Native Magma verification scripts (`verify_all_18_exact_optimal.m`, `verify_all_397_parents.m`, plus legacy `v1.0`/`v1.1` scripts).
- [`solvers/`](solvers/) — Representative RTX 3090 CUDA / Spectral Fourier / Householder Grassmannian / Symplectic / MP2-MP3 search solvers.

---

## 5. License

MIT License — Copyright (c) 2026 Katsutoshi Hashimoto.

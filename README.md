# Twenty-Nine Improved Bounds on Quantum Error-Correcting Codes (`CodeTables.de`)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24513985.svg)](https://doi.org/10.5281/zenodo.24513985)

**Author**: Katsutoshi Hashimoto (Independent Researcher, Japan — `ramuza0113@gmail.com` · [ORCID: `0009-0009-8292-5085`](https://orcid.org/0009-0009-8292-5085))  
**Date**: November 25, 2026 (`v1.1.0`)  
**Zenodo Concept DOI (all versions)**: [`10.5281/zenodo.24513985`](https://doi.org/10.5281/zenodo.24513985)  
**Paper**: [`paper/main.pdf`](paper/main.pdf) ([`paper/main.tex`](paper/main.tex))  
**Magma Script (all 18 codes)**: [`magma/verify_all_18_codes.m`](magma/verify_all_18_codes.m)

---

## 1. Summary of 29 Improved `CodeTables.de` Entries

Using $\lambda$-constacyclic ($\lambda^{q+1} = 1$) and $\beta$-bordered Hermitian self-orthogonal codes over $\mathbb{F}_{q^2}$ discovered via Meet-in-the-Middle norm matching and RTX 3090 CUDA minimum-weight screening, this repository (`v1.1.0`) provides **18 explicit pure quantum stabilizer codes** (including seven new distance-$7$ constructions across $q \in \{2, 4, 5, 7\}$) that strictly improve **29 minimum-distance lower bounds** in Markus Grassl's reference tables ([`CodeTables.de`](http://www.codetables.de/QECC/)), closing **5 open entries to exact optimality** ($d_{\mathrm{low}} = d_{\mathrm{up}}$).

| $q$ | Parameters $[[n, k]]_q$ | `CodeTables.de` (Nov 2026) | **New Bound** | Exact Optimal | Construction / Certificate |
| :-: | :--- | :---: | :---: | :-: | :--- |
| $2$ | **`[[33, 11]]_2`** | `6..8` | **`7..8`** | — | [`certificates/QECC_q2_n33_k11_d7.json`](certificates/QECC_q2_n33_k11_d7.json) |
| $2$ | **`[[34, 11]]_2`** | `6..9` | **`7..9`** | — | `Extend([[33,11,7]]_2, +1)` |
| $2$ | **`[[34, 12]]_2`** | `6..8` | **`7..8`** | — | [`certificates/QECC_q2_n34_k12_d7.json`](certificates/QECC_q2_n34_k12_d7.json) |
| $2$ | **`[[36, 26]]_2`** | `3..4` | **`4`** | **Yes** | [`certificates/QECC_q2_n36_k26_d4.json`](certificates/QECC_q2_n36_k26_d4.json) |
| $2$ | **`[[37, 26]]_2`** | `3..4` | **`4`** | **Yes** | `Extend([[36,26,4]]_2, +1)` |
| $2$ | **`[[49, 33]]_2`** | `4..6` | **`5..6`** | — | [`certificates/QECC_q2_n49_k33_d5.json`](certificates/QECC_q2_n49_k33_d5.json) |
| $3$ | **`[[20, 10]]_3`** | `4..5` | **`5`** | **Yes** | [`certificates/QECC_q3_n20_k10_d5.json`](certificates/QECC_q3_n20_k10_d5.json) |
| $4$ | **`[[18, 6]]_4`** | `5..7` | **`6..7`** | — | [`certificates/QECC_q4_n18_k6_d6.json`](certificates/QECC_q4_n18_k6_d6.json) |
| $4$ | **`[[19, 6]]_4`** | `5..7` | **`6..7`** | — | `Extend([[18,6,6]]_4, +1)` |
| $4$ | **`[[19, 7]]_4`** | `5..7` | **`6..7`** | — | [`certificates/QECC_q4_n19_k7_d6.json`](certificates/QECC_q4_n19_k7_d6.json) |
| $4$ | **`[[20, 7]]_4`** | `5..7` | **`6..7`** | — | `Extend([[19,7,6]]_4, +1)` |
| $4$ | **`[[20, 10]]_4`** | `4..6` | **`5..6`** | — | [`certificates/QECC_q4_n20_k10_d5.json`](certificates/QECC_q4_n20_k10_d5.json) |
| $4$ | **`[[24, 8]]_4`** | `6..8` | **`7..8`** | — | [`certificates/QECC_q4_n24_k8_d7.json`](certificates/QECC_q4_n24_k8_d7.json) |
| $4$ | **`[[25, 8]]_4`** | `6..9` | **`7..9`** | — | `Extend([[24,8,7]]_4, +1)` |
| $4$ | **`[[25, 9]]_4`** | `6..8` | **`7..8`** | — | [`certificates/QECC_q4_n25_k9_d7.json`](certificates/QECC_q4_n25_k9_d7.json) |
| $4$ | **`[[25, 14]]_4`** | `4..6` | **`5..6`** | — | `PureSubcode([[25,15,5]]_4, -1k)` |
| $4$ | **`[[25, 15]]_4`** | `4..5` | **`5`** | **Yes** | [`certificates/QECC_q4_n25_k15_d5.json`](certificates/QECC_q4_n25_k15_d5.json) |
| $4$ | **`[[26, 9]]_4`** | `6..9` | **`7..9`** | — | `Extend([[25,9,7]]_4, +1)` |
| $4$ | **`[[26, 16]]_4`** | `4..5` | **`5`** | **Yes** | [`certificates/QECC_q4_n26_k16_d5.json`](certificates/QECC_q4_n26_k16_d5.json) |
| $4$ | **`[[27, 16]]_4`** | `4..6` | **`5..6`** | — | `Extend([[26,16,5]]_4, +1)` |
| $5$ | **`[[19, 7]]_5`** | `5..7` | **`6..7`** | — | [`certificates/QECC_q5_n19_k7_d6.json`](certificates/QECC_q5_n19_k7_d6.json) |
| $5$ | **`[[24, 12]]_5`** | `5..7` | **`6..7`** | — | [`certificates/QECC_q5_n24_k12_d6.json`](certificates/QECC_q5_n24_k12_d6.json) |
| $5$ | **`[[25, 12]]_5`** | `5..7` | **`6..7`** | — | `Extend([[24,12,6]]_5, +1)` |
| $5$ | **`[[32, 16]]_5`** | `6..9` | **`7..9`** | — | [`certificates/QECC_q5_n32_k16_d7.json`](certificates/QECC_q5_n32_k16_d7.json) |
| $5$ | **`[[33, 17]]_5`** | `6..8` | **`7..8`** | — | [`certificates/QECC_q5_n33_k17_d7.json`](certificates/QECC_q5_n33_k17_d7.json) |
| $7$ | **`[[21, 7]]_7`** | `6..8` | **`7..8`** | — | [`certificates/QECC_q7_n21_k7_d7.json`](certificates/QECC_q7_n21_k7_d7.json) |
| $7$ | **`[[22, 7]]_7`** | `6..8` | **`7..8`** | — | `Extend([[21,7,7]]_7, +1)` |
| $7$ | **`[[22, 8]]_7`** | `6..8` | **`7..8`** | — | [`certificates/QECC_q7_n22_k8_d7.json`](certificates/QECC_q7_n22_k8_d7.json) |
| $7$ | **`[[23, 8]]_7`** | `6..8` | **`7..8`** | — | `Extend([[22,8,7]]_7, +1)` |

---

## 2. Quick Clean-Room Verification (~5s)

### A. One-Command Standalone Python Verifier (Python + NumPy, Zero Symmetry Assumed)

```bash
python verify_all_qecc.py
```

Verifies SHA-256 digests, $\mathbb{F}_{q^2}$ RREF rank, Hermitian self-orthogonality $G G^\dagger = 0$, prime-field symplectic self-orthogonality $X Z^\top - Z X^\top \equiv 0 \pmod q$, and exact minimum distance $d(C^{\perp_H}) \ge d_{\mathrm{new}}$ across all 18 explicit parent codes.

### B. Magma Verification (`QuantumCode` / `HermitianDual`)

```bash
magma magma/verify_all_18_codes.m
```

---

## 3. SHA-256 Digests of Hermitian Generator Matrices (`G_hermitian_gf_q2`)

- `QECC_q2_n33_k11_d7.json`: `ec7fdf4fa65f50a828ceefa0ad2c10e222cc294e95473d79dda7772dad2befeb`
- `QECC_q2_n34_k12_d7.json`: `f31c1d11f5972912308a4d62970beb5ceb1da01d0a9c81d7090e896b1efc65d3`
- `QECC_q2_n36_k26_d4.json`: `579c5890fd4ccb6bb6d2716a2ce829dd88198836dab3d6f8db525245aff81b6d`
- `QECC_q2_n49_k33_d5.json`: `a376c6184aa62b87092f0ac42e74164fe86ad24c8579bc32d700ac2bfbe061cf`
- `QECC_q3_n20_k10_d5.json`: `4be77100499b51dddc666546c68b000341d06b1d17c18cfc46b6614f33206104`
- `QECC_q4_n18_k6_d6.json`: `17224a6584e9dbc42e1c4ecfd2e9bfc2af213e6bbf59e450e0c74ae704dc5b31`
- `QECC_q4_n19_k7_d6.json`: `cc69171ee33fd48beb4f2b36478ae60c4c59e60f6089e8a9176afca2a4d01f3e`
- `QECC_q4_n20_k10_d5.json`: `03a0cde8eac4244514e012c76ea2c02028df76479ee9ecb48e5895ff11ab3224`
- `QECC_q4_n24_k8_d7.json`: `ea7fa3b41aea0216ff380eb1835352fc8c7fbd3136f9d18ea68a82f2717f8283`
- `QECC_q4_n25_k9_d7.json`: `f4f281398af42637c805b2a88d86e1cd979c46fd280817c5bbfa88ec538dde98`
- `QECC_q4_n25_k15_d5.json`: `6198f35bf566e1d75d83af764e8512bf9bb24118d136a3f8ed2978f8b1c0c1e1`
- `QECC_q4_n26_k16_d5.json`: `739c1d90d60827a52fa2f6d3d6940130f259afffac98fff5dbf421191876acff`
- `QECC_q5_n19_k7_d6.json`: `9fa484151233195fbb67ab86ae3cd68040ab9f4824947fc1fe49a69be4e6042a`
- `QECC_q5_n24_k12_d6.json`: `dabbb85aa7a7966169b48f7855377fab2e29f4f90f7dee63d40199c6265a1021`
- `QECC_q5_n32_k16_d7.json`: `a2e90ae2c92b160bf8506ea6aa4fc32375fcbbdb6b2ab58fb3874445c4ac73f4`
- `QECC_q5_n33_k17_d7.json`: `964e83d397a325b97de51e83fc8362d099020d610a0cc96a883a3991fb7be665`
- `QECC_q7_n21_k7_d7.json`: `6a53ddd351fd6f10425ebc4a7bdba2cedfba2c28ee8a33251f1dffddd6fe41cc`
- `QECC_q7_n22_k8_d7.json`: `d78ddd233bd9b6b88a69415fb6d7dcb5b7f73b8b313f25a73d172108a4acbc5c`

---

## 4. License

MIT License — Copyright (c) 2026 Katsutoshi Hashimoto.

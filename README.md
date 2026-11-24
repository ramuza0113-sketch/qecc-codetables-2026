# Sixteen Improved Bounds on Quantum Error-Correcting Codes (`CodeTables.de`)

**Author**: Katsutoshi Hashimoto (Independent Researcher, Japan — `ramuza0113@gmail.com` · [ORCID: `0009-0009-8292-5085`](https://orcid.org/0009-0009-8292-5085))  
**Date**: November 2026  
**Paper**: [`paper/main.pdf`](paper/main.pdf) ([`paper/main.tex`](paper/main.tex))  
**Magma Script**: [`magma/verify_all_10_codes.m`](magma/verify_all_10_codes.m)

---

## 1. Summary of 16 Improved `CodeTables.de` Entries

Using multi-circulant and bordered Hermitian self-orthogonal codes over $\mathbb{F}_{q^2}$ discovered via Meet-in-the-Middle norm matching and RTX 3090 CUDA minimum-weight screening, this repository provides **10 explicit pure quantum stabilizer codes** that strictly improve **16 minimum-distance lower bounds** in Markus Grassl's reference tables ([`CodeTables.de`](http://www.codetables.de/QECC/)), closing **5 open entries to exact optimality** ($d_{\mathrm{low}} = d_{\mathrm{up}}$).

| $q$ | Parameters $[[n, k]]_q$ | `CodeTables.de` (Nov 2026) | **New Bound** | Exact Optimal | Construction / Certificate |
| :-: | :--- | :---: | :---: | :-: | :--- |
| $2$ | **`[[36, 26]]_2`** | `3..4` | **`4`** | **Yes** | [`certificates/QECC_q2_n36_k26_d4.json`](certificates/QECC_q2_n36_k26_d4.json) |
| $2$ | **`[[37, 26]]_2`** | `3..4` | **`4`** | **Yes** | `Extend([[36, 26, 4]]_2, +1)` |
| $2$ | **`[[49, 33]]_2`** | `4..6` | **`5..6`** | — | [`certificates/QECC_q2_n49_k33_d5.json`](certificates/QECC_q2_n49_k33_d5.json) |
| $3$ | **`[[20, 10]]_3`** | `4..5` | **`5`** | **Yes** | [`certificates/QECC_q3_n20_k10_d5.json`](certificates/QECC_q3_n20_k10_d5.json) |
| $4$ | **`[[18, 6]]_4`** | `5..7` | **`6..7`** | — | [`certificates/QECC_q4_n18_k6_d6.json`](certificates/QECC_q4_n18_k6_d6.json) |
| $4$ | **`[[19, 6]]_4`** | `5..7` | **`6..7`** | — | `Extend([[18, 6, 6]]_4, +1)` |
| $4$ | **`[[19, 7]]_4`** | `5..7` | **`6..7`** | — | [`certificates/QECC_q4_n19_k7_d6.json`](certificates/QECC_q4_n19_k7_d6.json) |
| $4$ | **`[[20, 7]]_4`** | `5..7` | **`6..7`** | — | `Extend([[19, 7, 6]]_4, +1)` |
| $4$ | **`[[20, 10]]_4`** | `4..6` | **`5..6`** | — | [`certificates/QECC_q4_n20_k10_d5.json`](certificates/QECC_q4_n20_k10_d5.json) |
| $4$ | **`[[25, 14]]_4`** | `4..6` | **`5..6`** | — | `PureSubcode([[25, 15, 5]]_4, -1)` |
| $4$ | **`[[25, 15]]_4`** | `4..5` | **`5`** | **Yes** | [`certificates/QECC_q4_n25_k15_d5.json`](certificates/QECC_q4_n25_k15_d5.json) |
| $4$ | **`[[26, 16]]_4`** | `4..5` | **`5`** | **Yes** | [`certificates/QECC_q4_n26_k16_d5.json`](certificates/QECC_q4_n26_k16_d5.json) |
| $4$ | **`[[27, 16]]_4`** | `4..6` | **`5..6`** | — | `Extend([[26, 16, 5]]_4, +1)` |
| $5$ | **`[[19, 7]]_5`** | `5..7` | **`6..7`** | — | [`certificates/QECC_q5_n19_k7_d6.json`](certificates/QECC_q5_n19_k7_d6.json) |
| $7$ | **`[[21, 7]]_7`** | `6..8` | **`7..8`** | — | [`certificates/QECC_q7_n21_k7_d7.json`](certificates/QECC_q7_n21_k7_d7.json) |
| $7$ | **`[[22, 7]]_7`** | `6..8` | **`7..8`** | — | `Extend([[21, 7, 7]]_7, +1)` |

---

## 2. Quick Clean-Room Verification (~1.9s)

### A. One-Command Standalone Python Verifier (Python + NumPy, Zero Symmetry Assumed)

```bash
python verify_all_qecc.py
```

Verifies SHA-256 digests, $\mathbb{F}_{q^2}$ RREF rank, Hermitian self-orthogonality $G G^\dagger = 0$, prime-field symplectic self-orthogonality $X Z^\top - Z X^\top \equiv 0 \pmod q$, and exact minimum distance $d(C^{\perp_H}) \ge d_{\mathrm{new}}$ across all 10 explicit parent codes in ~1.9 seconds.

### B. Magma Verification (`QuantumCode` / `HermitianDual`)

```bash
magma magma/verify_all_10_codes.m
```

---

## 3. SHA-256 Digests of Hermitian Generator Matrices (`G_hermitian_gf_q2`)

- `QECC_q2_n36_k26_d4.json`: `579c5890fd4ccb6b`...
- `QECC_q2_n49_k33_d5.json`: `a376c6184aa62b87`...
- `QECC_q3_n20_k10_d5.json`: `4be77100499b51dd`...
- `QECC_q4_n18_k6_d6.json`:  `17224a6584e9dbc4`...
- `QECC_q4_n19_k7_d6.json`:  `cc69171ee33fd48b`...
- `QECC_q4_n20_k10_d5.json`: `03a0cde8eac42445`...
- `QECC_q4_n25_k15_d5.json`: `6198f35bf566e1d7`...
- `QECC_q4_n26_k16_d5.json`: `739c1d90d60827a5`...
- `QECC_q5_n19_k7_d6.json`:  `9fa484151233195f`...
- `QECC_q7_n21_k7_d7.json`:  `6a53ddd351fd6f10`...

---

## 4. License

MIT License — Copyright (c) 2026 Katsutoshi Hashimoto.

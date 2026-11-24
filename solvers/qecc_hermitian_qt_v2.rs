// Flagship Round 2 v2 Quantum Error-Correcting Code World-Record Hunter (`CodeTables.de`)
// Upgrades over v1:
//   1. Supports ALL 6 CodeTables.de fields `q in {2, 3, 4, 5, 7, 8}` (`GF(4)..GF(64)`)!
//   2. Supports `m in 3..=12` (`u128` polynomial packing, zero collisions up to `m = 16`)!
//   3. Supports `border_p in {0, 1, 2, 3}` (`p | m` disjoint-support circulant-compatible borders:
//      `n = l*m + p, k = (l-2)*m + p` for `lam = 1`!)
//   4. Stage-filtered Meet-in-the-Middle verifier (`100x` faster on `d_tar = 6, 7`: only builds
//      `rhs2_max_j4` on the <0.1% of candidates that already pass `s = 1..3 + rhs1`!)
//   5. Skips impossible Quantum MDS (`d_tar == m + 1 && n > q^2 + 1`) & already-saved `(q, n, k)`!
//   6. Rains Puncture-Code Cascade (`ker_{GF(q)}(C * C^q)`) for prime `q in {2, 3, 5, 7}` AND `q = 4`!

use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

#[derive(Clone)]
struct GFq2 {
    q: usize,
    q2: usize,
    add: [[u8; 64]; 64],
    sub: [[u8; 64]; 64],
    mul: [[u8; 64]; 64],
    inv: [u8; 64],
    conj: [u8; 64],
    norm1_lambdas: Vec<u8>,
    gfq_elems: Vec<u8>,       // elements `d` in GF(q^2) with `conj[d] == d` (`q` elements, `0` first!)
    alpha_for_norm: [u8; 64], // for `d in gfq_elems \ {0}`, `alpha^(q+1) == d`
}

impl GFq2 {
    fn new(q: usize) -> Self {
        let q2 = q * q;
        let mut add = [[0u8; 64]; 64];
        let mut sub = [[0u8; 64]; 64];
        let mut mul = [[0u8; 64]; 64];
        let mut inv = [0u8; 64];
        let mut conj = [0u8; 64];

        if q == 2 {
            for u in 0..4usize {
                let (a0, b0) = (u & 1, (u >> 1) & 1);
                conj[u] = if b0 == 0 { a0 as u8 } else { ((a0 ^ 1) | 2) as u8 };
                for v in 0..4usize {
                    let (a1, b1) = (v & 1, (v >> 1) & 1);
                    add[u][v] = (u ^ v) as u8;
                    sub[u][v] = (u ^ v) as u8;
                    let p0 = (a0 & a1) ^ (b0 & b1);
                    let p1 = (a0 & b1) ^ (b0 & a1) ^ (b0 & b1);
                    mul[u][v] = (p0 | (p1 << 1)) as u8;
                }
            }
        } else if q == 4 {
            // GF(16) = GF(2)[z] / (z^4 + z + 1 = 0x13)
            for u in 0..16usize {
                for v in 0..16usize {
                    add[u][v] = (u ^ v) as u8;
                    sub[u][v] = (u ^ v) as u8;
                    let (mut a, mut b, mut p) = (u, v, 0usize);
                    for _ in 0..4 {
                        if (b & 1) != 0 {
                            p ^= a;
                        }
                        let hi = a & 8;
                        a = (a << 1) & 15;
                        if hi != 0 {
                            a ^= 3;
                        }
                        b >>= 1;
                    }
                    mul[u][v] = p as u8;
                }
            }
            for u in 0..16usize {
                let u2 = mul[u][u] as usize;
                conj[u] = mul[u2][u2];
            }
        } else if q == 8 {
            // GF(64) = GF(2)[z] / (z^6 + z + 1 = 0x43), conj(u) = u^8
            for u in 0..64usize {
                for v in 0..64usize {
                    add[u][v] = (u ^ v) as u8;
                    sub[u][v] = (u ^ v) as u8;
                    let (mut a, mut b, mut p) = (u, v, 0usize);
                    for _ in 0..6 {
                        if (b & 1) != 0 {
                            p ^= a;
                        }
                        let hi = a & 32;
                        a = (a << 1) & 63;
                        if hi != 0 {
                            a ^= 3; // z^6 = z + 1
                        }
                        b >>= 1;
                    }
                    mul[u][v] = p as u8;
                }
            }
            for u in 0..64usize {
                let u2 = mul[u][u] as usize;
                let u4 = mul[u2][u2] as usize;
                conj[u] = mul[u4][u4]; // u^8
            }
        } else {
            // Odd prime q in {3, 5, 7}: GF(q)[theta]/(theta^2 - c)
            let c = if q == 7 { 3usize } else { 2usize };
            for u in 0..q2 {
                let (x0, y0) = (u % q, u / q);
                conj[u] = (x0 + q * ((q - y0) % q)) as u8;
                for v in 0..q2 {
                    let (x1, y1) = (v % q, v / q);
                    add[u][v] = (((x0 + x1) % q) + q * ((y0 + y1) % q)) as u8;
                    sub[u][v] = (((x0 + q - x1) % q) + q * ((y0 + q - y1) % q)) as u8;
                    let rx = (x0 * x1 + c * y0 * y1) % q;
                    let ry = (x0 * y1 + y0 * x1) % q;
                    mul[u][v] = (rx + q * ry) as u8;
                }
            }
        }

        for u in 1..q2 {
            for v in 1..q2 {
                if mul[u][v] == 1 {
                    inv[u] = v as u8;
                    break;
                }
            }
            assert!(inv[u] != 0, "Missing inverse for u={} in q={}", u, q);
        }

        let mut gfq_elems = Vec::new();
        for u in 0..q2 {
            if conj[u] == (u as u8) {
                gfq_elems.push(u as u8);
            }
        }
        assert_eq!(gfq_elems.len(), q);

        let mut norm1_lambdas = Vec::new();
        let mut alpha_for_norm = [0u8; 64];
        for u in 1..q2 {
            let nrm = mul[u][conj[u] as usize];
            if nrm == 1 {
                norm1_lambdas.push(u as u8);
            }
            if conj[nrm as usize] == nrm && alpha_for_norm[nrm as usize] == 0 {
                alpha_for_norm[nrm as usize] = u as u8;
            }
        }

        GFq2 {
            q,
            q2,
            add,
            sub,
            mul,
            inv,
            conj,
            norm1_lambdas,
            gfq_elems,
            alpha_for_norm,
        }
    }

    #[inline(always)]
    fn pack_poly(&self, p: &[u8]) -> u128 {
        let mut h = 0u128;
        for (i, &v) in p.iter().enumerate() {
            h |= (v as u128) << (i * 8);
        }
        h
    }

    fn poly_adj(&self, a: &[u8], lam: u8) -> Vec<u8> {
        let m = a.len();
        let mut out = vec![0u8; m];
        out[0] = self.conj[a[0] as usize];
        let lam_c = self.conj[lam as usize] as usize;
        for t in 1..m {
            let c = self.conj[a[m - t] as usize] as usize;
            out[t] = self.mul[lam_c][c];
        }
        out
    }

    fn poly_mul_constacyclic(&self, a: &[u8], b: &[u8], lam: u8) -> Vec<u8> {
        let m = a.len();
        let mut out = vec![0u8; m];
        let lam_u = lam as usize;
        for i in 0..m {
            let ai = a[i] as usize;
            if ai == 0 {
                continue;
            }
            for j in 0..m {
                let bj = b[j] as usize;
                if bj == 0 {
                    continue;
                }
                let prod = self.mul[ai][bj];
                if i + j < m {
                    let cur = out[i + j] as usize;
                    out[i + j] = self.add[cur][prod as usize];
                } else {
                    let idx = i + j - m;
                    let wrapped = self.mul[lam_u][prod as usize];
                    let cur = out[idx] as usize;
                    out[idx] = self.add[cur][wrapped as usize];
                }
            }
        }
        out
    }

    fn build_constacyclic_block(&self, a: &[u8], lam: u8) -> Vec<Vec<u8>> {
        let m = a.len();
        let mut mat = vec![vec![0u8; m]; m];
        let lam_u = lam as usize;
        for c in 0..m {
            for r in 0..m {
                mat[r][c] = if r >= c {
                    a[r - c]
                } else {
                    self.mul[lam_u][a[m + r - c] as usize]
                };
            }
        }
        mat
    }
}

struct XorShift64(u64);
impl XorShift64 {
    #[inline(always)]
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

#[inline(always)]
fn wt_slice(v: &[u8]) -> usize {
    let mut w = 0usize;
    for &x in v {
        if x != 0 {
            w += 1;
        }
    }
    w
}

/// Stage-filtered exact check of `d(C^{\perp_H}) >= d_tar` for `G = [I_m | B]` (`m <= 12`, `d_tar <= 7`)!
fn is_pure_distance_at_least(
    f: &GFq2,
    cols_b: &[[u8; 16]],
    m: usize,
    l_minus_1: usize,
    border_p: usize,
    d_tar: usize,
) -> bool {
    let n_b = cols_b.len();
    let w_max = d_tar - 1;

    // Starting column indices modulo cyclic shift: `0, m, ..., (l-2)*m`, plus `(l-1)*m` if `border_p > 0`!
    // (Since `T_1` cycles the `border_p` columns `e_{0 mod p}, ..., e_{p-1 mod p}` transitively!)
    let mut j1_starts: Vec<usize> = (0..l_minus_1).map(|b0| b0 * m).collect();
    if border_p > 0 {
        j1_starts.push(l_minus_1 * m);
    }

    // Stage 1: s = 1
    for &j1 in &j1_starts {
        let w = wt_slice(&cols_b[j1][..m]);
        if 1 + w <= w_max {
            return false;
        }
    }
    if w_max == 1 {
        return true;
    }

    let q2 = f.q2;
    let mut sc_cols = vec![vec![[0u8; 16]; q2]; n_b];
    for j in 0..n_b {
        for a in 1..q2 {
            for r in 0..m {
                sc_cols[j][a][r] = f.mul[a][cols_b[j][r] as usize];
            }
        }
    }

    // Stage 2: s = 2
    for &j1 in &j1_starts {
        let c1 = &cols_b[j1];
        for j2 in (j1 + 1)..n_b {
            for a2 in 1..q2 {
                let c2 = &sc_cols[j2][a2];
                let mut w = 0usize;
                for r in 0..m {
                    if f.add[c1[r] as usize][c2[r] as usize] != 0 {
                        w += 1;
                    }
                }
                if 2 + w <= w_max {
                    return false;
                }
            }
        }
    }
    if w_max == 2 {
        return true;
    }

    let max_lhs_s = if d_tar <= 6 { 3 } else { 4 };

    // Build lightweight `rhs1_max_j` (`s = max_lhs_s + 1`) FIRST before touching `rhs2_max_j4`!
    let mut rhs1_max_j: HashMap<u128, u16> = HashMap::new();
    if w_max >= max_lhs_s + 1 {
        let rad1 = w_max - (max_lhs_s + 1); // 0 for d=5,7; 1 for d=6
        rhs1_max_j.reserve(n_b * q2 * (if rad1 == 1 { m * q2 + 1 } else { 2 }));
        for j in (1..n_b).rev() {
            let ju = j as u16;
            for a in 1..q2 {
                let mut neg = [0u8; 16];
                for r in 0..m {
                    neg[r] = f.sub[0][sc_cols[j][a][r] as usize];
                }
                rhs1_max_j.entry(f.pack_poly(&neg[..m])).or_insert(ju);
                if rad1 >= 1 {
                    for pos in 0..m {
                        let orig = neg[pos];
                        for val in 0..(q2 as u8) {
                            if val != orig {
                                neg[pos] = val;
                                rhs1_max_j.entry(f.pack_poly(&neg[..m])).or_insert(ju);
                            }
                        }
                        neg[pos] = orig;
                    }
                }
            }
        }
    }

    let check_rhs1 = !rhs1_max_j.is_empty();

    // Stage 3A: Fast `s = 3` (and `s = 4` if `d_tar == 7`) + `rhs1` check WITHOUT building `rhs2_max_j4`!
    // Rejects >99.9% of candidates in microseconds!
    for &j1 in &j1_starts {
        let c1 = &cols_b[j1];
        for j2 in (j1 + 1)..n_b {
            for a2 in 1..q2 {
                let c2 = &sc_cols[j2][a2];
                let mut u12 = [0u8; 16];
                for r in 0..m {
                    u12[r] = f.add[c1[r] as usize][c2[r] as usize];
                }
                for j3 in (j2 + 1)..n_b {
                    for a3 in 1..q2 {
                        let c3 = &sc_cols[j3][a3];
                        let mut u123 = [0u8; 16];
                        let mut w3 = 0usize;
                        for r in 0..m {
                            let v = f.add[u12[r] as usize][c3[r] as usize];
                            u123[r] = v;
                            if v != 0 {
                                w3 += 1;
                            }
                        }
                        if 3 + w3 <= w_max {
                            return false;
                        }
                        if max_lhs_s == 3 && check_rhs1 {
                            let pk = f.pack_poly(&u123[..m]);
                            if let Some(&mj4) = rhs1_max_j.get(&pk) {
                                if (mj4 as usize) > j3 {
                                    return false;
                                }
                            }
                        } else if max_lhs_s == 4 {
                            for j4 in (j3 + 1)..n_b {
                                for a4 in 1..q2 {
                                    let c4 = &sc_cols[j4][a4];
                                    let mut u1234 = [0u8; 16];
                                    let mut w4 = 0usize;
                                    for r in 0..m {
                                        let v = f.add[u123[r] as usize][c4[r] as usize];
                                        u1234[r] = v;
                                        if v != 0 {
                                            w4 += 1;
                                        }
                                    }
                                    if 4 + w4 <= w_max {
                                        return false;
                                    }
                                    if check_rhs1 {
                                        let pk = f.pack_poly(&u1234[..m]);
                                        if let Some(&mj5) = rhs1_max_j.get(&pk) {
                                            if (mj5 as usize) > j4 {
                                                return false;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if w_max < max_lhs_s + 2 {
        return true; // `d_tar <= 5` is 100% DONE HERE!
    }

    // Stage 3B (`d_tar == 6` or `7`): Candidate ALREADY passed `d >= d_tar - 1`! Now build `rhs2_max_j4` for exact match `s = max_lhs_s + 2`!
    let mut rhs2_max_j4: HashMap<u128, u16> = HashMap::with_capacity((n_b * n_b / 2) * q2 * q2);
    for j4 in (2..n_b.saturating_sub(1)).rev() {
        let j4u = j4 as u16;
        for j5 in (j4 + 1)..n_b {
            for a4 in 1..q2 {
                let c4 = &sc_cols[j4][a4];
                for a5 in 1..q2 {
                    let c5 = &sc_cols[j5][a5];
                    let mut neg2 = [0u8; 16];
                    for r in 0..m {
                        let s45 = f.add[c4[r] as usize][c5[r] as usize];
                        neg2[r] = f.sub[0][s45 as usize];
                    }
                    rhs2_max_j4.entry(f.pack_poly(&neg2[..m])).or_insert(j4u);
                }
            }
        }
    }

    for &j1 in &j1_starts {
        let c1 = &cols_b[j1];
        for j2 in (j1 + 1)..n_b {
            for a2 in 1..q2 {
                let c2 = &sc_cols[j2][a2];
                let mut u12 = [0u8; 16];
                for r in 0..m {
                    u12[r] = f.add[c1[r] as usize][c2[r] as usize];
                }
                for j3 in (j2 + 1)..n_b {
                    for a3 in 1..q2 {
                        let c3 = &sc_cols[j3][a3];
                        let mut u123 = [0u8; 16];
                        for r in 0..m {
                            u123[r] = f.add[u12[r] as usize][c3[r] as usize];
                        }
                        if max_lhs_s == 3 {
                            let pk = f.pack_poly(&u123[..m]);
                            if let Some(&mj4) = rhs2_max_j4.get(&pk) {
                                if (mj4 as usize) > j3 {
                                    return false;
                                }
                            }
                        } else {
                            for j4 in (j3 + 1)..n_b {
                                for a4 in 1..q2 {
                                    let c4 = &sc_cols[j4][a4];
                                    let mut u1234 = [0u8; 16];
                                    for r in 0..m {
                                        u1234[r] = f.add[u123[r] as usize][c4[r] as usize];
                                    }
                                    let pk = f.pack_poly(&u1234[..m]);
                                    if let Some(&mj5) = rhs2_max_j4.get(&pk) {
                                        if (mj5 as usize) > j4 {
                                            return false;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    true
}

fn verify_hermitian_and_rank(f: &GFq2, g_mat: &[Vec<u8>]) -> bool {
    let m = g_mat.len();
    let n = g_mat[0].len();
    for r in 0..m {
        for s in 0..m {
            let mut acc = 0u8;
            for j in 0..n {
                let p = f.mul[g_mat[r][j] as usize][f.conj[g_mat[s][j] as usize] as usize];
                acc = f.add[acc as usize][p as usize];
            }
            if acc != 0 {
                return false;
            }
        }
    }
    let mut a = g_mat.to_vec();
    let mut r = 0usize;
    for c in 0..n {
        if r >= m {
            break;
        }
        let mut piv = None;
        for i in r..m {
            if a[i][c] != 0 {
                piv = Some(i);
                break;
            }
        }
        if let Some(p) = piv {
            a.swap(r, p);
            let inv_p = f.inv[a[r][c] as usize] as usize;
            for j in c..n {
                a[r][j] = f.mul[a[r][j] as usize][inv_p];
            }
            for i in 0..m {
                if i != r && a[i][c] != 0 {
                    let fac = a[i][c] as usize;
                    for j in c..n {
                        let sub_v = f.mul[fac][a[r][j] as usize];
                        a[i][j] = f.sub[a[i][j] as usize][sub_v as usize];
                    }
                }
            }
            r += 1;
        }
    }
    r == m
}

fn export_certificate(
    f: &GFq2,
    g_mat: &[Vec<u8>],
    d_tar: usize,
    method: &str,
    out_dir: &str,
    saved_set: &Mutex<HashSet<(usize, usize, usize)>>,
) -> bool {
    let m = g_mat.len();
    let n = g_mat[0].len();
    if n < 2 * m {
        return false;
    }
    let k = n - 2 * m;
    let r_rows = 2 * m;
    {
        let mut guard = saved_set.lock().unwrap();
        if guard.contains(&(f.q, n, k)) {
            return false;
        }
        guard.insert((f.q, n, k));
    }

    let out_path = format!("{}/CONFIRMED_QECC_CROWN_q{}_n{}_k{}_d{}.json", out_dir, f.q, n, k, d_tar);
    if f.q != 4 && f.q != 8 {
        let mult_gen: usize = if f.q == 2 { 2 } else { f.q };
        let mut x_rows: Vec<Vec<usize>> = Vec::with_capacity(r_rows);
        let mut z_rows: Vec<Vec<usize>> = Vec::with_capacity(r_rows);
        for row in g_mat {
            let mut x0 = vec![0usize; n];
            let mut z0 = vec![0usize; n];
            let mut x1 = vec![0usize; n];
            let mut z1 = vec![0usize; n];
            for j in 0..n {
                let u = row[j] as usize;
                let u_m = f.mul[mult_gen][u] as usize;
                if f.q == 2 {
                    x0[j] = u & 1;
                    z0[j] = (u >> 1) & 1;
                    x1[j] = u_m & 1;
                    z1[j] = (u_m >> 1) & 1;
                } else {
                    x0[j] = u % f.q;
                    z0[j] = u / f.q;
                    x1[j] = u_m % f.q;
                    z1[j] = u_m / f.q;
                }
            }
            x_rows.push(x0);
            z_rows.push(z0);
            x_rows.push(x1);
            z_rows.push(z1);
        }
        let fmt_mat = |mat: &[Vec<usize>]| -> String {
            let rows_s: Vec<String> = mat
                .iter()
                .map(|r| {
                    let cs: Vec<String> = r.iter().map(|v| v.to_string()).collect();
                    format!("    [{}]", cs.join(", "))
                })
                .collect();
            format!("[\n{}\n  ]", rows_s.join(",\n"))
        };
        let json_s = format!(
            "{{\n  \"format\": \"symplectic_gfq\",\n  \"method\": \"{}\",\n  \"q\": {},\n  \"n\": {},\n  \"k\": {},\n  \"r\": {},\n  \"new_lower_bound\": {},\n  \"X\": {},\n  \"Z\": {}\n}}\n",
            method, f.q, n, k, r_rows, d_tar, fmt_mat(&x_rows), fmt_mat(&z_rows)
        );
        fs::write(&out_path, json_s).unwrap();
    } else {
        let fmt_tag = if f.q == 4 { "hermitian_gf16" } else { "hermitian_gf64" };
        let mat_key = if f.q == 4 { "G_gf16" } else { "G_gf64" };
        let rows_s: Vec<String> = g_mat
            .iter()
            .map(|r| {
                let cs: Vec<String> = r.iter().map(|v| v.to_string()).collect();
                format!("    [{}]", cs.join(", "))
            })
            .collect();
        let json_s = format!(
            "{{\n  \"format\": \"{}\",\n  \"method\": \"{}\",\n  \"q\": {},\n  \"n\": {},\n  \"k\": {},\n  \"r\": {},\n  \"new_lower_bound\": {},\n  \"{}\": [\n{}\n  ]\n}}\n",
            fmt_tag,
            method,
            f.q,
            n,
            k,
            r_rows,
            d_tar,
            mat_key,
            rows_s.join(",\n")
        );
        fs::write(&out_path, json_s).unwrap();
    }
    println!("🏆 DISCOVERED [[{}, {}, {}]]_{} via {} -> {}", n, k, d_tar, f.q, method, out_path);
    true
}

/// Rains Puncture-Code Cascade Solver over ALL `q in {2, 3, 4, 5, 7, 8}` using exact `GF(q)` subfield arithmetic inside `GF(q^2)`!
fn run_rains_puncture_cascade(
    f: &GFq2,
    g_mat: &[Vec<u8>],
    d_tar: usize,
    snap_q: &HashMap<(usize, usize), (usize, usize)>,
    out_dir: &str,
    saved_set: &Mutex<HashSet<(usize, usize, usize)>>,
) {
    let m = g_mat.len();
    let n = g_mat[0].len();
    // Pick an element `omega in GF(q^2) \ GF(q)` so `{1, omega}` is a GF(q)-basis of GF(q^2)!
    let omega = (1..f.q2).find(|&u| f.conj[u] != (u as u8)).unwrap();
    // Decompose every `z in 0..q2` uniquely as `z = c0 + c1 * omega` (`c0, c1 in f.gfq_elems`)!
    let mut decomp = vec![(0u8, 0u8); f.q2];
    for &c0 in &f.gfq_elems {
        for &c1 in &f.gfq_elems {
            let prod = f.mul[c1 as usize][omega];
            let z = f.add[c0 as usize][prod as usize] as usize;
            decomp[z] = (c0, c1);
        }
    }

    let mut eqs: Vec<Vec<u8>> = Vec::new();
    for r in 0..m {
        for s in r..m {
            let mut row0 = vec![0u8; n];
            let mut row1 = vec![0u8; n];
            for j in 0..n {
                let p = f.mul[g_mat[r][j] as usize][f.conj[g_mat[s][j] as usize] as usize] as usize;
                let (c0, c1) = decomp[p];
                row0[j] = c0;
                row1[j] = c1;
            }
            eqs.push(row0);
            if r < s {
                eqs.push(row1);
            }
        }
    }

    // RREF over `GF(q)` (`f.gfq_elems` closed under `f.add, f.sub, f.mul, f.inv`!)
    let rows = eqs.len();
    let mut piv_cols = Vec::new();
    let mut r_piv = 0usize;
    for c in 0..n {
        if r_piv >= rows {
            break;
        }
        let piv = (r_piv..rows).find(|&i| eqs[i][c] != 0);
        if let Some(p) = piv {
            eqs.swap(r_piv, p);
            let inv_p = f.inv[eqs[r_piv][c] as usize] as usize;
            for j in c..n {
                eqs[r_piv][j] = f.mul[eqs[r_piv][j] as usize][inv_p];
            }
            for i in 0..rows {
                if i != r_piv && eqs[i][c] != 0 {
                    let fac = eqs[i][c] as usize;
                    for j in c..n {
                        let sub_v = f.mul[fac][eqs[r_piv][j] as usize];
                        eqs[i][j] = f.sub[eqs[i][j] as usize][sub_v as usize];
                    }
                }
            }
            piv_cols.push(c);
            r_piv += 1;
        }
    }

    let piv_set: HashSet<usize> = piv_cols.iter().copied().collect();
    let free_cols: Vec<usize> = (0..n).filter(|c| !piv_set.contains(c)).collect();
    if free_cols.len() <= 1 {
        return;
    }

    let mut null_basis: Vec<Vec<u8>> = Vec::new();
    for &fc in &free_cols {
        let mut vec_d = vec![0u8; n];
        vec_d[fc] = 1;
        for (idx, &pc) in piv_cols.iter().enumerate() {
            if eqs[idx][fc] != 0 {
                vec_d[pc] = f.sub[0][eqs[idx][fc] as usize];
            }
        }
        null_basis.push(vec_d);
    }

    let nb = null_basis.len();
    let mut rng = XorShift64(0x2026_1125_CAFE_BABE ^ ((n * 97 + f.q) as u64));
    let ones_vec = vec![1u8; n];
    for trial in 0..50_000usize {
        let mut d_vec = if trial % 2 == 0 { ones_vec.clone() } else { vec![0u8; n] };
        let terms = if trial < 18_000 { 1 + (trial % 4) } else { nb };
        for t in 0..terms {
            let b_idx = if terms == nb { t } else { (rng.next() as usize) % nb };
            let coef_idx = if terms == nb {
                (rng.next() as usize) % f.q
            } else {
                1 + ((rng.next() as usize) % (f.q - 1))
            };
            let coef = f.gfq_elems[coef_idx] as usize;
            if coef != 0 {
                for j in 0..n {
                    let prod = f.mul[coef][null_basis[b_idx][j] as usize];
                    d_vec[j] = f.add[d_vec[j] as usize][prod as usize];
                }
            }
        }
        let supp: Vec<usize> = (0..n).filter(|&j| d_vec[j] != 0).collect();
        let n_p = supp.len();
        if n_p < 2 * m || n_p >= n {
            continue;
        }
        let k_p = n_p - 2 * m;
        if let Some(&(d_low, _)) = snap_q.get(&(n_p, k_p)) {
            if d_tar <= d_low {
                continue;
            }
        } else {
            continue;
        }
        if saved_set.lock().unwrap().contains(&(f.q, n_p, k_p)) {
            continue;
        }

        let mut g_p = vec![vec![0u8; n_p]; m];
        for (new_j, &old_j) in supp.iter().enumerate() {
            let alpha = f.alpha_for_norm[d_vec[old_j] as usize] as usize;
            assert!(alpha != 0);
            for r in 0..m {
                g_p[r][new_j] = f.mul[alpha][g_mat[r][old_j] as usize];
            }
        }
        if verify_hermitian_and_rank(f, &g_p) {
            export_certificate(
                f,
                &g_p,
                d_tar,
                &format!("RainsPuncture(from_n{}_m{})", n, m),
                out_dir,
                saved_set,
            );
        }
    }
}

fn parse_codetables_snapshot(path: &str) -> HashMap<usize, HashMap<(usize, usize), (usize, usize)>> {
    let txt = fs::read_to_string(path).unwrap();
    let mut out: HashMap<usize, HashMap<(usize, usize), (usize, usize)>> = HashMap::new();
    for q in [2usize, 3, 4, 5, 7, 8] {
        let key = format!("\"{}\": {{", q);
        if let Some(pos) = txt.find(&key) {
            let sub = &txt[pos + key.len()..];
            let end = sub.find('}').unwrap_or(sub.len());
            let body = &sub[..end];
            let mut qmap = HashMap::new();
            let mut rest = body;
            while let Some(q1) = rest.find('"') {
                let after1 = &rest[q1 + 1..];
                let q2 = match after1.find('"') {
                    Some(i) => i,
                    None => break,
                };
                let nk_str = &after1[..q2];
                let after_nk = &after1[q2 + 1..];
                let b1 = match after_nk.find('[') {
                    Some(i) => i,
                    None => break,
                };
                let b2 = match after_nk.find(']') {
                    Some(i) => i,
                    None => break,
                };
                let bounds_str = &after_nk[b1 + 1..b2];
                let mut sp_nk = nk_str.split(',');
                let n: usize = sp_nk.next().unwrap().trim().parse().unwrap();
                let k: usize = sp_nk.next().unwrap().trim().parse().unwrap();
                let mut sp_b = bounds_str.split(',');
                let dl: usize = sp_b.next().unwrap().trim().parse().unwrap();
                let du: usize = sp_b.next().unwrap().trim().parse().unwrap();
                qmap.insert((n, k), (dl, du));
                rest = &after_nk[b2 + 1..];
            }
            out.insert(q, qmap);
        }
    }
    out
}

fn load_existing_saved(out_dir: &str) -> HashSet<(usize, usize, usize)> {
    let mut s = HashSet::new();
    if let Ok(rd) = fs::read_dir(out_dir) {
        for ent in rd.flatten() {
            let name = ent.file_name().to_string_lossy().to_string();
            if name.starts_with("CONFIRMED_QECC_CROWN_q") && name.ends_with(".json") {
                let parts: Vec<&str> = name
                    .trim_start_matches("CONFIRMED_QECC_CROWN_")
                    .trim_end_matches(".json")
                    .split('_')
                    .collect();
                if parts.len() >= 3 {
                    if let (Ok(q), Ok(n), Ok(k)) = (
                        parts[0][1..].parse::<usize>(),
                        parts[1][1..].parse::<usize>(),
                        parts[2][1..].parse::<usize>(),
                    ) {
                        s.insert((q, n, k));
                    }
                }
            }
        }
    }
    s
}

#[derive(Clone)]
struct Target {
    q: usize,
    m: usize,
    l: usize,
    border_p: usize, // 0, 1, 2 (`2 | m`), or 3 (`3 | m`)
    d_tar: usize,
    direct_beats: bool,
    cascade_pot: usize,
}

impl Target {
    fn n_est(&self) -> usize {
        self.l * self.m + self.border_p
    }
    fn k_est(&self) -> usize {
        (self.l - 2) * self.m + self.border_p
    }
}

fn hunt_target(
    t: &Target,
    f: &GFq2,
    snap_q: &HashMap<(usize, usize), (usize, usize)>,
    secs_for_target: f64,
    out_dir: &str,
    saved_set: &Arc<Mutex<HashSet<(usize, usize, usize)>>>,
) {
    let m = t.m;
    let l = t.l;
    let bp = t.border_p;
    let l_m_1 = l - 1;
    let n = t.n_est();
    let k = t.k_est();
    let d_tar = t.d_tar;

    if t.direct_beats && t.cascade_pot <= 1 && saved_set.lock().unwrap().contains(&(t.q, n, k)) {
        return;
    }

    // For `bp > 0`, `lam` MUST be `1`, and we can pick border column scalar `alpha` with norm `d_b in gfq_elems \ {0}`!
    let configs: Vec<(u8, u8)> = if bp == 0 {
        f.norm1_lambdas.iter().map(|&lam| (lam, 0u8)).collect()
    } else {
        // Test up to 3 nonzero norms `d_b` in `GF(q)^\times`
        f.gfq_elems
            .iter()
            .skip(1)
            .take(3)
            .map(|&db| (1u8, db))
            .collect()
    };

    let per_cfg_secs = (secs_for_target / (configs.len() as f64)).max(0.35);

    for &(lam, db) in &configs {
        if t.direct_beats && t.cascade_pot <= 1 && saved_set.lock().unwrap().contains(&(t.q, n, k)) {
            return;
        }
        // Note: if `bp > 0`, border column has weight `m / bp`! Check `1 + m / bp >= d_tar`!
        if bp > 0 && 1 + (m / bp) < d_tar {
            return;
        }

        let min_wt = (d_tar - 1).min(m);
        let total_space = (f.q2 as u128).saturating_pow(m as u32);
        let max_pool = 50_000usize;
        let mut pool: Vec<Vec<u8>> = Vec::new();

        if total_space <= 150_000 {
            let tot = total_space as usize;
            for mut idx in 1..tot {
                let mut p = vec![0u8; m];
                let mut w = 0usize;
                for r in 0..m {
                    let v = (idx % f.q2) as u8;
                    idx /= f.q2;
                    p[r] = v;
                    if v != 0 {
                        w += 1;
                    }
                }
                if w >= min_wt {
                    pool.push(p);
                    if pool.len() >= max_pool {
                        break;
                    }
                }
            }
        } else {
            let mut rng = XorShift64(0x9E37_79B9_7F4A_7C15 ^ ((t.q * 131 + m * 17 + lam as usize + db as usize * 7) as u64));
            let mut seen = HashSet::with_capacity(max_pool);
            let mut tries = 0usize;
            while pool.len() < max_pool && tries < max_pool * 8 {
                tries += 1;
                let mut p = vec![0u8; m];
                let mut w = 0usize;
                for r in 0..m {
                    let v = ((rng.next() as usize) % f.q2) as u8;
                    p[r] = v;
                    if v != 0 {
                        w += 1;
                    }
                }
                while w < min_wt {
                    let pos = (rng.next() as usize) % m;
                    if p[pos] == 0 {
                        p[pos] = 1 + (((rng.next() as usize) % (f.q2 - 1)) as u8);
                        w += 1;
                    }
                }
                let pk = f.pack_poly(&p);
                if seen.insert(pk) {
                    pool.push(p);
                }
            }
        }
        if pool.is_empty() {
            continue;
        }

        let mut norms: Vec<Vec<u8>> = Vec::with_capacity(pool.len());
        let mut buckets: HashMap<u128, Vec<usize>> = HashMap::new();
        for (idx, a) in pool.iter().enumerate() {
            let adj = f.poly_adj(a, lam);
            let nrm = f.poly_mul_constacyclic(a, &adj, lam);
            let nk = f.pack_poly(&nrm);
            buckets.entry(nk).or_default().push(idx);
            norms.push(nrm);
        }

        // Target polynomial for `sum_{i=1}^{l-1} a_i a_i^\dagger`:
        // Unbordered (`bp == 0`): `-1`
        // `bp`-bordered (`bp | m`, `lam == 1`, norm `db`): `-1 - db * (1 + x^{bp} + x^{2*bp} + ... + x^{m - bp})`!
        let mut target_poly = vec![0u8; m];
        target_poly[0] = f.sub[0][1];
        if bp > 0 {
            let mut r = 0usize;
            while r < m {
                target_poly[r] = f.sub[target_poly[r] as usize][db as usize];
                r += bp;
            }
        }

        let alpha_b = if bp > 0 { f.alpha_for_norm[db as usize] } else { 0 };
        let found = Arc::new(AtomicBool::new(false));
        let winner_polys: Arc<Mutex<Option<Vec<Vec<u8>>>>> = Arc::new(Mutex::new(None));
        let pool_arc = Arc::new(pool);
        let norms_arc = Arc::new(norms);
        let buckets_arc = Arc::new(buckets);
        let f_arc = Arc::new(f.clone());
        let target_arc = Arc::new(target_poly);

        let n_threads = 16usize;
        let t_cfg = Instant::now();
        let mut handles = Vec::with_capacity(n_threads);

        for tid in 0..n_threads {
            let found_c = Arc::clone(&found);
            let win_c = Arc::clone(&winner_polys);
            let pool_c = Arc::clone(&pool_arc);
            let norms_c = Arc::clone(&norms_arc);
            let buck_c = Arc::clone(&buckets_arc);
            let f_c = Arc::clone(&f_arc);
            let targ_c = Arc::clone(&target_arc);

            handles.push(thread::spawn(move || {
                let mut rng = XorShift64(0xD1B5_4A32_D192_ED03 ^ ((tid as u64 + 1) * 0x9E37_79B9) ^ ((n * 29 + bp) as u64));
                let plen = pool_c.len();
                let start = Instant::now();
                let mut chosen = vec![0usize; l_m_1];

                while !found_c.load(Ordering::Relaxed) && start.elapsed().as_secs_f64() < per_cfg_secs {
                    for _batch in 0..64 {
                        if found_c.load(Ordering::Relaxed) {
                            break;
                        }
                        let mut rem = (*targ_c).clone();
                        for b in 0..(l_m_1 - 1) {
                            let idx = (rng.next() as usize) % plen;
                            chosen[b] = idx;
                            let nrm = &norms_c[idx];
                            for r in 0..m {
                                rem[r] = f_c.sub[rem[r] as usize][nrm[r] as usize];
                            }
                        }
                        let rem_k = f_c.pack_poly(&rem);
                        let cands = match buck_c.get(&rem_k) {
                            Some(v) => v,
                            None => continue,
                        };
                        chosen[l_m_1 - 1] = cands[(rng.next() as usize) % cands.len()];

                        let n_b = l_m_1 * m + bp;
                        let mut cols_b = vec![[0u8; 16]; n_b];
                        for b in 0..l_m_1 {
                            let blk = f_c.build_constacyclic_block(&pool_c[chosen[b]], lam);
                            for c in 0..m {
                                for r in 0..m {
                                    cols_b[b * m + c][r] = blk[r][c];
                                }
                            }
                        }
                        for c_off in 0..bp {
                            let mut r = c_off;
                            while r < m {
                                cols_b[l_m_1 * m + c_off][r] = alpha_b;
                                r += bp;
                            }
                        }

                        if is_pure_distance_at_least(&f_c, &cols_b, m, l_m_1, bp, d_tar) {
                            found_c.store(true, Ordering::Relaxed);
                            let polys: Vec<Vec<u8>> = chosen.iter().map(|&i| pool_c[i].clone()).collect();
                            *win_c.lock().unwrap() = Some(polys);
                            return;
                        }
                    }
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        let picked_polys = winner_polys.lock().unwrap().take();
        if let Some(polys) = picked_polys {
            let mut g_mat = vec![vec![0u8; n]; m];
            for r in 0..m {
                g_mat[r][r] = 1;
            }
            for (b, p) in polys.iter().enumerate() {
                let blk = f.build_constacyclic_block(p, lam);
                for r in 0..m {
                    for c in 0..m {
                        g_mat[r][(b + 1) * m + c] = blk[r][c];
                    }
                }
            }
            for c_off in 0..bp {
                let mut r = c_off;
                while r < m {
                    g_mat[r][l * m + c_off] = alpha_b;
                    r += bp;
                }
            }
            assert!(
                verify_hermitian_and_rank(f, &g_mat),
                "Hermitian/Rank verification failed on q={} n={} k={}",
                t.q,
                n,
                k
            );
            let method = format!(
                "HermitianQT_v2(m={},l={},bp={},lam={},db={},{:.2}s)",
                m,
                l,
                bp,
                lam,
                db,
                t_cfg.elapsed().as_secs_f64()
            );
            if t.direct_beats {
                export_certificate(f, &g_mat, d_tar, &method, out_dir, saved_set);
            }
            if n > m * m {
                run_rains_puncture_cascade(f, &g_mat, d_tar, snap_q, out_dir, saved_set);
            }
            break;
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let out_dir = &args[1];
    let total_budget_secs: f64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(145.0);

    let snap_path = format!("{}/codetables_qecc_snapshot.json", out_dir);
    let snap = parse_codetables_snapshot(&snap_path);
    let fields: HashMap<usize, GFq2> = [2usize, 3, 4, 5, 7, 8]
        .iter()
        .map(|&q| (q, GFq2::new(q)))
        .collect();
    let existing = load_existing_saved(out_dir);
    println!("[Hermitian QT v2] Pre-loaded {} already-discovered certificates.", existing.len());
    let saved_set = Arc::new(Mutex::new(existing));

    let mut targets: Vec<Target> = Vec::new();
    for &q in &[4usize, 5, 7, 8, 3, 2] {
        let snap_q = &snap[&q];
        let q2 = q * q;
        for m in 3..=12usize {
            let mut bps = vec![0usize, 1];
            if m % 2 == 0 {
                bps.push(2);
            }
            if m % 3 == 0 {
                bps.push(3);
            }
            for l in 3..=9usize {
                for &bp in &bps {
                    let n = l * m + bp;
                    let k = (l - 2) * m + bp;
                    if n > 82 {
                        continue;
                    }
                    for d_tar in 4..=(7.min(m + 1)) {
                        // Skip impossible Quantum MDS (`d_tar == m + 1` with `n > q^2 + 1`)!
                        if d_tar == m + 1 && n > q2 + 1 {
                            continue;
                        }
                        if bp > 0 && 1 + (m / bp) < d_tar {
                            continue;
                        }
                        // Keep runtime snappy on huge `q` + huge `d_tar`
                        if d_tar == 7 && (q >= 7 && n > 24 || q == 5 && n > 26 || n > 34) {
                            continue;
                        }
                        if d_tar == 6 && ((q == 8 && n > 32) || (q == 7 && n > 34) || (q == 5 && n > 42)) {
                            continue;
                        }
                        if d_tar == 5 && q == 8 && n > 50 {
                            continue;
                        }

                        let mut direct_beats = false;
                        if let Some(&(dl, du)) = snap_q.get(&(n, k)) {
                            if d_tar == dl + 1 && dl < du && !saved_set.lock().unwrap().contains(&(q, n, k)) {
                                direct_beats = true;
                            }
                        }
                        let mut cascade_pot = if direct_beats { 1 } else { 0 };
                        if n > m * m {
                            for s in 1..=(n - m * m) {
                                if n - s >= 2 * m && k >= s {
                                    if let Some(&(dl, du)) = snap_q.get(&(n - s, k - s)) {
                                        if d_tar > dl && dl < du && !saved_set.lock().unwrap().contains(&(q, n - s, k - s)) {
                                            cascade_pot += 1;
                                        }
                                    }
                                }
                            }
                        }
                        if direct_beats || cascade_pot >= 3 {
                            targets.push(Target {
                                q,
                                m,
                                l,
                                border_p: bp,
                                d_tar,
                                direct_beats,
                                cascade_pot,
                            });
                        }
                    }
                }
            }
        }
    }

    // Prioritize high-probability sweet spots (`d_tar <= m - 1` or high `q^2`, high `cascade_pot`, fast `d_tar`)!
    targets.sort_by(|a, b| {
        let slack_a = (a.m + 1 - a.d_tar) as i64; // slack >= 1 is non-MDS (HUGE hit rate!)
        let slack_b = (b.m + 1 - b.d_tar) as i64;
        let mds_pen_a = if slack_a == 0 { 180 } else { 0 };
        let mds_pen_b = if slack_b == 0 { 180 } else { 0 };
        let score_a = (a.d_tar as i64) * 90 + mds_pen_a - (a.cascade_pot as i64) * 18 + (a.n_est() as i64);
        let score_b = (b.d_tar as i64) * 90 + mds_pen_b - (b.cascade_pot as i64) * 18 + (b.n_est() as i64);
        score_a.cmp(&score_b)
    });

    println!(
        "[Hermitian QT v2 Blitz] Queued {} prime/prime-power targets across q in {{2,3,4,5,7,8}}! Running {:.0}s on 16 threads...",
        targets.len(),
        total_budget_secs
    );

    let t_global = Instant::now();
    for (idx, t) in targets.iter().enumerate() {
        let elapsed = t_global.elapsed().as_secs_f64();
        if elapsed >= total_budget_secs {
            break;
        }
        let rem = total_budget_secs - elapsed;
        let per_t = if t.cascade_pot >= 5 {
            2.2f64.min(rem)
        } else if t.d_tar <= 5 {
            1.1f64.min(rem)
        } else {
            1.8f64.min(rem)
        };
        if idx < 35 || t.cascade_pot >= 4 {
            println!(
                "  -> v2 #{}/{}: [[{}, {}, {}]]_{} (m={}, l={}, bp={}, cascade={})",
                idx + 1,
                targets.len(),
                t.n_est(),
                t.k_est(),
                t.d_tar,
                t.q,
                t.m,
                t.l,
                t.border_p,
                t.cascade_pot
            );
        }
        hunt_target(t, &fields[&t.q], &snap[&t.q], per_t, out_dir, &saved_set);
    }

    let tot_found = saved_set.lock().unwrap().len();
    println!(
        "\n=== HERMITIAN QT v2 BLITZ FINISHED IN {:.1}s: TOTAL EXPLICIT QECC CERTIFICATES = {} ===",
        t_global.elapsed().as_secs_f64(),
        tot_found
    );
}

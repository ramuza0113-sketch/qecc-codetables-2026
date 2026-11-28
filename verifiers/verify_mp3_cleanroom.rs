// Standalone Clean-Room First-Principles Verifier for 3-Block Quasi-Unitary Matrix-Product Quantum Codes (`verify_qecc_mp3_cleanroom.rs`)
// Supports ALL fields GF(q^2) for q in {2, 3, 4, 5, 7, 8, 9} and ANY target distance d_tar >= 4!
//
// Verifies 100% from raw matrix entries with ZERO Reed-Solomon, Vandermonde, or group-symmetry assumptions:
//   1. Exact dimensions: n == 3 * n0, m == m1 + m2 + m3, k == n - 2 * m; entries in 0..q^2-1.
//   2. Exact 3x3 upper-triangular block-structure identity over GF(q^2):
//        G == [ G1 , G1 * diag(D1) , G1 * diag(D2) ]
//             [  0 ,      G2       , G2 * diag(E)  ]
//             [  0 ,       0       ,      G3       ]
//      with EVERY coordinate j in 0..n0-1 satisfying NSC non-degeneracy:
//        D1[j] != 0, D2[j] != 0, E[j] != 0, AND D1[j] * E[j] != D2[j] in GF(q^2).
//   3. Exact full GF(q^2)-rank: rank(G) == m via full Gaussian elimination.
//   4. Exact full Hermitian self-orthogonality: G * G^dagger == 0 (all m x m entries == 0 over GF(q^2)).
//   5. Exact pure minimum distance d(C^{perp_H}) >= d_tar via 20-thread exhaustive column-subset rank DFS on G1, G2, G3:
//      - Exhaustively proves EVERY column subset of G1 (m1 x n0) of size <= (d_tar - 1) is linearly independent
//        over GF(q^2), hence d(C1^{perp_H}) >= d_tar.
//      - Exhaustively proves EVERY column subset of G2 (m2 x n0) of size <= ((d_tar - 1) / 2) is linearly independent
//        over GF(q^2), hence 2 * d(C2^{perp_H}) >= d_tar.
//      - Exhaustively proves EVERY column subset of G3 (m3 x n0) of size <= ((d_tar - 1) / 3) is linearly independent
//        over GF(q^2), hence 3 * d(C3^{perp_H}) >= d_tar.
//      - By block orthogonality G * (u1, u2, u3)^dagger == 0:
//          w1 := u1 + conj(D1)*u2 + conj(D2)*u3 in C1^{perp_H},
//          w2 := u2 + conj(E)*u3 in C2^{perp_H},
//          w3 := u3 in C3^{perp_H}:
//          * Case w1 != 0: wt(u1)+wt(u2)+wt(u3) >= wt(w1) >= d(C1^{perp_H}) >= d_tar.
//          * Case w1 == 0, w2 != 0: since D1[j] != 0 and D2[j] != 0, at least 2 of (u1[j],u2[j],u3[j]) are nonzero
//            for every j in supp(w2), so wt(u1)+wt(u2)+wt(u3) >= 2 * wt(w2) >= 2 * d(C2^{perp_H}) >= d_tar.
//          * Case w1 == 0, w2 == 0, w3 != 0: u2 = -conj(E)*w3 and u1 = (conj(D1)*conj(E) - conj(D2))*w3;
//            since E[j] != 0 and D1[j]*E[j] != D2[j], ALL 3 of (u1[j],u2[j],u3[j]) are nonzero for every j in supp(w3),
//            so wt(u1)+wt(u2)+wt(u3) = 3 * wt(w3) >= 3 * d(C3^{perp_H}) >= d_tar.
//
// Prints "VALID_PURE_MP3_QECC_CLEANROOM" and exits 0 iff 100% proven, else exits 1.

use std::env;
use std::fs;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

const MAX_Q2: usize = 82;
const MAX_M: usize = 42;

#[derive(Clone)]
struct Field {
    q2: usize,
    add: [[u8; MAX_Q2]; MAX_Q2],
    sub: [[u8; MAX_Q2]; MAX_Q2],
    mul: [[u8; MAX_Q2]; MAX_Q2],
    inv: [u8; MAX_Q2],
    conj: [u8; MAX_Q2],
}

impl Field {
    fn new(q: usize) -> Self {
        assert!(matches!(q, 2 | 3 | 4 | 5 | 7 | 8 | 9), "Unsupported q={}", q);
        let q2 = q * q;
        let mut add = [[0u8; MAX_Q2]; MAX_Q2];
        let mut sub = [[0u8; MAX_Q2]; MAX_Q2];
        let mut mul = [[0u8; MAX_Q2]; MAX_Q2];
        let mut inv = [0u8; MAX_Q2];
        let mut conj = [0u8; MAX_Q2];

        if q == 2 {
            for u in 0..4usize {
                let (a0, b0) = (u & 1, (u >> 1) & 1);
                conj[u] = (if b0 == 0 { a0 } else { (a0 ^ 1) | 2 }) as u8;
                for v in 0..4usize {
                    let (a1, b1) = (v & 1, (v >> 1) & 1);
                    add[u][v] = (u ^ v) as u8;
                    sub[u][v] = (u ^ v) as u8;
                    mul[u][v] = (((a0 & a1) ^ (b0 & b1)) | (((a0 & b1) ^ (b0 & a1) ^ (b0 & b1)) << 1)) as u8;
                }
            }
        } else if q == 4 || q == 8 {
            let deg = if q == 4 { 4 } else { 6 };
            for u in 0..q2 {
                for v in 0..q2 {
                    add[u][v] = (u ^ v) as u8;
                    sub[u][v] = (u ^ v) as u8;
                    let (mut a, mut b, mut p) = (u, v, 0usize);
                    for _ in 0..deg {
                        if (b & 1) != 0 { p ^= a; }
                        let hi = a & (1 << (deg - 1));
                        a = (a << 1) & (q2 - 1);
                        if hi != 0 { a ^= 3; }
                        b >>= 1;
                    }
                    mul[u][v] = p as u8;
                }
            }
            for u in 0..q2 {
                let mut p = u;
                for _ in 0..(deg / 2) { p = mul[p][p] as usize; }
                conj[u] = p as u8;
            }
        } else if q == 9 {
            for u in 0..81usize {
                let mut pu = [0usize; 4];
                let mut tu = u;
                for i in 0..4 { pu[i] = tu % 3; tu /= 3; }
                for v in 0..81usize {
                    let mut pv = [0usize; 4];
                    let mut tv = v;
                    for i in 0..4 { pv[i] = tv % 3; tv /= 3; }
                    let (mut sa, mut sb, mut p3) = (0usize, 0usize, 1usize);
                    for i in 0..4 {
                        sa += ((pu[i] + pv[i]) % 3) * p3;
                        sb += ((pu[i] + 3 - pv[i]) % 3) * p3;
                        p3 *= 3;
                    }
                    add[u][v] = sa as u8;
                    sub[u][v] = sb as u8;
                    let mut pr = [0usize; 7];
                    for i in 0..4 {
                        for j in 0..4 { pr[i + j] = (pr[i + j] + pu[i] * pv[j]) % 3; }
                    }
                    for &d in &[6usize, 5, 4] {
                        let c = pr[d];
                        if c != 0 {
                            pr[d] = 0;
                            pr[d - 3] = (pr[d - 3] + 2 * c) % 3;
                            pr[d - 4] = (pr[d - 4] + c) % 3;
                        }
                    }
                    let (mut sm, mut pm) = (0usize, 1usize);
                    for i in 0..4 { sm += pr[i] * pm; pm *= 3; }
                    mul[u][v] = sm as u8;
                }
            }
            for u in 0..81usize {
                let mut p = u;
                for _ in 0..2 {
                    let p2 = mul[p][p] as usize;
                    p = mul[p2][p] as usize;
                }
                conj[u] = p as u8;
            }
        } else {
            let c_ns = if q == 7 { 3 } else { 2 };
            for u in 0..q2 {
                let (x0, y0) = (u % q, u / q);
                conj[u] = (x0 + q * ((q - y0) % q)) as u8;
                for v in 0..q2 {
                    let (x1, y1) = (v % q, v / q);
                    add[u][v] = (((x0 + x1) % q) + q * ((y0 + y1) % q)) as u8;
                    sub[u][v] = (((x0 + q - x1) % q) + q * ((y0 + q - y1) % q)) as u8;
                    mul[u][v] = (((x0 * x1 + c_ns * y0 * y1) % q) + q * ((x0 * y1 + y0 * x1) % q)) as u8;
                }
            }
        }

        for u in 1..q2 {
            let mut ok = false;
            for v in 1..q2 {
                if mul[u][v] == 1 { inv[u] = v as u8; ok = true; break; }
            }
            assert!(ok, "No inverse for {} in GF({}^2)", u, q);
            assert_eq!(conj[conj[u] as usize] as usize, u);
        }
        Self { q2, add, sub, mul, inv, conj }
    }
}

fn parse_int(txt: &str, key: &str) -> usize {
    let pat = format!("\"{}\":", key);
    let idx = txt.find(&pat).unwrap_or_else(|| panic!("missing key {}", key)) + pat.len();
    let rest = txt[idx..].trim_start();
    let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
    rest[..end].parse().unwrap()
}

fn parse_vec(txt: &str, key: &str) -> Vec<u8> {
    let pat = format!("\"{}\":", key);
    let idx = txt.find(&pat).expect("missing vec key") + pat.len();
    let s = txt[idx..].trim_start();
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut num = String::new();
    for ch in s.chars() {
        match ch {
            '[' => depth += 1,
            ']' => {
                if !num.is_empty() { out.push(num.parse::<u8>().unwrap()); num.clear(); }
                depth -= 1;
                if depth == 0 { break; }
            }
            '0'..='9' => num.push(ch),
            _ => {
                if !num.is_empty() { out.push(num.parse::<u8>().unwrap()); num.clear(); }
            }
        }
    }
    out
}

fn parse_matrix(txt: &str, key: &str) -> Vec<Vec<u8>> {
    let pat = format!("\"{}\":", key);
    let idx = txt.find(&pat).expect("missing mat key") + pat.len();
    let s = txt[idx..].trim_start();
    let mut mat = Vec::new();
    let mut cur = Vec::new();
    let mut depth = 0i32;
    let mut num = String::new();
    for ch in s.chars() {
        match ch {
            '[' => { depth += 1; if depth == 2 { cur.clear(); } }
            ']' => {
                if !num.is_empty() { cur.push(num.parse::<u8>().unwrap()); num.clear(); }
                if depth == 2 { mat.push(cur.clone()); }
                depth -= 1;
                if depth == 0 { break; }
            }
            '0'..='9' => num.push(ch),
            _ => {
                if !num.is_empty() { cur.push(num.parse::<u8>().unwrap()); num.clear(); }
            }
        }
    }
    mat
}

fn check_rank(f: &Field, g: &[Vec<u8>], rows: usize, cols: usize) -> usize {
    let mut a = g.to_vec();
    let mut rk = 0usize;
    for col in 0..cols {
        if rk == rows { break; }
        let mut piv = None;
        for r in rk..rows {
            if a[r][col] != 0 { piv = Some(r); break; }
        }
        let p = match piv { Some(v) => v, None => continue };
        a.swap(rk, p);
        let inv_p = f.inv[a[rk][col] as usize] as usize;
        for c in col..cols { a[rk][c] = f.mul[a[rk][c] as usize][inv_p]; }
        for r in 0..rows {
            if r != rk && a[r][col] != 0 {
                let sc = a[r][col] as usize;
                for c in col..cols {
                    let prod = f.mul[sc][a[rk][c] as usize] as usize;
                    a[r][c] = f.sub[a[r][c] as usize][prod];
                }
            }
        }
        rk += 1;
    }
    rk
}

#[allow(clippy::too_many_arguments)]
fn dfs_subset_indep(
    f: &Field,
    cols: &[[u8; MAX_M]],
    n0: usize,
    m_rows: usize,
    w_max: usize,
    start_c: usize,
    depth: usize,
    basis: &[[u8; MAX_M]; MAX_M],
    pivs: &[usize; MAX_M],
    fail: &AtomicBool,
) {
    if fail.load(Ordering::Relaxed) { return; }
    for j in start_c..n0 {
        if fail.load(Ordering::Relaxed) { return; }
        let mut v = cols[j];
        for b in 0..depth {
            let p = pivs[b];
            let coeff = v[p] as usize;
            if coeff != 0 {
                for r in 0..m_rows {
                    let prod = f.mul[coeff][basis[b][r] as usize] as usize;
                    v[r] = f.sub[v[r] as usize][prod];
                }
            }
        }
        let mut new_piv = None;
        for r in 0..m_rows {
            if v[r] != 0 { new_piv = Some(r); break; }
        }
        let np = match new_piv {
            Some(r) => r,
            None => {
                fail.store(true, Ordering::Relaxed);
                return;
            }
        };
        if depth + 1 < w_max {
            let inv_v = f.inv[v[np] as usize] as usize;
            for r in 0..m_rows {
                v[r] = f.mul[v[r] as usize][inv_v];
            }
            let mut next_basis = *basis;
            let mut next_pivs = *pivs;
            next_basis[depth] = v;
            next_pivs[depth] = np;
            dfs_subset_indep(f, cols, n0, m_rows, w_max, j + 1, depth + 1, &next_basis, &next_pivs, fail);
        }
    }
}

fn verify_all_subsets_indep(f: &Field, mat: &[Vec<u8>], m_rows: usize, n0: usize, w_max: usize) -> Result<(), String> {
    if w_max == 0 { return Ok(()); }
    if w_max > m_rows {
        return Err(format!("w_max={} > m_rows={} cannot be independent", w_max, m_rows));
    }
    let mut cols = vec![[0u8; MAX_M]; n0];
    for j in 0..n0 {
        let mut nz = false;
        for r in 0..m_rows {
            cols[j][r] = mat[r][j];
            if mat[r][j] != 0 { nz = true; }
        }
        if !nz {
            return Err(format!("zero column at j={}", j));
        }
    }
    if w_max == 1 { return Ok(()); }

    let f_arc = Arc::new(f.clone());
    let c_arc = Arc::new(cols);
    let fail = Arc::new(AtomicBool::new(false));
    let n_th = 20usize;
    let mut handles = Vec::new();

    for tid in 0..n_th {
        let ff = Arc::clone(&f_arc);
        let cc = Arc::clone(&c_arc);
        let fl = Arc::clone(&fail);
        handles.push(thread::spawn(move || {
            for j0 in (tid..n0).step_by(n_th) {
                if fl.load(Ordering::Relaxed) { return; }
                let mut v = cc[j0];
                let p0 = (0..m_rows).find(|&r| v[r] != 0).unwrap();
                let inv_v = ff.inv[v[p0] as usize] as usize;
                for r in 0..m_rows {
                    v[r] = ff.mul[v[r] as usize][inv_v];
                }
                let mut basis = [[0u8; MAX_M]; MAX_M];
                let mut pivs = [0usize; MAX_M];
                basis[0] = v;
                pivs[0] = p0;
                dfs_subset_indep(&ff, &cc, n0, m_rows, w_max, j0 + 1, 1, &basis, &pivs, &fl);
            }
        }));
    }
    for h in handles { h.join().unwrap(); }
    if fail.load(Ordering::Relaxed) {
        Err(format!("dependent column subset of size <= {} found", w_max))
    } else {
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn verify_mp3(
    q: usize,
    n: usize,
    m1: usize,
    m2: usize,
    m3: usize,
    d_tar: usize,
    g: &[Vec<u8>],
    g1: &[Vec<u8>],
    g2: &[Vec<u8>],
    g3: &[Vec<u8>],
    d1: &[u8],
    d2: &[u8],
    e_diag: &[u8],
) -> Result<(), String> {
    if n % 3 != 0 { return Err("n must be divisible by 3".into()); }
    let n0 = n / 3;
    let m = m1 + m2 + m3;
    if m > MAX_M || n <= 2 * m { return Err("invalid m or k <= 0".into()); }
    let f = Field::new(q);

    if g.len() != m || g1.len() != m1 || g2.len() != m2 || g3.len() != m3
        || d1.len() != n0 || d2.len() != n0 || e_diag.len() != n0
    {
        return Err("dimension mismatch".into());
    }
    for r in 0..m {
        if g[r].len() != n || g[r].iter().any(|&v| (v as usize) >= f.q2) {
            return Err("invalid G row".into());
        }
    }
    for j in 0..n0 {
        let (x1, x2, xe) = (d1[j] as usize, d2[j] as usize, e_diag[j] as usize);
        if x1 == 0 || x2 == 0 || xe == 0 || x1 >= f.q2 || x2 >= f.q2 || xe >= f.q2 {
            return Err(format!("invalid diagonal entry at j={}", j));
        }
        if f.mul[x1][xe] as usize == x2 {
            return Err(format!("NSC non-degeneracy violated: D1[{}]*E[{}] == D2[{}]", j, j, j));
        }
    }

    // Verify exact 3x3 upper-triangular block structure!
    for r in 0..m1 {
        if g1[r].len() != n0 { return Err("invalid G1 row".into()); }
        for j in 0..n0 {
            if g[r][j] != g1[r][j] { return Err("G(1,1) != G1".into()); }
            if g[r][n0 + j] != f.mul[g1[r][j] as usize][d1[j] as usize] { return Err("G(1,2) != G1*D1".into()); }
            if g[r][2 * n0 + j] != f.mul[g1[r][j] as usize][d2[j] as usize] { return Err("G(1,3) != G1*D2".into()); }
        }
    }
    for r in 0..m2 {
        if g2[r].len() != n0 { return Err("invalid G2 row".into()); }
        for j in 0..n0 {
            if g[m1 + r][j] != 0 { return Err("G(2,1) != 0".into()); }
            if g[m1 + r][n0 + j] != g2[r][j] { return Err("G(2,2) != G2".into()); }
            if g[m1 + r][2 * n0 + j] != f.mul[g2[r][j] as usize][e_diag[j] as usize] { return Err("G(2,3) != G2*E".into()); }
        }
    }
    for r in 0..m3 {
        if g3[r].len() != n0 { return Err("invalid G3 row".into()); }
        for j in 0..n0 {
            if g[m1 + m2 + r][j] != 0 { return Err("G(3,1) != 0".into()); }
            if g[m1 + m2 + r][n0 + j] != 0 { return Err("G(3,2) != 0".into()); }
            if g[m1 + m2 + r][2 * n0 + j] != g3[r][j] { return Err("G(3,3) != G3".into()); }
        }
    }

    let rk = check_rank(&f, g, m, n);
    if rk != m { return Err(format!("rank(G)={} != m={}", rk, m)); }

    for r1 in 0..m {
        for r2 in r1..m {
            let mut acc = 0usize;
            for c in 0..n {
                let cj = f.conj[g[r2][c] as usize] as usize;
                let prod = f.mul[g[r1][c] as usize][cj] as usize;
                acc = f.add[acc][prod] as usize;
            }
            if acc != 0 {
                return Err(format!("G non-Hermitian-orthogonal at ({},{}): {}", r1, r2, acc));
            }
        }
    }

    let w1 = d_tar - 1;
    let w2 = (d_tar - 1) / 2;
    let w3 = (d_tar - 1) / 3;
    verify_all_subsets_indep(&f, g1, m1, n0, w1)?;
    verify_all_subsets_indep(&f, g2, m2, n0, w2)?;
    verify_all_subsets_indep(&f, g3, m3, n0, w3)?;
    Ok(())
}

fn run_selftest() {
    for &q in &[2usize, 3, 4, 5, 7, 8, 9] {
        let _ = Field::new(q);
    }
    // Self-test on q=4 (`GF(16)`), n0=16 (affine GF(16)), m1=5 (`d1=6`), m2=2 (`2*d2=6`), m3=1 (`3*d3=6`) -> `[[48, 32, 6]]_4`!
    // Check orthogonality on full GF(16): sum_{x in GF(16)} x^e == 0 for 0 <= e < 15!
    // Pick constant d1, d2, e with |e|^2 == 1 (char 2: -1 == 1!), 1 ^ |d1|^2 ^ |d2|^2 == 0, and d1*e != d2!
    let f = Field::new(4);
    let mut norm = [0u8; 16];
    for u in 1..16usize {
        norm[u] = f.mul[u][f.conj[u] as usize];
    }
    let e0 = (1..16u8).find(|&u| norm[u as usize] == 1 && u != 1).unwrap();
    let mut found_pair = None;
    for a in 1..16u8 {
        for b in 1..16u8 {
            if (1 ^ norm[a as usize] ^ norm[b as usize]) == 0 && f.mul[a as usize][e0 as usize] != b {
                found_pair = Some((a, b));
                break;
            }
        }
        if found_pair.is_some() { break; }
    }
    let (d1_0, d2_0) = found_pair.unwrap();
    let n0 = 16usize;
    let (m1, m2, m3) = (5usize, 2usize, 1usize);
    let mut g1 = vec![vec![0u8; n0]; m1];
    let mut g2 = vec![vec![0u8; n0]; m2];
    let g3 = vec![vec![1u8; n0]; m3];
    for x in 0..n0 {
        let mut p = 1u8;
        for a in 0..m1 {
            g1[a][x] = p;
            if a < m2 { g2[a][x] = p; }
            p = f.mul[p as usize][x];
        }
    }
    let d1 = vec![d1_0; n0];
    let d2 = vec![d2_0; n0];
    let ed = vec![e0; n0];
    let mut g = vec![vec![0u8; 3 * n0]; m1 + m2 + m3];
    for r in 0..m1 {
        for j in 0..n0 {
            g[r][j] = g1[r][j];
            g[r][n0 + j] = f.mul[g1[r][j] as usize][d1_0 as usize];
            g[r][2 * n0 + j] = f.mul[g1[r][j] as usize][d2_0 as usize];
        }
    }
    for r in 0..m2 {
        for j in 0..n0 {
            g[m1 + r][n0 + j] = g2[r][j];
            g[m1 + r][2 * n0 + j] = f.mul[g2[r][j] as usize][e0 as usize];
        }
    }
    for j in 0..n0 { g[m1 + m2][2 * n0 + j] = g3[0][j]; }

    assert!(verify_mp3(4, 48, 5, 2, 1, 6, &g, &g1, &g2, &g3, &d1, &d2, &ed).is_ok(), "[[48,32,6]]_4 MP3 selftest failed");
    assert!(verify_mp3(4, 48, 5, 2, 1, 7, &g, &g1, &g2, &g3, &d1, &d2, &ed).is_err(), "[[48,32,7]]_4 MP3 must fail");
    println!("SELFTEST_MP3_CLEANROOM_OK");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() >= 2 && args[1] == "--selftest" {
        run_selftest();
        return;
    }
    if args.len() < 2 {
        eprintln!("Usage: verify_qecc_mp3_cleanroom <json_path>");
        std::process::exit(1);
    }
    let t0 = Instant::now();
    let txt = fs::read_to_string(&args[1]).expect("failed to read json");
    let q = parse_int(&txt, "q");
    let n = parse_int(&txt, "n");
    let m1 = parse_int(&txt, "m1");
    let m2 = parse_int(&txt, "m2");
    let m3 = parse_int(&txt, "m3");
    let d_tar = parse_int(&txt, "d_tar");
    let g = parse_matrix(&txt, "G");
    let g1 = parse_matrix(&txt, "G1");
    let g2 = parse_matrix(&txt, "G2");
    let g3 = parse_matrix(&txt, "G3");
    let d1 = parse_vec(&txt, "D1");
    let d2 = parse_vec(&txt, "D2");
    let ed = parse_vec(&txt, "E");

    match verify_mp3(q, n, m1, m2, m3, d_tar, &g, &g1, &g2, &g3, &d1, &d2, &ed) {
        Ok(()) => {
            println!(
                "VALID_PURE_MP3_QECC_CLEANROOM q={} n={} k={} d={} (m1={},m2={},m3={}) elapsed={:.3}s",
                q, n, n - 2 * (m1 + m2 + m3), d_tar, m1, m2, m3, t0.elapsed().as_secs_f64()
            );
            std::process::exit(0);
        }
        Err(msg) => {
            eprintln!("FAIL: {}", msg);
            std::process::exit(1);
        }
    }
}

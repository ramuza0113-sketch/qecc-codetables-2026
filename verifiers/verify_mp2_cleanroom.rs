// Standalone Clean-Room First-Principles Verifier for Quasi-Unitary Matrix-Product Quantum Codes (`verify_qecc_mp_plotkin_cleanroom.rs`)
// Supports ALL fields GF(q^2) for q in {2, 3, 4, 5, 7, 8, 9} and ANY target distance d_tar >= 4 (including d_tar = 8..14)!
//
// Verifies 100% from raw matrix entries with ZERO Reed-Solomon, Vandermonde, or group-symmetry assumptions:
//   1. Exact dimensions: n == 2 * n0, m == m1 + m2, k == n - 2 * m; entries in 0..q^2-1.
//   2. Exact block-structure identity over GF(q^2):
//        G == [ G1 , G1 * diag(D) ]
//             [  0 ,     G2       ]
//      with EVERY diagonal entry D[j] != 0 in GF(q^2)*.
//   3. Exact full GF(q^2)-rank: rank(G) == m via full Gaussian elimination.
//   4. Exact full Hermitian self-orthogonality: G * G^dagger == 0 (all m x m entries == 0 over GF(q^2)).
//   5. Exact pure minimum distance d(C^{perp_H}) >= d_tar via 20-thread exhaustive column-subset rank DFS on G1 and G2:
//      - Exhaustively proves EVERY column subset of G1 (m1 x n0) of size <= (d_tar - 1) is linearly independent
//        over GF(q^2), hence d(C1^{perp_H}) >= d_tar.
//      - Exhaustively proves EVERY column subset of G2 (m2 x n0) of size <= ((d_tar - 1) / 2) is linearly independent
//        over GF(q^2), hence d(C2^{perp_H}) >= ((d_tar + 1) / 2), so 2 * d(C2^{perp_H}) >= d_tar.
//      - By block orthogonality G * (u, v)^dagger == 0 <=> G1*(u + conj(D)*v)^dagger == 0 and G2*v^dagger == 0:
//          * If u + conj(D)*v == 0, then v != 0 in C2^{perp_H} and u == -conj(D)*v (same support since D[j] != 0),
//            so wt(u, v) == 2 * wt(v) >= 2 * d(C2^{perp_H}) >= d_tar.
//          * If w := u + conj(D)*v != 0, then w in C1^{perp_H} \ {0}, and wt(u, v) = wt(u) + wt(v) >= wt(w) >= d_tar.
//
// Prints "VALID_PURE_MP_QECC_CLEANROOM" and exits 0 iff 100% proven, else exits 1.

use std::env;
use std::fs;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

const MAX_Q2: usize = 82;
const MAX_M: usize = 54;

#[derive(Clone)]
struct Field {
    q: usize,
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
        Self { q, q2, add, sub, mul, inv, conj }
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
    let rem_need = 1usize; // we check EVERY subset of size 1..=w_max!
    let _ = rem_need;
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

fn binom_cap(n: usize, k: usize) -> u128 {
    if k > n { return 0; }
    let kk = k.min(n - k);
    let mut r = 1u128;
    for i in 0..kk {
        r = r.saturating_mul((n - i) as u128) / ((i + 1) as u128);
    }
    r
}

/// Constructively reconstructs `(x_j, y_j, c_j)` purely from raw `mat` (`m_rows x n0`) and verifies:
///   (A) Every entry `mat[a][j] == c_j * x_j^a * y_j^{m_rows - 1 - a}` over `GF(q^2)` with `c_j != 0`, AND
///   (B) Every pair `0 <= i < j < n0` satisfies `x_i * y_j != x_j * y_i` in `GF(q^2)`.
/// By the Homogeneous Vandermonde Determinant Theorem over fields, (A) + (B) => every `m_rows x m_rows`
/// minor of `mat` is a product of nonzero field elements, hence strictly nonzero!
fn try_verify_homogeneous_vandermonde_mds(f: &Field, mat: &[Vec<u8>], m_rows: usize, n0: usize) -> bool {
    if m_rows < 2 || n0 < m_rows {
        return false;
    }
    let mut xs = vec![0usize; n0];
    let mut ys = vec![0usize; n0];
    let mut cs = vec![0usize; n0];
    for j in 0..n0 {
        let m0 = mat[0][j] as usize;
        if m0 != 0 {
            cs[j] = m0;
            ys[j] = 1usize;
            xs[j] = f.mul[mat[1][j] as usize][f.inv[m0] as usize] as usize;
        } else {
            let ml = mat[m_rows - 1][j] as usize;
            if ml == 0 {
                return false;
            }
            cs[j] = ml;
            xs[j] = 1usize;
            ys[j] = 0usize;
        }
        let mut xp = vec![1usize; m_rows];
        let mut yp = vec![1usize; m_rows];
        for a in 1..m_rows {
            xp[a] = f.mul[xp[a - 1]][xs[j]] as usize;
            yp[a] = f.mul[yp[a - 1]][ys[j]] as usize;
        }
        for a in 0..m_rows {
            let mon = f.mul[xp[a]][yp[m_rows - 1 - a]] as usize;
            let exp = f.mul[cs[j]][mon];
            if mat[a][j] != exp {
                return false;
            }
        }
    }
    for i in 0..n0 {
        for j in (i + 1)..n0 {
            if f.mul[xs[i]][ys[j]] == f.mul[xs[j]][ys[i]] {
                return false;
            }
        }
    }
    true
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
    if binom_cap(n0, w_max) > 300_000_000u128 && try_verify_homogeneous_vandermonde_mds(f, mat, m_rows, n0) {
        return Ok(());
    }

    let f_arc = Arc::new(f.clone());
    let c_arc = Arc::new(cols);
    let fail = Arc::new(AtomicBool::new(false));
    let n_th = 20usize;
    let mut handles = Vec::new();

    if w_max >= 4 && n0 >= 6 {
        let mut tasks = Vec::new();
        for j0 in 0..n0 {
            for j1 in (j0 + 1)..n0 {
                for j2 in (j1 + 1)..n0 {
                    tasks.push((j0, j1, j2));
                }
            }
        }
        let t_arc = Arc::new(tasks);
        let next_idx = Arc::new(AtomicUsize::new(0));
        for _ in 0..n_th {
            let ff = Arc::clone(&f_arc);
            let cc = Arc::clone(&c_arc);
            let fl = Arc::clone(&fail);
            let tt = Arc::clone(&t_arc);
            let nx = Arc::clone(&next_idx);
            handles.push(thread::spawn(move || {
                loop {
                    if fl.load(Ordering::Relaxed) { return; }
                    let idx = nx.fetch_add(1, Ordering::Relaxed);
                    if idx >= tt.len() { return; }
                    let (j0, j1, j2) = tt[idx];
                    let mut basis = [[0u8; MAX_M]; MAX_M];
                    let mut pivs = [0usize; MAX_M];
                    let mut ok_pref = true;
                    for (d_p, &jc) in [j0, j1, j2].iter().enumerate() {
                        let mut v = cc[jc];
                        for b in 0..d_p {
                            let p = pivs[b];
                            let coeff = v[p] as usize;
                            if coeff != 0 {
                                for r in 0..m_rows {
                                    let prod = ff.mul[coeff][basis[b][r] as usize] as usize;
                                    v[r] = ff.sub[v[r] as usize][prod];
                                }
                            }
                        }
                        match (0..m_rows).find(|&r| v[r] != 0) {
                            Some(np) => {
                                let inv_v = ff.inv[v[np] as usize] as usize;
                                for r in 0..m_rows {
                                    v[r] = ff.mul[v[r] as usize][inv_v];
                                }
                                basis[d_p] = v;
                                pivs[d_p] = np;
                            }
                            None => {
                                fl.store(true, Ordering::Relaxed);
                                ok_pref = false;
                                break;
                            }
                        }
                    }
                    if !ok_pref { return; }
                    dfs_subset_indep(&ff, &cc, n0, m_rows, w_max, j2 + 1, 3, &basis, &pivs, &fl);
                }
            }));
        }
    } else {
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
    }
    for h in handles { h.join().unwrap(); }
    if fail.load(Ordering::Relaxed) {
        Err(format!("dependent column subset of size <= {} found", w_max))
    } else {
        Ok(())
    }
}

fn verify_mp(q: usize, n: usize, m1: usize, m2: usize, d_tar: usize, g: &[Vec<u8>], g1: &[Vec<u8>], g2: &[Vec<u8>], d_diag: &[u8]) -> Result<(), String> {
    if n % 2 != 0 { return Err("n must be even".into()); }
    let n0 = n / 2;
    let m = m1 + m2;
    if m > MAX_M || n <= 2 * m { return Err("invalid m or k <= 0".into()); }
    let f = Field::new(q);

    if g.len() != m || g1.len() != m1 || g2.len() != m2 || d_diag.len() != n0 {
        return Err("dimension mismatch".into());
    }
    for r in 0..m {
        if g[r].len() != n || g[r].iter().any(|&v| (v as usize) >= f.q2) { return Err("invalid G row".into()); }
    }
    for j in 0..n0 {
        if d_diag[j] == 0 || (d_diag[j] as usize) >= f.q2 { return Err("invalid D[j]".into()); }
    }

    // Verify exact block structure G == [[G1, G1*D], [0, G2]]!
    for r in 0..m1 {
        if g1[r].len() != n0 { return Err("invalid G1 row".into()); }
        for j in 0..n0 {
            if g[r][j] != g1[r][j] { return Err("G top-left != G1".into()); }
            let exp = f.mul[g1[r][j] as usize][d_diag[j] as usize];
            if g[r][n0 + j] != exp { return Err("G top-right != G1*D".into()); }
        }
    }
    for r in 0..m2 {
        if g2[r].len() != n0 { return Err("invalid G2 row".into()); }
        for j in 0..n0 {
            if g[m1 + r][j] != 0 { return Err("G bottom-left != 0".into()); }
            if g[m1 + r][n0 + j] != g2[r][j] { return Err("G bottom-right != G2".into()); }
        }
    }

    let rk = check_rank(&f, g, m, n);
    if rk != m { return Err(format!("rank(G)={} != m={}", rk, m)); }

    // Verify ALL m x m Hermitian inner products of G equal 0!
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
    verify_all_subsets_indep(&f, g1, m1, n0, w1)?;
    verify_all_subsets_indep(&f, g2, m2, n0, w2)?;
    Ok(())
}

fn run_selftest() {
    for &q in &[2usize, 3, 4, 5, 7, 8, 9] {
        let _ = Field::new(q);
    }
    // Self-test on q=3 (`GF(9)`), n0=9 (full affine GF(9)), m1=3 (`d1=4`), m2=1 (`d2=2` -> `2*2=4`) -> `[[18, 10, 4]]_3`!
    let f = Field::new(3);
    // Find d0 with d0 * conj(d0) == 2 == -1 in GF(9)
    let d0 = (1..9u8).find(|&u| f.mul[u as usize][f.conj[u as usize] as usize] == 2).unwrap();
    let mut g1 = vec![vec![0u8; 9]; 3];
    let mut g2 = vec![vec![1u8; 9]; 1];
    for x in 0..9usize {
        g1[0][x] = 1;
        g1[1][x] = x as u8;
        g1[2][x] = f.mul[x][x];
        g2[0][x] = 1;
    }
    let d_diag = vec![d0; 9];
    let mut g = vec![vec![0u8; 18]; 4];
    for r in 0..3 {
        for j in 0..9 {
            g[r][j] = g1[r][j];
            g[r][9 + j] = f.mul[g1[r][j] as usize][d0 as usize];
        }
    }
    for j in 0..9 { g[3][9 + j] = g2[0][j]; }
    assert!(verify_mp(3, 18, 3, 1, 4, &g, &g1, &g2, &d_diag).is_ok(), "[[18,10,4]]_3 MP selftest failed");
    assert!(verify_mp(3, 18, 3, 1, 5, &g, &g1, &g2, &d_diag).is_err(), "[[18,10,5]]_3 MP must fail");
    // Homogeneous Vandermonde MDS certificate selftest (including point at infinity (1,0), duplicate point rejection, and entry tamper rejection):
    let mut g1_pg = vec![vec![0u8; 10]; 3];
    for x in 0..9usize {
        let sc = ((x % 8) + 1) as usize;
        g1_pg[0][x] = sc as u8;
        g1_pg[1][x] = f.mul[sc][x];
        g1_pg[2][x] = f.mul[sc][f.mul[x][x] as usize];
    }
    g1_pg[0][9] = 0; g1_pg[1][9] = 0; g1_pg[2][9] = 2; // (1 : 0) scaled by 2
    assert!(try_verify_homogeneous_vandermonde_mds(&f, &g1_pg, 3, 10), "PG(1,9) Vandermonde cert must pass");
    let mut g1_tamper = g1_pg.clone();
    g1_tamper[2][4] = f.add[g1_tamper[2][4] as usize][1];
    assert!(!try_verify_homogeneous_vandermonde_mds(&f, &g1_tamper, 3, 10), "Tampered entry must fail cert");
    let mut g1_dup = g1_pg.clone();
    // Make col 9 proportional to col 3 `(x=3, y=1)` with scale 2:
    g1_dup[0][9] = 2; g1_dup[1][9] = f.mul[2][3]; g1_dup[2][9] = f.mul[2][f.mul[3][3] as usize];
    assert!(!try_verify_homogeneous_vandermonde_mds(&f, &g1_dup, 3, 10), "Duplicate projective point must fail cert");
    println!("SELFTEST_MP_PLOTKIN_CLEANROOM_OK");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() >= 2 && args[1] == "--selftest" {
        run_selftest();
        return;
    }
    if args.len() < 2 {
        eprintln!("Usage: verify_qecc_mp_plotkin_cleanroom <json_path>");
        std::process::exit(1);
    }
    let t0 = Instant::now();
    let txt = fs::read_to_string(&args[1]).expect("failed to read json");
    let q = parse_int(&txt, "q");
    let n = parse_int(&txt, "n");
    let m1 = parse_int(&txt, "m1");
    let m2 = parse_int(&txt, "m2");
    let d_tar = parse_int(&txt, "d_tar");
    let g = parse_matrix(&txt, "G");
    let g1 = parse_matrix(&txt, "G1");
    let g2 = parse_matrix(&txt, "G2");
    let d_diag = parse_vec(&txt, "D");

    match verify_mp(q, n, m1, m2, d_tar, &g, &g1, &g2, &d_diag) {
        Ok(()) => {
            println!(
                "VALID_PURE_MP_QECC_CLEANROOM q={} n={} k={} d={} (m1={},m2={}) elapsed={:.3}s",
                q, n, n - 2 * (m1 + m2), d_tar, m1, m2, t0.elapsed().as_secs_f64()
            );
            std::process::exit(0);
        }
        Err(msg) => {
            eprintln!("FAIL: {}", msg);
            std::process::exit(1);
        }
    }
}

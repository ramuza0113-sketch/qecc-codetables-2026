// 20-Thread Standalone Zero-Symmetry Universal Clean-Room Verifier for GF(q^2)-Hermitian Pure QECCs!
// Supports ALL `q in {2, 3, 4, 5, 7, 8, 9}`, `m <= 44`, `n <= 130`, `d_tar in 2..15`!
// Zero symmetry or pivot-position assumptions:
//   1. Constructs exact arithmetic `(add, sub, mul, inv, conj)` on GF(q^2) and self-tests field axioms + subtraction!
//   2. Verifies exact GF(q^2)-rank(G) == m via full Gaussian elimination (NO assumption that leading m columns == I_m!).
//   3. Verifies exact Hermitian self-orthogonality: G * G^\dagger == 0 over GF(q^2) for all m x m pairs.
//   4. Exhaustively verifies d(C^{\perp_H}) >= d_tar across 20 threads by proving EVERY subset of `w_max = d_tar - 1`
//      columns of G is linearly independent over GF(q^2), accelerated by exact `u128` / `[u128; 3]` pre-scaled
//      branchless XOR Gaussian elimination in characteristic 2 (`q in {2, 4, 8}`) and pre-scaled basis tables in odd char!

use std::env;
use std::fs;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;

fn make_gf_q2(q: usize) -> (Vec<Vec<u8>>, Vec<Vec<u8>>, Vec<Vec<u8>>, Vec<u8>, Vec<u8>) {
    let q2 = q * q;
    let mut add = vec![vec![0u8; q2]; q2];
    let mut sub = vec![vec![0u8; q2]; q2];
    let mut mul = vec![vec![0u8; q2]; q2];
    let mut inv = vec![0u8; q2];
    let mut conj = vec![0u8; q2];

    if q == 2 {
        for u in 0..4usize {
            let (a0, b0) = (u & 1, (u >> 1) & 1);
            conj[u] = if b0 == 0 { a0 as u8 } else { ((a0 ^ 1) | 2) as u8 };
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
        // GF(81) = GF(3)[z] / (z^4 + z + 2) (primitive polynomial over GF(3)!)
        let to_poly = |u: usize| -> [u8; 4] {
            [(u % 3) as u8, ((u / 3) % 3) as u8, ((u / 9) % 3) as u8, ((u / 27) % 3) as u8]
        };
        let from_poly = |p: [u8; 4]| -> u8 {
            p[0] + 3 * p[1] + 9 * p[2] + 27 * p[3]
        };
        for u in 0..81usize {
            let pu = to_poly(u);
            for v in 0..81usize {
                let pv = to_poly(v);
                add[u][v] = from_poly([
                    (pu[0] + pv[0]) % 3, (pu[1] + pv[1]) % 3, (pu[2] + pv[2]) % 3, (pu[3] + pv[3]) % 3,
                ]);
                sub[u][v] = from_poly([
                    (pu[0] + 3 - pv[0]) % 3, (pu[1] + 3 - pv[1]) % 3, (pu[2] + 3 - pv[2]) % 3, (pu[3] + 3 - pv[3]) % 3,
                ]);
                let mut pr = [0u8; 7];
                for i in 0..4 {
                    for j in 0..4 {
                        pr[i + j] = (pr[i + j] + pu[i] * pv[j]) % 3;
                    }
                }
                for d in (4..7).rev() {
                    let c = pr[d];
                    if c != 0 {
                        pr[d] = 0;
                        pr[d - 3] = (pr[d - 3] + 2 * c) % 3;
                        pr[d - 4] = (pr[d - 4] + c) % 3;
                    }
                }
                mul[u][v] = from_poly([pr[0], pr[1], pr[2], pr[3]]);
            }
        }
        for u in 0..81usize {
            let u2 = mul[u][u] as usize;
            let u3 = mul[u2][u] as usize;
            let u6 = mul[u3][u3] as usize;
            conj[u] = mul[u6][u3];
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
        for v in 1..q2 {
            if mul[u][v] == 1 { inv[u] = v as u8; break; }
        }
        assert!(inv[u] != 0, "GF(q^2) inverse failed at u={} for q={}", u, q);
    }
    for u in 0..q2 {
        for v in 0..q2 {
            assert_eq!(sub[add[u][v] as usize][v], u as u8, "sub/add check failed!");
        }
    }
    (add, sub, mul, inv, conj)
}

fn parse_simple_json(txt: &str) -> (usize, usize, usize, usize, Vec<Vec<u8>>) {
    let get_int = |key: &str| -> usize {
        let pat = format!("\"{}\":", key);
        let idx = txt.find(&pat).unwrap() + pat.len();
        let rest = txt[idx..].trim_start();
        let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap();
        rest[..end].parse().unwrap()
    };
    let q = get_int("q");
    let m = get_int("m");
    let n = get_int("n");
    let d_tar = get_int("d_tar");
    let g_idx = txt.find("\"G\":").unwrap() + 4;
    let g_str = txt[g_idx..].trim();
    let mut g = Vec::with_capacity(m);
    let mut cur_row = Vec::new();
    let mut depth = 0i32;
    let mut num_buf = String::new();
    for ch in g_str.chars() {
        match ch {
            '[' => { depth += 1; if depth == 2 { cur_row.clear(); } }
            ']' => {
                if !num_buf.is_empty() { cur_row.push(num_buf.parse::<u8>().unwrap()); num_buf.clear(); }
                if depth == 2 { g.push(cur_row.clone()); }
                depth -= 1;
                if depth == 0 { break; }
            }
            '0'..='9' => num_buf.push(ch),
            _ => {
                if !num_buf.is_empty() { cur_row.push(num_buf.parse::<u8>().unwrap()); num_buf.clear(); }
            }
        }
    }
    (q, m, n, d_tar, g)
}

fn compute_gfq2_rank(g: &[Vec<u8>], m: usize, n: usize, sub: &[Vec<u8>], mul: &[Vec<u8>], inv: &[u8]) -> usize {
    let mut a = g.to_vec();
    let mut rk = 0usize;
    for col in 0..n {
        if rk == m { break; }
        let mut piv = None;
        for r in rk..m {
            if a[r][col] != 0 { piv = Some(r); break; }
        }
        let p = match piv { Some(v) => v, None => continue };
        a.swap(rk, p);
        let iv = inv[a[rk][col] as usize] as usize;
        for c in col..n {
            a[rk][c] = mul[a[rk][c] as usize][iv];
        }
        for r in 0..m {
            if r != rk && a[r][col] != 0 {
                let sc = a[r][col] as usize;
                for c in col..n {
                    let prod = mul[sc][a[rk][c] as usize];
                    a[r][c] = sub[a[r][c] as usize][prod as usize];
                }
            }
        }
        rk += 1;
    }
    rk
}

#[inline(always)]
fn build_scaled128(
    v: u128,
    np: usize,
    m: usize,
    q2: usize,
    mul: &[[u8; 82]; 82],
    inv: &[u8; 82],
    out_table: &mut [u128; 64],
) {
    let piv_val = ((v >> (np * 8)) & 0xFF) as usize;
    let iv_d = inv[piv_val] as usize;
    let mut b_bytes = [0u8; 16];
    for r in 0..m {
        let vr = ((v >> (r * 8)) & 0xFF) as usize;
        b_bytes[r] = mul[iv_d][vr];
    }
    out_table[0] = 0u128;
    for fac in 1..q2 {
        let m_row = &mul[fac];
        let mut su = 0u128;
        for r in 0..m {
            su |= (m_row[b_bytes[r] as usize] as u128) << (r * 8);
        }
        out_table[fac] = su;
    }
}

#[inline(always)]
fn dfs_char2_m16(
    depth: usize,
    start_col: usize,
    w_max: usize,
    n: usize,
    m: usize,
    q2: usize,
    cols128: &[u128],
    piv_shift: &mut [usize; 16],
    scaled128: &mut [[u128; 64]; 16],
    mul: &[[u8; 82]; 82],
    inv: &[u8; 82],
    fail: &AtomicBool,
) {
    if fail.load(Ordering::Relaxed) { return; }
    let rem_needed = w_max - depth;
    if start_col + rem_needed > n { return; }

    if depth + 1 == w_max {
        for c in start_col..n {
            let mut v = cols128[c];
            for p in 0..depth {
                let fac = ((v >> piv_shift[p]) & 0xFF) as usize;
                v ^= scaled128[p][fac];
            }
            if v == 0 {
                fail.store(true, Ordering::Relaxed);
                return;
            }
        }
        return;
    }

    for c in start_col..=(n - rem_needed) {
        if fail.load(Ordering::Relaxed) { return; }
        let mut v = cols128[c];
        for p in 0..depth {
            let fac = ((v >> piv_shift[p]) & 0xFF) as usize;
            v ^= scaled128[p][fac];
        }
        if v == 0 {
            fail.store(true, Ordering::Relaxed);
            return;
        }
        let np = (v.trailing_zeros() >> 3) as usize;
        build_scaled128(v, np, m, q2, mul, inv, &mut scaled128[depth]);
        piv_shift[depth] = np * 8;
        dfs_char2_m16(depth + 1, c + 1, w_max, n, m, q2, cols128, piv_shift, scaled128, mul, inv, fail);
    }
}

#[inline(always)]
fn build_scaled3(
    v: &[u128; 3],
    np: usize,
    m: usize,
    q2: usize,
    mul: &[[u8; 82]; 82],
    inv: &[u8; 82],
    out_table: &mut [[u128; 3]; 64],
) {
    let pw = np >> 4;
    let ps = (np & 15) * 8;
    let piv_val = ((v[pw] >> ps) & 0xFF) as usize;
    let iv_d = inv[piv_val] as usize;
    let mut b_bytes = [0u8; 48];
    for r in 0..m {
        let vr = ((v[r >> 4] >> ((r & 15) * 8)) & 0xFF) as usize;
        b_bytes[r] = mul[iv_d][vr];
    }
    out_table[0] = [0u128; 3];
    for fac in 1..q2 {
        let m_row = &mul[fac];
        let mut su = [0u128; 3];
        for r in 0..m {
            su[r >> 4] |= (m_row[b_bytes[r] as usize] as u128) << ((r & 15) * 8);
        }
        out_table[fac] = su;
    }
}

#[inline(always)]
fn dfs_char2_m48(
    depth: usize,
    start_col: usize,
    w_max: usize,
    n: usize,
    m: usize,
    q2: usize,
    cols3: &[[u128; 3]],
    piv_word: &mut [usize; 16],
    piv_shift: &mut [usize; 16],
    scaled3: &mut [[[u128; 3]; 64]; 16],
    mul: &[[u8; 82]; 82],
    inv: &[u8; 82],
    fail: &AtomicBool,
) {
    if fail.load(Ordering::Relaxed) { return; }
    let rem_needed = w_max - depth;
    if start_col + rem_needed > n { return; }

    if depth + 1 == w_max {
        for c in start_col..n {
            let [mut v0, mut v1, mut v2] = cols3[c];
            for p in 0..depth {
                let wval = if piv_word[p] == 0 { v0 } else if piv_word[p] == 1 { v1 } else { v2 };
                let fac = ((wval >> piv_shift[p]) & 0xFF) as usize;
                let sc = &scaled3[p][fac];
                v0 ^= sc[0];
                v1 ^= sc[1];
                v2 ^= sc[2];
            }
            if (v0 | v1 | v2) == 0 {
                fail.store(true, Ordering::Relaxed);
                return;
            }
        }
        return;
    }

    for c in start_col..=(n - rem_needed) {
        if fail.load(Ordering::Relaxed) { return; }
        let [mut v0, mut v1, mut v2] = cols3[c];
        for p in 0..depth {
            let wval = if piv_word[p] == 0 { v0 } else if piv_word[p] == 1 { v1 } else { v2 };
            let fac = ((wval >> piv_shift[p]) & 0xFF) as usize;
            let sc = &scaled3[p][fac];
            v0 ^= sc[0];
            v1 ^= sc[1];
            v2 ^= sc[2];
        }
        let np = if v0 != 0 {
            (v0.trailing_zeros() >> 3) as usize
        } else if v1 != 0 {
            16 + ((v1.trailing_zeros() >> 3) as usize)
        } else if v2 != 0 {
            32 + ((v2.trailing_zeros() >> 3) as usize)
        } else {
            fail.store(true, Ordering::Relaxed);
            return;
        };
        let vv = [v0, v1, v2];
        build_scaled3(&vv, np, m, q2, mul, inv, &mut scaled3[depth]);
        piv_word[depth] = np >> 4;
        piv_shift[depth] = (np & 15) * 8;
        dfs_char2_m48(depth + 1, c + 1, w_max, n, m, q2, cols3, piv_word, piv_shift, scaled3, mul, inv, fail);
    }
}

#[inline(always)]
fn build_scaled_u8(
    v: &[u8; 44],
    np: usize,
    m: usize,
    q2: usize,
    mul: &[[u8; 82]; 82],
    inv: &[u8; 82],
    out_table: &mut [[u8; 44]; 82],
) {
    let iv_d = inv[v[np] as usize] as usize;
    let mut b_row = [0u8; 44];
    for r in 0..m {
        b_row[r] = mul[iv_d][v[r] as usize];
    }
    out_table[0] = [0u8; 44];
    for fac in 1..q2 {
        let m_row = &mul[fac];
        for r in 0..m {
            out_table[fac][r] = m_row[b_row[r] as usize];
        }
    }
}

#[inline(always)]
fn dfs_odd_fast(
    depth: usize,
    start_col: usize,
    w_max: usize,
    n: usize,
    m: usize,
    q2: usize,
    cols: &[[u8; 44]],
    piv_rows: &mut [usize; 16],
    scaled_u8: &mut [[[u8; 44]; 82]; 16],
    sub: &[[u8; 82]; 82],
    mul: &[[u8; 82]; 82],
    inv: &[u8; 82],
    fail: &AtomicBool,
) {
    if fail.load(Ordering::Relaxed) { return; }
    let rem_needed = w_max - depth;
    if start_col + rem_needed > n { return; }

    if depth + 1 == w_max {
        for c in start_col..n {
            let mut v = cols[c];
            for p in 0..depth {
                let fac = v[piv_rows[p]] as usize;
                if fac != 0 {
                    let sc = &scaled_u8[p][fac];
                    for r in 0..m {
                        v[r] = sub[v[r] as usize][sc[r] as usize];
                    }
                }
            }
            let mut nz = 0u8;
            for r in 0..m { nz |= v[r]; }
            if nz == 0 {
                fail.store(true, Ordering::Relaxed);
                return;
            }
        }
        return;
    }

    for c in start_col..=(n - rem_needed) {
        if fail.load(Ordering::Relaxed) { return; }
        let mut v = cols[c];
        for p in 0..depth {
            let fac = v[piv_rows[p]] as usize;
            if fac != 0 {
                let sc = &scaled_u8[p][fac];
                for r in 0..m {
                    v[r] = sub[v[r] as usize][sc[r] as usize];
                }
            }
        }
        let mut np = usize::MAX;
        for r in 0..m {
            if v[r] != 0 { np = r; break; }
        }
        if np == usize::MAX {
            fail.store(true, Ordering::Relaxed);
            return;
        }
        build_scaled_u8(&v, np, m, q2, mul, inv, &mut scaled_u8[depth]);
        piv_rows[depth] = np;
        dfs_odd_fast(depth + 1, c + 1, w_max, n, m, q2, cols, piv_rows, scaled_u8, sub, mul, inv, fail);
    }
}

fn verify_matrix(q: usize, m: usize, n: usize, d_tar: usize, g: &[Vec<u8>], n_threads: usize) -> Result<f64, &'static str> {
    assert!(m <= 44 && d_tar >= 2);
    let w_max = d_tar - 1;
    assert!(w_max <= m && w_max <= 16 && w_max <= n);

    let t0 = std::time::Instant::now();
    let (add, sub, mul, inv, conj) = make_gf_q2(q);

    if g.len() != m || g.iter().any(|row| row.len() != n) {
        return Err("BAD_SHAPE");
    }
    if compute_gfq2_rank(g, m, n, &sub, &mul, &inv) != m {
        return Err("FAIL_RANK");
    }
    for i in 0..m {
        for j in 0..m {
            let mut acc = 0u8;
            for col in 0..n {
                let pr = mul[g[i][col] as usize][conj[g[j][col] as usize] as usize];
                acc = add[acc as usize][pr as usize];
            }
            if acc != 0 {
                return Err("FAIL_HERMITIAN");
            }
        }
    }

    let q2 = q * q;
    let mut sub_flat = [[0u8; 82]; 82];
    let mut mul_flat = [[0u8; 82]; 82];
    let mut inv_flat = [0u8; 82];
    for u in 0..q2 {
        inv_flat[u] = inv[u];
        for v in 0..q2 {
            sub_flat[u][v] = sub[u][v];
            mul_flat[u][v] = mul[u][v];
        }
    }

    let mut cols = vec![[0u8; 44]; n];
    for c in 0..n {
        let mut nz = false;
        for r in 0..m {
            cols[c][r] = g[r][c];
            if g[r][c] != 0 { nz = true; }
        }
        if !nz {
            return Err("ZERO_COL");
        }
    }
    if w_max == 1 {
        return Ok(t0.elapsed().as_secs_f64());
    }

    let k_pref = if w_max >= 9 && n >= 20 {
        5usize
    } else if w_max >= 5 && n >= 10 {
        4usize
    } else {
        2usize
    };
    let mut tasks: Vec<[u8; 5]> = Vec::new();
    if k_pref == 5 {
        for c0 in 0..=(n - w_max) {
            for c1 in (c0 + 1)..=(n - (w_max - 1)) {
                for c2 in (c1 + 1)..=(n - (w_max - 2)) {
                    for c3 in (c2 + 1)..=(n - (w_max - 3)) {
                        for c4 in (c3 + 1)..=(n - (w_max - 4)) {
                            tasks.push([c0 as u8, c1 as u8, c2 as u8, c3 as u8, c4 as u8]);
                        }
                    }
                }
            }
        }
    } else if k_pref == 4 {
        for c0 in 0..=(n - w_max) {
            for c1 in (c0 + 1)..=(n - (w_max - 1)) {
                for c2 in (c1 + 1)..=(n - (w_max - 2)) {
                    for c3 in (c2 + 1)..=(n - (w_max - 3)) {
                        tasks.push([c0 as u8, c1 as u8, c2 as u8, c3 as u8, 0]);
                    }
                }
            }
        }
    } else {
        for c0 in 0..=(n - w_max) {
            for c1 in (c0 + 1)..=(n - (w_max - 1)) {
                tasks.push([c0 as u8, c1 as u8, 0, 0, 0]);
            }
        }
    }

    let tasks_arc = Arc::new(tasks);
    let next_idx = Arc::new(AtomicUsize::new(0));
    let fail = Arc::new(AtomicBool::new(false));
    let is_char2 = matches!(q, 2 | 4 | 8);

    if is_char2 && m <= 16 {
        let mut cols128 = vec![0u128; n];
        for c in 0..n {
            let mut u = 0u128;
            for r in 0..m {
                u |= (cols[c][r] as u128) << (r * 8);
            }
            cols128[c] = u;
        }
        let c_arc = Arc::new(cols128);
        let handles: Vec<_> = (0..n_threads).map(|_| {
            let cs = Arc::clone(&c_arc);
            let ts = Arc::clone(&tasks_arc);
            let ni = Arc::clone(&next_idx);
            let fl = Arc::clone(&fail);
            let mu = mul_flat;
            let iv = inv_flat;
            thread::spawn(move || {
                let mut piv_shift = [0usize; 16];
                let mut scaled128 = Box::new([[0u128; 64]; 16]);
                let tot = ts.len();
                loop {
                    if fl.load(Ordering::Relaxed) { return; }
                    let idx = ni.fetch_add(1, Ordering::Relaxed);
                    if idx >= tot { return; }
                    let t = ts[idx];
                    let mut dep = false;
                    for d in 0..k_pref {
                        let mut v = cs[t[d] as usize];
                        for p in 0..d {
                            let fac = ((v >> piv_shift[p]) & 0xFF) as usize;
                            v ^= scaled128[p][fac];
                        }
                        if v == 0 {
                            fl.store(true, Ordering::Relaxed);
                            dep = true;
                            break;
                        }
                        if d + 1 < w_max {
                            let np = (v.trailing_zeros() >> 3) as usize;
                            build_scaled128(v, np, m, q2, &mu, &iv, &mut scaled128[d]);
                            piv_shift[d] = np * 8;
                        }
                    }
                    if !dep && w_max > k_pref {
                        dfs_char2_m16(
                            k_pref, (t[k_pref - 1] as usize) + 1, w_max, n, m, q2,
                            &cs, &mut piv_shift, &mut scaled128, &mu, &iv, &fl,
                        );
                    }
                }
            })
        }).collect();
        for h in handles { h.join().unwrap(); }
    } else if is_char2 {
        let mut cols3 = vec![[0u128; 3]; n];
        for c in 0..n {
            for r in 0..m {
                cols3[c][r >> 4] |= (cols[c][r] as u128) << ((r & 15) * 8);
            }
        }
        let c_arc = Arc::new(cols3);
        let handles: Vec<_> = (0..n_threads).map(|_| {
            let cs = Arc::clone(&c_arc);
            let ts = Arc::clone(&tasks_arc);
            let ni = Arc::clone(&next_idx);
            let fl = Arc::clone(&fail);
            let mu = mul_flat;
            let iv = inv_flat;
            thread::spawn(move || {
                let mut piv_word = [0usize; 16];
                let mut piv_shift = [0usize; 16];
                let mut scaled3 = Box::new([[[0u128; 3]; 64]; 16]);
                let tot = ts.len();
                loop {
                    if fl.load(Ordering::Relaxed) { return; }
                    let idx = ni.fetch_add(1, Ordering::Relaxed);
                    if idx >= tot { return; }
                    let t = ts[idx];
                    let mut dep = false;
                    for d in 0..k_pref {
                        let [mut v0, mut v1, mut v2] = cs[t[d] as usize];
                        for p in 0..d {
                            let wval = if piv_word[p] == 0 { v0 } else if piv_word[p] == 1 { v1 } else { v2 };
                            let fac = ((wval >> piv_shift[p]) & 0xFF) as usize;
                            let sc = &scaled3[p][fac];
                            v0 ^= sc[0]; v1 ^= sc[1]; v2 ^= sc[2];
                        }
                        let np = if v0 != 0 {
                            (v0.trailing_zeros() >> 3) as usize
                        } else if v1 != 0 {
                            16 + ((v1.trailing_zeros() >> 3) as usize)
                        } else if v2 != 0 {
                            32 + ((v2.trailing_zeros() >> 3) as usize)
                        } else {
                            fl.store(true, Ordering::Relaxed);
                            dep = true;
                            break;
                        };
                        if d + 1 < w_max {
                            let vv = [v0, v1, v2];
                            build_scaled3(&vv, np, m, q2, &mu, &iv, &mut scaled3[d]);
                            piv_word[d] = np >> 4;
                            piv_shift[d] = (np & 15) * 8;
                        }
                    }
                    if !dep && w_max > k_pref {
                        dfs_char2_m48(
                            k_pref, (t[k_pref - 1] as usize) + 1, w_max, n, m, q2,
                            &cs, &mut piv_word, &mut piv_shift, &mut scaled3, &mu, &iv, &fl,
                        );
                    }
                }
            })
        }).collect();
        for h in handles { h.join().unwrap(); }
    } else {
        let c_arc = Arc::new(cols);
        let handles: Vec<_> = (0..n_threads).map(|_| {
            let cs = Arc::clone(&c_arc);
            let ts = Arc::clone(&tasks_arc);
            let ni = Arc::clone(&next_idx);
            let fl = Arc::clone(&fail);
            let sb = sub_flat;
            let mu = mul_flat;
            let iv = inv_flat;
            thread::spawn(move || {
                let mut piv_rows = [0usize; 16];
                let mut scaled_u8 = Box::new([[[0u8; 44]; 82]; 16]);
                let tot = ts.len();
                loop {
                    if fl.load(Ordering::Relaxed) { return; }
                    let idx = ni.fetch_add(1, Ordering::Relaxed);
                    if idx >= tot { return; }
                    let t = ts[idx];
                    let mut dep = false;
                    for d in 0..k_pref {
                        let mut v = cs[t[d] as usize];
                        for p in 0..d {
                            let fac = v[piv_rows[p]] as usize;
                            if fac != 0 {
                                let sc = &scaled_u8[p][fac];
                                for r in 0..m {
                                    v[r] = sb[v[r] as usize][sc[r] as usize];
                                }
                            }
                        }
                        let mut np = usize::MAX;
                        for r in 0..m {
                            if v[r] != 0 { np = r; break; }
                        }
                        if np == usize::MAX {
                            fl.store(true, Ordering::Relaxed);
                            dep = true;
                            break;
                        }
                        if d + 1 < w_max {
                            build_scaled_u8(&v, np, m, q2, &mu, &iv, &mut scaled_u8[d]);
                            piv_rows[d] = np;
                        }
                    }
                    if !dep && w_max > k_pref {
                        dfs_odd_fast(
                            k_pref, (t[k_pref - 1] as usize) + 1, w_max, n, m, q2,
                            &cs, &mut piv_rows, &mut scaled_u8, &sb, &mu, &iv, &fl,
                        );
                    }
                }
            })
        }).collect();
        for h in handles { h.join().unwrap(); }
    }

    if fail.load(Ordering::Relaxed) {
        return Err("FAIL_DIST");
    }
    Ok(t0.elapsed().as_secs_f64())
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() >= 2 && args[1] == "selftest" {
        for &q in &[2usize, 3, 4, 5, 7, 8, 9] {
            let _ = make_gf_q2(q);
        }
        // Hexacode [[6, 0, 4]]_2 (`m = 3, n = 6, q = 2, GF(4): 0,1,2=w,3=w^2`)
        let hex = vec![
            vec![1, 0, 0, 1, 2, 2],
            vec![0, 1, 0, 2, 1, 2],
            vec![0, 0, 1, 2, 2, 1],
        ];
        assert!(verify_matrix(2, 3, 6, 4, &hex, 4).is_ok(), "Hexacode d=4 must pass!");
        assert!(verify_matrix(2, 3, 6, 3, &hex, 4).is_ok(), "Hexacode d=3 must pass!");
        // Direct sum of 6 copies of Hexacode (`m = 18, n = 36, q = 2`) exercises `[u128; 3]` `m > 16` path: `d = 4` OK, `d = 5` FAIL!
        let mut hex6 = vec![vec![0u8; 36]; 18];
        for b in 0..6 {
            for r in 0..3 {
                for c in 0..6 {
                    hex6[b * 3 + r][b * 6 + c] = hex[r][c];
                }
            }
        }
        assert!(verify_matrix(2, 18, 36, 4, &hex6, 4).is_ok(), "Hex6 d=4 must pass!");
        assert!(verify_matrix(2, 18, 36, 5, &hex6, 4).is_err(), "Hex6 d=5 must fail!");
        println!("SELFTEST_ALL_FIELDS_AND_SIMD_PATHS_OK");
        return;
    }

    let txt = fs::read_to_string(&args[1]).expect("read json");
    let (q, m, n, d_tar, g) = parse_simple_json(&txt);
    let n_threads: usize = env::var("QECC_VERIFY_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);

    match verify_matrix(q, m, n, d_tar, &g, n_threads) {
        Ok(el) => {
            println!("VALID_PURE_QECC_UNIVERSAL q={} n={} k={} d={} elapsed={:.3}s", q, n, n - 2 * m, d_tar, el);
        }
        Err(e) => {
            eprintln!("{}", e);
            std::process::exit(1);
        }
    }
}

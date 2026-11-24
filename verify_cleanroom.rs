// 100% Standalone Zero-Shared-Code, Zero-Symmetry Multi-Threaded Clean-Room Verifier
// Verifies EVERY `CONFIRMED_QECC_CROWN_*.json` from first principles:
//   1. Exact dimensions & entry bounds.
//   2. Exact Gaussian-elimination RREF rank (`rank == r` over GF(q), or `rank == m` over GF(16)/GF(64)).
//   3. Exact symplectic (`X*Z^T - Z*X^T == 0 mod q`) or Hermitian (`G*G^\dagger == 0`) self-orthogonality.
//   4. 100% EXHAUSTIVE ZERO-SYMMETRY check over ALL `C(n, w)` column subsets (`1 <= w <= d_tar - 1`)
//      and ALL projective Pauli / Hermitian rays across 20 CPU threads!
//   5. Exact comparison against Markus Grassl's `codetables_qecc_snapshot.json` (+ `ExtendCode` propagation).

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::env;
use std::fs;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

fn parse_usize_field(txt: &str, key: &str) -> usize {
    let pat = format!("\"{}\":", key);
    let p = txt.find(&pat).unwrap() + pat.len();
    let rest = txt[p..].trim_start();
    let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
    rest[..end].parse().unwrap()
}

fn parse_str_field(txt: &str, key: &str) -> String {
    let pat = format!("\"{}\":", key);
    let p = match txt.find(&pat) {
        Some(i) => i + pat.len(),
        None => return String::new(),
    };
    let rest = txt[p..].trim_start();
    if !rest.starts_with('"') {
        return String::new();
    }
    let after = &rest[1..];
    let end = after.find('"').unwrap();
    after[..end].to_string()
}

fn parse_2d_matrix(txt: &str, key: &str) -> Vec<Vec<usize>> {
    let pat = format!("\"{}\":", key);
    let p = txt.find(&pat).unwrap() + pat.len();
    let rest = txt[p..].trim_start();
    let mut rows = Vec::new();
    let mut cur = Vec::new();
    let mut num = String::new();
    let mut depth = 0i32;
    for ch in rest.chars() {
        match ch {
            '[' => {
                depth += 1;
                if depth == 2 {
                    cur.clear();
                }
            }
            ']' => {
                if !num.is_empty() {
                    cur.push(num.parse::<usize>().unwrap());
                    num.clear();
                }
                if depth == 2 {
                    rows.push(cur.clone());
                }
                depth -= 1;
                if depth == 0 {
                    break;
                }
            }
            ',' => {
                if !num.is_empty() {
                    cur.push(num.parse::<usize>().unwrap());
                    num.clear();
                }
            }
            c if c.is_ascii_digit() => num.push(c),
            _ => {}
        }
    }
    rows
}

fn parse_snapshot(path: &str) -> HashMap<usize, HashMap<(usize, usize), (usize, usize)>> {
    let txt = fs::read_to_string(path).unwrap();
    let mut out = HashMap::new();
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
                let b1 = after_nk.find('[').unwrap();
                let b2 = after_nk.find(']').unwrap();
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

fn mod_inv_prime(a: usize, q: usize) -> usize {
    for x in 1..q {
        if (a * x) % q == 1 {
            return x;
        }
    }
    panic!("No inverse for {} mod {}", a, q);
}

fn verify_symplectic_prime(txt: &str, path: &str) -> (usize, usize, usize, usize, bool, u64) {
    let q = parse_usize_field(txt, "q");
    let n = parse_usize_field(txt, "n");
    let k = parse_usize_field(txt, "k");
    let r = parse_usize_field(txt, "r");
    let d_tar = parse_usize_field(txt, "new_lower_bound");
    assert_eq!(r, n - k, "r != n - k in {}", path);

    let x_mat = parse_2d_matrix(txt, "X");
    let z_mat = parse_2d_matrix(txt, "Z");
    assert_eq!(x_mat.len(), r);
    assert_eq!(z_mat.len(), r);
    for i in 0..r {
        assert_eq!(x_mat[i].len(), n);
        assert_eq!(z_mat[i].len(), n);
        for j in 0..n {
            assert!(x_mat[i][j] < q && z_mat[i][j] < q);
        }
    }

    // 1. Exact symplectic self-orthogonality: X * Z^T - Z * X^T == 0 mod q
    for i in 0..r {
        for j in 0..r {
            let mut acc = 0usize;
            for col in 0..n {
                acc = (acc + x_mat[i][col] * z_mat[j][col] + q * q - z_mat[i][col] * x_mat[j][col]) % q;
            }
            assert_eq!(acc, 0, "Symplectic orthogonality failed at ({},{}) in {}", i, j, path);
        }
    }

    // 2. Exact RREF of H = [X | Z] (`r x 2n`) over GF(q)
    let mut rref = vec![vec![0usize; 2 * n]; r];
    for i in 0..r {
        for col in 0..n {
            rref[i][col] = x_mat[i][col];
            rref[i][n + col] = z_mat[i][col];
        }
    }
    let mut pivots = Vec::with_capacity(r);
    let mut row = 0usize;
    for col in 0..(2 * n) {
        if row >= r {
            break;
        }
        if let Some(piv) = (row..r).find(|&i| rref[i][col] % q != 0) {
            rref.swap(row, piv);
            let ip = mod_inv_prime(rref[row][col], q);
            for j in col..(2 * n) {
                rref[row][j] = (rref[row][j] * ip) % q;
            }
            for i in 0..r {
                if i != row && rref[i][col] != 0 {
                    let fac = rref[i][col];
                    for j in col..(2 * n) {
                        rref[i][j] = (rref[i][j] + q * q - fac * rref[row][j]) % q;
                    }
                }
            }
            pivots.push(col);
            row += 1;
        }
    }
    assert_eq!(pivots.len(), r, "Rank mismatch in {}: expected {}, got {}", path, r, pivots.len());

    // 3. Exhaustive zero-symmetry ray check across 20 threads!
    // Projective first-column rays (`q + 1`) and other nonzero rays (`q^2 - 1`):
    let mut first_rays = Vec::with_capacity(q + 1);
    for b in 0..q {
        first_rays.push((1usize, b));
    }
    first_rays.push((0usize, 1usize));
    let mut other_rays = Vec::with_capacity(q * q - 1);
    for a in 0..q {
        for b in 0..q {
            if a != 0 || b != 0 {
                other_rays.push((a, b));
            }
        }
    }

    // Precompute `col_synd[col][a][b]` (`[u8; 32]`, `r <= 32`)
    assert!(r <= 32);
    let mut col_synd = vec![vec![vec![[0u8; 32]; q]; q]; n];
    for col in 0..n {
        for a in 0..q {
            for b in 0..q {
                for i in 0..r {
                    col_synd[col][a][b][i] = ((b * x_mat[i][col] + q * q - a * z_mat[i][col]) % q) as u8;
                }
            }
        }
    }

    let in_rowspan = |vec2n: &[usize]| -> bool {
        let mut v = vec2n.to_vec();
        for (idx_p, &pc) in pivots.iter().enumerate() {
            let c = v[pc] % q;
            if c != 0 {
                for j in pc..(2 * n) {
                    v[j] = (v[j] + q * q - c * rref[idx_p][j]) % q;
                }
            }
        }
        v.iter().all(|&x| x % q == 0)
    };

    let rays_checked = Arc::new(AtomicU64::new(0));
    let degen_cnt = Arc::new(AtomicU64::new(0));

    // Partition over first active column `c0 in 0..n` across 20 threads!
    let col_synd_arc = Arc::new(col_synd);
    let first_arc = Arc::new(first_rays);
    let other_arc = Arc::new(other_rays);
    let rref_arc = Arc::new(rref.clone());
    let piv_arc = Arc::new(pivots.clone());

    let n_threads = 20usize;
    let mut handles = Vec::with_capacity(n_threads);
    for tid in 0..n_threads {
        let cs_c = Arc::clone(&col_synd_arc);
        let fr_c = Arc::clone(&first_arc);
        let or_c = Arc::clone(&other_arc);
        let rr_c = Arc::clone(&rref_arc);
        let pv_c = Arc::clone(&piv_arc);
        let rc_c = Arc::clone(&rays_checked);
        let dg_c = Arc::clone(&degen_cnt);
        let p_str = path.to_string();

        handles.push(thread::spawn(move || {
            let mut local_rays = 0u64;
            let mut local_degen = 0u64;
            let check_membership = |cols: &[usize], ab: &[(usize, usize)]| {
                let mut v = vec![0usize; 2 * n];
                for (&col, &(a, b)) in cols.iter().zip(ab.iter()) {
                    v[col] = a;
                    v[n + col] = b;
                }
                for (idx_p, &pc) in pv_c.iter().enumerate() {
                    let c = v[pc] % q;
                    if c != 0 {
                        for j in pc..(2 * n) {
                            v[j] = (v[j] + q * q - c * rr_c[idx_p][j]) % q;
                        }
                    }
                }
                assert!(
                    v.iter().all(|&x| x % q == 0),
                    "FATAL LOGICAL ERROR OF WEIGHT {} AT {:?} IN {}!",
                    cols.len(),
                    cols,
                    p_str
                );
            };

            fn dfs_symp(
                depth: usize,
                w: usize,
                last_col: usize,
                n: usize,
                q: usize,
                r: usize,
                cur_s: [u8; 32],
                cols: &mut [usize; 8],
                ab: &mut [(usize, usize); 8],
                cs: &[Vec<Vec<[u8; 32]>>],
                other_rays: &[(usize, usize)],
                local_rays: &mut u64,
                local_degen: &mut u64,
                check_mem: &dyn Fn(&[usize], &[(usize, usize)]),
            ) {
                if depth == w {
                    *local_rays += 1;
                    if cur_s[..r].iter().all(|&x| x == 0) {
                        check_mem(&cols[..w], &ab[..w]);
                        *local_degen += 1;
                    }
                    return;
                }
                let rem_needed = w - depth;
                for nxt_c in (last_col + 1)..=(n - rem_needed) {
                    cols[depth] = nxt_c;
                    for &(a, b) in other_rays {
                        ab[depth] = (a, b);
                        let add_s = &cs[nxt_c][a][b];
                        let mut nxt_s = [0u8; 32];
                        let qu = q as u8;
                        for i in 0..r {
                            let v = cur_s[i] + add_s[i];
                            nxt_s[i] = if v >= qu { v - qu } else { v };
                        }
                        dfs_symp(
                            depth + 1,
                            w,
                            nxt_c,
                            n,
                            q,
                            r,
                            nxt_s,
                            cols,
                            ab,
                            cs,
                            other_rays,
                            local_rays,
                            local_degen,
                            check_mem,
                        );
                    }
                }
            }

            let mut c0 = tid;
            while c0 < n {
                let mut cols = [0usize; 8];
                let mut ab = [(0usize, 0usize); 8];
                cols[0] = c0;
                for w in 1..d_tar {
                    if c0 + w > n {
                        continue;
                    }
                    for &(a0, b0) in fr_c.iter() {
                        ab[0] = (a0, b0);
                        let s0 = cs_c[c0][a0][b0];
                        dfs_symp(
                            1,
                            w,
                            c0,
                            n,
                            q,
                            r,
                            s0,
                            &mut cols,
                            &mut ab,
                            &cs_c,
                            &or_c,
                            &mut local_rays,
                            &mut local_degen,
                            &check_membership,
                        );
                    }
                }
                c0 += n_threads;
            }
            rc_c.fetch_add(local_rays, Ordering::Relaxed);
            dg_c.fetch_add(local_degen, Ordering::Relaxed);
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    let _ = in_rowspan;
    (
        q,
        n,
        k,
        d_tar,
        degen_cnt.load(Ordering::Relaxed) == 0,
        rays_checked.load(Ordering::Relaxed),
    )
}

fn verify_hermitian_gf2e(txt: &str, path: &str, q: usize) -> (usize, usize, usize, usize, bool, u64) {
    // q == 4 (`GF(16)`, poly 0x13, conj u^4) or q == 8 (`GF(64)`, poly 0x43, conj u^8)
    let q2 = q * q;
    let deg = if q == 4 { 4usize } else { 6usize };
    let red = 3usize; // z^4 + z + 1 or z^6 + z + 1 -> lower bits `z + 1 = 3`!

    let mut mul = vec![vec![0u8; q2]; q2];
    let mut inv = vec![0u8; q2];
    let mut conj = vec![0u8; q2];
    for u in 0..q2 {
        for v in 0..q2 {
            let (mut a, mut b, mut p) = (u, v, 0usize);
            for _ in 0..deg {
                if (b & 1) != 0 {
                    p ^= a;
                }
                let hi = a & (1 << (deg - 1));
                a = (a << 1) & (q2 - 1);
                if hi != 0 {
                    a ^= red;
                }
                b >>= 1;
            }
            mul[u][v] = p as u8;
        }
    }
    for u in 1..q2 {
        let mut p = u;
        for _ in 0..(deg / 2) {
            p = mul[p][p] as usize; // u^(2^(deg/2)) = u^4 or u^8
        }
        conj[u] = p as u8;
        inv[u] = (1..q2).find(|&v| mul[u][v] == 1).unwrap() as u8;
    }

    let n = parse_usize_field(txt, "n");
    let k = parse_usize_field(txt, "k");
    let r = parse_usize_field(txt, "r");
    let d_tar = parse_usize_field(txt, "new_lower_bound");
    let m = r / 2;
    assert_eq!(r, 2 * m);
    assert_eq!(r, n - k);

    let key = if q == 4 { "G_gf16" } else { "G_gf64" };
    let g_mat = parse_2d_matrix(txt, key);
    assert_eq!(g_mat.len(), m);
    for i in 0..m {
        assert_eq!(g_mat[i].len(), n);
        for j in 0..n {
            assert!(g_mat[i][j] < q2);
        }
    }

    // 1. Exact Hermitian self-orthogonality: G * G^\dagger == 0
    for i in 0..m {
        for j in 0..m {
            let mut acc = 0u8;
            for col in 0..n {
                acc ^= mul[g_mat[i][col]][conj[g_mat[j][col]] as usize];
            }
            assert_eq!(acc, 0, "Hermitian orthogonality failed at ({},{}) in {}", i, j, path);
        }
    }

    // 2. Exact RREF of G over GF(q^2)
    let mut rref: Vec<Vec<u8>> = g_mat
        .iter()
        .map(|row| row.iter().map(|&x| x as u8).collect())
        .collect();
    let mut pivots = Vec::with_capacity(m);
    let mut row = 0usize;
    for col in 0..n {
        if row >= m {
            break;
        }
        if let Some(piv) = (row..m).find(|&i| rref[i][col] != 0) {
            rref.swap(row, piv);
            let ip = inv[rref[row][col] as usize] as usize;
            for j in col..n {
                rref[row][j] = mul[rref[row][j] as usize][ip];
            }
            for i in 0..m {
                if i != row && rref[i][col] != 0 {
                    let fac = rref[i][col] as usize;
                    for j in col..n {
                        rref[i][j] ^= mul[fac][rref[row][j] as usize];
                    }
                }
            }
            pivots.push(col);
            row += 1;
        }
    }
    assert_eq!(pivots.len(), m, "GF({}) rank mismatch in {}", q2, path);

    // 3. Exhaustive zero-symmetry check over ALL projective vectors in GF(q^2)^n of weight `1 .. d_tar - 1`!
    // Pack each column's `m` entries (`m <= 16`) as a `u128` (`8 bits * m`), so XOR in GF(2^e)^m is ONE SINGLE `u128` XOR INSTRUCTION!!
    assert!(m <= 16);
    let mut col_sc = vec![vec![0u128; q2]; n];
    for col in 0..n {
        for a in 1..q2 {
            let mut pk = 0u128;
            for i in 0..m {
                let v = mul[g_mat[i][col]][a] as u128;
                pk |= v << (i * 8);
            }
            col_sc[col][a] = pk;
        }
    }

    let rays_checked = Arc::new(AtomicU64::new(0));
    let degen_cnt = Arc::new(AtomicU64::new(0));
    let cs_arc = Arc::new(col_sc);
    let rr_arc = Arc::new(rref);
    let pv_arc = Arc::new(pivots);
    let mul_arc = Arc::new(mul);
    let conj_arc = Arc::new(conj);

    let n_threads = 20usize;
    let mut handles = Vec::with_capacity(n_threads);
    for tid in 0..n_threads {
        let cs_c = Arc::clone(&cs_arc);
        let rr_c = Arc::clone(&rr_arc);
        let pv_c = Arc::clone(&pv_arc);
        let ml_c = Arc::clone(&mul_arc);
        let cj_c = Arc::clone(&conj_arc);
        let rc_c = Arc::clone(&rays_checked);
        let dg_c = Arc::clone(&degen_cnt);
        let p_str = path.to_string();

        handles.push(thread::spawn(move || {
            let mut local_rays = 0u64;
            let mut local_degen = 0u64;
            let check_mem = |cols: &[usize], vals: &[usize]| {
                // `G * u == 0` => vector in `C^{\perp_H}` is `conj(u)`! Check `conj(u) in rowspan(G)`!
                let mut v = vec![0u8; n];
                for (&c, &a) in cols.iter().zip(vals.iter()) {
                    v[c] = cj_c[a];
                }
                for (idx_p, &pc) in pv_c.iter().enumerate() {
                    let fac = v[pc] as usize;
                    if fac != 0 {
                        for j in pc..n {
                            v[j] ^= ml_c[fac][rr_c[idx_p][j] as usize];
                        }
                    }
                }
                assert!(
                    v.iter().all(|&x| x == 0),
                    "FATAL HERMITIAN LOGICAL ERROR OF WEIGHT {} AT {:?} IN {}!",
                    cols.len(),
                    cols,
                    p_str
                );
            };

            fn dfs_herm(
                depth: usize,
                w: usize,
                last_col: usize,
                n: usize,
                q2: usize,
                cur_s: u128,
                cols: &mut [usize; 8],
                vals: &mut [usize; 8],
                cs: &[Vec<u128>],
                local_rays: &mut u64,
                local_degen: &mut u64,
                check_mem: &dyn Fn(&[usize], &[usize]),
            ) {
                if depth == w {
                    *local_rays += 1;
                    if cur_s == 0 {
                        check_mem(&cols[..w], &vals[..w]);
                        *local_degen += 1;
                    }
                    return;
                }
                let rem = w - depth;
                for nxt_c in (last_col + 1)..=(n - rem) {
                    cols[depth] = nxt_c;
                    let sc_nxt = &cs[nxt_c];
                    for a in 1..q2 {
                        vals[depth] = a;
                        dfs_herm(
                            depth + 1,
                            w,
                            nxt_c,
                            n,
                            q2,
                            cur_s ^ sc_nxt[a],
                            cols,
                            vals,
                            cs,
                            local_rays,
                            local_degen,
                            check_mem,
                        );
                    }
                }
            }

            let mut c0 = tid;
            while c0 < n {
                let mut cols = [0usize; 8];
                let mut vals = [0usize; 8];
                cols[0] = c0;
                vals[0] = 1; // projective normalization `u[c0] = 1`!
                let s0 = cs_c[c0][1];
                for w in 1..d_tar {
                    if c0 + w > n {
                        continue;
                    }
                    dfs_herm(
                        1,
                        w,
                        c0,
                        n,
                        q2,
                        s0,
                        &mut cols,
                        &mut vals,
                        &cs_c,
                        &mut local_rays,
                        &mut local_degen,
                        &check_mem,
                    );
                }
                c0 += n_threads;
            }
            rc_c.fetch_add(local_rays, Ordering::Relaxed);
            dg_c.fetch_add(local_degen, Ordering::Relaxed);
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    (
        q,
        n,
        k,
        d_tar,
        degen_cnt.load(Ordering::Relaxed) == 0,
        rays_checked.load(Ordering::Relaxed),
    )
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let out_dir = &args[1];
    let snap = parse_snapshot(&format!("{}/codetables_qecc_snapshot.json", out_dir));

    let mut files: Vec<String> = fs::read_dir(out_dir)
        .unwrap()
        .flatten()
        .filter_map(|e| {
            let nm = e.file_name().to_string_lossy().to_string();
            if nm.starts_with("CONFIRMED_QECC_CROWN_") && nm.ends_with(".json") {
                Some(e.path().to_string_lossy().to_string())
            } else {
                None
            }
        })
        .collect();
    files.sort();

    println!("=== STANDALONE ZERO-SHARED-CODE CLEAN-ROOM VERIFIER ===");
    println!("Verifying {} QECC certificate files across 20 CPU threads...\n", files.len());

    let t_all = Instant::now();
    // Track best new lower bound per `(q, n, k)`: `(d_old, d_up, d_new, source_desc)`
    let mut cell_improvements: BTreeMap<(usize, usize, usize), (usize, usize, usize, String)> = BTreeMap::new();
    let mut direct_verified: Vec<(usize, usize, usize, usize, usize, usize, bool, u64, String)> = Vec::new();
    let mut total_rays = 0u64;

    for fpath in &files {
        let t0 = Instant::now();
        let txt = fs::read_to_string(fpath).unwrap();
        let fmt = parse_str_field(&txt, "format");
        let method = parse_str_field(&txt, "method");
        let q_f = parse_usize_field(&txt, "q");

        let (q, n, k, d_tar, is_pure, rays) = if fmt == "hermitian_gf16" || fmt == "hermitian_gf64" {
            verify_hermitian_gf2e(&txt, fpath, q_f)
        } else {
            verify_symplectic_prime(&txt, fpath)
        };
        total_rays += rays;

        let &(d_old, d_up) = snap[&q].get(&(n, k)).unwrap_or(&(0, 99));
        let mut beaten_here = BTreeSet::new();
        if d_tar > d_old {
            beaten_here.insert((n, k, d_old, d_up, d_tar, format!("Direct [[{},{},{}]]_{}", n, k, d_tar, q)));
        }
        // Trivial explicit `ExtendCode` (`[[n + s, k, >= d_tar]]_q`: appends `s` unentangled `Z` stabilizers)
        for n_ext in (n + 1)..=120 {
            if let Some(&(dl, du)) = snap[&q].get(&(n_ext, k)) {
                if d_tar > dl {
                    beaten_here.insert((
                        n_ext,
                        k,
                        dl,
                        du,
                        d_tar,
                        format!("Extend([[{},{},{}]]_{} + {}Z)", n, k, d_tar, q, n_ext - n),
                    ));
                }
            }
        }

        for (bn, bk, b_old, b_up, b_new, src) in &beaten_here {
            let entry = cell_improvements
                .entry((q, *bn, *bk))
                .or_insert((*b_old, *b_up, *b_new, src.clone()));
            if *b_new > entry.2 || (*b_new == entry.2 && src.starts_with("Direct")) {
                *entry = (*b_old, *b_up, *b_new, src.clone());
            }
        }

        let fname = std::path::Path::new(fpath).file_name().unwrap().to_string_lossy().to_string();
        if d_tar > d_old || !beaten_here.is_empty() {
            println!(
                "✅ VERIFIED [[{}, {}, {}]]_{} ({}) | CodeTables: {}..{} | rays={:>11} ({:.2}s) | +{} cells | {}",
                n,
                k,
                d_tar,
                q,
                if is_pure { "PURE" } else { "DEGEN" },
                d_old,
                d_up,
                rays,
                t0.elapsed().as_secs_f64(),
                beaten_here.len(),
                fname
            );
            direct_verified.push((q, n, k, d_old, d_up, d_tar, is_pure, rays, method));
        }
    }

    println!("\n======================================================================");
    println!(
        "🏆 GRAND TOTAL CLEAN-ROOM VERIFIED: {} EXPLICIT BASE CODES -> {} DISTINCT CODETABLES.DE CELLS IMPROVED!",
        direct_verified.len(),
        cell_improvements.len()
    );
    println!("   Total exhaustive zero-symmetry rays checked: {} in {:.2}s", total_rays, t_all.elapsed().as_secs_f64());
    println!("======================================================================\n");

    let mut summary_lines = Vec::new();
    summary_lines.push("{\n  \"total_base_certificates\": ".to_string() + &direct_verified.len().to_string() + ",");
    summary_lines.push("  \"total_improved_cells\": ".to_string() + &cell_improvements.len().to_string() + ",");
    summary_lines.push("  \"total_rays_checked\": ".to_string() + &total_rays.to_string() + ",");
    summary_lines.push("  \"improved_cells\": [".to_string());
    let items: Vec<String> = cell_improvements
        .iter()
        .map(|(&(q, n, k), (d_old, d_up, d_new, src))| {
            format!(
                "    {{\"q\": {}, \"n\": {}, \"k\": {}, \"old_bounds\": \"{}..{}\", \"new_lower_bound\": {}, \"exact_optimal\": {}, \"source\": \"{}\"}}",
                q,
                n,
                k,
                d_old,
                d_up,
                d_new,
                d_new == d_up,
                src
            )
        })
        .collect();
    summary_lines.push(items.join(",\n"));
    summary_lines.push("  ]\n}".to_string());
    let sum_path = format!("{}/VERIFIED_QECC_GRAND_SLAM_SUMMARY.json", out_dir);
    fs::write(&sum_path, summary_lines.join("\n")).unwrap();
}

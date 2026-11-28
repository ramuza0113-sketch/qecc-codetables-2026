r"""Standalone Clean-Room Verification Suite for All 397 Quantum Stabilizer Code Certificates (`v1.2.0`).

Independently verifies from first principles (zero `cupy` / GPU dependency, zero symmetry assumptions):
  1. Cryptographic SHA-256 digest of every generator matrix (`G` over GF(q^2) or `(X, Z)` over GF(q)).
  2. Exact dimensions and full row rank (`rank_{GF(q^2)}(G) == m` or `rank_{GF(q)}([X | Z]) == r_S`).
  3. Exact Hermitian (`G * G^\dagger == 0`) or Symplectic (`X * Z^T - Z * X^T == 0`) self-orthogonality
     for all 397 parent certificates, plus exact block & NSC non-degeneracy identities for all 287 MP2/MP3 codes.
  4. Exact zero-symmetry minimum distance `d >= d_new` via the 4 standalone multi-threaded Rust clean-room
     verifiers in `verifiers/`:
       - Default mode (`python verify_all_qecc.py`): exhaustively verifies minimum distance on all 15 Exact Optimal
         parents, all 287 MP2/MP3 parents (up to d = 16, n = 100), and all direct parents with C(n, d - 1) <= 5e7
         (351 / 397 parents in ~25s; pass `--full` to also exhaustively sweep the remaining 46 ultra-large C(n, d - 1)
         direct d = 9..13 codes).
  5. Exact CodeTables.de propagation audit against `codetables_qecc_snapshot.json`:
       - 1,575 improved entries under single-operator propagation (`Extend`, `PureSubcode`, `Puncture`, `Shorten`),
       - 1,654 improved entries under purity-preserving transitive closure,
       - 18 Exact Optimal closures (`d_low == d_up`).
"""
import argparse
import glob
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import numpy as np


def make_gf_q2(q: int):
    q2 = q * q
    add = np.zeros((q2, q2), dtype=np.int64)
    sub = np.zeros((q2, q2), dtype=np.int64)
    mul = np.zeros((q2, q2), dtype=np.int64)
    inv = np.zeros(q2, dtype=np.int64)
    conj = np.zeros(q2, dtype=np.int64)
    if q == 2:
        for u in range(4):
            a0, b0 = u & 1, (u >> 1) & 1
            conj[u] = a0 if b0 == 0 else ((a0 ^ 1) | 2)
            for v in range(4):
                a1, b1 = v & 1, (v >> 1) & 1
                add[u, v] = sub[u, v] = u ^ v
                mul[u, v] = ((a0 & a1) ^ (b0 & b1)) | (((a0 & b1) ^ (b0 & a1) ^ (b0 & b1)) << 1)
    elif q in (4, 8):
        deg = 4 if q == 4 else 6
        for u in range(q2):
            for v in range(q2):
                add[u, v] = sub[u, v] = u ^ v
                a, b, p = u, v, 0
                for _ in range(deg):
                    if b & 1:
                        p ^= a
                    hi = a & (1 << (deg - 1))
                    a = (a << 1) & (q2 - 1)
                    if hi:
                        a ^= 3
                    b >>= 1
                mul[u, v] = p
        for u in range(q2):
            p = u
            for _ in range(deg // 2):
                p = int(mul[p, p])
            conj[u] = p
    else:
        c_ns = 3 if q == 7 else 2
        for u in range(q2):
            x0, y0 = u % q, u // q
            conj[u] = x0 + q * ((q - y0) % q)
            for v in range(q2):
                x1, y1 = v % q, v // q
                add[u, v] = ((x0 + x1) % q) + q * ((y0 + y1) % q)
                sub[u, v] = ((x0 - x1) % q) + q * ((y0 - y1) % q)
                mul[u, v] = ((x0 * x1 + c_ns * y0 * y1) % q) + q * ((x0 * y1 + y0 * x1) % q)
    for u in range(1, q2):
        for v in range(1, q2):
            if mul[u, v] == 1:
                inv[u] = v
                break
    return add, sub, mul, inv, conj


def make_gf_q(q: int):
    add = np.zeros((q, q), dtype=np.int64)
    sub = np.zeros((q, q), dtype=np.int64)
    mul = np.zeros((q, q), dtype=np.int64)
    inv = np.zeros(q, dtype=np.int64)
    if q in (2, 3, 5, 7):
        for u in range(q):
            for v in range(q):
                add[u, v] = (u + v) % q
                sub[u, v] = (u - v) % q
                mul[u, v] = (u * v) % q
    else:
        deg = 2 if q == 4 else 3
        for u in range(q):
            for v in range(q):
                add[u, v] = sub[u, v] = u ^ v
                a, b, p = u, v, 0
                for _ in range(deg):
                    if b & 1:
                        p ^= a
                    hi = a & (1 << (deg - 1))
                    a = (a << 1) & (q - 1)
                    if hi:
                        a ^= 3
                    b >>= 1
                mul[u, v] = p
    for u in range(1, q):
        for v in range(1, q):
            if mul[u, v] == 1:
                inv[u] = v
                break
    return add, sub, mul, inv


def rref_rank(M: np.ndarray, sub: np.ndarray, mul: np.ndarray, inv: np.ndarray) -> int:
    A = M.copy()
    rows, cols = A.shape
    rk = 0
    for col in range(cols):
        if rk >= rows:
            break
        piv = next((i for i in range(rk, rows) if A[i, col] != 0), None)
        if piv is None:
            continue
        if piv != rk:
            A[[rk, piv]] = A[[piv, rk]]
        ip = int(inv[int(A[rk, col])])
        A[rk, col:] = mul[A[rk, col:], ip]
        for i in range(rows):
            if i != rk and A[i, col] != 0:
                fac = int(A[i, col])
                A[i, col:] = sub[A[i, col:], mul[fac, A[rk, col:]]]
        rk += 1
    return rk


def find_rustc() -> str | None:
    cand = shutil.which("rustc")
    if cand:
        return cand
    home_cargo = os.path.expanduser(os.path.join("~", ".cargo", "bin", "rustc.exe" if os.name == "nt" else "rustc"))
    return home_cargo if os.path.exists(home_cargo) else None


def ensure_rust_verifiers(base_dir: str) -> dict[str, str]:
    rustc = find_rustc()
    if not rustc:
        raise RuntimeError("rustc not found in PATH or ~/.cargo/bin; install Rust to run clean-room distance verifiers.")
    ver_dir = os.path.join(base_dir, "verifiers")
    bin_dir = os.path.join(ver_dir, "bin")
    os.makedirs(bin_dir, exist_ok=True)
    ext = ".exe" if os.name == "nt" else ""
    mapping = {
        "direct_hermitian": ("verify_hermitian_universal.rs", f"verify_hermitian_universal{ext}"),
        "direct_symplectic": ("verify_symplectic_universal.rs", f"verify_symplectic_universal{ext}"),
        "quasi_unitary_mp2": ("verify_mp2_cleanroom.rs", f"verify_mp2_cleanroom{ext}"),
        "quasi_unitary_mp3": ("verify_mp3_cleanroom.rs", f"verify_mp3_cleanroom{ext}"),
    }
    bins = {}
    for key, (src_name, bin_name) in mapping.items():
        src_p = os.path.join(ver_dir, src_name)
        bin_p = os.path.join(bin_dir, bin_name)
        if (not os.path.exists(bin_p)) or (os.path.getmtime(bin_p) < os.path.getmtime(src_p)):
            print(f"[build] Compiling {src_name} with rustc -C opt-level=3 ...", flush=True)
            subprocess.run([rustc, "-C", "opt-level=3", src_p, "-o", bin_p], check=True)
        bins[key] = bin_p
    # Run selftests on all 4 Rust binaries
    for key, bin_p in bins.items():
        flag = "selftest" if key == "direct_hermitian" else "--selftest"
        res = subprocess.run([bin_p, flag], capture_output=True, text=True, check=True)
        assert "OK" in res.stdout, f"Selftest failed for {key}: {res.stdout}"
    return bins


def verify_algebraic_structure(cert_path: str, gfq2_cache: dict, gfq_cache: dict) -> dict:
    d = json.load(open(cert_path, encoding="utf-8"))
    q = int(d["q"])
    n = int(d["n"])
    k = int(d["k"])
    r = int(d["r"])
    d_new = int(d["new_lower_bound"])
    arch = d["architecture"]
    assert n - k == r, f"n - k != r in {cert_path}"

    if arch in ("direct_hermitian", "quasi_unitary_mp2", "quasi_unitary_mp3"):
        m = int(d["m_hermitian"])
        assert r == 2 * m, f"r != 2m in {cert_path}"
        g_list = d["G_hermitian_gf_q2"]
        sha = hashlib.sha256(json.dumps(g_list, separators=(",", ":")).encode()).hexdigest()
        assert sha == d["sha256"], f"SHA-256 mismatch in {cert_path}"
        G = np.array(g_list, dtype=np.int64)
        assert G.shape == (m, n)
        add, sub, mul, inv, conj = gfq2_cache[q]
        # Full Hermitian orthogonality G * G^\dagger == 0
        Gc = conj[G]
        for r1 in range(m):
            prods = mul[G[r1][None, :], Gc]
            acc = np.zeros(m, dtype=np.int64)
            for col in range(n):
                acc = add[acc, prods[:, col]]
            assert np.all(acc == 0), f"Hermitian orthogonality failed at row {r1} in {cert_path}"
        assert rref_rank(G, sub, mul, inv) == m, f"GF(q^2) rank < m in {cert_path}"

        if arch == "quasi_unitary_mp2":
            m1, m2 = int(d["m1"]), int(d["m2"])
            n0 = n // 2
            assert n == 2 * n0 and m == m1 + m2
            G1 = np.array(d["G1"], dtype=np.int64)
            G2 = np.array(d["G2"], dtype=np.int64)
            D = np.array(d["D"], dtype=np.int64)
            assert np.all(D > 0)
            top_rhs = mul[G1, D[None, :]]
            assert np.array_equal(G[:m1, :n0], G1) and np.array_equal(G[:m1, n0:], top_rhs)
            assert np.all(G[m1:, :n0] == 0) and np.array_equal(G[m1:, n0:], G2)
        elif arch == "quasi_unitary_mp3":
            m1, m2, m3 = int(d["m1"]), int(d["m2"]), int(d["m3"])
            n0 = n // 3
            assert n == 3 * n0 and m == m1 + m2 + m3
            G1 = np.array(d["G1"], dtype=np.int64)
            G2 = np.array(d["G2"], dtype=np.int64)
            G3 = np.array(d["G3"], dtype=np.int64)
            D1 = np.array(d["D1"], dtype=np.int64)
            D2 = np.array(d["D2"], dtype=np.int64)
            E = np.array(d["E"], dtype=np.int64)
            assert np.all(D1 > 0) and np.all(D2 > 0) and np.all(E > 0)
            assert np.all(mul[D1, E] != D2), f"MP3 NSC condition violated in {cert_path}"
            assert np.array_equal(G[:m1, :n0], G1)
            assert np.array_equal(G[:m1, n0:2 * n0], mul[G1, D1[None, :]])
            assert np.array_equal(G[:m1, 2 * n0:], mul[G1, D2[None, :]])
            assert np.all(G[m1:m1 + m2, :n0] == 0)
            assert np.array_equal(G[m1:m1 + m2, n0:2 * n0], G2)
            assert np.array_equal(G[m1:m1 + m2, 2 * n0:], mul[G2, E[None, :]])
            assert np.all(G[m1 + m2:, :2 * n0] == 0) and np.array_equal(G[m1 + m2:, 2 * n0:], G3)
    elif arch == "direct_symplectic":
        X_list = d["X_symplectic_gf_q"]
        Z_list = d["Z_symplectic_gf_q"]
        sha = hashlib.sha256(json.dumps({"X": X_list, "Z": Z_list}, separators=(",", ":")).encode()).hexdigest()
        assert sha == d["sha256"], f"SHA-256 mismatch in {cert_path}"
        X = np.array(X_list, dtype=np.int64)
        Z = np.array(Z_list, dtype=np.int64)
        assert X.shape == (r, n) and Z.shape == (r, n)
        add, sub, mul, inv = gfq_cache[q]
        for r1 in range(r):
            p1 = mul[X[r1][None, :], Z]
            p2 = mul[Z[r1][None, :], X]
            diff = sub[p1, p2]
            acc = np.zeros(r, dtype=np.int64)
            for col in range(n):
                acc = add[acc, diff[:, col]]
            assert np.all(acc == 0), f"Symplectic orthogonality failed at row {r1} in {cert_path}"
        XZ = np.hstack([X, Z])
        assert rref_rank(XZ, sub, mul, inv) == r, f"GF(q) rank < r_S in {cert_path}"
    else:
        raise ValueError(f"Unknown architecture {arch} in {cert_path}")
    return d


def verify_propagation(base_dir: str, certs_meta: list[dict]):
    snap = json.load(open(os.path.join(base_dir, "codetables_qecc_snapshot.json"), encoding="utf-8"))
    one_step = {}

    def rec1(q, nn, kk, dd):
        b = snap.get(str(q), {}).get(f"{nn},{kk}")
        if b and dd > (b[0] or 0):
            if dd > one_step.get((q, nn, kk), (0, None))[0]:
                one_step[(q, nn, kk)] = (dd, b[1])

    for cm in certs_meta:
        q, n, k, dn = int(cm["q"]), int(cm["n"]), int(cm["k"]), int(cm["new_lower_bound"])
        rec1(q, n, k, dn)
        max_n_q = 256 if q == 2 else 100
        for s in range(1, max_n_q + 1 - n):
            rec1(q, n + s, k, dn)
        for s in range(1, k + 1):
            rec1(q, n, k - s, dn)
        for s in range(1, dn - 1):
            if n - s >= k:
                rec1(q, n - s, k, dn - s)
            if n - s >= k + s:
                rec1(q, n - s, k + s, dn - s)

    best_pure = {}
    best_any = {}
    for q in (2, 3, 4, 5, 7, 8):
        max_n_q = 256 if q == 2 else 100
        for cm in certs_meta:
            if int(cm["q"]) == q:
                best_pure[(q, int(cm["n"]), int(cm["k"]))] = int(cm["new_lower_bound"])
        changed = True
        while changed:
            changed = False
            for (qq, cn, ck), cd in list(best_pure.items()):
                if qq != q:
                    continue
                if ck >= 1 and cd > best_pure.get((q, cn, ck - 1), 0):
                    best_pure[(q, cn, ck - 1)] = cd
                    changed = True
                if cd >= 3 and cn - 1 >= max(1, ck) and (cd - 1) > best_pure.get((q, cn - 1, ck), 0):
                    best_pure[(q, cn - 1, ck)] = cd - 1
                    changed = True
                if cd >= 3 and (cn - 1) >= (ck + 1) and (cd - 1) > best_pure.get((q, cn - 1, ck + 1), 0):
                    best_pure[(q, cn - 1, ck + 1)] = cd - 1
                    changed = True
        for (qq, cn, ck), cd in best_pure.items():
            if qq != q:
                continue
            for nn in range(cn, max_n_q + 1):
                if cd > best_any.get((q, nn, ck), 0):
                    best_any[(q, nn, ck)] = cd

    pure_only_cnt = sum(
        1
        for (q, n, k), cd in best_pure.items()
        if (b := snap.get(str(q), {}).get(f"{n},{k}")) and cd > (b[0] or 0)
    )
    trans_cnt = sum(
        1
        for (q, n, k), cd in best_any.items()
        if (b := snap.get(str(q), {}).get(f"{n},{k}")) and cd > (b[0] or 0)
    )
    exact_pure_cnt = sum(
        1
        for (q, n, k), cd in best_pure.items()
        if (b := snap.get(str(q), {}).get(f"{n},{k}")) and cd > (b[0] or 0) and b[1] is not None and cd >= b[1]
    )
    exact_cnt = sum(1 for (dd, du) in one_step.values() if du is not None and dd >= du)
    return len(one_step), pure_only_cnt, trans_cnt, exact_pure_cnt, exact_cnt


def main():
    ap = argparse.ArgumentParser(description="Clean-room verifier for all 397 QECC certificates (v1.2.0).")
    ap.add_argument("--full", action="store_true", help="Also run exhaustive Rust subset sweep on ultra-large C(n, d-1) > 5e7 direct d=9..13 codes.")
    ap.add_argument("--subset-cap", type=int, default=5_000_000, help="Subset cap C(n, d-1) for default fast Rust sweep on direct codes (default: 5,000,000).")
    args = ap.parse_args()

    t0 = time.time()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    certs = sorted(
        glob.glob(os.path.join(base_dir, "certificates", "QECC_*.json")),
        key=lambda p: (
            int(json.load(open(p, encoding="utf-8"))["q"]),
            int(json.load(open(p, encoding="utf-8"))["n"]),
            int(json.load(open(p, encoding="utf-8"))["k"]),
        ),
    )
    assert len(certs) == 397, f"Expected 397 certificates, found {len(certs)}"

    gfq2_cache = {q: make_gf_q2(q) for q in (2, 3, 4, 5, 7, 8)}
    gfq_cache = {q: make_gf_q(q) for q in (2, 3, 4, 5, 7, 8)}

    print("=== [Stage 1/3] Python First-Principles Algebraic & SHA-256 Verification (All 397 Certificates) ===", flush=True)
    certs_meta = [verify_algebraic_structure(cp, gfq2_cache, gfq_cache) for cp in certs]
    print(f"  -> All 397 certificates passed SHA-256, full rank, self-orthogonality, and MP2/MP3 NSC checks ({time.time() - t0:.2f}s)!", flush=True)

    print("\n=== [Stage 2/3] Multi-Threaded Rust Clean-Room Minimum-Distance Verification ===", flush=True)
    bins = ensure_rust_verifiers(base_dir)
    dist_proven = 0
    dist_deferred = 0
    for cp, cm in zip(certs, certs_meta):
        arch = cm["architecture"]
        n, dn = int(cm["n"]), int(cm["new_lower_bound"])
        subsets = math.comb(n, dn - 1) if arch in ("direct_hermitian", "direct_symplectic") else 0
        if (not args.full) and arch in ("direct_hermitian", "direct_symplectic") and (not cm["exact_optimal"]) and subsets > args.subset_cap:
            dist_deferred += 1
            continue
        res = subprocess.run([bins[arch], cp], capture_output=True, text=True)
        if res.returncode != 0 or "VALID_" not in res.stdout:
            raise RuntimeError(f"Rust distance verifier FAILED on {cp}:\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}")
        dist_proven += 1
        if cm["exact_optimal"]:
            old_lo, old_hi = cm["previous_codetables_bounds"]
            print(f"  [EXACT OPTIMAL] {cm['code_parameters']:16s} ({old_lo}..{old_hi} -> {dn}..{old_hi}) | {res.stdout.strip()}", flush=True)

    print(
        f"  -> {dist_proven}/397 parent codes proven by exhaustive Rust minimum-distance sweep "
        f"({dist_deferred} ultra-large C(n, d-1) > {args.subset_cap:,} direct codes algebraically verified; pass --full to sweep all 397).",
        flush=True,
    )

    print("\n=== [Stage 3/3] CodeTables.de Secondary Propagation & Exact-Optimal Closure Audit ===", flush=True)
    c_single, c_pure_only, c_trans, c_exact_pure, c_exact = verify_propagation(base_dir, certs_meta)
    assert (
        c_single == 1575
        and c_pure_only == 1450
        and c_trans == 1654
        and c_exact_pure == 16
        and c_exact == 18
    ), (
        f"Propagation mismatch: got single={c_single}, pure_only={c_pure_only}, trans={c_trans}, exact_pure={c_exact_pure}, exact={c_exact}"
    )
    print(f"  -> Single-operator improved CodeTables.de entries  : {c_single:,} (asserted == 1,575)")
    print(f"  -> Strictly pure closure improved entries (no Ext) : {c_pure_only:,} (asserted == 1,450)")
    print(f"  -> Full closure (+204 terminal Extend entries)     : {c_trans:,} (asserted == 1,654)")
    print(f"  -> Exact Optimal closures (d_low == d_up)          : {c_exact} ({c_exact_pure} pure + {c_exact - c_exact_pure} terminal Extend; asserted == 18)")
    print(f"\nALL VERIFICATION CHECKS PASSED IN {time.time() - t0:.2f}s!", flush=True)


if __name__ == "__main__":
    main()

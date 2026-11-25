"""RTX 3090 CUDA QECC Wave 3 Sweep:
Sweeps:
  1. Remaining targets #66..#140 from Wave 2 (`q in {2,3,4,5,7,8}`, `d_tar in {5, 6, 7}`),
  2. Bordered Double-Circulant `l = 2` (`bp in {1, 2, 3, 4}` where `bp | m`, giving high-distance low-rate `[[2m + bp, bp, d]]_q`!),
  3. Large block sizes `m = 11, 12` (`l in {2, 3, 4}`, `n_b <= 43`)!
"""
import glob
import json
import os
import random
import sys
import time
import numpy as np
import cupy as cp
from rtx3090_qecc_cuda_blitz import CUDA_QECC_KERNEL, GFq2Host, export_winner


def main():
    out_d = sys.argv[1]
    budget_secs = float(sys.argv[2]) if len(sys.argv) > 2 else 115.0

    mod = cp.RawModule(code=CUDA_QECC_KERNEL)
    k_eval = mod.get_function("eval_batch_hermitian_qt")

    snap = json.load(open(os.path.join(out_d, "codetables_qecc_snapshot.json")))
    saved = set()
    for p in glob.glob(os.path.join(out_d, "CONFIRMED_QECC_CROWN_q*.json")):
        d = json.load(open(p))
        saved.add((int(d["q"]), int(d["n"]), int(d["k"])))

    # Also skip `(q, n, k)` pairs already exhaustively searched in Waves 1 & 2 (`#0..#66` of Wave 1/2 list)
    fields = {q: GFq2Host(q) for q in (2, 3, 4, 5, 7, 8)}

    # Reconstruct Wave 2 list first to know which 66 were already tried:
    tried_w12 = set()
    w2_tmp = []
    for q in (2, 3, 4, 5, 7, 8):
        sq = snap[str(q)]
        q2 = q * q
        for m in range(4, 11):
            bps = [0, 1] + ([2] if m % 2 == 0 else []) + ([3] if m % 3 == 0 else [])
            for l in range(3, 8):
                for bp in bps:
                    n = l * m + bp
                    k = (l - 2) * m + bp
                    n_b = (l - 1) * m + bp
                    if n_b > 43:
                        continue
                    bounds = sq.get(f"{n},{k}")
                    if not bounds:
                        continue
                    dl, du = bounds
                    d_tar = dl + 1
                    if dl >= du or not (4 <= d_tar <= min(7, m + 1)):
                        continue
                    if d_tar == m + 1 and n > q2 + 1:
                        continue
                    if bp > 0 and 1 + (m // bp) < d_tar:
                        continue
                    if d_tar == 7 and q >= 7 and n > 24:
                        continue
                    ext_bonus = 1 + sum(1 for nx in range(n + 1, 95) if sq.get(f"{nx},{k}", [99])[0] < d_tar)
                    slack = (m + 1) - d_tar
                    w2_tmp.append((d_tar, 0 if slack >= 1 else 1, -ext_bonus, n, q, m, l, bp))
    w2_tmp.sort()
    for item in w2_tmp[:67]:
        _, _, _, n, q, m, l, bp = item
        tried_w12.add((q, m, l, bp))

    targets = []
    for q in (2, 3, 4, 5, 7, 8):
        sq = snap[str(q)]
        q2 = q * q
        for m in range(4, 13):
            bps = [0] + [bp for bp in (1, 2, 3, 4, 5, 6) if m % bp == 0]
            for l in range(2, 8):
                for bp in bps:
                    if l == 2 and bp == 0:
                        continue  # k=0 pure states (or keep if k>=0? Let's include k == 0 too if open!)
                    n = l * m + bp
                    k = (l - 2) * m + bp
                    n_b = (l - 1) * m + bp
                    if n_b > 43 or (q, n, k) in saved or (q, m, l, bp) in tried_w12:
                        continue
                    bounds = sq.get(f"{n},{k}")
                    if not bounds:
                        continue
                    dl, du = bounds
                    d_tar = dl + 1
                    if dl >= du or not (4 <= d_tar <= min(7, m + 1)):
                        continue
                    if d_tar == m + 1 and n > q2 + 1:
                        continue
                    if bp > 0 and 1 + (m // bp) < d_tar:
                        continue
                    if d_tar == 7 and ((q >= 7 and n > 22) or (q == 5 and n > 25) or n > 34):
                        continue
                    if d_tar == 6 and q == 8 and n > 28:
                        continue
                    ext_bonus = 1 + sum(1 for nx in range(n + 1, 95) if sq.get(f"{nx},{k}", [99])[0] < d_tar)
                    slack = (m + 1) - d_tar
                    # Prioritize small `l` (`l=2,3,4` fast & high hit probability!), non-MDS (`slack >= 1`), and high `ext_bonus`!
                    targets.append((0 if l <= 4 else 1, d_tar, 0 if slack >= 1 else 1, -ext_bonus, n, q, m, l, bp))

    targets.sort()
    print(f"[RTX 3090 Wave 3] Pre-loaded {len(saved)} crowns; queued {len(targets)} fresh targets (incl. l=2, m=11..12, q=5,7,8)! Running {budget_secs:.0f}s...", flush=True)

    t_global = time.time()
    rng = random.Random(2026112599)
    batch_size = 512
    tot_eval = 0

    for idx, (_, d_tar, is_mds, neg_ext, n, q, m, l, bp) in enumerate(targets):
        elapsed = time.time() - t_global
        if elapsed >= budget_secs:
            break
        k = (l - 2) * m + bp
        if (q, n, k) in saved:
            continue

        f = fields[q]
        lm1 = l - 1
        n_b = lm1 * m + bp
        per_t = min(1.5 if l <= 3 else 2.2, budget_secs - elapsed)
        print(f"  [W3 #{idx+1}/{len(targets)}] Hunting [[{n}, {k}, {d_tar}]]_{q} (m={m}, l={l}, bp={bp}, +{-neg_ext} cells)...", flush=True)

        configs = [(lam, 0) for lam in f.norm1_lambdas[:4]] if bp == 0 else [(1, db) for db in f.gfq_elems[1:4]]
        per_cfg = max(0.35, per_t / len(configs))

        j1_starts = [b0 * m for b0 in range(lm1)] + ([lm1 * m] if bp > 0 else [])
        d_j1 = cp.asarray(np.array(j1_starts, dtype=np.int32))
        d_out = cp.zeros(batch_size, dtype=cp.int32)

        for lam, db in configs:
            if (q, n, k) in saved:
                break
            min_wt = min(m, d_tar - 1)
            pool_polys, pool_blks, buckets = [], [], {}
            tot_sp = f.q2 ** m
            max_p = 12000
            if tot_sp <= 35000:
                for val in range(1, tot_sp):
                    p = np.zeros(m, dtype=np.uint8)
                    vt = val
                    for r in range(m):
                        p[r] = vt % f.q2
                        vt //= f.q2
                    if int(np.count_nonzero(p)) >= min_wt:
                        nrm = f.poly_norm(p, lam)
                        ip = len(pool_polys)
                        pool_polys.append(p)
                        pool_blks.append(f.build_block(p, lam).T)
                        buckets.setdefault(nrm, []).append(ip)
                        if len(pool_polys) >= max_p:
                            break
            else:
                seen_p = set()
                for _ in range(max_p * 3):
                    if len(pool_polys) >= max_p:
                        break
                    p = np.random.randint(0, f.q2, size=m, dtype=np.uint8)
                    while int(np.count_nonzero(p)) < min_wt:
                        p[rng.randrange(m)] = rng.randint(1, f.q2 - 1)
                    tp = tuple(int(x) for x in p)
                    if tp in seen_p:
                        continue
                    seen_p.add(tp)
                    nrm = f.poly_norm(p, lam)
                    ip = len(pool_polys)
                    pool_polys.append(p)
                    pool_blks.append(f.build_block(p, lam).T)
                    buckets.setdefault(nrm, []).append(ip)

            if not pool_polys:
                continue

            pool_norms = np.zeros((len(pool_polys), m), dtype=np.uint8)
            for nrm_k, idxs in buckets.items():
                arr_n = np.array(nrm_k, dtype=np.uint8)
                for ii in idxs:
                    pool_norms[ii] = arr_n
            blks_arr = np.array(pool_blks, dtype=np.uint8)

            targ = np.zeros(m, dtype=np.uint8)
            targ[0] = f.sub[0, 1]
            if bp > 0:
                for r in range(0, m, bp):
                    targ[r] = f.sub[targ[r], db]
            alpha_b = f.alpha_for_norm[db] if bp > 0 else 0

            # Special fast path when `lm1 == 1` (`l == 2`): `a_1 * a_1^\dagger == targ` directly in `buckets[targ]`!
            if lm1 == 1:
                cands = buckets.get(tuple(int(x) for x in targ), [])
                if not cands:
                    continue
                for off in range(0, len(cands), batch_size):
                    sl = cands[off : off + batch_size]
                    vb = len(sl)
                    batch_cols = np.zeros((vb, n_b, m), dtype=np.uint8)
                    for bi_b, ix in enumerate(sl):
                        batch_cols[bi_b, :m, :] = blks_arr[ix]
                        for c_off in range(bp):
                            batch_cols[bi_b, m + c_off, c_off::bp] = alpha_b
                    d_cols = cp.asarray(batch_cols.reshape(-1))
                    k_eval((vb,), (256,), (d_cols, f.d_add, f.d_sub, f.d_mul, f.d_inv, d_j1, np.int32(len(j1_starts)), np.int32(n_b), np.int32(m), np.int32(f.q2), np.int32(d_tar), d_out))
                    cp.cuda.Stream.null.synchronize()
                    tot_eval += vb
                    wins = np.where(d_out[:vb].get() == 1)[0]
                    if len(wins) > 0:
                        ix = sl[int(wins[0])]
                        g_mat = np.zeros((m, n), dtype=np.uint8)
                        for r in range(m):
                            g_mat[r, r] = 1
                        g_mat[:, m : 2 * m] = blks_arr[ix].T
                        for c_off in range(bp):
                            g_mat[c_off::bp, 2 * m + c_off] = alpha_b
                        export_winner(f, g_mat, d_tar, f"RTX3090_CUDA_DoubleCirc(m={m},bp={bp},db={db})", out_d, saved)
                        break
                continue

            t_c = time.time()
            plen = len(pool_polys)
            while time.time() - t_c < per_cfg:
                batch_cols = np.zeros((batch_size, n_b, m), dtype=np.uint8)
                batch_choices = []
                valid_b = 0
                tries_b = 0
                while valid_b < batch_size and tries_b < batch_size * 8:
                    tries_b += 1
                    rem = targ.copy()
                    ch = []
                    for _ in range(lm1 - 1):
                        ix = rng.randrange(plen)
                        ch.append(ix)
                        for r in range(m):
                            rem[r] = f.sub[rem[r], pool_norms[ix, r]]
                    cands = buckets.get(tuple(int(x) for x in rem))
                    if not cands:
                        continue
                    last_ix = cands[rng.randrange(len(cands))]
                    ch.append(last_ix)
                    for bi, ix in enumerate(ch):
                        batch_cols[valid_b, bi * m : (bi + 1) * m, :] = blks_arr[ix]
                    if bp > 0:
                        for c_off in range(bp):
                            batch_cols[valid_b, lm1 * m + c_off, c_off::bp] = alpha_b
                    batch_choices.append(ch)
                    valid_b += 1
                if valid_b == 0:
                    break
                d_cols = cp.asarray(batch_cols[:valid_b].reshape(-1))
                k_eval((valid_b,), (256,), (d_cols, f.d_add, f.d_sub, f.d_mul, f.d_inv, d_j1, np.int32(len(j1_starts)), np.int32(n_b), np.int32(m), np.int32(f.q2), np.int32(d_tar), d_out))
                cp.cuda.Stream.null.synchronize()
                tot_eval += valid_b
                wins = np.where(d_out[:valid_b].get() == 1)[0]
                if len(wins) > 0:
                    wi = int(wins[0])
                    ch = batch_choices[wi]
                    g_mat = np.zeros((m, n), dtype=np.uint8)
                    for r in range(m):
                        g_mat[r, r] = 1
                    for bi, ix in enumerate(ch):
                        g_mat[:, (bi + 1) * m : (bi + 2) * m] = blks_arr[ix].T
                    if bp > 0:
                        for c_off in range(bp):
                            g_mat[c_off::bp, l * m + c_off] = alpha_b
                    export_winner(f, g_mat, d_tar, f"RTX3090_CUDA_Wave3(m={m},l={l},bp={bp},lam={lam})", out_d, saved)
                    break

    print(f"\n=== RTX 3090 WAVE 3 FINISHED IN {time.time()-t_global:.1f}s: EVALUATED {tot_eval:,} CODES ON GPU -> TOTAL SAVED = {len(saved)} ===", flush=True)


if __name__ == "__main__":
    main()

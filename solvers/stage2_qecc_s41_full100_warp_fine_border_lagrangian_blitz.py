r"""【第2段階・Stream 41 FULL POWER（全域 `n <= 100 / 128` 解放 × 192bit ワープ完全同期 微細＋境界同変ラグランジュ手術 `q=2,3,4,5,7,8`, `r_S=10..44`, `d=5..14` 大掃討）】
  1,543マス到達後の公式 `CodeTables.de` 全域調査で判明した **572件の超有望未踏ギャップ**
  （特にフロベニウス PGRS が届かない `q=5: n=54..100` の `[[95,70,8]]_5` 波及6マス、`[[96,71,8]]_5` 波及6マス、
  `[[84,62,7]]_5`〜`[[100,78,7]]_5` 各3マス連鎖、`q=4: n=54..100`, `q=3: n=30..100`, `q=2: n=50..128`、および奇数 `n, k, r_S` 全般）を
  192bit 3ワード・ワープ完全同期カーネル（`n <= 128, r_S <= 44`）＋ 微細1スカラー境界解放手術 ＋ `[u128; 3]` SWAR `v5` 検算器で一挙制圧する！
"""
import json
import math
import os
import random
import sys
import time
import numpy as np
import cupy as cp

os.environ["QECC_VERIFY_THREADS"] = "20"

SCRATCH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRATCH)

from gpu_qecc_v8_quasicyclic_symplectic_odd_nk import get_all_beaten
from stage2_qecc_core import calc_domino_gain, log_q2
from stage2_qecc_s23_odd_nk_high_dist_symplectic_d8_d12 import GFqBase
from stage2_qecc_s27_gf7_gf5_nib64_simd_lagrangian import verify_and_export_swar
from stage2_qecc_s34_fine_border_subgroup_lagrangian_surgery import FineBorderLagrangianState
from stage2_qecc_s36_warp_coherent_8tup_d13_d14_grand_slam import (
    CUDA_WARP_COHERENT_192,
    eval_warp_coherent,
)
from step2_gpu80_qecc_100cells import count_total_codetables_cells
from stagnation_guard import is_blacklisted, record_attempt

ENGINE_S41 = "STAGE2_QECC_FULL100_WARP_FINE_BORDER_LAGRANGIAN_BLITZ"
STREAM_TAG = "Stream41-Full100-WarpFineLag"

# Upgrade shared memory buffers from `[96]` to `[128]` so `n <= 128` is 100% supported!
CUDA_WARP_COHERENT_N128 = (
    CUDA_WARP_COHERENT_192
    .replace("s_x_lo[96], s_x_mi[96], s_x_hi[96]", "s_x_lo[128], s_x_mi[128], s_x_hi[128]")
    .replace("s_z_lo[96], s_z_mi[96], s_z_hi[96]", "s_z_lo[128], s_z_mi[128], s_z_hi[128]")
)


def choose_top_mcs_full100(r_s: int, k: int, d_tar: int, max_choices: int = 2) -> list[int]:
    """Selects up to `max_choices` valid cycle lengths `mc in 3..24` satisfying fixed-subspace pigeonhole condition."""
    w_max = d_tar - 1
    valid = []
    for mc in range(min(r_s, 24), 2, -1):
        lr, br = divmod(r_s, mc)
        lk, bk = divmod(k, mc)
        if lr < 1 or lk < 1:
            continue
        if br + 2 * bk > lr + br:
            continue
        ok = True
        for a in range(w_max // mc + 1):
            for b in range(min(br + bk, w_max - a * mc) + 1):
                if a == 0 and b == 0:
                    continue
                if a * mc + b <= w_max and 2 * (a + b) > lr + br:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            valid.append(mc)
            if len(valid) >= max_choices:
                break
    return valid


def build_warp_ktuples_anyd(n: int, r_s: int, k: int, mc: int, w_max: int, rng_np: np.random.Generator) -> tuple[np.ndarray, int]:
    lr, br = divmod(r_s, mc)
    lk, bk = divmod(k, mc)
    anchors = (
        [a * mc for a in range(lr)]
        + [lr * mc + i for i in range(br)]
        + [r_s + t * mc for t in range(lk)]
        + [r_s + lk * mc + j for j in range(bk)]
    )
    valid_a = [a for a in anchors if a <= n - w_max]
    if not valid_a:
        return np.zeros(0, dtype=np.uint64), 1

    if w_max <= 4:
        k_pref = 2
    elif w_max <= 6:
        k_pref = 3 if n > 75 else 4
    elif w_max <= 8:
        k_pref = 4 if n > 75 else 5
    elif w_max >= 12 and n <= 50:
        k_pref = 7
    elif w_max >= 10 and n <= 58:
        k_pref = 6
    else:
        k_pref = 5
    k_pref = min(k_pref, w_max)

    cur = np.array([[a] for a in valid_a], dtype=np.uint8)
    for step in range(1, k_pref):
        max_c = n - (w_max - step)
        counts = np.maximum(0, (max_c + 1) - (cur[:, -1].astype(np.int32) + 1))
        p_idx = np.repeat(np.arange(len(cur), dtype=np.int32), counts)
        if len(p_idx) == 0:
            return np.zeros(0, dtype=np.uint64), k_pref
        starts = np.cumsum(counts) - counts
        c_vals = (
            cur[p_idx, -1].astype(np.int32)
            + 1
            + (np.arange(len(p_idx), dtype=np.int32) - np.repeat(starts, counts))
        ).astype(np.uint8)
        nxt = np.empty((len(p_idx), step + 1), dtype=np.uint8)
        nxt[:, :step] = cur[p_idx]
        nxt[:, step] = c_vals
        cur = nxt

    padded8 = np.zeros((len(cur), 8), dtype=np.uint8)
    padded8[:, :k_pref] = cur
    packed = np.ascontiguousarray(padded8).view(np.uint64).reshape(-1)
    last_col = cur[:, k_pref - 1]
    order = np.argsort(last_col, kind="mergesort")
    s_u64 = packed[order]
    s_last = last_col[order]
    _, splits = np.unique(s_last, return_index=True)

    warp_chunks = []
    pad_val = np.uint64(0xFFFFFFFFFFFFFFFF)
    for b_arr in np.split(s_u64, splits[1:]):
        if len(b_arr) == 0:
            continue
        rng_np.shuffle(b_arr)
        rem = len(b_arr) % 32
        if rem != 0:
            b_arr = np.concatenate([b_arr, np.full(32 - rem, pad_val, dtype=np.uint64)])
        warp_chunks.append(b_arr.reshape(-1, 32))
    if not warp_chunks:
        return np.zeros(0, dtype=np.uint64), k_pref
    all_w = np.vstack(warp_chunks)
    rng_np.shuffle(all_w, axis=0)
    return np.ascontiguousarray(all_w.reshape(-1)), k_pref


def main():
    mod = cp.RawModule(code=CUDA_WARP_COHERENT_N128)
    k_ker = mod.get_function("check_symp_warp_coherent_8tup")
    d_pen = cp.zeros(1, dtype=cp.int32)

    snap = json.load(open(os.path.join(SCRATCH, "codetables_qecc_snapshot.json"), encoding="utf-8"))
    fqs = {q: GFqBase(q) for q in (2, 3, 4, 5, 7, 8)}
    rng = random.Random(2026112841)
    rng_np = np.random.default_rng(2026112841)

    beaten = get_all_beaten(snap)
    cells0, exact0 = count_total_codetables_cells(snap)
    targets = []

    for q in (5, 4, 3, 7, 8, 2):
        sq = snap[str(q)]
        max_n = 128 if q == 2 else 100
        for key_s, (dl, du) in sq.items():
            n, k = map(int, key_s.split(","))
            r_s = n - k
            if n > max_n or k <= 0 or not (10 <= r_s <= 44):
                continue
            cur_lb = max(dl, beaten.get((q, n, k), dl))
            if cur_lb >= du:
                continue
            d_tar = cur_lb + 1
            w_max = d_tar - 1
            if not (5 <= d_tar <= 13) or 2 * w_max > r_s:
                continue
            c_tot = math.comb(n, w_max)
            # Keep clean-room Rust `v5` verification fast (`<= 22B` subsets -> `0.1s..90s`!)
            if c_tot > 22_000_000_000:
                continue
            ev = c_tot * ((q * q - 1) ** w_max) / ((q * q - 1) * (q ** r_s))
            if ev > 24.0:
                continue
            mcs = choose_top_mcs_full100(r_s, k, d_tar, max_choices=2)
            if not mcs:
                continue
            gain, _ = calc_domino_gain(snap, beaten, q, n, k, d_tar)
            if gain <= 0:
                continue
            for mc in mcs:
                if (c_tot / mc) > 2_000_000_000:
                    continue
                tag = f"S41_WarpFineLag_q{q}_n{n}_k{k}_d{d_tar}_mc{mc}"
                if is_blacklisted(tag, ENGINE_S41):
                    continue
                # Tier 0: `C <= 5B` AND (`gain >= 2` or `E[v] <= 4.0`) — instant multi-cell blitz!
                tier = 0 if (c_tot <= 5_000_000_000 and (gain >= 2 or ev <= 4.0)) else (1 if gain >= 3 or ev <= 5.0 else 2)
                targets.append((tier, ev / (gain ** 1.6), c_tot, -gain, ev, q, n, k, r_s, d_tar, mc, tag))

    targets.sort()
    log_q2(
        STREAM_TAG,
        f"🔥🔥 Stream 41 起動（全域 `n<=100/128` × 192bit ワープ完全同期 微細＋境界ラグランジュ手術 `q=2..8`）："
        f"現在セル={cells0}（完全決着={exact0}）｜厳選 **{len(targets)} 件**！",
    )

    for tier, _, c_tot, neg_gain, ev, q, n, k, r_s, d_tar, mc, tag in targets:
        beaten = get_all_beaten(snap)
        gain, _ = calc_domino_gain(snap, beaten, q, n, k, d_tar)
        if gain <= 0 or is_blacklisted(tag, ENGINE_S41):
            continue

        fq = fqs[q]
        w_max = d_tar - 1
        t_gen = time.time()
        tups64, k_pref = build_warp_ktuples_anyd(n, r_s, k, mc, w_max, rng_np)
        if len(tups64) == 0:
            continue
        d_tups64 = cp.asarray(tups64)
        n_tups = len(tups64)

        log_q2(
            STREAM_TAG,
            f"⚡ 全域ラグランジュ狙撃 (Tier{tier},E[v]={ev:.2f},r_S={r_s},mc={mc},kp={k_pref},warps={n_tups//32:,},gen={time.time()-t_gen:.2f}s) "
            f"[[{n}, {k}, {d_tar}]]_{q} (波及={gain}セル)...",
        )

        t0 = time.time()
        budget = 65.0 if (gain >= 3 or ev <= 4.0) else 40.0
        global_best = 999999
        cur_st = None
        cur_pen = 999999
        stagnant = 0
        won = False

        while time.time() - t0 < budget:
            if cur_st is None or stagnant >= 110:
                cand_st = FineBorderLagrangianState(fq, r_s, k, mc, rng_np)
                cur_st = None
                cur_pen = 999999
                stagnant = 0
            else:
                cand_st = cur_st.mutate_fine(rng, rng_np)

            X, Z = cand_st.materialize()
            cutoff = max(2, min(cur_pen + 2, 16))
            pen = eval_warp_coherent(k_ker, fq, X, Z, d_tups64, k_pref, n_tups, n, w_max, cutoff, d_pen)

            if pen == 0:
                m_str = f"Full100_WarpFineBorder_EquivLagrangian(r_S={r_s},k={k},mc={mc})"
                if verify_and_export_swar(q, X, Z, d_tar, m_str, snap):
                    won = True
                    break
                global_best = min(global_best, 1)
                cur_st = None
                continue

            if pen < global_best:
                global_best = pen
            if pen <= cur_pen or (pen == cur_pen + 1 and rng.random() < 0.025):
                stagnant = 0 if pen < cur_pen else stagnant + 1
                cur_pen = pen
                cur_st = cand_st
            else:
                stagnant += 1

        if not won:
            record_attempt(tag, ENGINE_S41, f"S41Lag [[{n},{k},{d_tar}]]_{q}", global_best, global_best, time.time() - t0, max_stalled_rounds=1)

    cells_f, exact_f = count_total_codetables_cells(snap)
    log_q2(STREAM_TAG, f"🏁 Stream 41 完了！CodeTables.de 公式表更新セル合計: **{cells_f} マス**（完全決着: {exact_f} マス）！！")


if __name__ == "__main__":
    main()

r"""【第2段階・Stream 54（v1.2.0 永久蓋 決戦ハンター⑦：準ツイスト制約完全撤廃！純 Witt 等方グラスマン多様体 $\mathrm{IGr}(m, \mathbb{F}_{q^2}^n)$ 直接 Householder $U(2/3/4)$ 探索 ＆ ライブ王冠ラダークライマー）】
  Stream 53 がわずか 16 秒で **`[[11, 1, 6]]_5`（永久蓋 #14）** と **`[[12, 0, 7]]_5`（永久蓋 #15・初の自己双対QMDS永久蓋！）** を連続撃破した一方、
  `bp = 2, 3, 4, 5` のターゲット群（`[[14, 2, 7]]_5`, `[[14, 2, 7]]_7`, `[[14, 2, 7]]_8`, `[[13, 3, 6]]_5..8`, `[[10..16, 2..8, 5]]_4`, `[[12..17, 2..7, 6]]_4`, `[[15, 3, 6]]_3`, `[[11, 3, 5]]_3`, `[[37..39, 27..29, 4]]_2` 等）が
  「Stage-A 準ツイストの境界列重み制限 `1 + m // bp < d_tar` で `QT=999999` スキップ」されていたボトルネックを **100% 撤廃**！

  【Witt 拡張定理（1937）による数学的完全性】：
  - $\mathbb{F}_{q^2}^n$ 上の階数 $m$ の全等方部分空間 $G G^\dagger = 0$ はすべて単一の $U(n, \mathbb{F}_{q^2})$ 軌道に属する！
  - したがって準ツイスト Stage-A は一切不要：
    1) **王冠ラダークライム初期化**：既知の短尺王冠 $G_0 \in \mathbb{F}_{q^2}^{m \times n_0}$ にゼロ列をパディングした $[G_0 \mid 0_{m \times (n-n_0)}]$（厳密に $G G^\dagger = 0$！）へ、ゼロ列と既存列を混ぜるランダム Householder 鏡映を 6〜10 回適用！
       （例：さきほど勝った `[[11, 1, 6]]_5` → `[[13, 3, 6]]_5` → `[[15, 5, 6]]_5`、`[[12, 0, 7]]_5` → `[[14, 2, 7]]_5` → `[[16, 4, 7]]_5`！）
    2) **カノニカル等方基底スクランブル初期化**：$[I_m \mid \alpha I_m \mid 0_{m \times k}]$（$|\alpha|^2 = -1$）へ $d \in \{3, 4\}$ ランダム Householder 鏡映を 28 回適用し、256 ブロック独立な高密度等方グラスマン初期点を瞬時に生成！
  - さらに **ライブ連鎖昇格**：`[[n, k, d]]_q` が 1 個当たるたびに、その解行列 `Gw` を即座に `crowns[(q, m)]` に登録し、直上の `n+1, n+2` 永久蓋ターゲットを最優先で連続攻略する！
"""
import ctypes
import glob
import heapq
import itertools
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
from stage2_qecc_core import GFq2Ext, calc_domino_gain, log_q2
from stage2_qecc_s22_fullpower_lucas_hensel_and_d10_d13_slam import verify_and_export_20thread
from stage2_qecc_s53_householder_u234_universal_d4_d9_lid_finisher import (
    CUDA_S53_HOUSEHOLDER_KERNEL,
    build_exact_wmax_subsets,
    build_householder_tau_tables,
    pack_G64_16,
    run_selftest,
    unpack_G64_16,
)
from step2_gpu80_qecc_100cells import count_total_codetables_cells
from stagnation_guard import is_blacklisted, record_attempt

ENGINE_S54_DIRECT_WITT_IGR = "STAGE2_QECC_DIRECT_WITT_IGR_NO_QT_LADDER_LID_V1"
STREAM_TAG = "Stream54-DirectWittIGr-Ladder"


def load_all_hermitian_crowns(fields: dict[int, GFq2Ext]) -> dict[tuple[int, int], list[tuple[int, int, np.ndarray]]]:
    crowns: dict[tuple[int, int], list[tuple[int, int, np.ndarray]]] = {}
    for fp in sorted(glob.glob(os.path.join(SCRATCH, "CONFIRMED_QECC_CROWN_q*.json"))):
        try:
            d = json.load(open(fp, encoding="utf-8"))
            q = int(d["q"])
            if q not in fields:
                continue
            f = fields[q]
            n = int(d["n"])
            dn = int(d["new_lower_bound"])
            g_raw = d.get("G_hermitian") or d.get(f"G_gf{f.q2}") or d.get("G")
            G = None
            if g_raw is not None:
                G = np.array(g_raw, dtype=np.uint8)
            elif q == 2 and "X" in d and "Z" in d and int(d.get("r", 1)) % 2 == 0:
                # Reconstruct GF(4) matrix from interleaved [x0, x1], [z0, z1] if Hermitian!
                X = np.array(d["X"], dtype=np.uint8)
                Z = np.array(d["Z"], dtype=np.uint8)
                cand_G = (X[0::2] | (Z[0::2] << 1)).astype(np.uint8)
                if f.mat_herm_zero(cand_G):
                    G = cand_G
            if G is not None and G.shape[1] == n and f.mat_herm_zero(G):
                m = G.shape[0]
                crowns.setdefault((q, m), []).append((n, dn, G))
        except Exception:
            continue
    return crowns


def cpu_apply_householder(
    f: GFq2Ext, G: np.ndarray, cols: list[int], tau_tab_np: np.ndarray, tau_cnt_np: np.ndarray, rng: random.Random
) -> bool:
    d_ar = len(cols)
    m = G.shape[0]
    for _ in range(16):
        u = [rng.randint(1, f.q2 - 1) for _ in range(d_ar)]
        cu = [int(f.conj[x]) for x in u]
        eta = 0
        for x, cx in zip(u, cu):
            eta = int(f.add[eta, f.mul[x, cx]])
        cnt = int(tau_cnt_np[eta])
        if cnt <= 0:
            continue
        tau = int(tau_tab_np[eta, rng.randrange(cnt)])
        if d_ar == 2 and int(f.mul[tau, f.mul[u[0], cu[0]]]) == 1:
            continue
        w = np.zeros(m, dtype=np.uint8)
        for c_idx, cx in zip(cols, cu):
            w = f.add[w, f.mul[cx, G[:, c_idx]]]
        new_cols = []
        ok = True
        for c_idx, ut in zip(cols, u):
            tu = int(f.mul[tau, ut])
            nc = f.sub[G[:, c_idx], f.mul[tu, w]]
            if not np.any(nc):
                ok = False
                break
            new_cols.append(nc)
        if ok:
            for c_idx, nc in zip(cols, new_cols):
                G[:, c_idx] = nc
            return True
    return False


def build_diverse_igr_population(
    f: GFq2Ext,
    m: int,
    n: int,
    d_tar: int,
    NBLK: int,
    smaller_crowns: list[tuple[int, int, np.ndarray]],
    tau_tab_np: np.ndarray,
    tau_cnt_np: np.ndarray,
    rng: random.Random,
) -> np.ndarray:
    init_G = np.zeros((NBLK, 1024), dtype=np.uint8)
    neg1 = int(f.sub[0, 1])
    alpha_neg1 = int(f.alpha_for_norm[neg1])

    # Best smaller crowns `[G0 | 0]` (`prefer largest n0 < n with d0 >= d_tar - 1`)
    usable_crowns = [
        (n0, d0, G0) for (n0, d0, G0) in smaller_crowns
        if n0 < n and (n - n0) <= 6 and d0 >= max(4, d_tar - 1)
    ]
    usable_crowns.sort(key=lambda x: (-x[0], -x[1]))

    for b in range(NBLK):
        G = np.zeros((m, n), dtype=np.uint8)
        if usable_crowns and b < (NBLK * 3) // 5:
            n0, _, G0 = usable_crowns[b % min(8, len(usable_crowns))]
            G[:, :n0] = G0
            # Mix each padded zero column `c_pad in n0..n-1` into isotropic subspace via Householder reflections!
            for c_pad in range(n0, n):
                d_ar = 4 if f.q == 2 else 3
                others = rng.sample(range(c_pad), d_ar - 1)
                cpu_apply_householder(f, G, [c_pad] + others, tau_tab_np, tau_cnt_np, rng)
            # Extra light diversification (`1..3` reflections)
            n_extra = 1 + (b % 3)
            for _ in range(n_extra):
                d_ar = 4 if (f.q == 2 and rng.random() < 0.5) else 3
                cpu_apply_householder(f, G, rng.sample(range(n), d_ar), tau_tab_np, tau_cnt_np, rng)
        else:
            # Canonical Witt isotropic frame `[I_m | alpha*I_m | 0]` + 24 random U(3)/U(4) Householder scrambles!
            for r in range(m):
                G[r, r] = 1
                G[r, m + r] = alpha_neg1
            # First activate all zero columns `2m .. n-1`
            for c_pad in range(2 * m, n):
                d_ar = 4 if f.q == 2 else 3
                others = rng.sample(range(c_pad), d_ar - 1)
                cpu_apply_householder(f, G, [c_pad] + others, tau_tab_np, tau_cnt_np, rng)
            # Deep isotropic scramble (`22` reflections)
            for _ in range(22):
                d_ar = (3 if rng.random() < 0.55 else 4) if f.q == 2 else rng.choice((2, 3, 4))
                if d_ar <= n:
                    cpu_apply_householder(f, G, rng.sample(range(n), d_ar), tau_tab_np, tau_cnt_np, rng)

        init_G[b] = pack_G64_16(G)
    return init_G


def main():
    try:
        hp = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.kernel32.SetPriorityClass(hp, 0x00000020)
        ctypes.windll.kernel32.SetProcessAffinityMask(hp, (1 << 24) - 1)
    except Exception:
        pass

    mod_hh = cp.RawModule(code=CUDA_S53_HOUSEHOLDER_KERNEL)
    k_hh = mod_hh.get_function("search_householder_u234_finisher")
    run_selftest(k_hh)

    if len(sys.argv) >= 2 and sys.argv[1] == "--selftest":
        return

    snap = json.load(open(os.path.join(SCRATCH, "codetables_qecc_snapshot.json"), encoding="utf-8"))
    fields = {q: GFq2Ext(q) for q in (2, 3, 4, 5, 7, 8)}
    hh_gpu = {}
    hh_cpu = {}
    for q in (2, 3, 4, 5, 7, 8):
        d_conj, d_tt, d_tc = build_householder_tau_tables(fields[q])
        hh_gpu[q] = (d_conj, d_tt, d_tc)
        hh_cpu[q] = (d_tt.get().reshape(82, 16), d_tc.get())

    rng = random.Random(2026112854)
    rng_np = np.random.default_rng(2026112854)

    crowns = load_all_hermitian_crowns(fields)
    beaten = get_all_beaten(snap)
    cells0, exact0 = count_total_codetables_cells(snap)

    # Build priority heap sorted by: (has_crown_seed, c_tot_tier, c_tot, -ex_gain, q, m, n, k, d_tar, tag)
    pq = []
    queued_tags = set()

    def try_enqueue(q: int, m: int, n: int, cur_b: dict, boost: int = 0):
        if n < 2 * m or n > 42 or m < 3 or m > 12:
            return
        k = n - 2 * m
        sq = snap[str(q)]
        bb = sq.get(f"{n},{k}")
        cand_dts = set()
        if bb and bb[1] is not None:
            cur_lb = max(bb[0] or 0, cur_b.get((q, n, k), 0))
            if cur_lb < bb[1]:
                cand_dts.add(cur_lb + 1)
                if bb[1] <= m + 1:
                    cand_dts.add(bb[1])
        bb_sh = sq.get(f"{n-1},{k+1}")
        if bb_sh and bb_sh[1] is not None:
            cur_sh = max(bb_sh[0] or 0, cur_b.get((q, n - 1, k + 1), 0))
            if cur_sh < bb_sh[1]:
                cand_dts.add(bb_sh[1] + 1)

        for d_tar in sorted(cand_dts):
            if not (4 <= d_tar <= min(9, m + 1)):
                continue
            if q == 2 and m == 5 and d_tar == 4 and n >= 42:
                continue
            if q == 3 and m == 3 and d_tar == 4 and n >= 11:
                continue
            w_max = d_tar - 1
            c_tot = math.comb(n, w_max)
            if c_tot > 45_000:
                continue
            sp_sum = sum(math.comb(n, j) * ((q * q - 1) ** j) for j in range((d_tar - 1) // 2 + 1))
            if sp_sum > (q ** (2 * m)):
                continue
            gain, ex_gain = calc_domino_gain(snap, cur_b, q, n, k, d_tar)
            if ex_gain <= 0:
                continue
            tag = f"S54DirectIGr_q{q}_m{m}_n{n}_k{k}_d{d_tar}"
            if tag in queued_tags or is_blacklisted(tag, ENGINE_S54_DIRECT_WITT_IGR):
                continue
            queued_tags.add(tag)
            has_seed = 0 if any(n0 < n and (n - n0) <= 4 and d0 >= d_tar for (n0, d0, _) in crowns.get((q, m), [])) else 1
            prio = (-boost, 0 if c_tot <= 6_500 else 1, has_seed, c_tot, -ex_gain, q, m, n, k, d_tar, tag)
            heapq.heappush(pq, prio)

    for q in (5, 7, 8, 4, 3, 2):
        for m in range(3, 11):
            for n in range(2 * m, 42):
                try_enqueue(q, m, n, beaten, boost=0)

    log_q2(
        STREAM_TAG,
        f"🔥🔥 Stream 54 起動（QT制約完全撤廃！純 Witt 等方グラスマン多様体 $\\mathrm{{IGr}}(m, \\mathbb{{F}}_{{q^2}}^n)$ 直接 Householder ＆ ライブラダークライマー）："
        f"現在セル={cells0}（**永久蓋={exact0}**）｜未踏 `{len(pq)}` 件（最小部分列 210〜45,000 順）！",
    )

    while pq:
        _, _, _, c_tot, _, q, m, n, k, d_tar, tag = heapq.heappop(pq)
        beaten = get_all_beaten(snap)
        gain_now, ex_now = calc_domino_gain(snap, beaten, q, n, k, d_tar)
        if ex_now <= 0 or is_blacklisted(tag, ENGINE_S54_DIRECT_WITT_IGR):
            continue

        f = fields[q]
        d_conj, d_tt, d_tc = hh_gpu[q]
        tau_tab_np, tau_cnt_np = hh_cpu[q]
        t0 = time.time()

        NBLK = 256 if c_tot <= 15_000 else 164
        sm_crowns = crowns.get((q, m), [])
        init_G = build_diverse_igr_population(f, m, n, d_tar, NBLK, sm_crowns, tau_tab_np, tau_cnt_np, rng)

        subs_arr = build_exact_wmax_subsets(n, d_tar - 1, rng_np)
        d_subs = cp.asarray(subs_arr.reshape(-1))
        n_subs = len(subs_arr)

        d_pop_G = cp.asarray(init_G.reshape(-1))
        d_best_G = d_pop_G.copy()
        d_pop_pen = cp.full(NBLK, 9999999, dtype=cp.int32)
        d_best_pen = cp.full(NBLK, 9999999, dtype=cp.int32)
        d_vcols = cp.zeros((NBLK, 16), dtype=cp.int16)
        d_tabu = cp.full((NBLK, 6), -1, dtype=cp.int16)
        d_plat = cp.zeros(NBLK, dtype=cp.int32)
        d_sol = cp.full(1, -1, dtype=cp.int32)
        d_win = cp.zeros(1024, dtype=cp.uint8)

        n_macro = 36 if c_tot <= 2_000 else (24 if c_tot <= 8_500 else (10 if c_tot <= 22_000 else 5))
        t_budget = 20.0 if c_tot <= 6_500 else 26.0
        has_cl = any(n0 < n and (n - n0) <= 4 and d0 >= d_tar for (n0, d0, _) in sm_crowns)
        mode_lbl = "CrownLadder+IGr" if has_cl else "DirectIGr"
        log_q2(
            STREAM_TAG,
            f"🎯 {mode_lbl} Householder $U(2/3/4)$ 直接探索 [[{n}, {k}, {d_tar}]]_{q} (m={m}, subsets={c_tot:,}, **永久蓋+{ex_now}個**, 総波及={gain_now})...",
        )

        launches = 0
        best_indiv = 9999999
        won = False
        while time.time() - t0 < t_budget:
            d_sol.fill(-1)
            seed = (int(time.time() * 1000000) ^ (launches * 0x9E3779B9) ^ (n * 307 + q * 59)) & 0xFFFFFFFF
            k_hh(
                (NBLK,), (256,),
                (
                    d_pop_G, d_pop_pen, d_vcols, d_tabu, d_plat, d_best_G, d_best_pen,
                    f.d_add, f.d_sub, f.d_mul, f.d_inv, d_conj, d_tt, d_tc,
                    d_subs, np.int32(n_subs),
                    np.int32(q), np.int32(f.q2), np.int32(n), np.int32(m), np.int32(d_tar - 1),
                    np.int32(n_macro), np.int32(4), np.uint32(seed), d_sol, d_win,
                ),
            )
            cp.cuda.Stream.null.synchronize()
            launches += 1

            if int(d_sol.get()[0]) >= 0:
                Gw = unpack_G64_16(d_win.get(), m, n)
                if verify_and_export_20thread(f, Gw, d_tar, f"DirectWittIGr_HouseholderU234({mode_lbl},m={m})", snap):
                    c_now, ex_tot = count_total_codetables_cells(snap)
                    log_q2(
                        STREAM_TAG,
                        f"🔒🔒🔒【永久蓋 LOCKED (DirectWittIGr 対称性ゼロ直接構成)！！】[[{n}, {k}, {d_tar}]]_{q} "
                        f"⇒ 永久蓋累計: **{ex_tot} マス**（総更新: {c_now} マス）！！",
                    )
                    crowns.setdefault((q, m), []).append((n, d_tar, Gw))
                    fresh_b = get_all_beaten(snap)
                    # Immediately boost n+1, n+2, n+3 onto top of priority heap (`LIVE LADDER CLIMB!!`)
                    for step_up in (1, 2, 3):
                        try_enqueue(q, m, n + step_up, fresh_b, boost=10)
                    won = True
                    break
                d_sol.fill(-1)

            bp_min = int(d_best_pen.get().min())
            if bp_min < best_indiv:
                best_indiv = bp_min

        if not won:
            record_attempt(
                tag, ENGINE_S54_DIRECT_WITT_IGR,
                f"S54DirectIGr({mode_lbl},Indiv={best_indiv}) [[{n},{k},{d_tar}]]_{q}",
                best_indiv, best_indiv, time.time() - t0, max_stalled_rounds=1,
            )


if __name__ == "__main__":
    main()

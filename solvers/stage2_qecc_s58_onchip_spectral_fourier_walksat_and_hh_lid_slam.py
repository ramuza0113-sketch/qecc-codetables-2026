r"""【第2段階・Stream 58（v1.2.0 永久蓋 決戦ハンター⑪：100% 全空間・棄却ゼロ！オンチップ $\mathbb{F}_{q^2}$ スペクトルフーリエ Tabu WalkSAT × 櫛なし固有モード境界 (`bp=0..m-1`) × 貯水池タイブレーク Householder 仕上げスラム）】
  これまでの準ツイスト探索（`VRAM656QT`・`v10.0`）が抱えていた 3 大限界を一挙に完全粉砕する新世代エンジン：

  1. **`656` ブロック制限の完全撤廃（全 $(q^2)^m$ ブロック空間 100.000% カバー × VRAM 0 バイト）**：
     - 旧 `Stream 42 (VRAM656QT)` は疎多項式 656 個しかプールに持てず、`q=4, m=5`（全 $16^5 = 1,048,576$ 個の 0.06%）、
       `q=5, m=6`（全 $2.44 \times 10^8$ 個の 0.00027%）、`q=7, m=4`（全 $5.76 \times 10^6$ 個の 0.011%）しか探索できなかったため、`[[29,19,5]]_4` で頭打ちになっていた！
     - `m | (q^2 - 1)` を満たす全高レート永久蓋回廊（`q=3: m=4,8` / `q=4: m=3,5,15` / `q=5: m=4,6,8,12` / `q=7: m=4,6,8,12` / `q=8: m=7,9`）において、
       定数巡回環 $\mathbb{F}_{q^2}[x]/(x^m - \lambda)$ をオンチップ共有メモリ上の逆ファンデルモンド変換 $V^{-1}$ で完全対角化！
     - 1 固有モードずつフーリエ係数 $\hat{p}_{t, s}$ を摂動し最終ブロックを $O(1)$ 閉形式（自己双対根はノルム表引き、共役対根は1回乗算）で解く
       **「オンチップ Tabu スペクトルフーリエ WalkSAT」** により、全 $(q^2)^m$ 密多項式空間を棄却率 0.00%・超高速勾配降下で探索！
  2. **旧 `v10.0` の盲目ランダム射撃 (`0/1` 判定のみ・先頭42件打ち切り) を完全刷新**：
     - 厳密な重み付き違反ペナルティ `s_pen` (`d_tar in {4, 5, 6, 7}`) ＋ 櫛なし多重固有モード境界 (`CombFreeEchar`, `bp = 0 .. min(6, m-1)`) ＋
       ホスト同期エリート増幅により、`[[30..68, 20..58, 5]]_4`、`[[27..60, n-8, 5]]_5`、`[[24..60, n-12, 6]]_5`、`[[51..85, n-8, 5]]_7`、`[[55..85, n-12, 6]]_7`、`[[15..50, n-14, 7]]_8` 等の永久蓋群を総なめにする！
  3. **貯水池タイブレーク付き Stage-B Householder $U(2/3/4)$ 対称性破壊フィニッシャー**：
     - スペクトル Stage-A が `Orb <= 4`（かつ `C(n, d-1) <= 45,000`）の超ニアミスに到達した瞬間、そのエリート生成行列群を
       **均等貯水池タイブレーク（`Indiv = 1` 等高面上のエルゴード水平歩行！）付き 20-Probe Householder アサシン** へ直結し、定数巡回対称性を局所破壊して `0` へ仕留める！
"""
import ctypes
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
from stage2_qecc_s2_fast_mds_and_4arch_tabu import get_constacyclic_fibers
from stage2_qecc_s8_frontier_crown_sniper_and_fullpool import make_border_from_bspec
from stage2_qecc_s22_fullpower_lucas_hensel_and_d10_d13_slam import verify_and_export_20thread
from stage2_qecc_s42_exact_optimal_vram656_qt_permanent_lid_hunter import is_comb_free_xis
from stage2_qecc_s53_householder_u234_universal_d4_d9_lid_finisher import (
    build_exact_wmax_subsets,
    build_householder_tau_tables,
    pack_G64_16,
    unpack_G64_16,
)
from stage2_qecc_s54_direct_witt_igr_no_qt_qmds_mass_lid_harvester import cpu_apply_householder
from step2_gpu80_qecc_100cells import count_total_codetables_cells
from stagnation_guard import is_blacklisted, record_attempt

ENGINE_S58_SPECTRAL_WS = "STAGE2_QECC_ONCHIP_SPECTRAL_FOURIER_WALKSAT_PLUS_HH_LID_V1"
STREAM_TAG = "Stream58-SpectralFourierWS-Lid"


CUDA_S58_KERNELS = r'''
extern "C" {
__device__ __forceinline__ unsigned int xorshift32(unsigned int &state) {
    unsigned int x = state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    state = x; return x;
}

// ============================================================================
// KERNEL 1: On-Chip Spectral Fourier Persistent Tabu WalkSAT (`100% Zero-Reject`)
// ============================================================================
__global__ void search_onchip_spectral_fourier_walksat(
    const unsigned char* __restrict__ g_V_inv,     // [m * m]
    const unsigned char* __restrict__ g_pi,        // [m]: root involution s -> pi(s)
    const unsigned char* __restrict__ g_norm_buck, // [82 * 12]: elements u in GF(q^2) with u*conj(u) == nv
    const unsigned char* __restrict__ g_norm_cnt,  // [82]
    const unsigned char* __restrict__ g_targ_hat,  // [m]: DFT(targ_poly)
    const unsigned char* __restrict__ g_bcols,     // [bp * m]
    const unsigned char* __restrict__ g_add,       // [82 * 82]
    const unsigned char* __restrict__ g_sub,       // [82 * 82]
    const unsigned char* __restrict__ g_mul,       // [82 * 82]
    const unsigned char* __restrict__ g_inv,       // [82]
    const unsigned char* __restrict__ g_conj,      // [82]
    const int* __restrict__ g_j1_starts,           // [n_starts]
    const int n_starts,
    const int lm1,                                 // 1 .. 20
    const int bp,                                  // 0 .. 15
    const int m,                                   // 3 .. 16
    const int lam,
    const int q2,
    const int d_tar,                               // 4, 5, 6, or 7
    const int n_steps,
    const unsigned int base_seed,
    const unsigned char* __restrict__ g_elites,    // [n_elites * 20 * 16]
    const int n_elites,
    int* __restrict__ g_best_pen_blk,              // [gridDim.x]
    unsigned char* __restrict__ g_best_hats_blk,   // [gridDim.x * 320]
    int* __restrict__ g_solved_bid,
    unsigned char* __restrict__ g_win_hats         // [320]
) {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int bdim = blockDim.x;
    const int n_b = lm1 * m + bp;
    const int w_max = d_tar - 1;

    __shared__ unsigned char s_add[82 * 82];
    __shared__ unsigned char s_sub[82 * 82];
    __shared__ unsigned char s_mul[82 * 82];
    __shared__ unsigned char s_inv[82];
    __shared__ unsigned char s_conj[82];
    __shared__ unsigned char s_Vinv[16 * 16];
    __shared__ unsigned char s_pi[16];
    __shared__ unsigned char s_targ_hat[16];
    __shared__ unsigned char s_nbuck[82 * 12];
    __shared__ unsigned char s_ncnt[82];
    __shared__ unsigned char s_hats[20 * 16];
    __shared__ unsigned char s_cur_hats[20 * 16];
    __shared__ unsigned char s_best_hats[20 * 16];
    __shared__ unsigned char s_polys[20 * 16];
    __shared__ unsigned char s_cols[112 * 16];
    __shared__ short s_tabu[4];
    __shared__ int s_cur_pen, s_best_pen, s_pen, s_plat, s_last_mode;
    __shared__ bool s_done;

    for (int i = tid; i < 82 * 82; i += bdim) {
        s_add[i] = g_add[i];
        s_sub[i] = g_sub[i];
        s_mul[i] = g_mul[i];
    }
    for (int i = tid; i < 82 * 12; i += bdim) s_nbuck[i] = g_norm_buck[i];
    if (tid < 82) {
        s_inv[tid] = g_inv[tid];
        s_conj[tid] = g_conj[tid];
        s_ncnt[tid] = g_norm_cnt[tid];
    }
    if (tid < m * m) s_Vinv[tid] = g_V_inv[tid];
    if (tid < m) {
        s_pi[tid] = g_pi[tid];
        s_targ_hat[tid] = g_targ_hat[tid];
    }
    for (int i = tid; i < bp * m; i += bdim) {
        s_cols[(lm1 * m) * m + i] = g_bcols[i];
    }
    if (tid == 0) {
        s_done = (*g_solved_bid >= 0);
        s_cur_pen = 999999;
        s_best_pen = 999999;
        s_plat = 0;
        for (int k = 0; k < 4; ++k) s_tabu[k] = -1;
    }
    __syncthreads();
    if (s_done) return;

    unsigned int rng = base_seed ^ ((unsigned int)(bid * 2654435761U + tid * 2246822519U + 97U));
    xorshift32(rng);

    for (int step = 0; step < n_steps; ++step) {
        if (tid == 0) {
            s_done = (*g_solved_bid >= 0);
            if (!s_done) {
                bool fresh = (s_cur_pen >= 900000) || (s_plat >= 34);
                if (fresh) {
                    s_plat = 0;
                    if (n_elites > 0 && (xorshift32(rng) & 3U) != 0U) {
                        int ei = (int)(xorshift32(rng) % (unsigned int)n_elites);
                        const unsigned char* src = &g_elites[ei * 320];
                        for (int i = 0; i < 320; ++i) s_hats[i] = src[i];
                        // Apply 2 mode perturbations to diverge from elite!
                        for (int mut_k = 0; mut_k < 2; ++mut_k) {
                            int sm = (int)(xorshift32(rng) % (unsigned int)m);
                            int ps = (int)s_pi[sm];
                            if (ps == sm) {
                                int rem = (int)s_targ_hat[sm];
                                if (lm1 >= 2) {
                                    int t0 = (int)(xorshift32(rng) % (unsigned int)(lm1 - 1));
                                    s_hats[t0 * 16 + sm] = (unsigned char)(xorshift32(rng) % (unsigned int)q2);
                                    for (int t = 0; t < lm1 - 1; ++t) {
                                        int v = (int)s_hats[t * 16 + sm];
                                        int nrm = (int)s_mul[v * 82 + (int)s_conj[v]];
                                        rem = (int)s_sub[rem * 82 + nrm];
                                    }
                                }
                                int cnt = (int)s_ncnt[rem];
                                s_hats[(lm1 - 1) * 16 + sm] = s_nbuck[rem * 12 + (int)(xorshift32(rng) % (unsigned int)cnt)];
                            } else {
                                int s_lo = (sm < ps) ? sm : ps;
                                int s_hi = (sm < ps) ? ps : sm;
                                int rem = (int)s_targ_hat[s_lo];
                                if (lm1 >= 2) {
                                    int t0 = (int)(xorshift32(rng) % (unsigned int)(lm1 - 1));
                                    s_hats[t0 * 16 + s_lo] = (unsigned char)(xorshift32(rng) % (unsigned int)q2);
                                    s_hats[t0 * 16 + s_hi] = (unsigned char)(xorshift32(rng) % (unsigned int)q2);
                                    for (int t = 0; t < lm1 - 1; ++t) {
                                        int vs = (int)s_hats[t * 16 + s_lo];
                                        int vsp = (int)s_hats[t * 16 + s_hi];
                                        int prd = (int)s_mul[vs * 82 + (int)s_conj[vsp]];
                                        rem = (int)s_sub[rem * 82 + prd];
                                    }
                                }
                                int vs_last = 1 + (int)(xorshift32(rng) % (unsigned int)(q2 - 1));
                                s_hats[(lm1 - 1) * 16 + s_lo] = (unsigned char)vs_last;
                                int cv = (int)s_mul[(int)s_inv[vs_last] * 82 + rem];
                                s_hats[(lm1 - 1) * 16 + s_hi] = s_conj[cv];
                            }
                        }
                        s_cur_pen = 999999;
                        s_last_mode = 0;
                    } else {
                        // 100% Fresh Zero-Rejection Spectral Sample!
                        for (int s = 0; s < m; ++s) {
                            int ps = (int)s_pi[s];
                            if (ps == s) {
                                int rem = (int)s_targ_hat[s];
                                for (int t = 0; t < lm1 - 1; ++t) {
                                    int v = (int)(xorshift32(rng) % (unsigned int)q2);
                                    s_hats[t * 16 + s] = (unsigned char)v;
                                    int nrm = (int)s_mul[v * 82 + (int)s_conj[v]];
                                    rem = (int)s_sub[rem * 82 + nrm];
                                }
                                int cnt = (int)s_ncnt[rem];
                                s_hats[(lm1 - 1) * 16 + s] = s_nbuck[rem * 12 + (int)(xorshift32(rng) % (unsigned int)cnt)];
                            } else if (s < ps) {
                                int rem = (int)s_targ_hat[s];
                                for (int t = 0; t < lm1 - 1; ++t) {
                                    int vs = (int)(xorshift32(rng) % (unsigned int)q2);
                                    int vsp = (int)(xorshift32(rng) % (unsigned int)q2);
                                    s_hats[t * 16 + s] = (unsigned char)vs;
                                    s_hats[t * 16 + ps] = (unsigned char)vsp;
                                    int prd = (int)s_mul[vs * 82 + (int)s_conj[vsp]];
                                    rem = (int)s_sub[rem * 82 + prd];
                                }
                                int vs_last = 1 + (int)(xorshift32(rng) % (unsigned int)(q2 - 1));
                                s_hats[(lm1 - 1) * 16 + s] = (unsigned char)vs_last;
                                int cv = (int)s_mul[(int)s_inv[vs_last] * 82 + rem];
                                s_hats[(lm1 - 1) * 16 + ps] = s_conj[cv];
                            }
                        }
                        s_cur_pen = 999999;
                        s_last_mode = 0;
                    }
                } else {
                    // Persistent Single-Eigenmode Surgical WalkSAT Step!
                    for (int i = 0; i < 320; ++i) s_hats[i] = s_cur_hats[i];
                    int sm = -1;
                    for (int tr_m = 0; tr_m < 6; ++tr_m) {
                        int cand = (int)(xorshift32(rng) % (unsigned int)m);
                        bool tb = false;
                        for (int k = 0; k < 2; ++k) if (s_tabu[k] == (short)cand) { tb = true; break; }
                        if (!tb || tr_m == 5) { sm = cand; break; }
                    }
                    s_last_mode = sm;
                    int ps = (int)s_pi[sm];
                    if (ps == sm) {
                        // With 35% chance, pure phase rotation on 1 block (`norm unchanged -> 1-block change!`)
                        if ((xorshift32(rng) % 100U) < 35U) {
                            int t0 = (int)(xorshift32(rng) % (unsigned int)lm1);
                            int old_v = (int)s_hats[t0 * 16 + sm];
                            int nrm = (int)s_mul[old_v * 82 + (int)s_conj[old_v]];
                            int cnt = (int)s_ncnt[nrm];
                            if (cnt >= 2) {
                                s_hats[t0 * 16 + sm] = s_nbuck[nrm * 12 + (int)(xorshift32(rng) % (unsigned int)cnt)];
                            } else if (lm1 >= 2) {
                                int t1 = (t0 == lm1 - 1) ? 0 : t0;
                                s_hats[t1 * 16 + sm] = (unsigned char)(1 + (xorshift32(rng) % (unsigned int)(q2 - 1)));
                                int rem = (int)s_targ_hat[sm];
                                for (int t = 0; t < lm1 - 1; ++t) {
                                    int v = (int)s_hats[t * 16 + sm];
                                    rem = (int)s_sub[rem * 82 + (int)s_mul[v * 82 + (int)s_conj[v]]];
                                }
                                int c2 = (int)s_ncnt[rem];
                                s_hats[(lm1 - 1) * 16 + sm] = s_nbuck[rem * 12 + (int)(xorshift32(rng) % (unsigned int)c2)];
                            }
                        } else {
                            int rem = (int)s_targ_hat[sm];
                            if (lm1 >= 2) {
                                // Swap any chosen block `t_close` as the closed block!
                                int t_mut = (int)(xorshift32(rng) % (unsigned int)lm1);
                                int t_close = (t_mut + 1 + (int)(xorshift32(rng) % (unsigned int)(lm1 - 1))) % lm1;
                                s_hats[t_mut * 16 + sm] = (unsigned char)(xorshift32(rng) % (unsigned int)q2);
                                for (int t = 0; t < lm1; ++t) {
                                    if (t == t_close) continue;
                                    int v = (int)s_hats[t * 16 + sm];
                                    rem = (int)s_sub[rem * 82 + (int)s_mul[v * 82 + (int)s_conj[v]]];
                                }
                                int cnt = (int)s_ncnt[rem];
                                s_hats[t_close * 16 + sm] = s_nbuck[rem * 12 + (int)(xorshift32(rng) % (unsigned int)cnt)];
                            } else {
                                int cnt = (int)s_ncnt[rem];
                                s_hats[sm] = s_nbuck[rem * 12 + (int)(xorshift32(rng) % (unsigned int)cnt)];
                            }
                        }
                    } else {
                        int s_lo = (sm < ps) ? sm : ps;
                        int s_hi = (sm < ps) ? ps : sm;
                        int rem = (int)s_targ_hat[s_lo];
                        if (lm1 >= 2) {
                            int t_mut = (int)(xorshift32(rng) % (unsigned int)lm1);
                            int t_close = (t_mut + 1 + (int)(xorshift32(rng) % (unsigned int)(lm1 - 1))) % lm1;
                            if (sm == s_lo) s_hats[t_mut * 16 + s_lo] = (unsigned char)(xorshift32(rng) % (unsigned int)q2);
                            else s_hats[t_mut * 16 + s_hi] = (unsigned char)(xorshift32(rng) % (unsigned int)q2);
                            for (int t = 0; t < lm1; ++t) {
                                if (t == t_close) continue;
                                int vs = (int)s_hats[t * 16 + s_lo];
                                int vsp = (int)s_hats[t * 16 + s_hi];
                                rem = (int)s_sub[rem * 82 + (int)s_mul[vs * 82 + (int)s_conj[vsp]]];
                            }
                            int vs_cl = (int)s_hats[t_close * 16 + s_lo];
                            if (vs_cl == 0 || (xorshift32(rng) & 3U) == 0U) {
                                vs_cl = 1 + (int)(xorshift32(rng) % (unsigned int)(q2 - 1));
                                s_hats[t_close * 16 + s_lo] = (unsigned char)vs_cl;
                            }
                            int cv = (int)s_mul[(int)s_inv[vs_cl] * 82 + rem];
                            s_hats[t_close * 16 + s_hi] = s_conj[cv];
                        } else {
                            int vs_cl = 1 + (int)(xorshift32(rng) % (unsigned int)(q2 - 1));
                            s_hats[s_lo] = (unsigned char)vs_cl;
                            int cv = (int)s_mul[(int)s_inv[vs_cl] * 82 + rem];
                            s_hats[s_hi] = s_conj[cv];
                        }
                    }
                }
                s_pen = 0;
            }
        }
        __syncthreads();
        if (s_done) break;

        // Step A: Parallel On-Chip IDFT `s_polys[t, r] = sum_s V_inv[r, s] * s_hats[t, s]`!
        int tot_tr = lm1 * m;
        for (int idx = tid; idx < tot_tr; idx += bdim) {
            int t = idx / m;
            int r = idx % m;
            unsigned char acc = 0;
            for (int s = 0; s < m; ++s) {
                unsigned char c_rs = s_Vinv[r * m + s];
                if (c_rs != 0) {
                    unsigned char prd = s_mul[(int)c_rs * 82 + (int)s_hats[t * 16 + s]];
                    acc = s_add[(int)acc * 82 + (int)prd];
                }
            }
            s_polys[t * 16 + r] = acc;
        }
        __syncthreads();

        // Step B: Materialize `lm1` lam-constacyclic blocks into `s_cols[col * m + r]`!
        int bulk_elems = lm1 * m * m;
        int lam_row = lam * 82;
        for (int idx = tid; idx < bulk_elems; idx += bdim) {
            int col = idx / m;
            int r = idx % m;
            int t = col / m;
            int c = col % m;
            unsigned char val;
            if (r >= c) val = s_polys[t * 16 + (r - c)];
            else val = s_mul[lam_row + (int)s_polys[t * 16 + (m + r - c)]];
            s_cols[idx] = val;
        }
        __syncthreads();

        int pen_cap = (s_cur_pen < 900000) ? (s_cur_pen + 5) : ((w_max <= 4) ? 45 : 16);
        if (s_best_pen < 25 && s_best_pen + 5 < pen_cap) pen_cap = s_best_pen + 5;
        if (pen_cap < 4) pen_cap = 4;

        // s = 1 check
        for (int si = tid; si < n_starts; si += bdim) {
            int j1 = g_j1_starts[si];
            int w = 0;
            for (int r = 0; r < m; ++r) {
                if (s_cols[j1 * m + r] != 0) ++w;
            }
            if (1 + w <= w_max) atomicAdd(&s_pen, 1000);
        }
        __syncthreads();

        // s = 2 check
        if (s_pen < pen_cap) {
            int pair_tasks = n_starts * n_b;
            for (int pt = tid; pt < pair_tasks && s_pen < pen_cap; pt += bdim) {
                int si = pt / n_b;
                int j2 = pt % n_b;
                int j1 = g_j1_starts[si];
                if (j2 <= j1) continue;
                for (int a2 = 1; a2 < q2; ++a2) {
                    int a2_row = a2 * 82;
                    int w = 0;
                    for (int r = 0; r < m; ++r) {
                        unsigned char v2 = s_mul[a2_row + s_cols[j2 * m + r]];
                        if (s_add[(int)s_cols[j1 * m + r] * 82 + v2] != 0) ++w;
                    }
                    if (2 + w <= w_max) { atomicAdd(&s_pen, 100); if (s_pen >= pen_cap) break; }
                }
            }
            __syncthreads();
        }

        // s = 3 .. w_max exact forced-scalar check (`d_tar in {4, 5, 6, 7}`)
        if (s_pen < pen_cap && w_max <= 3) {
            int s2_outer = n_starts * n_b * (q2 - 1);
            for (int idx = tid; idx < s2_outer && s_pen < pen_cap; idx += bdim) {
                int a2 = 1 + (idx % (q2 - 1));
                int tmp = idx / (q2 - 1);
                int j2 = tmp % n_b;
                int j1 = g_j1_starts[tmp / n_b];
                if (j2 <= j1) continue;
                unsigned char u12[16];
                int r0 = -1;
                int a2_row = a2 * 82;
                for (int r = 0; r < m; ++r) {
                    unsigned char v = s_add[(int)s_cols[j1 * m + r] * 82 + s_mul[a2_row + s_cols[j2 * m + r]]];
                    u12[r] = v;
                    if (v != 0 && r0 < 0) r0 = r;
                }
                if (r0 < 0) { atomicAdd(&s_pen, 100); continue; }
                unsigned char neg0 = s_sub[u12[r0]];
                for (int j3 = j2 + 1; j3 < n_b; ++j3) {
                    unsigned char c3 = s_cols[j3 * m + r0];
                    if (c3 == 0) continue;
                    int a3_row = (int)s_mul[(int)neg0 * 82 + s_inv[c3]] * 82;
                    bool z = true;
                    for (int r = 0; r < m; ++r) {
                        if (s_add[(int)u12[r] * 82 + s_mul[a3_row + s_cols[j3 * m + r]]] != 0) { z = false; break; }
                    }
                    if (z) { atomicAdd(&s_pen, 1); if (s_pen >= pen_cap) break; }
                }
            }
            __syncthreads();
        } else if (s_pen < pen_cap) {
            int s3_outer = n_starts * n_b * (q2 - 1);
            for (int idx = tid; idx < s3_outer && s_pen < pen_cap; idx += bdim) {
                int a2 = 1 + (idx % (q2 - 1));
                int tmp = idx / (q2 - 1);
                int j2 = tmp % n_b;
                int j1 = g_j1_starts[tmp / n_b];
                if (j2 <= j1) continue;
                unsigned char u12[16];
                int a2_row = a2 * 82;
                for (int r = 0; r < m; ++r) {
                    u12[r] = s_add[(int)s_cols[j1 * m + r] * 82 + s_mul[a2_row + s_cols[j2 * m + r]]];
                }
                for (int j3 = j2 + 1; j3 < n_b && s_pen < pen_cap; ++j3) {
                    for (int a3 = 1; a3 < q2 && s_pen < pen_cap; ++a3) {
                        unsigned char u123[16];
                        int w3 = 0, r0 = -1;
                        int a3_row = a3 * 82;
                        for (int r = 0; r < m; ++r) {
                            unsigned char v = s_add[(int)u12[r] * 82 + s_mul[a3_row + s_cols[j3 * m + r]]];
                            u123[r] = v;
                            if (v != 0) { if (r0 < 0) r0 = r; ++w3; }
                        }
                        if (3 + w3 <= w_max) { atomicAdd(&s_pen, 10); continue; }
                        if (w_max == 4) {
                            unsigned char neg0 = s_sub[u123[r0]];
                            for (int j4 = j3 + 1; j4 < n_b; ++j4) {
                                unsigned char c4 = s_cols[j4 * m + r0];
                                if (c4 == 0) continue;
                                int a4_row = (int)s_mul[(int)neg0 * 82 + s_inv[c4]] * 82;
                                bool z = true;
                                for (int r = 0; r < m; ++r) {
                                    if (s_add[(int)u123[r] * 82 + s_mul[a4_row + s_cols[j4 * m + r]]] != 0) { z = false; break; }
                                }
                                if (z) { atomicAdd(&s_pen, 1); if (s_pen >= pen_cap) break; }
                            }
                        } else {
                            for (int j4 = j3 + 1; j4 < n_b && s_pen < pen_cap; ++j4) {
                                for (int a4 = 1; a4 < q2 && s_pen < pen_cap; ++a4) {
                                    unsigned char u4[16];
                                    int w4 = 0, p0 = -1, p1 = -1;
                                    int a4_row = a4 * 82;
                                    for (int r = 0; r < m; ++r) {
                                        unsigned char v = s_add[(int)u123[r] * 82 + s_mul[a4_row + s_cols[j4 * m + r]]];
                                        u4[r] = v;
                                        if (v != 0) {
                                            if (p0 < 0) p0 = r;
                                            else if (p1 < 0) p1 = r;
                                            ++w4;
                                        }
                                    }
                                    if (4 + w4 <= w_max) { atomicAdd(&s_pen, 4); continue; }
                                    if (w_max == 5) {
                                        unsigned char neg0 = s_sub[u4[p0]];
                                        for (int j5 = j4 + 1; j5 < n_b; ++j5) {
                                            unsigned char c5 = s_cols[j5 * m + p0];
                                            if (c5 == 0) continue;
                                            int a5_row = (int)s_mul[(int)neg0 * 82 + s_inv[c5]] * 82;
                                            bool z = true;
                                            for (int r = 0; r < m; ++r) {
                                                if (s_add[(int)u4[r] * 82 + s_mul[a5_row + s_cols[j5 * m + r]]] != 0) { z = false; break; }
                                            }
                                            if (z) { atomicAdd(&s_pen, 1); if (s_pen >= pen_cap) break; }
                                        }
                                    } else if (w_max == 6) {
                                        unsigned char neg_p0 = s_sub[u4[p0]];
                                        unsigned char neg_p1 = s_sub[u4[p1]];
                                        for (int j5 = j4 + 1; j5 < n_b && s_pen < pen_cap; ++j5) {
                                            unsigned char c5_p0 = s_cols[j5 * m + p0];
                                            unsigned char c5_p1 = s_cols[j5 * m + p1];
                                            int cand_a[2]; int n_ca = 0;
                                            if (c5_p0 != 0) cand_a[n_ca++] = (int)s_mul[(int)neg_p0 * 82 + s_inv[c5_p0]];
                                            if (c5_p1 != 0) {
                                                int alt = (int)s_mul[(int)neg_p1 * 82 + s_inv[c5_p1]];
                                                if (n_ca == 0 || alt != cand_a[0]) cand_a[n_ca++] = alt;
                                            }
                                            for (int ci = 0; ci < n_ca; ++ci) {
                                                int ar = cand_a[ci] * 82, w5 = 0;
                                                for (int r = 0; r < m; ++r) {
                                                    if (s_add[(int)u4[r] * 82 + s_mul[ar + s_cols[j5 * m + r]]] != 0) ++w5;
                                                }
                                                if (w5 <= 1) { atomicAdd(&s_pen, 2); break; }
                                            }
                                            if (s_pen >= pen_cap) break;
                                            for (int a5 = 1; a5 < q2 && s_pen < pen_cap; ++a5) {
                                                int a5_r = a5 * 82, q0 = -1;
                                                unsigned char u5[16];
                                                for (int r = 0; r < m; ++r) {
                                                    unsigned char v = s_add[(int)u4[r] * 82 + s_mul[a5_r + s_cols[j5 * m + r]]];
                                                    u5[r] = v;
                                                    if (v != 0 && q0 < 0) q0 = r;
                                                }
                                                if (q0 < 0) { atomicAdd(&s_pen, 2); break; }
                                                unsigned char nq0 = s_sub[u5[q0]];
                                                for (int j6 = j5 + 1; j6 < n_b; ++j6) {
                                                    unsigned char c6 = s_cols[j6 * m + q0];
                                                    if (c6 == 0) continue;
                                                    int a6_r = (int)s_mul[(int)nq0 * 82 + s_inv[c6]] * 82;
                                                    bool z6 = true;
                                                    for (int r = 0; r < m; ++r) {
                                                        if (s_add[(int)u5[r] * 82 + s_mul[a6_r + s_cols[j6 * m + r]]] != 0) { z6 = false; break; }
                                                    }
                                                    if (z6) { atomicAdd(&s_pen, 1); if (s_pen >= pen_cap) break; }
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
            __syncthreads();
        }

        __shared__ bool s_take, s_new_b;
        if (tid == 0) {
            s_take = false;
            s_new_b = false;
            if (s_pen == 0) {
                s_best_pen = 0;
                if (atomicCAS(g_solved_bid, -1, bid) == -1) {
                    for (int i = 0; i < 320; ++i) g_win_hats[i] = s_hats[i];
                }
                s_done = true;
            } else {
                if (s_pen < s_best_pen) {
                    s_best_pen = s_pen;
                    s_new_b = true;
                }
                bool allow_up = (s_plat >= 10 && (s_plat & 3) == 0 && s_pen <= s_cur_pen + 2);
                if (s_pen < s_cur_pen || (s_pen == s_cur_pen && (xorshift32(rng) & 1U) == 0U) || allow_up) {
                    if (s_pen < s_cur_pen) s_plat = 0;
                    else ++s_plat;
                    s_cur_pen = s_pen;
                    s_take = true;
                    for (int k = 3; k >= 1; --k) s_tabu[k] = s_tabu[k - 1];
                    s_tabu[0] = (short)s_last_mode;
                } else {
                    ++s_plat;
                }
            }
        }
        __syncthreads();
        if (s_done) break;
        if (s_new_b) {
            for (int i = tid; i < 320; i += bdim) s_best_hats[i] = s_hats[i];
        }
        if (s_take) {
            for (int i = tid; i < 320; i += bdim) s_cur_hats[i] = s_hats[i];
        }
        __syncthreads();
    }

    if (tid == 0) g_best_pen_blk[bid] = s_best_pen;
    if (s_best_pen < 999999) {
        for (int i = tid; i < 320; i += bdim) g_best_hats_blk[bid * 320 + i] = s_best_hats[i];
    }
}

// ============================================================================
// KERNEL 2: Reservoir-Tiebreak 20-Probe Householder U(2/3/4) Finisher (`n <= 64`)
// Fixes the Indiv == 1 freeze by sampling uniformly among tied best probes!
// ============================================================================
__device__ void eval_subset_rank_s58(
    const unsigned char* s_G,
    const unsigned char* s_sub,
    const unsigned char* s_mul,
    const unsigned char* s_inv,
    const unsigned char* __restrict__ g_subs,
    const int n_subs,
    const int qi_offset,
    const int m,
    const int w_max,
    const int pen_cap,
    int &s_pen,
    short* s_vcols
) {
    const int tid = threadIdx.x;
    const int bdim = blockDim.x;
    if (tid == 0) {
        s_pen = 0;
        if (s_vcols != nullptr) s_vcols[15] = 0;
    }
    __syncthreads();

    unsigned char basis[8][16];
    unsigned char piv[8];

    for (int step_q = tid; step_q < n_subs; step_q += bdim) {
        if (s_pen >= pen_cap) break;
        int qi = step_q + qi_offset;
        if (qi >= n_subs) qi -= n_subs;
        const unsigned char* tup = &g_subs[((size_t)qi) * 8ULL];
        for (int d = 0; d < w_max; ++d) {
            unsigned char v[16];
            const unsigned char* src = &s_G[(int)tup[d] * 16];
            for (int r = 0; r < m; ++r) v[r] = src[r];
            for (int p = 0; p < d; ++p) {
                int pr = (int)piv[p];
                unsigned char fac = v[pr];
                if (fac != 0) {
                    int m_off = (int)fac * 82;
                    for (int r = 0; r < m; ++r) {
                        unsigned char prod = s_mul[m_off + (int)basis[p][r]];
                        v[r] = s_sub[(int)v[r] * 82 + (int)prod];
                    }
                }
            }
            int np = -1;
            for (int r = 0; r < m; ++r) {
                if (v[r] != 0) { np = r; break; }
            }
            if (np < 0) {
                int weight_pen = (d == 0) ? 500 : ((d == 1) ? 50 : 1);
                int old = atomicAdd(&s_pen, weight_pen);
                if (old == 0 && s_vcols != nullptr) {
                    int cnt = d + 1;
                    for (int k = 0; k < cnt; ++k) s_vcols[k] = (short)tup[k];
                    s_vcols[15] = (short)cnt;
                }
                break;
            }
            if (d + 1 < w_max) {
                int iv_off = (int)s_inv[v[np]] * 82;
                for (int r = 0; r < m; ++r) basis[d][r] = s_mul[iv_off + (int)v[r]];
                piv[d] = (unsigned char)np;
            }
        }
    }
    __syncthreads();
}

__device__ bool sample_and_apply_hh_s58(
    unsigned char* s_mat,
    const int* cols,
    const int d_ar,
    const int m,
    const int q2,
    const unsigned char* s_add,
    const unsigned char* s_sub,
    const unsigned char* s_mul,
    const unsigned char* s_conj,
    const unsigned char* g_tau_table,
    const unsigned char* g_tau_cnt,
    unsigned int &rng
) {
    unsigned char u[4], cu[4];
    for (int tr = 0; tr < 12; ++tr) {
        int eta = 0;
        for (int t = 0; t < d_ar; ++t) {
            int ut = 1 + (int)(xorshift32(rng) % (unsigned int)(q2 - 1));
            u[t] = (unsigned char)ut;
            cu[t] = s_conj[ut];
            int nrm = (int)s_mul[ut * 82 + (int)cu[t]];
            eta = (int)s_add[eta * 82 + nrm];
        }
        int cnt = (int)g_tau_cnt[eta];
        if (cnt <= 0) continue;
        int tau = (int)g_tau_table[eta * 16 + (int)(xorshift32(rng) % (unsigned int)cnt)];
        if (d_ar == 2) {
            int n0 = (int)s_mul[(int)u[0] * 82 + (int)cu[0]];
            if ((int)s_mul[tau * 82 + n0] == 1) continue;
        }
        unsigned char w[16];
        for (int r = 0; r < m; ++r) w[r] = 0;
        for (int t = 0; t < d_ar; ++t) {
            int c_off = cols[t] * 16;
            int m_off = (int)cu[t] * 82;
            for (int r = 0; r < m; ++r) {
                unsigned char pr = s_mul[m_off + (int)s_mat[c_off + r]];
                w[r] = s_add[(int)w[r] * 82 + (int)pr];
            }
        }
        unsigned char new_cols[4][16];
        bool all_nz = true;
        for (int t = 0; t < d_ar; ++t) {
            int tu = (int)s_mul[tau * 82 + (int)u[t]];
            int m_off = tu * 82;
            int c_off = cols[t] * 16;
            bool nz = false;
            for (int r = 0; r < m; ++r) {
                unsigned char sv = s_mul[m_off + (int)w[r]];
                unsigned char nv = s_sub[(int)s_mat[c_off + r] * 82 + (int)sv];
                new_cols[t][r] = nv;
                if (nv != 0) nz = true;
            }
            if (!nz) { all_nz = false; break; }
        }
        if (!all_nz) continue;
        for (int t = 0; t < d_ar; ++t) {
            int c_off = cols[t] * 16;
            for (int r = 0; r < m; ++r) s_mat[c_off + r] = new_cols[t][r];
        }
        return true;
    }
    return false;
}

__global__ void search_reservoir_hh_finisher(
    unsigned char* __restrict__ g_pop_G,
    int* __restrict__ g_pop_pen,
    short* __restrict__ g_pop_vcols,
    short* __restrict__ g_pop_tabu,
    int* __restrict__ g_pop_plat,
    unsigned char* __restrict__ g_best_G_blk,
    int* __restrict__ g_best_pen_blk,
    const unsigned char* __restrict__ g_add,
    const unsigned char* __restrict__ g_sub,
    const unsigned char* __restrict__ g_mul,
    const unsigned char* __restrict__ g_inv,
    const unsigned char* __restrict__ g_conj,
    const unsigned char* __restrict__ g_tau_table,
    const unsigned char* __restrict__ g_tau_cnt,
    const unsigned char* __restrict__ g_subs,
    const int n_subs,
    const int q,
    const int q2,
    const int n,
    const int m,
    const int w_max,
    const int n_macro,
    const int k_probes,
    const unsigned int base_seed,
    int* __restrict__ g_solved_bid,
    unsigned char* __restrict__ g_win_G
) {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int bdim = blockDim.x;

    __shared__ unsigned char s_add[82 * 82];
    __shared__ unsigned char s_sub[82 * 82];
    __shared__ unsigned char s_mul[82 * 82];
    __shared__ unsigned char s_inv[82];
    __shared__ unsigned char s_conj[82];
    __shared__ unsigned char s_cur_G[1024];
    __shared__ unsigned char s_cand_G[1024];
    __shared__ unsigned char s_best_probe_G[1024];
    __shared__ short s_cur_vcols[16], s_cand_vcols[16], s_best_p_vcols[16];
    __shared__ short s_tabu[6];
    __shared__ int s_cur_pen, s_best_pen, s_plat, s_pen, s_qi_off;
    __shared__ int s_best_p_pen, s_best_p_c0, s_ties;
    __shared__ bool s_done, s_need_init, s_probe_ok;

    for (int i = tid; i < 82 * 82; i += bdim) {
        s_add[i] = g_add[i];
        s_sub[i] = g_sub[i];
        s_mul[i] = g_mul[i];
    }
    if (tid < 82) {
        s_inv[tid] = g_inv[tid];
        s_conj[tid] = g_conj[tid];
    }
    for (int i = tid; i < 1024; i += bdim) s_cur_G[i] = g_pop_G[((size_t)bid) * 1024ULL + i];
    if (tid < 16) s_cur_vcols[tid] = g_pop_vcols[bid * 16 + tid];
    if (tid < 6) s_tabu[tid] = g_pop_tabu[bid * 6 + tid];
    if (tid == 0) {
        s_cur_pen = g_pop_pen[bid];
        s_best_pen = g_best_pen_blk[bid];
        s_plat = g_pop_plat[bid];
        s_done = (*g_solved_bid >= 0);
        s_need_init = (s_cur_pen >= 9000000);
    }
    __syncthreads();
    if (s_done) return;

    unsigned int rng = base_seed ^ ((unsigned int)(bid * 2654435761U + tid * 2246822519U + 131U));
    xorshift32(rng);

    if (s_need_init) {
        if (tid == 0) s_qi_off = (int)(xorshift32(rng) % (unsigned int)n_subs);
        __syncthreads();
        eval_subset_rank_s58(s_cur_G, s_sub, s_mul, s_inv, g_subs, n_subs, s_qi_off, m, w_max, 9999999, s_pen, s_cur_vcols);
        if (tid == 0) {
            s_cur_pen = s_pen;
            if (s_pen < s_best_pen) s_best_pen = s_pen;
            if (s_pen == 0 && atomicCAS(g_solved_bid, -1, bid) == -1) {
                for (int i = 0; i < 1024; ++i) g_win_G[i] = s_cur_G[i];
            }
            s_done = (*g_solved_bid >= 0);
        }
        if (s_cur_pen == s_best_pen) {
            for (int i = tid; i < 1024; i += bdim) g_best_G_blk[((size_t)bid) * 1024ULL + i] = s_cur_G[i];
        }
        __syncthreads();
        if (s_done) return;
    }

    for (int step = 0; step < n_macro; ++step) {
        if (tid == 0) {
            s_done = (*g_solved_bid >= 0);
            s_best_p_pen = 9999999;
            s_best_p_c0 = -1;
            s_ties = 0;
        }
        __syncthreads();
        if (s_done) break;

        for (int pr = 0; pr < k_probes; ++pr) {
            for (int i = tid; i < 1024; i += bdim) s_cand_G[i] = s_cur_G[i];
            __syncthreads();

            __shared__ int s_c0_used;
            if (tid == 0) {
                s_qi_off = (int)(xorshift32(rng) % (unsigned int)n_subs);
                int vc_cnt = (int)s_cur_vcols[15];
                int c0 = -1;
                if (vc_cnt > 0 && (xorshift32(rng) % 100U) < 92U) {
                    for (int tr = 0; tr < 6; ++tr) {
                        int cand_c = (int)s_cur_vcols[(pr + tr) % vc_cnt];
                        bool tb = false;
                        for (int k = 0; k < 3; ++k) if (s_tabu[k] == (short)cand_c) { tb = true; break; }
                        if (!tb || tr == 5) { c0 = cand_c; break; }
                    }
                }
                if (c0 < 0) c0 = (int)(xorshift32(rng) % (unsigned int)n);
                s_c0_used = c0;

                int d_ar = 2;
                unsigned int r100 = xorshift32(rng) % 100U;
                if (q == 2) d_ar = (r100 < 58U) ? 3 : 4;
                else if (s_cur_pen <= 2) d_ar = (r100 < 60U) ? 2 : ((r100 < 88U) ? 3 : 4);
                else d_ar = (r100 < 45U) ? 2 : ((r100 < 80U) ? 3 : 4);
                if (d_ar > n) d_ar = n;

                int cols[4];
                cols[0] = c0;
                for (int t = 1; t < d_ar; ++t) {
                    while (true) {
                        int cc = (int)(xorshift32(rng) % (unsigned int)n);
                        bool dup = false;
                        for (int prev = 0; prev < t; ++prev) if (cols[prev] == cc) { dup = true; break; }
                        if (!dup) { cols[t] = cc; break; }
                    }
                }
                s_probe_ok = sample_and_apply_hh_s58(
                    s_cand_G, cols, d_ar, m, q2, s_add, s_sub, s_mul, s_conj, g_tau_table, g_tau_cnt, rng
                );
            }
            __syncthreads();
            if (!s_probe_ok) continue;

            int cutoff = s_cur_pen + 4;
            if (s_best_p_pen + 1 < cutoff) cutoff = s_best_p_pen + 1;
            if (cutoff < 3) cutoff = 3;

            eval_subset_rank_s58(s_cand_G, s_sub, s_mul, s_inv, g_subs, n_subs, s_qi_off, m, w_max, cutoff, s_pen, s_cand_vcols);

            if (tid == 0 && s_pen == 0) {
                if (atomicCAS(g_solved_bid, -1, bid) == -1) {
                    for (int i = 0; i < 1024; ++i) g_win_G[i] = s_cand_G[i];
                }
                s_done = true;
            }
            __shared__ bool s_take_p;
            if (tid == 0) {
                s_take_p = false;
                if (s_pen < s_best_p_pen) {
                    s_best_p_pen = s_pen;
                    s_ties = 1;
                    s_take_p = true;
                } else if (s_pen == s_best_p_pen) {
                    ++s_ties;
                    if ((xorshift32(rng) % (unsigned int)s_ties) == 0U) s_take_p = true;
                }
                if (s_take_p) {
                    s_best_p_c0 = s_c0_used;
                    for (int k = 0; k < 16; ++k) s_best_p_vcols[k] = s_cand_vcols[k];
                }
            }
            __syncthreads();
            if (s_take_p) {
                for (int i = tid; i < 1024; i += bdim) s_best_probe_G[i] = s_cand_G[i];
            }
            __syncthreads();
            if (s_done) break;
        }
        if (s_done) break;

        __shared__ int s_act;
        if (tid == 0) {
            s_act = 0;
            bool allow_up = ((s_plat >= 4 && (s_plat & 1) == 1) && (s_best_p_pen <= s_cur_pen + 2))
                            || (s_plat >= 11 && (s_best_p_pen <= s_cur_pen + 4));
            if (s_best_p_c0 >= 0 && (s_best_p_pen <= s_cur_pen || allow_up)) {
                s_act = 1;
                if (s_best_p_pen < s_cur_pen) s_plat = 0;
                else ++s_plat;
                s_cur_pen = s_best_p_pen;
                for (int k = 5; k >= 1; --k) s_tabu[k] = s_tabu[k - 1];
                s_tabu[0] = (short)s_best_p_c0;
                for (int k = 0; k < 16; ++k) s_cur_vcols[k] = s_best_p_vcols[k];
                if (s_cur_pen < s_best_pen) s_best_pen = s_cur_pen;
                if (s_plat >= 28) {
                    s_act = 2;
                    s_plat = 0;
                }
            } else {
                ++s_plat;
                if (s_plat >= 24) {
                    s_act = 2;
                    s_plat = 0;
                }
            }
        }
        __syncthreads();

        if (s_act == 1) {
            for (int i = tid; i < 1024; i += bdim) s_cur_G[i] = s_best_probe_G[i];
            __syncthreads();
            if (s_cur_pen == s_best_pen) {
                for (int i = tid; i < 1024; i += bdim) g_best_G_blk[((size_t)bid) * 1024ULL + i] = s_cur_G[i];
            }
            __syncthreads();
        } else if (s_act == 2) {
            for (int i = tid; i < 1024; i += bdim) s_cur_G[i] = g_best_G_blk[((size_t)bid) * 1024ULL + i];
            __syncthreads();
            if (tid == 0) {
                int cols[3];
                cols[0] = (int)(xorshift32(rng) % (unsigned int)n);
                cols[1] = (cols[0] + 1 + (int)(xorshift32(rng) % (unsigned int)(n - 1))) % n;
                cols[2] = (cols[1] + 1 + (int)(xorshift32(rng) % (unsigned int)(n - 2))) % n;
                if (cols[2] == cols[0]) cols[2] = (cols[2] + 1) % n;
                int d_k = (q > 2 && s_best_pen <= 2) ? 2 : 3;
                sample_and_apply_hh_s58(s_cur_G, cols, d_k, m, q2, s_add, s_sub, s_mul, s_conj, g_tau_table, g_tau_cnt, rng);
                for (int k = 0; k < 6; ++k) s_tabu[k] = -1;
                s_qi_off = (int)(xorshift32(rng) % (unsigned int)n_subs);
            }
            __syncthreads();
            eval_subset_rank_s58(s_cur_G, s_sub, s_mul, s_inv, g_subs, n_subs, s_qi_off, m, w_max, 9999999, s_pen, s_cur_vcols);
            if (tid == 0) {
                s_cur_pen = s_pen;
                if (s_pen == 0 && atomicCAS(g_solved_bid, -1, bid) == -1) {
                    for (int i = 0; i < 1024; ++i) g_win_G[i] = s_cur_G[i];
                    s_done = true;
                }
            }
            __syncthreads();
        }
    }

    for (int i = tid; i < 1024; i += bdim) g_pop_G[((size_t)bid) * 1024ULL + i] = s_cur_G[i];
    if (tid < 16) g_pop_vcols[bid * 16 + tid] = s_cur_vcols[tid];
    if (tid < 6) g_pop_tabu[bid * 6 + tid] = s_tabu[tid];
    if (tid == 0) {
        g_pop_pen[bid] = s_cur_pen;
        g_best_pen_blk[bid] = s_best_pen;
        g_pop_plat[bid] = s_plat;
    }
}
}
'''


class SpectralDecompositionExt:
    """Exact GFq2Ext-native Fourier diagonalization of GF(q^2)[x]/(x^m - lam) (`m | q^2 - 1`)."""

    def __init__(self, f: GFq2Ext, m: int, lam: int):
        self.f = f
        self.m = m
        self.lam = lam
        q2 = f.q2

        roots = [u for u in range(1, q2) if f.pow(u, m) == lam]
        self.valid = (len(roots) == m)
        if not self.valid:
            return
        self.roots = np.array(roots, dtype=np.uint8)
        r_to_i = {int(rv): idx for idx, rv in enumerate(roots)}
        self.pi = np.array([r_to_i[int(f.inv[f.conj[rv]])] for rv in roots], dtype=np.uint8)

        V = np.zeros((m, m), dtype=np.uint8)
        for s, rv in enumerate(roots):
            cur = 1
            for r in range(m):
                V[s, r] = cur
                cur = int(f.mul[cur, rv])
        self.V = V
        self.V_inv = self._invert(V)
        self.d_V_inv = cp.asarray(self.V_inv.reshape(-1))
        self.d_pi = cp.asarray(self.pi)

        nb_arr = np.zeros((82, 12), dtype=np.uint8)
        nc_arr = np.zeros(82, dtype=np.uint8)
        buckets = {}
        for u in range(q2):
            nv = int(f.mul[u, f.conj[u]])
            buckets.setdefault(nv, []).append(u)
        for nv, lst in buckets.items():
            nc_arr[nv] = len(lst)
            for idx_u, u in enumerate(lst[:12]):
                nb_arr[nv, idx_u] = u
        self.d_nbuck = cp.asarray(nb_arr.reshape(-1))
        self.d_ncnt = cp.asarray(nc_arr)

    def _invert(self, mat: np.ndarray) -> np.ndarray:
        f = self.f
        m = mat.shape[0]
        aug = np.zeros((m, 2 * m), dtype=np.uint8)
        aug[:, :m] = mat
        for i in range(m):
            aug[i, m + i] = 1
        for piv in range(m):
            sel = next(r for r in range(piv, m) if aug[r, piv] != 0)
            if sel != piv:
                aug[[piv, sel]] = aug[[sel, piv]]
            iv = int(f.inv[aug[piv, piv]])
            for c in range(2 * m):
                aug[piv, c] = f.mul[aug[piv, c], iv]
            for r in range(m):
                if r != piv and aug[r, piv] != 0:
                    fac = int(aug[r, piv])
                    for c in range(2 * m):
                        aug[r, c] = f.sub[aug[r, c], f.mul[fac, aug[piv, c]]]
        return np.ascontiguousarray(aug[:, m:], dtype=np.uint8)

    def dft(self, poly: np.ndarray) -> np.ndarray:
        f = self.f
        m = self.m
        out = np.zeros(m, dtype=np.uint8)
        for s in range(m):
            acc = 0
            for r in range(m):
                acc = int(f.add[acc, f.mul[self.V[s, r], poly[r]]])
            out[s] = acc
        return out

    def materialize_G_from_hats(self, hats_320: np.ndarray, lm1: int, bcols: np.ndarray) -> np.ndarray:
        """Converts `hats_320` (`[20 * 16]`) + `bcols` (`[bp, m]`) to exact `m x n` generator matrix `G`!"""
        f = self.f
        m = self.m
        bp = bcols.shape[0] if bcols.ndim == 2 else 0
        n = (lm1 + 1) * m + bp
        hats = hats_320.reshape(20, 16)[:lm1, :m]
        G = np.zeros((m, n), dtype=np.uint8)
        for r in range(m):
            G[r, r] = 1
        for t in range(lm1):
            poly = np.zeros(m, dtype=np.uint8)
            for r in range(m):
                acc = 0
                for s in range(m):
                    acc = int(f.add[acc, f.mul[self.V_inv[r, s], hats[t, s]]])
                poly[r] = acc
            for c in range(m):
                col = (t + 1) * m + c
                for r in range(m):
                    G[r, col] = poly[r - c] if r >= c else f.mul[self.lam, poly[m + r - c]]
        if bp > 0:
            G[:, (lm1 + 1) * m : n] = bcols.T
        return G


def check_herm_orth(f: GFq2Ext, G: np.ndarray) -> bool:
    m, n = G.shape
    for r in range(m):
        for s in range(r, m):
            acc = 0
            for c in range(n):
                acc = int(f.add[acc, f.mul[G[r, c], f.conj[G[s, c]]]])
            if acc != 0:
                return False
    return True


def run_selftest(k_spec, k_hh):
    # Verify exact 0%-rejection on-chip Spectral Fourier + IDFT + orthogonality + solver across all 5 dividing fields!
    for q, m, l, bp, d_tar in [(3, 4, 3, 0, 4), (4, 5, 3, 2, 5), (5, 4, 3, 1, 4), (7, 4, 3, 0, 4), (8, 3, 3, 1, 4)]:
        f = GFq2Ext(q)
        fibs = get_constacyclic_fibers(f, m)
        lam, xis = next(iter(fibs.items()))
        sp = SpectralDecompositionExt(f, m, lam)
        assert sp.valid
        bspec = [(1, xis[i]) for i in range(bp)]
        targ, bcols = make_border_from_bspec(f, m, bspec)
        if bp == 0:
            bcols = np.zeros((0, m), dtype=np.uint8)
        targ_hat = sp.dft(targ)
        lm1 = l - 1
        j1s = [bi * m for bi in range(lm1)] + [lm1 * m + c for c in range(bp)]
        NBLK = 32
        d_bpen = cp.full(NBLK, 999999, dtype=cp.int32)
        d_bhats = cp.zeros((NBLK, 320), dtype=cp.uint8)
        d_sol = cp.full(1, -1, dtype=cp.int32)
        d_win = cp.zeros(320, dtype=cp.uint8)
        d_conj, d_tt, d_tc = build_householder_tau_tables(f)
        k_spec(
            (NBLK,), (128,),
            (
                sp.d_V_inv, sp.d_pi, sp.d_nbuck, sp.d_ncnt,
                cp.asarray(targ_hat), cp.asarray(bcols.reshape(-1)) if bp > 0 else cp.zeros(1, dtype=cp.uint8),
                f.d_add, f.d_sub, f.d_mul, f.d_inv, d_conj,
                cp.asarray(np.array(j1s, dtype=np.int32)), np.int32(len(j1s)),
                np.int32(lm1), np.int32(bp), np.int32(m), np.int32(lam), np.int32(f.q2),
                np.int32(d_tar), np.int32(25), np.uint32(12345 + q),
                cp.zeros(320, dtype=cp.uint8), np.int32(0),
                d_bpen, d_bhats, d_sol, d_win,
            ),
        )
        cp.cuda.Stream.null.synchronize()
        if int(d_sol.get()[0]) >= 0:
            Gw = sp.materialize_G_from_hats(d_win.get(), lm1, bcols)
            assert check_herm_orth(f, Gw), f"Spectral winner orthogonality failed on q={q}!"
        else:
            bh = d_bhats.get()
            Gw = sp.materialize_G_from_hats(bh[0], lm1, bcols)
            assert check_herm_orth(f, Gw), f"Spectral elite orthogonality failed on q={q}!"
    print("SELFTEST_S58_ONCHIP_SPECTRAL_FOURIER_AND_RESERVOIR_HH_OK", flush=True)


def run_stage_b_reservoir_hh(
    k_hh, f: GFq2Ext, d_conj, d_tt, d_tc, tau_tab_np, tau_cnt_np,
    seed_Gs: list[np.ndarray], m: int, n: int, d_tar: int, rng: random.Random, rng_np: np.random.Generator,
    budget_s: float = 11.0,
):
    c_tot = math.comb(n, d_tar - 1)
    NBLK = 224 if c_tot <= 15_000 else 128
    pop = np.zeros((NBLK, 1024), dtype=np.uint8)
    for b in range(NBLK):
        gm = seed_Gs[b % len(seed_Gs)].copy()
        if b >= len(seed_Gs):
            d_ar = 2 if (b % 3 != 0) else 3
            cpu_apply_householder(f, gm, rng.sample(range(n), d_ar), tau_tab_np, tau_cnt_np, rng)
        pop[b] = pack_G64_16(gm)

    subs_arr = build_exact_wmax_subsets(n, d_tar - 1, rng_np)
    d_subs = cp.asarray(subs_arr.reshape(-1))
    n_subs = len(subs_arr)

    d_pop_G = cp.asarray(pop.reshape(-1))
    d_best_G = d_pop_G.copy()
    d_pop_pen = cp.full(NBLK, 9999999, dtype=cp.int32)
    d_best_pen = cp.full(NBLK, 9999999, dtype=cp.int32)
    d_vcols = cp.zeros((NBLK, 16), dtype=cp.int16)
    d_tabu = cp.full((NBLK, 6), -1, dtype=cp.int16)
    d_plat = cp.zeros(NBLK, dtype=cp.int32)
    d_sol = cp.full(1, -1, dtype=cp.int32)
    d_win = cp.zeros(1024, dtype=cp.uint8)

    k_probes = 20 if c_tot <= 6_500 else (12 if c_tot <= 20_000 else 6)
    n_macro = 14 if c_tot <= 4_000 else (8 if c_tot <= 15_000 else 4)
    t0 = time.time()
    launches = 0
    best_indiv = 9999999
    while time.time() - t0 < budget_s:
        d_sol.fill(-1)
        seed = (int(time.time() * 1000000) ^ (launches * 0x9E3779B9) ^ (n * 401 + m * 53)) & 0xFFFFFFFF
        k_hh(
            (NBLK,), (256,),
            (
                d_pop_G, d_pop_pen, d_vcols, d_tabu, d_plat, d_best_G, d_best_pen,
                f.d_add, f.d_sub, f.d_mul, f.d_inv, d_conj, d_tt, d_tc,
                d_subs, np.int32(n_subs),
                np.int32(f.q), np.int32(f.q2), np.int32(n), np.int32(m), np.int32(d_tar - 1),
                np.int32(n_macro), np.int32(k_probes), np.uint32(seed), d_sol, d_win,
            ),
        )
        cp.cuda.Stream.null.synchronize()
        launches += 1
        if int(d_sol.get()[0]) >= 0:
            return unpack_G64_16(d_win.get(), m, n), 0

        pens = d_best_pen.get()
        bmin = int(pens.min())
        if bmin < best_indiv:
            best_indiv = bmin
        if launches % 3 == 0 and bmin <= 6:
            order_b = np.argsort(pens)
            top_b = order_b[:8]
            worst_b = order_b[64:]
            if int(pens[worst_b[0]]) > bmin:
                bg_h = d_best_G.get().reshape(NBLK, 1024)
                pg_h = d_pop_G.get().reshape(NBLK, 1024)
                pp_h = d_pop_pen.get()
                for wi, wb in enumerate(worst_b):
                    sb = int(top_b[wi % len(top_b)])
                    gm = unpack_G64_16(bg_h[sb], m, n)
                    if wi >= 8:
                        cpu_apply_householder(f, gm, rng.sample(range(n), 2), tau_tab_np, tau_cnt_np, rng)
                    pg_h[wb] = pack_G64_16(gm)
                    bg_h[wb] = bg_h[sb]
                    pens[wb] = pens[sb]
                    pp_h[wb] = 9999999
                d_pop_G = cp.asarray(pg_h.reshape(-1))
                d_best_G = cp.asarray(bg_h.reshape(-1))
                d_best_pen = cp.asarray(pens)
                d_pop_pen = cp.asarray(pp_h)
    return None, best_indiv


def main():
    try:
        hp = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.kernel32.SetPriorityClass(hp, 0x00000020)
        ctypes.windll.kernel32.SetProcessAffinityMask(hp, (1 << 24) - 1)
    except Exception:
        pass
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)

    mod = cp.RawModule(code=CUDA_S58_KERNELS)
    k_spec = mod.get_function("search_onchip_spectral_fourier_walksat")
    k_hh = mod.get_function("search_reservoir_hh_finisher")
    run_selftest(k_spec, k_hh)

    if len(sys.argv) >= 2 and sys.argv[1] == "--selftest":
        return

    snap = json.load(open(os.path.join(SCRATCH, "codetables_qecc_snapshot.json"), encoding="utf-8"))
    fields = {q: GFq2Ext(q) for q in (3, 4, 5, 7, 8)}
    hh_gpu = {q: build_householder_tau_tables(fields[q]) for q in fields}
    hh_cpu = {q: (hh_gpu[q][1].get().reshape(82, 16), hh_gpu[q][2].get()) for q in fields}
    rng = random.Random(2026112858)
    rng_np = np.random.default_rng(2026112858)

    beaten = get_all_beaten(snap)
    cells0, exact0 = count_total_codetables_cells(snap)

    # Enumerate ALL open Permanent-Lid targets (`exact_gain >= 1`) where `m | (q^2 - 1)` and `m <= 16`, `n <= 100`!
    pq = []
    spec_cache = {}
    for q in (4, 5, 7, 8, 3):
        f = fields[q]
        q2m1 = q * q - 1
        sq = snap[str(q)]
        for m in range(3, 17):
            if q2m1 % m != 0:
                continue
            fibs = get_constacyclic_fibers(f, m)
            if not fibs:
                continue
            for l in range(2, 22):
                if (l - 1) > 20:
                    continue
                for bp in range(0, min(7, m)):
                    n = l * m + bp
                    k = (l - 2) * m + bp
                    nb = (l - 1) * m + bp
                    if k < 0 or n < 10 or n > 102 or nb > 108:
                        continue
                    # Segre / Bose-Burton projective plane arc upper bound for m == 3, d_tar == 4:
                    # Over GF(q^2), max arc in PG(2, q^2) has size q^2 + 1 (if q odd) or q^2 + 2 (if q even)!
                    bb = sq.get(f"{n},{k}")
                    if not bb or bb[1] is None:
                        continue
                    cur_lb = max(bb[0] or 0, beaten.get((q, n, k), 0))
                    if cur_lb >= bb[1]:
                        continue
                    for d_tar in sorted({cur_lb + 1, int(bb[1])}):
                        if d_tar not in (4, 5, 6, 7) or (d_tar - 1) > m or d_tar > bb[1]:
                            continue
                        if m == 3 and d_tar == 4:
                            max_arc = (q * q + 2) if (q % 2 == 0) else (q * q + 1)
                            if n > max_arc:
                                continue
                        # Pure sphere-packing bound
                        t_pk = (d_tar - 1) // 2
                        if sum(math.comb(n, w) * ((q * q - 1) ** w) for w in range(t_pk + 1)) > (q ** (2 * m)):
                            continue
                        gain, ex_gain = calc_domino_gain(snap, beaten, q, n, k, d_tar)
                        if ex_gain <= 0:
                            continue
                        w_max = d_tar - 1
                        ev = math.comb(n, w_max) * ((q * q - 1) ** (w_max - 1)) / (q ** (2 * m))
                        if ev > 2500.0:
                            continue
                        if d_tar == 7 and (n > 30 or ev > 400.0):
                            continue

                        # Check valid comb-free eigencharacter borders (`bp == 0` always valid!)
                        valid_lams = []
                        for lm, xis in fibs.items():
                            if bp == 0:
                                valid_lams.append((lm, []))
                            elif len(xis) >= bp:
                                cf_t = [xt for xt in itertools.islice(itertools.combinations(xis, bp), 40) if is_comb_free_xis(f, m, xt, d_tar)]
                                if cf_t:
                                    valid_lams.append((lm, cf_t))
                        if not valid_lams:
                            continue
                        valid_lams.sort(key=lambda pair: (0 if pair[0] == 1 else 1, pair[0]))

                        tag = f"S58SpecWS_q{q}_m{m}_l{l}_bp{bp}_n{n}_k{k}_d{d_tar}"
                        if is_blacklisted(tag, ENGINE_S58_SPECTRAL_WS):
                            continue

                        # Prioritize proven sweet-spot high-rate corridors (`slack >= 1` first, especially q=4 m=5 n=30..68, q=5 m=4/6, q=7 m=4/6!)
                        is_qmds = 1 if m == (d_tar - 1) else 0
                        sweet = 0 if (is_qmds == 0 and d_tar in (4, 5, 6) and ev <= 350.0) else (1 if is_qmds == 0 else 2)
                        heapq.heappush(pq, (sweet, d_tar, ev, -ex_gain, n, q, m, l, bp, k, valid_lams[:3], tag))

    log_q2(
        STREAM_TAG,
        f"🔥🔥 Stream 58 起動（100%全空間オンチップ・スペクトルフーリエ Tabu WalkSAT ＋ 貯水池HH仕上げ）："
        f"現在セル={cells0}（**永久蓋={exact0}**）｜未踏スペクトル永久蓋 `{len(pq)}` 件！",
    )

    while pq:
        sweet, d_tar, ev, _, n, q, m, l, bp, k, valid_lams, tag = heapq.heappop(pq)
        beaten = get_all_beaten(snap)
        gain_now, ex_now = calc_domino_gain(snap, beaten, q, n, k, d_tar)
        if ex_now <= 0 or is_blacklisted(tag, ENGINE_S58_SPECTRAL_WS):
            continue

        f = fields[q]
        lm1 = l - 1
        j1_starts = [bi * m for bi in range(lm1)] + [lm1 * m + c for c in range(bp)]
        d_j1 = cp.asarray(np.array(j1_starts, dtype=np.int32))
        nonzero_gfq = f.gfq_elems[1:]

        configs = []
        for lam, cf_tups in valid_lams:
            skey = (q, m, lam)
            if skey not in spec_cache:
                spec_cache[skey] = SpectralDecompositionExt(f, m, lam)
            sp = spec_cache[skey]
            if not sp.valid:
                continue
            if bp == 0:
                targ, _ = make_border_from_bspec(f, m, [])
                bcols = np.zeros((0, m), dtype=np.uint8)
                configs.append((f"lam{lam}_bp0", sp, sp.dft(targ), bcols, f"bp=0,lam={lam}"))
            else:
                shuf = list(cf_tups)
                rng.shuffle(shuf)
                for xt in shuf[:5]:
                    for _ in range(3):
                        bs = [(rng.choice(nonzero_gfq), x) for x in xt]
                        targ, bcols = make_border_from_bspec(f, m, bs)
                        configs.append((f"lam{lam}_{bs}", sp, sp.dft(targ), bcols, f"lam={lam},bs={bs}"))
                        if len(configs) >= 14:
                            break
                    if len(configs) >= 14:
                        break

        if not configs:
            continue

        NBLK = 256 if d_tar <= 5 else 164
        n_steps = 44 if d_tar == 4 else (22 if d_tar == 5 else (6 if d_tar == 6 else 3))
        d_bpen = cp.full(NBLK, 999999, dtype=cp.int32)
        d_bhats = cp.zeros((NBLK, 320), dtype=cp.uint8)
        d_sol = cp.full(1, -1, dtype=cp.int32)
        d_win = cp.zeros(320, dtype=cp.uint8)
        d_conj, d_tt, d_tc = hh_gpu[q]
        tau_tab_np, tau_cnt_np = hh_cpu[q]

        t_budget = 22.0 if (sweet == 0 or ex_now >= 2) else 14.0
        t0 = time.time()
        launches = 0
        global_min = 999999
        elites_by_cfg = {}
        stage_b_mats = []
        won = False

        log_q2(
            STREAM_TAG,
            f"🎯 スペクトルWS狙撃 [[{n}, {k}, {d_tar}]]_{q} (m={m}, l={l}, bp={bp}, cfgs={len(configs)}, **永久蓋+{ex_now}個**, 総波及={gain_now})...",
        )

        while time.time() - t0 < t_budget:
            ckey, sp, targ_hat, bcols, cdesc = configs[launches % len(configs)]
            cur_el = elites_by_cfg.get(ckey, [])
            if cur_el:
                el_arr = np.stack([h for _, h in cur_el], axis=0)
                d_el = cp.asarray(el_arr.reshape(-1))
                n_el = len(cur_el)
            else:
                d_el = cp.zeros(320, dtype=cp.uint8)
                n_el = 0

            d_sol.fill(-1)
            d_bpen.fill(999999)
            seed = (int(time.time() * 1000000) ^ (launches * 0x9E3779B9) ^ (n * 197 + q * 41)) & 0xFFFFFFFF
            k_spec(
                (NBLK,), (256,),
                (
                    sp.d_V_inv, sp.d_pi, sp.d_nbuck, sp.d_ncnt,
                    cp.asarray(targ_hat), cp.asarray(bcols.reshape(-1)) if bp > 0 else cp.zeros(1, dtype=cp.uint8),
                    f.d_add, f.d_sub, f.d_mul, f.d_inv, d_conj,
                    d_j1, np.int32(len(j1_starts)),
                    np.int32(lm1), np.int32(bp), np.int32(m), np.int32(sp.lam), np.int32(f.q2),
                    np.int32(d_tar), np.int32(n_steps), np.uint32(seed),
                    d_el, np.int32(n_el),
                    d_bpen, d_bhats, d_sol, d_win,
                ),
            )
            cp.cuda.Stream.null.synchronize()
            launches += 1

            if int(d_sol.get()[0]) >= 0:
                Gw = sp.materialize_G_from_hats(d_win.get(), lm1, bcols)
                if verify_and_export_20thread(f, Gw, d_tar, f"OnChipSpectralFourierWS(m={m},l={l},bp={bp},{cdesc})", snap):
                    c_tot_now, ex_tot_now = count_total_codetables_cells(snap)
                    log_q2(
                        STREAM_TAG,
                        f"🔒🔒🔒【永久蓋 LOCKED (スペクトルフーリエWS)！！】[[{n}, {k}, {d_tar}]]_{q} "
                        f"⇒ 永久蓋累計: **{ex_tot_now} マス**（総更新: {c_tot_now} マス）！！",
                    )
                    won = True
                    break
                d_sol.fill(-1)

            pens = d_bpen.get()
            bh = d_bhats.get()
            bmin = int(pens.min())
            if bmin < global_min:
                global_min = bmin

            for idx_b in np.argsort(pens)[:8]:
                ii = int(idx_b)
                pv = int(pens[ii])
                if pv <= bmin + 3 and pv < 900000:
                    hc = bh[ii].copy()
                    cur_el.append((pv, hc))
                    if pv <= 4 and n <= 62 and math.comb(n, d_tar - 1) <= 45_000 and len(stage_b_mats) < 20:
                        stage_b_mats.append((pv, sp.materialize_G_from_hats(hc, lm1, bcols)))
            cur_el.sort(key=lambda x: x[0])
            elites_by_cfg[ckey] = cur_el[:12]

        # Stage-B Reservoir-Tiebreak Householder Finisher if Stage-A reached `Orb <= 4`!
        if not won and stage_b_mats and global_min <= 4:
            stage_b_mats.sort(key=lambda x: x[0])
            seed_Gs = [gm for _, gm in stage_b_mats[:12]]
            Gw_hh, best_indiv = run_stage_b_reservoir_hh(
                k_hh, f, d_conj, d_tt, d_tc, tau_tab_np, tau_cnt_np,
                seed_Gs, m, n, d_tar, rng, rng_np, budget_s=11.0,
            )
            if Gw_hh is not None:
                if verify_and_export_20thread(f, Gw_hh, d_tar, f"SpectralToReservoirHH(m={m},fromOrb={global_min})", snap):
                    c_tot_now, ex_tot_now = count_total_codetables_cells(snap)
                    log_q2(
                        STREAM_TAG,
                        f"🔒🔒🔒【永久蓋 LOCKED (スペクトル⇒貯水池HH対称性破壊)！！】[[{n}, {k}, {d_tar}]]_{q} "
                        f"⇒ 永久蓋累計: **{ex_tot_now} マス**（総更新: {c_tot_now} マス）！！",
                    )
                    won = True

        if not won:
            record_attempt(
                tag, ENGINE_S58_SPECTRAL_WS,
                f"S58SpecWS(Orb={global_min}) [[{n},{k},{d_tar}]]_{q}",
                global_min, global_min, time.time() - t0, max_stalled_rounds=1,
            )


if __name__ == "__main__":
    main()

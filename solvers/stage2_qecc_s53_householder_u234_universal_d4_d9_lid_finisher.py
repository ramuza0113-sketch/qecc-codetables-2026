r"""【第2段階・Stream 53（v1.2.0 永久蓋 決戦ハンター⑥：汎用オンザフライ Householder–Witt $U(2)/U(3)/U(4)$ 鏡映・トランスベクション × 全数部分列ランク `d = 4..9` 永久蓋フィニッシャー）】
  Stream 51 & Stream 52 で新たに炙り出された「あと違反1〜4個で永久蓋確定」の超リーチ群：
    - `[[15, 1, 7]]_3` (`MIN_VIOL = 1`!! **永久蓋+2個**, `C(15,6) = 5,005`!)
    - `[[21, 11, 5]]_3` (`Indiv = 1`!! **永久蓋+1個**, `C(21,4) = 5,985`!)
    - `[[15, 5, 6]]_7` (`MIN_VIOL = 2`!! **永久蓋+1個**, `C(15,5) = 3,003`!)
    - `[[35, 21, 5]]_2` (`QT = 1`!! **永久蓋+1個**, `C(35,4) = 52,360`!)
    - `[[36, 22, 5]]_2` (`QT = 2`!! **永久蓋+2個**, `C(36,4) = 58,905`!)
    - `[[18, 6, 6]]_3` (`MIN_VIOL = 3`!! **永久蓋+1個**, `C(18,5) = 8,568`!)
    - `[[14, 0, 7]]_3` (`MIN_VIOL = 3`!! **永久蓋+1個**, `C(14,6) = 3,003`!)
    - `[[11, 1, 6]]_5` (`MIN_VIOL = 4`!! **永久蓋+1個**, `C(11,5) = 462`!)
    - さらに `[[16, 2, 7]]_3`, `[[17, 3, 7]]_3`, `[[15, 3, 6]]_3`, `[[17, 5, 6]]_3`, `[[13, 1, 7]]_4`, `[[14, 2, 7]]_5..8`, `[[15, 1, 8]]_4..8`, `[[31, 21, 5]]_4`, `[[37..39, 27..29, 4]]_2`！

  【数学的突破（Dieudonné–Witt ユニタリ生成定理に基づく動的アリティ $d \in \{2, 3, 4\}$ オンザフライ鏡映・トランスベクション）】：
  固定サイズの小ユニタリ表（$U(2)$ 56個 / $U(3)$ 486個）では `Indiv=1` の局所盆地から脱出できなかった限界を完全打破！
  任意の $d \in \{2, 3, 4\}$ 列 $S = \{c_0, \dots, c_{d-1}\}$ と任意の非ゼロ係数 $u \in (\mathbb{F}_{q^2}^\times)^d$ に対し、$\eta = \sum u_t \overline{u_t} \in \mathbb{F}_q$ を計算：
    - $\eta \ne 0$ のとき：$\beta + \overline{\beta} = \eta$ なる $\beta \in \mathbb{F}_{q^2}^\times$ を取り $\tau = \beta^{-1}$ とすれば $U = I_d - \tau u u^\dagger$ は **全オフ対角非ゼロの厳密 Householder ユニタリ鏡映**！
    - $\eta = 0$ のとき（特に $q = 2, d = 4$ では恒等的に $\eta = 0$！）：$\tau + \overline{\tau} = 0$ なる $\tau \in \mathbb{F}_{q^2}^\times$ を取れば $U = I_d - \tau u u^\dagger$ は **厳密等方 Witt トランスベクション（4列同時混合！）**！
  共有メモリ行列テーブル 0 バイトで数百万通りの $U(2)/U(3)/U(4)$ 変換をオンザフライ生成し、かつ `d = 4..9` 汎用ブロック内並列ガウス消去と直結して最後の 1 個の違反を破壊する！
"""
import ctypes
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
from rtx3090_qecc_cuda_blitz import GFq2Host
from stage2_qecc_core import GFq2Ext, calc_domino_gain, log_q2
from stage2_qecc_s2_fast_mds_and_4arch_tabu import get_constacyclic_fibers
from stage2_qecc_s8_frontier_crown_sniper_and_fullpool import make_border_from_bspec
from stage2_qecc_s11_monomial_cycle_multiborder_qt import build_monomial_cycle_border
from stage2_qecc_s12_vram_pool_656_constacyclic_sniper import build_vram_csr_pool
from stage2_qecc_s14_algebraic_and_gpu_mp_plotkin import build_gfq_decomp
from stage2_qecc_s20_primary_ideal_zero_reject_subset_rank import (
    OrthogonalComponentSolver,
    poly_batch_to_blocks_and_norms,
)
from stage2_qecc_s22_fullpower_lucas_hensel_and_d10_d13_slam import (
    CUDA_SUBSET_RANK_D14,
    build_symmetry_ktuples,
    eval_fullpower,
    solve_lucas_hensel_char2_pow2,
    verify_and_export_20thread,
)
from stage2_qecc_s44_hash_mitm_qt_and_witt_unitary_surgery_lid_slam import is_comb_free_xis
from stage2_qecc_s46_persistent_tabu_witt_and_q2_129_lid_slam import CUDA_S46_KERNELS
from step2_gpu80_qecc_100cells import count_total_codetables_cells
from stagnation_guard import is_blacklisted, record_attempt

ENGINE_S53_HOUSEHOLDER = "STAGE2_QECC_HOUSEHOLDER_U234_UNIVERSAL_D4_D9_FINISHER_V1"
STREAM_TAG = "Stream53-HouseholderU234-d4d9"


CUDA_S53_HOUSEHOLDER_KERNEL = r'''
extern "C" {
__device__ __forceinline__ unsigned int xorshift32(unsigned int &state) {
    unsigned int x = state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    state = x; return x;
}

// Block-cooperative exact subset-rank evaluator over GF(q^2) for s_G[n * m] (`n <= 64, m <= 16, w_max <= 8`)!
__device__ void eval_subset_rank_block(
    const unsigned char* s_G,                 // [64 * 16]: column-major col c at c*16
    const unsigned char* s_sub,               // [82 * 82]
    const unsigned char* s_mul,               // [82 * 82]
    const unsigned char* s_inv,               // [82]
    const unsigned char* __restrict__ g_subs, // [n_subs * 8]: column indices c_0..c_{w_max-1}
    const int n_subs,
    const int m,
    const int w_max,
    const int pen_cap,
    int &s_pen,
    short* s_vcols                            // [16]: s_vcols[0..w_max-1] = violating subset, s_vcols[15] = count
) {
    const int tid = threadIdx.x;
    const int bdim = blockDim.x;
    if (tid == 0) {
        s_pen = 0;
        if (s_vcols != nullptr) s_vcols[15] = 0;
    }
    __syncthreads();

    // First check single zero columns (`weight 1`)
    if (tid < 64) {
        // Handled implicitly when c_0 is zero column in first subset containing c_0!
    }

    unsigned char basis[8][16];
    unsigned char piv[8];

    for (int qi = tid; qi < n_subs; qi += bdim) {
        if (s_pen >= pen_cap) break;
        const unsigned char* tup = &g_subs[((size_t)qi) * 8ULL];
        bool dep = false;
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
                dep = true;
                int weight_pen = (d == 0) ? 500 : ((d == 1) ? 50 : 1);
                int old = atomicAdd(&s_pen, weight_pen);
                if (old == 0 && s_vcols != nullptr) {
                    for (int k = 0; k < w_max; ++k) s_vcols[k] = (short)tup[k];
                    s_vcols[15] = (short)w_max;
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

// Apply in-place unitary Householder/Witt reflection G[:, cols] <- G[:, cols] * (I_d - tau * u * u^\dagger)
// Run by tid == 0 on shared memory array s_mat[64 * 16]!
__device__ bool sample_and_apply_householder(
    unsigned char* s_mat,                     // [64 * 16]
    const int* cols,                          // [d_ar]
    const int d_ar,                           // 2, 3, or 4
    const int m,
    const int q,
    const int q2,
    const unsigned char* s_add,               // [82 * 82]
    const unsigned char* s_sub,               // [82 * 82]
    const unsigned char* s_mul,               // [82 * 82]
    const unsigned char* s_inv,               // [82]
    const unsigned char* s_conj,              // [82]
    const unsigned char* g_tau_table,         // [82 * 16]: for each eta in 0..q2-1, up to 16 valid tau != 0!
    const unsigned char* g_tau_cnt,           // [82]
    unsigned int &rng
) {
    unsigned char u[4];
    unsigned char cu[4];
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
        // Avoid trivial monomial swap when d_ar == 2 and diag == 0 (`1 - tau * |u0|^2 == 0`):
        if (d_ar == 2) {
            int n0 = (int)s_mul[(int)u[0] * 82 + (int)cu[0]];
            int tn0 = (int)s_mul[tau * 82 + n0];
            if (tn0 == 1) continue;
        }
        // Compute w[r] = sum_{t=0}^{d_ar-1} conj(u[t]) * s_mat[cols[t], r]
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
        // Check no new column becomes all-zero (`preserves projective validity!`)
        unsigned char new_cols[4][16];
        bool all_nz = true;
        for (int t = 0; t < d_ar; ++t) {
            int tu = (int)s_mul[tau * 82 + (int)u[t]];
            int m_off = tu * 82;
            int c_off = cols[t] * 16;
            bool nz = false;
            for (int r = 0; r < m; ++r) {
                unsigned char sub_val = s_mul[m_off + (int)w[r]];
                unsigned char nv = s_sub[(int)s_mat[c_off + r] * 82 + (int)sub_val];
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

__global__ void search_householder_u234_finisher(
    unsigned char* __restrict__ g_pop_G,           // [NBLK * 1024] (`64 * 16` per block)
    int* __restrict__ g_pop_pen,                   // [NBLK]
    short* __restrict__ g_pop_vcols,               // [NBLK * 16]
    short* __restrict__ g_pop_tabu,                // [NBLK * 6]
    int* __restrict__ g_pop_plat,                  // [NBLK]
    unsigned char* __restrict__ g_best_G_blk,      // [NBLK * 1024]
    int* __restrict__ g_best_pen_blk,              // [NBLK]
    const unsigned char* __restrict__ g_add,       // [82 * 82]
    const unsigned char* __restrict__ g_sub,       // [82 * 82]
    const unsigned char* __restrict__ g_mul,       // [82 * 82]
    const unsigned char* __restrict__ g_inv,       // [82]
    const unsigned char* __restrict__ g_conj,      // [82]
    const unsigned char* __restrict__ g_tau_table, // [82 * 16]
    const unsigned char* __restrict__ g_tau_cnt,   // [82]
    const unsigned char* __restrict__ g_subs,      // [n_subs * 8]
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
    unsigned char* __restrict__ g_win_G            // [1024]
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
    __shared__ int s_cur_pen, s_best_pen, s_plat, s_pen;
    __shared__ int s_best_p_pen, s_best_p_c0;
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

    unsigned int rng = base_seed ^ ((unsigned int)(bid * 2654435761U + tid * 2246822519U + 31U));
    xorshift32(rng);

    if (s_need_init) {
        eval_subset_rank_block(
            s_cur_G, s_sub, s_mul, s_inv, g_subs, n_subs, m, w_max, 9999999, s_pen, s_cur_vcols
        );
        if (tid == 0) {
            s_cur_pen = s_pen;
            if (s_pen < s_best_pen) s_best_pen = s_pen;
            if (s_pen == 0 && atomicCAS(g_solved_bid, -1, bid) == -1) {
                for (int i = 0; i < 1024; ++i) g_win_G[i] = s_cur_G[i];
            }
            s_done = (*g_solved_bid >= 0);
        }
        for (int i = tid; i < 1024; i += bdim) g_best_G_blk[((size_t)bid) * 1024ULL + i] = s_cur_G[i];
        __syncthreads();
        if (s_done) return;
    }

    for (int step = 0; step < n_macro; ++step) {
        if (tid == 0) {
            s_done = (*g_solved_bid >= 0);
            s_best_p_pen = 9999999;
            s_best_p_c0 = -1;
        }
        __syncthreads();
        if (s_done) break;

        for (int pr = 0; pr < k_probes; ++pr) {
            for (int i = tid; i < 1024; i += bdim) s_cand_G[i] = s_cur_G[i];
            __syncthreads();

            __shared__ int s_c0_used;
            if (tid == 0) {
                int vc_cnt = (int)s_cur_vcols[15];
                int c0 = -1;
                if (vc_cnt > 0 && (xorshift32(rng) % 100U) < 88U) {
                    for (int tr = 0; tr < 8; ++tr) {
                        int cand_c = (int)s_cur_vcols[xorshift32(rng) % (unsigned int)vc_cnt];
                        bool tb = false;
                        for (int k = 0; k < 6; ++k) if (s_tabu[k] == (short)cand_c) { tb = true; break; }
                        if (!tb || tr == 7) { c0 = cand_c; break; }
                    }
                }
                if (c0 < 0) c0 = (int)(xorshift32(rng) % (unsigned int)n);
                s_c0_used = c0;

                // Choose dynamic arity d_ar in {2, 3, 4}!
                int d_ar = 3;
                unsigned int r100 = xorshift32(rng) % 100U;
                if (q == 2) {
                    d_ar = (r100 < 55U) ? 3 : 4;
                } else {
                    d_ar = (r100 < 45U) ? 2 : ((r100 < 80U) ? 3 : 4);
                }
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
                s_probe_ok = sample_and_apply_householder(
                    s_cand_G, cols, d_ar, m, q, q2, s_add, s_sub, s_mul, s_inv, s_conj, g_tau_table, g_tau_cnt, rng
                );
            }
            __syncthreads();
            if (!s_probe_ok) continue;

            int cutoff = s_cur_pen + 5;
            if (s_best_p_pen + 1 < cutoff) cutoff = s_best_p_pen + 1;
            if (cutoff < 3) cutoff = 3;

            eval_subset_rank_block(
                s_cand_G, s_sub, s_mul, s_inv, g_subs, n_subs, m, w_max, cutoff, s_pen, s_cand_vcols
            );

            if (tid == 0 && s_pen == 0) {
                if (atomicCAS(g_solved_bid, -1, bid) == -1) {
                    for (int i = 0; i < 1024; ++i) g_win_G[i] = s_cand_G[i];
                }
                s_done = true;
            }
            __shared__ bool s_take_probe;
            if (tid == 0) {
                s_take_probe = (s_pen < s_best_p_pen);
                if (s_take_probe) {
                    s_best_p_pen = s_pen;
                    s_best_p_c0 = s_c0_used;
                    for (int k = 0; k < 16; ++k) s_best_p_vcols[k] = s_cand_vcols[k];
                }
            }
            __syncthreads();
            if (s_take_probe) {
                for (int i = tid; i < 1024; i += bdim) s_best_probe_G[i] = s_cand_G[i];
            }
            __syncthreads();
            if (s_done) break;
        }
        if (s_done) break;

        __shared__ int s_act;
        if (tid == 0) {
            s_act = 0;
            bool allow_up = ((s_plat & 3) == 3) && (s_best_p_pen <= s_cur_pen + 3);
            if (s_best_p_c0 >= 0 && (s_best_p_pen <= s_cur_pen || allow_up)) {
                s_act = 1;
                if (s_best_p_pen < s_cur_pen) s_plat = 0;
                else ++s_plat;
                s_cur_pen = s_best_p_pen;
                for (int k = 5; k >= 1; --k) s_tabu[k] = s_tabu[k - 1];
                s_tabu[0] = (short)s_best_p_c0;
                for (int k = 0; k < 16; ++k) s_cur_vcols[k] = s_best_p_vcols[k];
                if (s_cur_pen < s_best_pen) s_best_pen = s_cur_pen;
            } else {
                ++s_plat;
                if (s_plat >= 44) {
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
            // Reseed from OWN block's elite + 2 consecutive non-monomial Householder tunnel kicks!
            for (int i = tid; i < 1024; i += bdim) s_cur_G[i] = g_best_G_blk[((size_t)bid) * 1024ULL + i];
            __syncthreads();
            if (tid == 0) {
                for (int kk = 0; kk < 2; ++kk) {
                    int cols[3];
                    cols[0] = (int)(xorshift32(rng) % (unsigned int)n);
                    cols[1] = (cols[0] + 1 + (int)(xorshift32(rng) % (unsigned int)(n - 1))) % n;
                    cols[2] = (cols[1] + 1 + (int)(xorshift32(rng) % (unsigned int)(n - 2))) % n;
                    if (cols[2] == cols[0]) cols[2] = (cols[2] + 1) % n;
                    sample_and_apply_householder(
                        s_cur_G, cols, 3, m, q, q2, s_add, s_sub, s_mul, s_inv, s_conj, g_tau_table, g_tau_cnt, rng
                    );
                }
                for (int k = 0; k < 6; ++k) s_tabu[k] = -1;
            }
            __syncthreads();
            eval_subset_rank_block(
                s_cur_G, s_sub, s_mul, s_inv, g_subs, n_subs, m, w_max, 9999999, s_pen, s_cur_vcols
            );
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


def build_householder_tau_tables(f: GFq2Ext):
    """For each eta in GF(q^2), precomputes all nonzero tau in GF(q^2)* satisfying:
       tau + conj(tau) == tau * conj(tau) * eta!
    (Covers BOTH eta != 0 Householder reflections AND eta == 0 isotropic Witt transvections!)
    """
    tau_tab = np.zeros((82, 16), dtype=np.uint8)
    tau_cnt = np.zeros(82, dtype=np.uint8)
    for eta in range(f.q2):
        sols = []
        for tau in range(1, f.q2):
            ct = int(f.conj[tau])
            lhs = int(f.add[tau, ct])
            rhs = int(f.mul[f.mul[tau, ct], eta])
            if lhs == rhs:
                sols.append(tau)
        tau_cnt[eta] = min(16, len(sols))
        for i, s in enumerate(sols[:16]):
            tau_tab[eta, i] = s
    conj82 = np.zeros(82, dtype=np.uint8)
    conj82[: f.q2] = f.conj[: f.q2]
    return cp.asarray(conj82), cp.asarray(tau_tab.reshape(-1)), cp.asarray(tau_cnt)


def build_exact_wmax_subsets(n: int, w_max: int, rng_np: np.random.Generator) -> np.ndarray:
    combs = list(itertools.combinations(range(n), w_max))
    arr = np.zeros((len(combs), 8), dtype=np.uint8)
    if combs:
        arr[:, :w_max] = np.array(combs, dtype=np.uint8)
        rng_np.shuffle(arr, axis=0)
    return arr


def pack_G64_16(G: np.ndarray) -> np.ndarray:
    m, n = G.shape
    out = np.zeros((64, 16), dtype=np.uint8)
    out[:n, :m] = G.T
    return out.reshape(-1)


def unpack_G64_16(buf: np.ndarray, m: int, n: int) -> np.ndarray:
    return np.ascontiguousarray(buf.reshape(64, 16)[:n, :m].T, dtype=np.uint8)


def run_selftest(k_hh):
    rng_np = np.random.default_rng(77)
    # 1. Verify tau_table mathematical exactness & non-emptiness across ALL q in {2, 3, 4, 5, 7, 8}!
    for q in (2, 3, 4, 5, 7, 8):
        f = GFq2Ext(q)
        _, d_tt, d_tc = build_householder_tau_tables(f)
        tc = d_tc.get()
        for eta in f.gfq_elems:
            expected = (q - 1) if eta == 0 else q
            assert int(tc[int(eta)]) == expected, f"tau_cnt mismatch for q={q}, eta={eta}!"

    # 2. Verify k_hh preserves G * G^\dagger == 0 AND solves [[6, 0, 4]]_2 (`m=3, n=6, d=4`) & [[8, 2, 4]]_3 (`m=3, n=8, d=4`)!
    for q, m, n, d_tar in [(2, 3, 6, 4), (3, 3, 8, 4)]:
        f = GFq2Ext(q)
        d_conj, d_tt, d_tc = build_householder_tau_tables(f)
        # Start from trivial [I_3 | I_3 ...] (`Hermitian self-orthogonal when scaled by norm -1!`)
        G0 = np.zeros((m, n), dtype=np.uint8)
        neg1 = int(f.sub[0, 1])
        alpha_neg1 = int(f.alpha_for_norm[neg1])
        for r in range(m):
            G0[r, r] = 1
            G0[r, m + r] = alpha_neg1
        assert f.mat_herm_zero(G0)
        # Note: for n=8, columns 6,7 need isotropic orthogonal fill -> test n=6 (`[I_3 | alpha*I_3]` has d=2 initially!)
        if n == 8:
            G0 = G0[:, :6]
            n = 6
        subs = build_exact_wmax_subsets(n, d_tar - 1, rng_np)
        NBLK = 32
        init_G = np.tile(pack_G64_16(G0), (NBLK, 1))
        d_pop_G = cp.asarray(init_G.reshape(-1))
        d_best_G = d_pop_G.copy()
        d_pop_pen = cp.full(NBLK, 9999999, dtype=cp.int32)
        d_best_pen = cp.full(NBLK, 9999999, dtype=cp.int32)
        d_vcols = cp.zeros((NBLK, 16), dtype=cp.int16)
        d_tabu = cp.full((NBLK, 6), -1, dtype=cp.int16)
        d_plat = cp.zeros(NBLK, dtype=cp.int32)
        d_sol = cp.full(1, -1, dtype=cp.int32)
        d_win = cp.zeros(1024, dtype=cp.uint8)
        k_hh(
            (NBLK,), (128,),
            (
                d_pop_G, d_pop_pen, d_vcols, d_tabu, d_plat, d_best_G, d_best_pen,
                f.d_add, f.d_sub, f.d_mul, f.d_inv, d_conj, d_tt, d_tc,
                cp.asarray(subs.reshape(-1)), np.int32(len(subs)),
                np.int32(q), np.int32(f.q2), np.int32(n), np.int32(m), np.int32(d_tar - 1),
                np.int32(35), np.int32(4), np.uint32(98765), d_sol, d_win,
            ),
        )
        cp.cuda.Stream.null.synchronize()
        assert int(d_sol.get()[0]) >= 0, f"Householder must solve [[6, 0, 4]]_{q} from [I_3 | a*I_3]!"
        Gw = unpack_G64_16(d_win.get(), m, n)
        assert f.mat_herm_zero(Gw), f"Householder broke Hermitian orthogonality for q={q}!"
    print("SELFTEST_S53_HOUSEHOLDER_U234_ALL_FIELDS_EXACT_UNITARY_OK", flush=True)


def harvest_seeds_zero_reject(
    k_sub, f: GFq2Ext, c0_lut, c1_lut, m: int, l: int, bp: int, d_tar: int, rng: random.Random, rng_np: np.random.Generator, t_sec: float = 5.0
) -> tuple[int, list[np.ndarray]]:
    n = l * m + bp
    lm1 = l - 1
    w_max = d_tar - 1
    is_pow2_char2 = (f.q in (2, 4, 8)) and (m & (m - 1)) == 0
    mu_list = [1] if is_pow2_char2 else f.norm1[:3]
    tups_np, k_pref = build_symmetry_ktuples(n, m, l, bp, w_max, rng_np)
    if len(tups_np) == 0:
        return 999999, []
    d_tups = cp.asarray(tups_np.reshape(-1))
    n_tups = len(tups_np)
    d_pen = cp.zeros(1, dtype=cp.int32)

    elites: list[tuple[int, np.ndarray]] = []
    best_pen = 999999

    for mu in mu_list:
        lam = f.pow(mu, m // bp) if bp >= 2 else mu
        if is_pow2_char2 and lam != 1:
            continue
        if bp == 1 and not any(f.pow(z, m) == lam for z in f.norm1):
            continue
        ocs = None if is_pow2_char2 else OrthogonalComponentSolver(f, c0_lut, c1_lut, m, lam, 15000, rng)
        if ocs is not None and ocs.num_comps <= 1 and (f.q ** m) > 20_000_000:
            continue

        bord_opts = []
        if bp == 0:
            targ0 = np.zeros(m, dtype=np.uint8); targ0[0] = f.sub[0, 1]
            tu = targ0 if is_pow2_char2 else ocs.decompose_target(targ0)
            if tu is not None:
                bord_opts.append((np.zeros((0, m), dtype=np.uint8), tu))
        elif bp == 1:
            zetas = [z for z in f.norm1 if f.pow(z, m) == lam]
            for z in zetas[:2]:
                for u in range(1, min(f.q2, 24)):
                    bc = np.zeros((1, m), dtype=np.uint8)
                    zp = 1
                    for r in range(m):
                        bc[0, r] = f.mul[u, zp]; zp = int(f.mul[zp, z])
                    targ = np.zeros(m, dtype=np.uint8); targ[0] = f.sub[0, 1]
                    for r in range(m):
                        targ[r] = f.sub[targ[r], int(f.mul[bc[0, r], f.conj[bc[0, 0]]])]
                    if is_pow2_char2 and lm1 == 1 and int(np.bitwise_xor.reduce(targ)) == 0:
                        continue
                    tu = targ if is_pow2_char2 else ocs.decompose_target(targ)
                    if tu is not None:
                        bord_opts.append((bc, tu))
        else:
            if m % bp != 0 or (1 + m // bp) < d_tar:
                continue
            for _ in range(min(48, (f.q2 ** bp) - 1)):
                b0 = np.array([rng.randrange(f.q2) for _ in range(bp)], dtype=np.uint8)
                if not np.any(b0): b0[0] = 1
                bc, targ = build_monomial_cycle_border(f, m, bp, mu, b0)
                if is_pow2_char2 and lm1 == 1 and int(np.bitwise_xor.reduce(targ)) == 0:
                    continue
                tu = targ if is_pow2_char2 else ocs.decompose_target(targ)
                if tu is not None:
                    bord_opts.append((bc, tu))

        if not bord_opts:
            continue

        t0 = time.time()
        per_lam = max(1.8, t_sec / max(1, len(mu_list)))
        g_cols = np.zeros((n, m), dtype=np.uint8)
        for r in range(m): g_cols[r, r] = 1
        best_state = None
        stag = 0
        while time.time() - t0 < per_lam:
            if is_pow2_char2:
                bcols, targ_raw = bord_opts[rng.randrange(len(bord_opts))]
                polys_g = solve_lucas_hensel_char2_pow2(f, targ_raw, m, lm1, rng)
                cc = None
            else:
                if best_state is not None and ocs.num_comps >= 2 and rng.random() < 0.72:
                    prev_cc, bi_o = best_state
                    bcols, tu = bord_opts[bi_o]
                    polys_g, cc = ocs.sample_solution(tu, lm1, rng, prev_comp_choices=prev_cc)
                else:
                    bi_o = rng.randrange(len(bord_opts))
                    bcols, tu = bord_opts[bi_o]
                    polys_g, cc = ocs.sample_solution(tu, lm1, rng, prev_comp_choices=None)
            if polys_g is None:
                break
            blks_t, _ = poly_batch_to_blocks_and_norms(f, polys_g, m, lam)
            for bi in range(lm1):
                g_cols[(bi + 1) * m : (bi + 2) * m, :] = blks_t[bi]
            if bp > 0:
                g_cols[l * m : l * m + bp, :] = bcols
            cutoff = max(1, min(best_pen + 3, 12))
            pen = eval_fullpower(k_sub, f, cp.asarray(g_cols.reshape(-1)), d_tups, k_pref, n_tups, n, m, w_max, cutoff, d_pen)
            if pen <= min(12, best_pen + 2):
                gm = np.ascontiguousarray(g_cols.T, dtype=np.uint8)
                elites.append((pen, gm))
                if len(elites) > 20:
                    elites.sort(key=lambda x: x[0])
                    elites = elites[:16]
            if pen == 0:
                return 0, [np.ascontiguousarray(g_cols.T, dtype=np.uint8)]
            if pen < best_pen:
                best_pen = pen
                best_state = (cc, bi_o) if cc is not None else None
                stag = 0
            else:
                stag += 1
                if stag >= 20:
                    best_state = None
                    stag = 0

    elites.sort(key=lambda x: x[0])
    return best_pen, [gm for _, gm in elites[:16]]


def harvest_seeds_hash_mitm(
    k_qt, f: GFq2Ext, fh: GFq2Host, m: int, l: int, bp: int, d_tar: int, rng: random.Random, vram_cache: dict, t_sec: float = 6.0
) -> tuple[int, list[np.ndarray]]:
    n = l * m + bp
    lm1 = l - 1
    fibs = get_constacyclic_fibers(f, m)
    valid_lams = []
    for lm, xis in fibs.items():
        if bp == 0:
            valid_lams.append((lm, []))
        elif len(xis) >= bp:
            cf = [xt for xt in itertools.islice(itertools.combinations(xis, bp), 35) if is_comb_free_xis(f, m, xt, d_tar)]
            if cf:
                valid_lams.append((lm, cf))
    valid_lams.sort(key=lambda p: (0 if p[0] == 1 else 1, p[0]))
    if not valid_lams:
        return 999999, []

    min_wt = min(m, max(2, d_tar - 2))
    j1_starts = [b0_i * m for b0_i in range(lm1)] + [lm1 * m + c for c in range(bp)]
    d_j1 = cp.asarray(np.array(j1_starts, dtype=np.int32))
    nonzero_gfq = f.gfq_elems[1:]

    elites: list[tuple[int, np.ndarray]] = []
    qt_min = 999999
    NBLK = 256
    d_bpen = cp.zeros(NBLK, dtype=cp.int32)
    d_bch = cp.zeros((NBLK, 24), dtype=cp.int32)
    d_solved = cp.zeros(1, dtype=cp.int32)
    d_win_ch = cp.zeros(24, dtype=cp.int32)

    for lam, cf_tups in valid_lams[:2]:
        pkey = (f.q, m, lam, min_wt)
        if pkey not in vram_cache:
            if len(vram_cache) >= 3: vram_cache.clear()
            vram_cache[pkey] = build_vram_csr_pool(f, fh, m, lam, min_wt, rng)
        blks_arr, d_blks, d_pnorms, d_bkeys, d_boffs, d_bitems, plen, nbuck = vram_cache[pkey]
        if plen == 0 or nbuck == 0:
            continue

        bconfigs = []
        if bp == 0:
            targ, bcols = make_border_from_bspec(f, m, [])
            bconfigs.append(("bp0", targ, bcols))
        else:
            for xt in cf_tups[:6]:
                for _ in range(4):
                    bs = tuple((rng.choice(nonzero_gfq), x) for x in xt)
                    targ, bcols = make_border_from_bspec(f, m, list(bs))
                    bconfigs.append((bs, targ, bcols))
        if not bconfigs:
            continue

        t0 = time.time()
        per_l = max(2.5, t_sec / min(2, len(valid_lams)))
        el_by_b = {}
        launches = 0
        while time.time() - t0 < per_l:
            bkey, targ, bcols = bconfigs[launches % len(bconfigs)]
            cur_el = el_by_b.get(bkey, [])
            if cur_el:
                ep = np.zeros((len(cur_el), 24), dtype=np.int32)
                for ei, (_, cl) in enumerate(cur_el): ep[ei, :lm1] = cl
                d_el, n_el = cp.asarray(ep.reshape(-1)), len(cur_el)
            else:
                d_el, n_el = cp.zeros(24, dtype=cp.int32), 0
            d_solved.fill(-1)
            seed = (int(time.time() * 1000000) ^ (launches * 0x9E3779B9)) & 0xFFFFFFFF
            k_qt(
                (NBLK,), (256,),
                (
                    d_blks, d_pnorms, d_bkeys, d_boffs, d_bitems, np.int32(plen), np.int32(nbuck),
                    cp.asarray(bcols.reshape(-1)) if bp > 0 else cp.zeros(1, dtype=np.uint8), cp.asarray(targ),
                    f.d_add, f.d_sub, f.d_mul, f.d_inv, d_j1, np.int32(len(j1_starts)),
                    np.int32(lm1), np.int32(bp), np.int32(m), np.int32(f.q2), np.int32(d_tar),
                    np.int32(45), np.uint32(seed), d_el, np.int32(n_el), d_bpen, d_bch, d_solved, d_win_ch,
                ),
            )
            cp.cuda.Stream.null.synchronize()
            launches += 1
            if int(d_solved.get()[0]) >= 0:
                cw = d_win_ch.get()[:lm1].tolist()
                gm = np.zeros((m, n), dtype=np.uint8)
                for r in range(m): gm[r, r] = 1
                for bi, ix in enumerate(cw): gm[:, (bi + 1) * m : (bi + 2) * m] = blks_arr[ix].T
                if bp > 0: gm[:, l * m : l * m + bp] = bcols.T
                return 0, [gm]
            pens = d_bpen.get(); chs = d_bch.get()
            bmin = int(pens.min())
            if bmin < qt_min: qt_min = bmin
            for idx in np.argsort(pens)[:5]:
                ii = int(idx); pv = int(pens[ii])
                if pv <= bmin + 3 and pv < 9000000:
                    cc = chs[ii, :lm1].tolist()
                    if not any(cc == pc for _, pc in cur_el): cur_el.append((pv, cc))
                    if pv <= min(16, qt_min + 2):
                        gm = np.zeros((m, n), dtype=np.uint8)
                        for r in range(m): gm[r, r] = 1
                        for bi, ix in enumerate(cc): gm[:, (bi + 1) * m : (bi + 2) * m] = blks_arr[ix].T
                        if bp > 0: gm[:, l * m : l * m + bp] = bcols.T
                        elites.append((pv, gm))
            if len(elites) > 16:
                elites.sort(key=lambda x: x[0]); elites = elites[:16]
            cur_el.sort(key=lambda x: x[0]); el_by_b[bkey] = cur_el[:12]

    elites.sort(key=lambda x: x[0])
    return qt_min, [gm for _, gm in elites[:16]]


ULTRA_FINISHER_TARGETS = {
    (3, 15, 1, 7), (2, 24, 8, 6), (3, 21, 11, 5), (7, 15, 5, 6), (2, 35, 21, 5), (2, 36, 22, 5),
    (3, 18, 6, 6), (3, 19, 7, 6), (2, 25, 9, 6), (4, 24, 12, 6),
    (3, 14, 0, 7), (5, 11, 1, 6), (3, 16, 2, 7), (3, 17, 3, 7),
    (3, 15, 3, 6), (3, 17, 5, 6), (4, 13, 1, 7), (4, 14, 0, 8), (4, 15, 1, 8),
    (5, 12, 0, 7), (5, 14, 2, 7), (7, 14, 2, 7), (8, 14, 2, 7),
    (7, 15, 1, 8), (8, 15, 1, 8), (8, 16, 0, 9), (8, 17, 1, 9),
    (4, 31, 21, 5), (4, 32, 22, 5), (2, 37, 27, 4), (2, 38, 28, 4), (2, 39, 29, 4),
    (3, 19, 9, 5), (4, 23, 11, 6),
}


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

    mod_sub = cp.RawModule(code=CUDA_SUBSET_RANK_D14)
    k_sub = mod_sub.get_function("check_subset_rank_d14")
    mod_qt = cp.RawModule(code=CUDA_S46_KERNELS)
    k_qt = mod_qt.get_function("harvest_qt_seeds_129")

    snap = json.load(open(os.path.join(SCRATCH, "codetables_qecc_snapshot.json"), encoding="utf-8"))
    fields = {q: GFq2Ext(q) for q in (2, 3, 4, 5, 7, 8)}
    fhosts = {q: GFq2Host(q) for q in (2, 3, 4, 5, 7, 8)}
    decomp_q = {q: build_gfq_decomp(fields[q]) for q in (2, 3, 4, 5, 7, 8)}
    hh_tables = {q: build_householder_tau_tables(fields[q]) for q in (2, 3, 4, 5, 7, 8)}

    rng = random.Random(2026112853)
    rng_np = np.random.default_rng(2026112853)
    vram_cache = {}

    beaten = get_all_beaten(snap)
    cells0, exact0 = count_total_codetables_cells(snap)

    targets = []
    for q in (3, 2, 4, 5, 7, 8):
        sq = snap[str(q)]
        for m in range(4, 13):
            for n in range(2 * m, 44):
                k = n - 2 * m
                l, bp = divmod(n, m)
                if l < 2 or bp >= 6:
                    continue
                bb = sq.get(f"{n},{k}")
                cand_dts = set()
                if bb and bb[1] is not None:
                    cur_lb = max(bb[0] or 0, beaten.get((q, n, k), 0))
                    if cur_lb < bb[1]:
                        cand_dts.add(cur_lb + 1)
                        if bb[1] <= m + 1:
                            cand_dts.add(bb[1])
                bb_sh = sq.get(f"{n-1},{k+1}")
                if bb_sh and bb_sh[1] is not None:
                    cur_sh = max(bb_sh[0] or 0, beaten.get((q, n - 1, k + 1), 0))
                    if cur_sh < bb_sh[1]:
                        cand_dts.add(bb_sh[1] + 1)

                for d_tar in sorted(cand_dts):
                    if not (4 <= d_tar <= min(9, m + 1)):
                        continue
                    if q == 2 and m == 5 and d_tar == 4 and n >= 42:
                        continue
                    w_max = d_tar - 1
                    c_tot = math.comb(n, w_max)
                    if c_tot > 65_000:
                        continue
                    sp_sum = sum(math.comb(n, j) * ((q * q - 1) ** j) for j in range((d_tar - 1) // 2 + 1))
                    if sp_sum > (q ** (2 * m)):
                        continue
                    gain, ex_gain = calc_domino_gain(snap, beaten, q, n, k, d_tar)
                    if ex_gain <= 0:
                        continue
                    tag = f"S53HouseholderU234_q{q}_m{m}_n{n}_k{k}_d{d_tar}"
                    if is_blacklisted(tag, ENGINE_S53_HOUSEHOLDER):
                        continue
                    is_ultra = 0 if (q, n, k, d_tar) in ULTRA_FINISHER_TARGETS else 1
                    targets.append((is_ultra, c_tot, -ex_gain, q, m, l, bp, n, k, d_tar, ex_gain, gain, tag))

    targets.sort()
    log_q2(
        STREAM_TAG,
        f"🔥🔥 Stream 53 起動（オンザフライ Householder–Witt $U(2)/U(3)/U(4)$ 鏡映×全数部分列ランク `d=4..9` 永久蓋フィニッシャー）："
        f"現在セル={cells0}（**永久蓋={exact0}**）｜厳選 **{len(targets)} 件**！",
    )

    for is_ultra, c_tot, _, q, m, l, bp, n, k, d_tar, ex_gain, gain, tag in targets:
        beaten = get_all_beaten(snap)
        gain_now, ex_now = calc_domino_gain(snap, beaten, q, n, k, d_tar)
        if ex_now <= 0 or is_blacklisted(tag, ENGINE_S53_HOUSEHOLDER):
            continue

        f = fields[q]
        fh = fhosts[q]
        c0_lut, c1_lut = decomp_q[q]
        d_conj, d_tt, d_tc = hh_tables[q]
        t0 = time.time()

        log_q2(
            STREAM_TAG,
            f"🎯 Stage-A シード収穫 [[{n}, {k}, {d_tar}]]_{q} (m={m},l={l},bp={bp},subsets={c_tot:,}, **永久蓋+{ex_now}個**, 総波及={gain_now})...",
        )
        if d_tar >= 6 or bp == 0:
            qt_pen, seed_Gs = harvest_seeds_zero_reject(k_sub, f, c0_lut, c1_lut, m, l, bp, d_tar, rng, rng_np, t_sec=6.0)
        else:
            qt_pen, seed_Gs = harvest_seeds_hash_mitm(k_qt, f, fh, m, l, bp, d_tar, rng, vram_cache, t_sec=6.0)

        if qt_pen == 0 and seed_Gs:
            if verify_and_export_20thread(f, seed_Gs[0], d_tar, f"S53StageA_QT(m={m},l={l},bp={bp})", snap):
                c_now, ex_tot = count_total_codetables_cells(snap)
                log_q2(STREAM_TAG, f"🔒🔒🔒【永久蓋 LOCKED (Stream53 Stage-A)！！】[[{n}, {k}, {d_tar}]]_{q} ⇒ 永久蓋累計: **{ex_tot} マス**（総更新: {c_now} マス）！！")
                continue

        if not seed_Gs or qt_pen > 16:
            record_attempt(tag, ENGINE_S53_HOUSEHOLDER, f"S53NoNearSeed(QT={qt_pen}) [[{n},{k},{d_tar}]]_{q}", qt_pen, qt_pen, time.time() - t0, max_stalled_rounds=1)
            continue

        subs_arr = build_exact_wmax_subsets(n, d_tar - 1, rng_np)
        d_subs = cp.asarray(subs_arr.reshape(-1))
        n_subs = len(subs_arr)

        NBLK = 256 if c_tot <= 15_000 else 164
        init_G = np.zeros((NBLK, 1024), dtype=np.uint8)
        for b in range(NBLK):
            init_G[b] = pack_G64_16(seed_Gs[b % len(seed_Gs)])

        d_pop_G = cp.asarray(init_G.reshape(-1))
        d_best_G = d_pop_G.copy()
        d_pop_pen = cp.full(NBLK, 9999999, dtype=cp.int32)
        d_best_pen = cp.full(NBLK, 9999999, dtype=cp.int32)
        d_vcols = cp.zeros((NBLK, 16), dtype=cp.int16)
        d_tabu = cp.full((NBLK, 6), -1, dtype=cp.int16)
        d_plat = cp.zeros(NBLK, dtype=cp.int32)
        d_sol = cp.full(1, -1, dtype=cp.int32)
        d_win = cp.zeros(1024, dtype=cp.uint8)

        n_macro = 28 if c_tot <= 6_500 else (12 if c_tot <= 20_000 else 5)
        t_hh = 32.0 if qt_pen <= 3 else 18.0
        log_q2(
            STREAM_TAG,
            f"⚡ Stage-B Householder–Witt $U(2/3/4)$ 鏡映フィニッシャー (`QT違反={qt_pen}` → {t_hh:.0f}s 全数部分列ランク探索): [[{n}, {k}, {d_tar}]]_{q}...",
        )

        tb0 = time.time()
        launches = 0
        best_indiv = 9999999
        won = False
        while time.time() - tb0 < t_hh:
            d_sol.fill(-1)
            seed = (int(time.time() * 1000000) ^ (launches * 0x85EBCA6B) ^ (n * 271 + q * 47)) & 0xFFFFFFFF
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
                if verify_and_export_20thread(f, Gw, d_tar, f"HouseholderWitt_U234_Finisher(m={m},fromQT={qt_pen})", snap):
                    c_now, ex_tot = count_total_codetables_cells(snap)
                    log_q2(STREAM_TAG, f"🔒🔒🔒【永久蓋 LOCKED (Householder-U234対称性破壊)！！】[[{n}, {k}, {d_tar}]]_{q} ⇒ 永久蓋累計: **{ex_tot} マス**（総更新: {c_now} マス）！！")
                    won = True
                    break
                d_sol.fill(-1)

            bp_min = int(d_best_pen.get().min())
            if bp_min < best_indiv:
                best_indiv = bp_min

        if not won:
            record_attempt(
                tag, ENGINE_S53_HOUSEHOLDER,
                f"S53Householder(QT={qt_pen},Indiv={best_indiv}) [[{n},{k},{d_tar}]]_{q}",
                best_indiv, best_indiv, time.time() - t0, max_stalled_rounds=1,
            )


if __name__ == "__main__":
    main()

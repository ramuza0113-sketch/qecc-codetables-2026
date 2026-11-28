r"""【第2段階・Stream 42（v1.2.0 永久蓋 専用ハンター①：全6体 `q=2,3,4,5,7,8` × `d_up in {4,5,6,7}` VRAM常駐 656ブロック MonomialCycleQT・VRAMPoolQT・CombFreeVandermonde・等ノルム手術 統合スナイパー）】
  方針転換：ここからは「`d_low = d_up`（理論上限ピタリの永久蓋 / Exact Optimal）が閉まるマス（`exact_gain >= 1`）」のみを 100% 狙い撃つ！

  【旧 Stream 11/12/13 から解放した 5 大盲点】：
  1. **`d_tar == 4` (`w_max == 3`) 専用超高速オンチップ分岐を新設**：`[[37..86, n-10, 4]]_2`（45個の永久蓋！）や `[[15..17, n-6, 4]]_4`, `[[27,21,4]]_5`, `[[51,45,4]]_7` を毎秒 1,000万符号の速度で走査！
  2. **`q = 7` (`GF(49)`) および `n = 54..96`（`l` 最大 22 ブロック：`s_ch[24], s_cols[96*20]`）を全面解放**：旧スクリプトで抜けていた `q=7, m=5`（46個の永久蓋！）や `q=8, m=5`（30個の永久蓋！）を一挙解禁！
  3. **CUDA オンチップ「1ブロック等ノルム手術（50%）＋ 2ブロック・ノルム交換手術（50%）」**：エリート解から 1 ブロックだけを同一ノルムバケット内の別多項式へ差し替える微細手術により `MIN_VIOL = 1..3` を確実に仕留める！
  4. **4種の境界表現を単一カーネルに完全統合**：
     - `bp = 0`（全 `lam in Norm_1` Constacyclic）
     - `MonomialCycleQT`（`bp >= 2, bp | m`, `1 + m/bp >= d_tar`）
     - `CombFreeEchar / VandermondeQT`（`bp >= 1`, 部分周期櫛 `bp + (e-bp+1)(m/e) >= d_tar` 検証済み！）
     - `UnitaryIdealQT`（`m` が素数かつ `gcd(m, q+1)=1` の `[[18,4,6]]_2`, `[[23,9,6]]_2`, `[[16,2,7]]_3` 等に対する自己双対イデアル正規直交化境界、`j1_starts = 全列` で偽陽性ゼロ！）
  5. **純粋量子球充填限界 (`Sphere-Packing`) ＆ `exact_gain >= 1` 厳密フィルタ**：永久蓋が 1 個以上確実に閉まる実現可能ターゲットのみを最前線順に高速周回（足踏み 1 パス即ブラックリスト厳守）！
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
from stage2_qecc_s14_algebraic_and_gpu_mp_plotkin import decompose_constacyclic_minimal_ideals
from stage2_qecc_s22_fullpower_lucas_hensel_and_d10_d13_slam import verify_and_export_20thread
from stage2_qecc_s38_multieigenmode_vandermonde_border_qt import build_vandermonde_eigen_border
from step2_gpu80_qecc_100cells import count_total_codetables_cells
from stagnation_guard import is_blacklisted, record_attempt

ENGINE_S42_EXACT_QT = "STAGE2_QECC_EXACT_OPTIMAL_VRAM656_QT_PERMANENT_LID"
STREAM_TAG = "Stream42-ExactLid-VRAM656QT"


CUDA_VRAM_POOL_EXACT_QT = r'''
extern "C" {
__device__ __forceinline__ unsigned int xorshift32(unsigned int &state) {
    unsigned int x = state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    state = x; return x;
}

__device__ __forceinline__ unsigned long long pack_norm64(const unsigned char* v, int m) {
    unsigned long long h = 14695981039346656037ULL;
    for (int i = 0; i < m; ++i) {
        h ^= (unsigned long long)v[i];
        h *= 1099511628211ULL;
    }
    return h;
}

__global__ void search_vram_pool_exact_qt(
    const unsigned char* __restrict__ g_pool_blks, // [plen * m * m]
    const unsigned char* __restrict__ g_pnorms,    // [plen * m]
    const unsigned long long* __restrict__ g_bkeys,// [n_buckets]
    const int* __restrict__ g_boffs,               // [n_buckets + 1]
    const int* __restrict__ g_bitems,              // [total_items]
    const int plen,
    const int n_buckets,
    const unsigned char* __restrict__ g_bcols,     // [bp * m]
    const unsigned char* __restrict__ g_targ,      // [m]
    const unsigned char* __restrict__ g_add,
    const unsigned char* __restrict__ g_sub,
    const unsigned char* __restrict__ g_mul,
    const unsigned char* __restrict__ g_inv,
    const int* __restrict__ g_j1_starts,
    const int n_starts,
    const int lm1,                                 // up to 24!
    const int bp,
    const int m,                                   // up to 20!
    const int q2,
    const int d_tar,                               // 4, 5, 6, or 7!
    const int n_trials,
    const unsigned int base_seed,
    const int* __restrict__ g_elites,              // [n_elites * 24]
    const int n_elites,
    int* __restrict__ g_best_pen_blk,              // [gridDim.x]
    int* __restrict__ g_best_ch_blk,               // [gridDim.x * 24]
    int* __restrict__ g_solved_bid,
    int* __restrict__ g_win_ch                     // [24]
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
    __shared__ unsigned char s_cols[96 * 20];
    __shared__ unsigned char s_targ[20];
    __shared__ int s_ch[24];
    __shared__ int s_best_ch[24];
    __shared__ int s_best_pen;
    __shared__ int s_pen;
    __shared__ bool s_valid;
    __shared__ bool s_done;

    for (int i = tid; i < 82 * 82; i += bdim) {
        s_add[i] = g_add[i];
        s_sub[i] = g_sub[i];
        s_mul[i] = g_mul[i];
    }
    if (tid < 82) s_inv[tid] = g_inv[tid];
    if (tid < m) s_targ[tid] = g_targ[tid];
    for (int i = tid; i < bp * m; i += bdim) {
        s_cols[(lm1 * m) * m + i] = g_bcols[i];
    }
    if (tid == 0) {
        s_done = (*g_solved_bid >= 0);
        s_best_pen = 999999;
    }
    __syncthreads();
    if (s_done) return;

    unsigned int rng = base_seed ^ ((unsigned int)(bid * 257 + tid + 1) * 2654435761U);

    for (int tr = 0; tr < n_trials; ++tr) {
        if (tid == 0) {
            s_done = (*g_solved_bid >= 0);
            s_valid = false;
            if (!s_done) {
                for (int attempt = 0; attempt < 28; ++attempt) {
                    // 1) Try 1-Block Same-Norm Unitary Surgery (50% when elites exist)!
                    if (n_elites > 0 && attempt < 10 && (xorshift32(rng) & 1U) == 0U) {
                        int ei = (int)(xorshift32(rng) % (unsigned int)n_elites);
                        for (int b = 0; b < lm1; ++b) s_ch[b] = g_elites[ei * 24 + b];
                        int pos = (int)(xorshift32(rng) % (unsigned int)lm1);
                        int old_ix = s_ch[pos];
                        const unsigned char* old_nrm = &g_pnorms[old_ix * m];
                        unsigned long long kh0 = pack_norm64(old_nrm, m);
                        int lo0 = 0, hi0 = n_buckets - 1, f0 = -1;
                        while (lo0 <= hi0) {
                            int mid0 = (lo0 + hi0) >> 1;
                            unsigned long long mk0 = g_bkeys[mid0];
                            if (mk0 == kh0) { f0 = mid0; break; }
                            if (mk0 < kh0) lo0 = mid0 + 1;
                            else hi0 = mid0 - 1;
                        }
                        if (f0 >= 0) {
                            int st0 = g_boffs[f0], en0 = g_boffs[f0 + 1];
                            int cnt0 = en0 - st0;
                            if (cnt0 >= 2) {
                                int pick0 = g_bitems[st0 + (int)(xorshift32(rng) % (unsigned int)cnt0)];
                                if (pick0 != old_ix) {
                                    bool ex0 = true;
                                    for (int r = 0; r < m; ++r) {
                                        if (g_pnorms[pick0 * m + r] != old_nrm[r]) { ex0 = false; break; }
                                    }
                                    if (ex0) {
                                        s_ch[pos] = pick0;
                                        s_valid = true;
                                        break;
                                    }
                                }
                            }
                        }
                    }

                    // 2) 2-Block Norm-Exchange Elite Surgery OR Fresh Random Norm Closure!
                    int close_pos = lm1 - 1;
                    unsigned char rem[20];
                    for (int r = 0; r < m; ++r) rem[r] = s_targ[r];

                    if (n_elites > 0 && lm1 >= 2 && (xorshift32(rng) & 3U) != 0U) {
                        int ei = (int)(xorshift32(rng) % (unsigned int)n_elites);
                        close_pos = (int)(xorshift32(rng) % (unsigned int)lm1);
                        for (int b = 0; b < lm1; ++b) s_ch[b] = g_elites[ei * 24 + b];
                        int mut_pos = (close_pos + 1 + (int)(xorshift32(rng) % (unsigned int)(lm1 - 1))) % lm1;
                        s_ch[mut_pos] = (int)(xorshift32(rng) % (unsigned int)plen);
                        if (lm1 >= 4 && (xorshift32(rng) & 7U) == 0U) {
                            int mut2 = (close_pos + 1 + (int)(xorshift32(rng) % (unsigned int)(lm1 - 1))) % lm1;
                            s_ch[mut2] = (int)(xorshift32(rng) % (unsigned int)plen);
                        }
                        for (int b = 0; b < lm1; ++b) {
                            if (b == close_pos) continue;
                            int ix = s_ch[b];
                            for (int r = 0; r < m; ++r) {
                                rem[r] = s_sub[(int)rem[r] * 82 + g_pnorms[ix * m + r]];
                            }
                        }
                    } else {
                        for (int b = 0; b < lm1 - 1; ++b) {
                            int ix = (int)(xorshift32(rng) % (unsigned int)plen);
                            s_ch[b] = ix;
                            for (int r = 0; r < m; ++r) {
                                rem[r] = s_sub[(int)rem[r] * 82 + g_pnorms[ix * m + r]];
                            }
                        }
                    }

                    unsigned long long kh = pack_norm64(rem, m);
                    int lo = 0, hi = n_buckets - 1, found = -1;
                    while (lo <= hi) {
                        int mid = (lo + hi) >> 1;
                        unsigned long long mk = g_bkeys[mid];
                        if (mk == kh) { found = mid; break; }
                        if (mk < kh) lo = mid + 1;
                        else hi = mid - 1;
                    }
                    if (found >= 0) {
                        int st_o = g_boffs[found];
                        int en_o = g_boffs[found + 1];
                        int cnt = en_o - st_o;
                        if (cnt > 0) {
                            int pick = g_bitems[st_o + (int)(xorshift32(rng) % (unsigned int)cnt)];
                            bool exact_m = true;
                            for (int r = 0; r < m; ++r) {
                                if (g_pnorms[pick * m + r] != rem[r]) { exact_m = false; break; }
                            }
                            if (exact_m) {
                                s_ch[close_pos] = pick;
                                s_valid = true;
                                break;
                            }
                        }
                    }
                }
                s_pen = 0;
            }
        }
        __syncthreads();
        if (s_done) break;
        if (!s_valid) continue;

        int bulk_elems = lm1 * m * m;
        for (int idx = tid; idx < bulk_elems; idx += bdim) {
            int blk = idx / (m * m);
            int rem_i = idx % (m * m);
            int ix = s_ch[blk];
            s_cols[idx] = g_pool_blks[ix * (m * m) + rem_i];
        }
        __syncthreads();

        int pen_cap = (s_best_pen < 50) ? (s_best_pen + 4) : ((w_max <= 4) ? 40 : 12);

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

        // d_tar == 4 (`w_max == 3`): ultra-fast s = 3 (`wt == 0`) forced scalar check!
        if (s_pen < pen_cap && w_max <= 3) {
            int s2_outer = n_starts * n_b * (q2 - 1);
            for (int idx = tid; idx < s2_outer && s_pen < pen_cap; idx += bdim) {
                int a2 = 1 + (idx % (q2 - 1));
                int tmp = idx / (q2 - 1);
                int j2 = tmp % n_b;
                int j1 = g_j1_starts[tmp / n_b];
                if (j2 <= j1) continue;
                unsigned char u12[20];
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
            // d_tar in {5, 6, 7} (`w_max in {4, 5, 6}`):
            int s3_outer = n_starts * n_b * (q2 - 1);
            for (int idx = tid; idx < s3_outer && s_pen < pen_cap; idx += bdim) {
                int a2 = 1 + (idx % (q2 - 1));
                int tmp = idx / (q2 - 1);
                int j2 = tmp % n_b;
                int j1 = g_j1_starts[tmp / n_b];
                if (j2 <= j1) continue;

                unsigned char u12[20];
                int a2_row = a2 * 82;
                for (int r = 0; r < m; ++r) {
                    u12[r] = s_add[(int)s_cols[j1 * m + r] * 82 + s_mul[a2_row + s_cols[j2 * m + r]]];
                }

                for (int j3 = j2 + 1; j3 < n_b && s_pen < pen_cap; ++j3) {
                    for (int a3 = 1; a3 < q2 && s_pen < pen_cap; ++a3) {
                        unsigned char u123[20];
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
                                    unsigned char u4[20];
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
                                                unsigned char u5[20];
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

        if (tid == 0) {
            if (s_pen == 0) {
                s_best_pen = 0;
                int old = atomicCAS(g_solved_bid, -1, bid);
                if (old == -1) {
                    for (int b = 0; b < lm1; ++b) g_win_ch[b] = s_ch[b];
                }
                s_done = true;
            } else if (s_pen < s_best_pen) {
                s_best_pen = s_pen;
                for (int b = 0; b < lm1; ++b) s_best_ch[b] = s_ch[b];
            }
        }
        __syncthreads();
        if (s_done) break;
    }

    if (tid == 0) {
        g_best_pen_blk[bid] = s_best_pen;
        if (s_best_pen < 999999) {
            for (int b = 0; b < lm1; ++b) g_best_ch_blk[bid * 24 + b] = s_best_ch[b];
        }
    }
}
}
'''


def is_comb_free_xis(f: GFq2Ext, m: int, xis: tuple[int, ...], d_tar: int) -> bool:
    bp = len(xis)
    if bp == 0:
        return True
    divs = [e for e in range(1, m + 1) if m % e == 0]
    x0 = xis[0]
    e_min = m
    for e in divs:
        p0 = f.pow(x0, e)
        if all(f.pow(xs, e) == p0 for xs in xis[1:]):
            e_min = e
            break
    if bp > e_min:
        return False
    return (bp + (e_min - bp + 1) * (m // e_min)) >= d_tar


def build_selfdual_ideal_unitary_borders(f: GFq2Ext, m: int, bp: int, lam: int, rng: random.Random) -> list[tuple[str, np.ndarray, np.ndarray, str]]:
    """Constructs non-monomial Unitary-Ideal borders `B` (`m x bp`) via Hermitian Gram congruence `B^dagger B = c * I_{bp}`
    on non-degenerate ideals of `GF(q^2)[x]/(x^m - lam)` (unlocks prime `m` with `gcd(m, q+1) == 1`!).
    """
    if bp <= 1 or math.gcd(m, f.q) != 1:
        return []
    orbs = decompose_constacyclic_minimal_ideals(f, m, lam, rng)
    dims = [ob.shape[0] for ob in orbs]
    out = []
    for r_sub in range(1, len(orbs) + 1):
        for comb in itertools.islice(itertools.combinations(range(len(orbs)), r_sub), 30):
            if sum(dims[i] for i in comb) != bp:
                continue
            B0 = np.vstack([orbs[i] for i in comb]).T.copy()  # m x bp
            # Gram matrix `G = B0^dagger * B0` (`bp x bp`)
            G = np.zeros((bp, bp), dtype=np.uint8)
            for i in range(bp):
                for j in range(bp):
                    acc = 0
                    for r in range(m):
                        acc = int(f.add[acc, f.mul[f.conj[B0[r, i]], B0[r, j]]])
                    G[i, j] = acc
            # Hermitian congruence `P^dagger G P = diag(d_0..d_{bp-1})`
            P = np.eye(bp, dtype=np.uint8)
            H = G.copy()
            ok = True
            for k_idx in range(bp):
                if H[k_idx, k_idx] == 0:
                    # Find nonzero off-diagonal `H[i_nz, k_idx] != 0` and add scalar multiple to make diagonal nonzero!
                    found_piv = False
                    for i_nz in range(k_idx + 1, bp):
                        if H[i_nz, i_nz] != 0:
                            H[[k_idx, i_nz]] = H[[i_nz, k_idx]]
                            H[:, [k_idx, i_nz]] = H[:, [i_nz, k_idx]]
                            P[:, [k_idx, i_nz]] = P[:, [i_nz, k_idx]]
                            found_piv = True
                            break
                    if not found_piv:
                        for i_nz in range(k_idx + 1, bp):
                            if H[k_idx, i_nz] != 0:
                                for alpha in range(1, f.q2):
                                    # new diag = alpha*H[i_nz, k] + conj(alpha)*H[k, i_nz] + |alpha|^2 H[i_nz, i_nz]
                                    t1 = int(f.mul[alpha, H[i_nz, k_idx]])
                                    t2 = int(f.mul[f.conj[alpha], H[k_idx, i_nz]])
                                    diag_new = int(f.add[t1, t2])
                                    if diag_new != 0:
                                        P[:, k_idx] = f.add[P[:, k_idx], f.mul[alpha, P[:, i_nz]]]
                                        # Recompute H = P^dagger G P cleanly
                                        for a in range(bp):
                                            for b in range(bp):
                                                ac2 = 0
                                                for u in range(bp):
                                                    for v in range(bp):
                                                        ac2 = int(f.add[ac2, f.mul[f.conj[P[u, a]], f.mul[G[u, v], P[v, b]]]])
                                                H[a, b] = ac2
                                        found_piv = True
                                        break
                                break
                    if not found_piv or H[k_idx, k_idx] == 0:
                        ok = False
                        break
                dk_inv = int(f.inv[H[k_idx, k_idx]])
                for j_el in range(k_idx + 1, bp):
                    if H[k_idx, j_el] != 0:
                        fac = int(f.sub[0, f.mul[H[k_idx, j_el], dk_inv]])
                        P[:, j_el] = f.add[P[:, j_el], f.mul[fac, P[:, k_idx]]]
                for a in range(bp):
                    for b in range(bp):
                        ac2 = 0
                        for u in range(bp):
                            for v in range(bp):
                                ac2 = int(f.add[ac2, f.mul[f.conj[P[u, a]], f.mul[G[u, v], P[v, b]]]])
                        H[a, b] = ac2
            if not ok:
                continue
            # Now H is diagonal with nonzero GF(q)* diagonal entries! Scale each col `j` so `H[j,j] == c_scale`!
            for c_sc in f.gfq_elems[1:4]:
                B_unit = np.zeros((m, bp), dtype=np.uint8)
                for j_el in range(bp):
                    ratio_g = int(f.mul[c_sc, f.inv[H[j_el, j_el]]])
                    alpha_j = int(f.alpha_for_norm[ratio_g])
                    col_p = f.mul[alpha_j, P[:, j_el]]
                    for r in range(m):
                        acc = 0
                        for u in range(bp):
                            acc = int(f.add[acc, f.mul[B0[r, u], col_p[u]]])
                        B_unit[r, j_el] = acc
                # Verify `HB = B_unit * B_unit^dagger` is `lam`-constacyclic!
                HB0 = np.zeros(m, dtype=np.uint8)
                for r in range(m):
                    acc = 0
                    for j_el in range(bp):
                        acc = int(f.add[acc, f.mul[B_unit[r, j_el], f.conj[B_unit[0, j_el]]]])
                    HB0[r] = acc
                is_cc = True
                for r in range(m):
                    for c in range(m):
                        acc = 0
                        for j_el in range(bp):
                            acc = int(f.add[acc, f.mul[B_unit[r, j_el], f.conj[B_unit[c, j_el]]]])
                        # Note `fh.build_block` convention: `HA[r, c] = HA[r-c, 0]` if `r>=c` else `conj(lam)*HA[m+r-c, 0]`
                        exp = HB0[r - c] if r >= c else int(f.mul[f.conj[lam], HB0[m + r - c]])
                        if acc != exp:
                            is_cc = False
                            break
                    if not is_cc:
                        break
                if is_cc:
                    targ = np.zeros(m, dtype=np.uint8)
                    targ[0] = f.sub[0, 1]
                    for r in range(m):
                        targ[r] = f.sub[targ[r], HB0[r]]
                    bcols = np.ascontiguousarray(B_unit.T, dtype=np.uint8)
                    out.append((f"uid_{comb}_{c_sc}", targ, bcols, f"UnitaryIdeal(comb={comb},sc={c_sc})"))
                    if len(out) >= 8:
                        return out
    return out


def main():
    try:
        hp = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.kernel32.SetPriorityClass(hp, 0x00000020)
        ctypes.windll.kernel32.SetProcessAffinityMask(hp, (1 << 24) - 1)
    except Exception:
        pass
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)

    snap = json.load(open(os.path.join(SCRATCH, "codetables_qecc_snapshot.json"), encoding="utf-8"))
    fields = {q: GFq2Ext(q) for q in (2, 3, 4, 5, 7, 8)}
    fhosts = {q: GFq2Host(q) for q in (2, 3, 4, 5, 7, 8)}
    mod = cp.RawModule(code=CUDA_VRAM_POOL_EXACT_QT)
    k_qt = mod.get_function("search_vram_pool_exact_qt")
    rng = random.Random(2026112842)

    beaten = get_all_beaten(snap)
    cells0, exact0 = count_total_codetables_cells(snap)

    # Collect ALL pure-feasible EVEN-`r_S = 2m` targets with `exact_gain >= 1` and `d_tar in {4, 5, 6, 7}`!
    cands = []
    for q in (2, 3, 4, 5, 7, 8):
        f = fields[q]
        sq = snap[str(q)]
        for m in range(3, 19):
            fibs = get_constacyclic_fibers(f, m)
            for l in range(2, 22):
                for bp in range(0, min(7, m)):
                    n = l * m + bp
                    k = (l - 2) * m + bp
                    nb = (l - 1) * m + bp
                    if k <= 0 or n < 11 or n > 98 or nb > 95 or (l - 1) > 23:
                        continue
                    bb = sq.get(f"{n},{k}")
                    if not bb:
                        continue
                    cur_lb = max(bb[0], beaten.get((q, n, k), bb[0]))
                    if cur_lb >= bb[1]:
                        continue
                    d_tar = cur_lb + 1
                    if d_tar not in (4, 5, 6, 7) or (d_tar - 1) > m:
                        continue
                    # Pure Sphere-Packing feasibility check:
                    t_pack = (d_tar - 1) // 2
                    vol = sum(math.comb(n, w) * ((q * q - 1) ** w) for w in range(t_pack + 1))
                    if vol > (q ** (2 * m)):
                        continue
                    gain, ex_gain = calc_domino_gain(snap, beaten, q, n, k, d_tar)
                    if ex_gain <= 0:
                        continue

                    w_max = d_tar - 1
                    ev = math.comb(n, w_max) * ((q * q - 1) ** (w_max - 1)) / (q ** (2 * m))
                    if ev > 1800.0:
                        continue
                    if d_tar == 7 and (n > 28 or q > 5 or ev > 350.0):
                        continue

                    # Sweet-spot tiering: prioritize `slack = m - (d_tar - 1) >= 1` (non-QMDS!) where QT permanent lids abound!
                    is_qmds = 1 if m == (d_tar - 1) else 0
                    if is_qmds == 0 and ((d_tar in (4, 5) and ev <= 350.0) or (d_tar == 6 and ev <= 260.0)):
                        tier = 0
                    elif is_qmds == 0 and d_tar in (4, 5, 6) and ev <= 950.0:
                        tier = 1
                    elif is_qmds == 0:
                        tier = 2
                    else:
                        tier = 3

                    # Mode A: MonomialCycleQT (`bp >= 2`, `m % bp == 0`, AND `1 + m // bp >= d_tar`!)
                    if bp >= 2 and (m % bp == 0) and (1 + m // bp) >= d_tar:
                        for mu in ([1] + [u for u in f.norm1 if u != 1][:3]):
                            lam = f.pow(mu, m // bp)
                            tag = f"S42ExMono_q{q}_m{m}_l{l}_bp{bp}_mu{mu}_n{n}_k{k}_d{d_tar}"
                            if not is_blacklisted(tag, ENGINE_S42_EXACT_QT):
                                cands.append((tier, d_tar, ev, n, -ex_gain, "mono", q, m, l, bp, mu, lam, [], k, tag))

                    # Mode B: Comb-Free Eigencharacter / Vandermonde QT (`bp == 0, 1` or comb-free `bp >= 2`)!
                    valid_lams = []
                    for lm, xis in fibs.items():
                        if bp == 0:
                            valid_lams.append((lm, []))
                        elif len(xis) >= bp:
                            cf_tuples = [xt for xt in itertools.islice(itertools.combinations(xis, bp), 35) if is_comb_free_xis(f, m, xt, d_tar)]
                            if cf_tuples:
                                valid_lams.append((lm, cf_tuples))
                    valid_lams.sort(key=lambda pair: (0 if pair[0] == 1 else 1, pair[0]))
                    for lam, cf_tups in valid_lams[:4]:
                        tag = f"S42ExEch_q{q}_m{m}_l{l}_bp{bp}_lam{lam}_n{n}_k{k}_d{d_tar}"
                        if not is_blacklisted(tag, ENGINE_S42_EXACT_QT):
                            cands.append((tier, d_tar, ev, n, -ex_gain, "echar", q, m, l, bp, 1, lam, cf_tups, k, tag))

                    # Mode C: Non-Monomial Unitary-Ideal Border (`bp >= 2`, small `n <= 32` when `m % bp != 0` and no comb-free echar!)
                    if bp >= 2 and not valid_lams and not (m % bp == 0 and (1 + m // bp) >= d_tar) and n <= 32 and math.gcd(m, q) == 1:
                        for lam in f.norm1[:2]:
                            tag = f"S42ExUId_q{q}_m{m}_l{l}_bp{bp}_lam{lam}_n{n}_k{k}_d{d_tar}"
                            if not is_blacklisted(tag, ENGINE_S42_EXACT_QT):
                                cands.append((tier, d_tar, ev, n, -ex_gain, "uideal", q, m, l, bp, 1, lam, [], k, tag))

    cands.sort(key=lambda x: x[:5])
    log_q2(
        STREAM_TAG,
        f"🔥🔥 Stream 42 起動（永久蓋 `d_low=d_up` 専用 VRAM656 MonomialCycleQT・CombFreeEchar・UnitaryIdeal 統合ハンター）："
        f"現在セル={cells0}（**永久蓋={exact0}**）｜厳選 **{len(cands)} 件**！",
    )

    vram_cache = {}
    d_solved = cp.zeros(1, dtype=cp.int32)
    d_win_ch = cp.zeros(24, dtype=cp.int32)
    corridor_failed_ns: dict[tuple[int, int, int], set[int]] = {}
    corridor_won_ns: dict[tuple[int, int, int], set[int]] = {}

    for tier, d_tar, _, n, _, bmode, q, m, l, bp, mu, lam, cf_tups, k, tag in cands:
        ckey = (q, m, d_tar)
        # Corridor Ceiling Guard: if >= 3 smaller `n`s in `(q, m, d_tar)` failed completely above any won `n`, skip larger `n`!
        max_won = max(corridor_won_ns.get(ckey, {0}))
        pure_fails = corridor_failed_ns.get(ckey, set()) - corridor_won_ns.get(ckey, set())
        fails_above = [fn for fn in pure_fails if max_won < fn < n]
        if len(fails_above) >= 3:
            continue
        # Exact PG(4, 4) cap ceiling: `m_2(4, 4) == 41` (no pure `[n, n-5, 4]_4` exists for `n >= 42`!)
        if q == 2 and m == 5 and d_tar == 4 and n >= 42:
            continue

        beaten = get_all_beaten(snap)
        gain_now, ex_now = calc_domino_gain(snap, beaten, q, n, k, d_tar)
        if ex_now <= 0 or is_blacklisted(tag, ENGINE_S42_EXACT_QT):
            continue

        f = fields[q]
        fh = fhosts[q]
        min_wt = min(m, max(2, d_tar - 2))
        pkey = (q, m, lam, min_wt)
        if pkey not in vram_cache:
            if len(vram_cache) >= 3:
                vram_cache.clear()
            vram_cache[pkey] = build_vram_csr_pool(f, fh, m, lam, min_wt, rng)
        blks_arr, d_blks, d_pnorms, d_bkeys, d_boffs, d_bitems, plen, nbuck = vram_cache[pkey]
        if plen == 0 or nbuck == 0:
            continue

        lm1 = l - 1
        nb = lm1 * m + bp
        if bmode == "uideal":
            j1_starts = list(range(nb))  # non-monomial border requires all columns as starts!
        else:
            j1_starts = [b0_i * m for b0_i in range(lm1)] + [lm1 * m + c for c in range(bp)]
        d_j1 = cp.asarray(np.array(j1_starts, dtype=np.int32))
        nonzero_gfq = f.gfq_elems[1:]

        border_configs = []
        if bp == 0:
            targ, bcols = make_border_from_bspec(f, m, [])
            border_configs.append(("bp0", targ, bcols, "bp=0"))
        elif bmode == "mono":
            for _ in range(22):
                b0 = np.array([rng.randint(0, f.q2 - 1) for _ in range(bp)], dtype=np.uint8)
                if not np.any(b0):
                    b0[0] = 1
                bcols, targ = build_monomial_cycle_border(f, m, bp, mu, b0)
                border_configs.append((tuple(b0.tolist()), targ, bcols, f"Mono(mu={mu},b0={b0.tolist()})"))
        elif bmode == "echar":
            seen_bs = set()
            rng_tups = list(cf_tups)
            rng.shuffle(rng_tups)
            for xt in rng_tups[:8]:
                for _ in range(10):
                    bs = tuple((rng.choice(nonzero_gfq), x) for x in xt)
                    if bs not in seen_bs:
                        seen_bs.add(bs)
                        targ, bcols = make_border_from_bspec(f, m, list(bs))
                        border_configs.append((bs, targ, bcols, f"CombFreeEchar(bspec={list(bs)})"))
                    if len(border_configs) >= 18:
                        break
                if len(border_configs) >= 18:
                    break
        else:
            border_configs = build_selfdual_ideal_unitary_borders(f, m, bp, lam, rng)

        if not border_configs:
            record_attempt(tag, ENGINE_S42_EXACT_QT, f"S42Ex({bmode}) [[{n},{k},{d_tar}]]_{q}", 999999, 999999, 0.1, max_stalled_rounds=1)
            continue

        NBLK = 328 if d_tar <= 5 else 164
        n_tr = 64 if d_tar == 4 else (28 if d_tar == 5 else (6 if d_tar == 6 else 2))
        d_bpen = cp.zeros(NBLK, dtype=cp.int32)
        d_bch = cp.zeros((NBLK, 24), dtype=cp.int32)

        log_q2(
            STREAM_TAG,
            f"🎯 永久蓋狙撃 [{bmode}:m={m},l={l},bp={bp},lam={lam}] [[{n}, {k}, {d_tar}]]_{q} (**永久蓋+{ex_now}個**, 総波及={gain_now})...",
        )
        t0 = time.time()
        t_lim = 6.0 if tier == 3 else (22.0 if (ex_now >= 2 or tier == 0) else 14.0)
        global_min = 999999
        elites_by_bkey = {}
        launches = 0
        won = False

        while time.time() - t0 < t_lim:
            bkey, targ, bcols, bdesc = border_configs[launches % len(border_configs)]
            d_targ = cp.asarray(targ)
            d_bcols = cp.asarray(bcols.reshape(-1)) if bp > 0 else cp.zeros(1, dtype=np.uint8)

            cur_elites = elites_by_bkey.get(bkey, [])
            if cur_elites:
                el_pad = np.zeros((len(cur_elites), 24), dtype=np.int32)
                for ei_idx, (_, ch_l) in enumerate(cur_elites):
                    el_pad[ei_idx, :lm1] = ch_l
                d_el = cp.asarray(el_pad.reshape(-1))
                n_el = len(cur_elites)
            else:
                d_el = cp.zeros(24, dtype=cp.int32)
                n_el = 0

            d_solved.fill(-1)
            seed = (int(time.time() * 1000000) ^ (launches * 0x9E3779B9) ^ (n * 131 + lam * 31 + q * 17)) & 0xFFFFFFFF
            k_qt(
                (NBLK,), (256,),
                (
                    d_blks, d_pnorms, d_bkeys, d_boffs, d_bitems,
                    np.int32(plen), np.int32(nbuck),
                    d_bcols, d_targ,
                    f.d_add, f.d_sub, f.d_mul, f.d_inv,
                    d_j1, np.int32(len(j1_starts)), np.int32(lm1), np.int32(bp),
                    np.int32(m), np.int32(f.q2), np.int32(d_tar), np.int32(n_tr), np.uint32(seed),
                    d_el, np.int32(n_el),
                    d_bpen, d_bch, d_solved, d_win_ch,
                ),
            )
            cp.cuda.Stream.null.synchronize()
            launches += 1

            if int(d_solved.get()[0]) >= 0:
                ch_w = d_win_ch.get()[:lm1].tolist()
                g_mat = np.zeros((m, n), dtype=np.uint8)
                for r in range(m):
                    g_mat[r, r] = 1
                for bi, ix in enumerate(ch_w):
                    g_mat[:, (bi + 1) * m : (bi + 2) * m] = blks_arr[ix].T
                if bp > 0:
                    g_mat[:, l * m : l * m + bp] = bcols.T
                m_str = f"ExactOptimal_VRAM656QT(m={m},l={l},bp={bp},lam={lam},{bdesc})"
                if verify_and_export_20thread(f, g_mat, d_tar, m_str, snap):
                    c_tot_now, ex_tot_now = count_total_codetables_cells(snap)
                    log_q2(STREAM_TAG, f"🔒🔒🔒【永久蓋 LOCKED！！】[[{n}, {k}, {d_tar}]]_{q} ⇒ 永久蓋累計: **{ex_tot_now} マス**（総更新: {c_tot_now} マス）！！")
                    corridor_won_ns.setdefault(ckey, set()).add(n)
                    won = True
                    break
                d_solved.fill(-1)

            pens = d_bpen.get()
            chs = d_bch.get()
            bmin = int(pens.min())
            if bmin < global_min:
                global_min = bmin

            for idx in np.argsort(pens)[:8]:
                ii = int(idx)
                pv = int(pens[ii])
                if pv <= bmin + 3 and pv < 900000:
                    ch_cand = chs[ii, :lm1].tolist()
                    if not any(ch_cand == prev_c for _, prev_c in cur_elites):
                        cur_elites.append((pv, ch_cand))
            cur_elites.sort(key=lambda x: x[0])
            elites_by_bkey[bkey] = cur_elites[:14]

        if not won:
            corridor_failed_ns.setdefault(ckey, set()).add(n)
            tot_codes = launches * NBLK * n_tr
            record_attempt(
                tag, ENGINE_S42_EXACT_QT,
                f"S42ExQT({bmode}) [[{n},{k},{d_tar}]]_{q} ({tot_codes:,})",
                global_min, global_min, time.time() - t0, max_stalled_rounds=1,
            )


if __name__ == "__main__":
    main()

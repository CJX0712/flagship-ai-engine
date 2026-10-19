/**
 * 旗舰AI引擎 · 非负矩阵分解（NMF）
 * 站在：Lee & Seung 1999《Learning the Parts of Objects by Non-negative Matrix
 *       Factorization》(Nature) · Lee & Seung 2001（NIPS，乘性更新算法）
 * 模型：V ≈ W·H，V ∈ ℝ^{m×n}_+，W ∈ ℝ^{m×k}_+，H ∈ ℝ^{k×n}_+（全部非负）。
 * 乘性更新（欧氏距离版，Lee & Seung 2001 定理 1/2）：
 *   H ← H ⊙ (WᵀV) / (WᵀWH + ε)      W ← W ⊙ (VHᵀ) / (WHHᵀ + ε)
 * 该更新的核心性质：**目标 ‖V − WH‖²_F 单调不增**（辅助函数/EM 式证明），
 * 且非负性天然保持（乘性、全正因子）——这是它区别于普通梯度下降的根本优势。
 * 不变量：
 *   ① 重构误差 ‖V − WH‖²_F 单调不增（Lee & Seung 2001 定理）
 *   ② 非负性全程保持（乘性更新的结构性保证）
 *   ③ 精确恢复：由 V = W₀H₀ 构造的秩匹配数据，误差 → 0
 *   ④ Eckart–Young 下界：NMF 误差 ≥ 同秩 SVD 最优逼近误差（非负约束只会更差不会更好）
 *   ⑤ 收敛到不动点：末步更新量 → 0（驻点条件）
 */

import { zeros, transpose, matmul, svd } from '../core/linalg.js';
import { mulberry32 } from '../core/stats.js';

/** 矩阵逐元素最大绝对差 */
function matMaxAbs(A, B) {
  let m = 0;
  for (let i = 0; i < A.length; i++) {
    for (let j = 0; j < A[i].length; j++) m = Math.max(m, Math.abs(A[i][j] - B[i][j]));
  }
  return m;
}

/**
 * NMF 分解（Lee-Seung 乘性更新）。
 * @param {number[][]} V 非负数据矩阵 m×n
 * @param {number} k 分解秩
 * @returns {{W:number[][], H:number[][], errors:number[], steps:number[], k:number}}
 */
export function nmf(V, k, { iters = 500, eps = 1e-9, seed = 42, W0 = null, H0 = null } = {}) {
  const m = V.length, n = V[0].length;
  const rng = mulberry32(seed);
  let W, H;
  if (W0 && H0) {
    W = W0.map((r) => [...r]);
    H = H0.map((r) => [...r]);
  } else {
    // 正随机初始化（严格 > 0，否则乘性更新会被"冻结"在 0）
    W = zeros(m, k);
    H = zeros(k, n);
    for (let i = 0; i < m; i++) for (let a = 0; a < k; a++) W[i][a] = 0.1 + rng() * 0.9;
    for (let a = 0; a < k; a++) for (let j = 0; j < n; j++) H[a][j] = 0.1 + rng() * 0.9;
  }
  const errors = [];
  const steps = [];
  /** 平方 Frobenius 重构误差 ‖V − WH‖²_F（Lee-Seung 目标函数本身） */
  const sqErr = () => {
    const WH = matmul(W, H);
    let s = 0;
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) s += (V[i][j] - WH[i][j]) ** 2;
    }
    return s;
  };
  errors.push(sqErr());
  for (let t = 0; t < iters; t++) {
    const WtV = matmul(transpose(W), V);
    const WtWH = matmul(matmul(transpose(W), W), H);
    for (let a = 0; a < k; a++) {
      for (let j = 0; j < n; j++) H[a][j] *= WtV[a][j] / (WtWH[a][j] + eps);
    }
    const VHt = matmul(V, transpose(H));
    const WHHt = matmul(matmul(W, H), transpose(H));
    for (let i = 0; i < m; i++) {
      for (let a = 0; a < k; a++) W[i][a] *= VHt[i][a] / (WHHt[i][a] + eps);
    }
    errors.push(sqErr());
    steps.push(relativeStep(W, VHt, WHHt, eps));
  }
  return { W, H, errors, steps, k };
}

/** 当前步的相对更新量（驻点判据：乘性因子偏离 1 的最大幅度） */
function relativeStep(W, VHt, WHHt, eps) {
  let worst = 0;
  for (let i = 0; i < W.length; i++) {
    for (let a = 0; a < W[i].length; a++) {
      const factor = VHt[i][a] / (WHHt[i][a] + eps);
      worst = Math.max(worst, Math.abs(1 - factor));
    }
  }
  return worst;
}

/** 重构误差：‖V − WH‖²_F */
export function nmfReconstructionError(V, W, H) {
  const WH = matmul(W, H);
  let s = 0;
  for (let i = 0; i < V.length; i++) {
    for (let j = 0; j < V[i].length; j++) s += (V[i][j] - WH[i][j]) ** 2;
  }
  return s;
}

/** 非负性检查：返回最负元素（应 ≥ 0） */
export function nmfMinEntry(...mats) {
  let mn = Infinity;
  for (const M of mats) {
    for (const row of M) for (const v of row) mn = Math.min(mn, v);
  }
  return mn;
}

/**
 * 同秩 SVD（截断）的最优逼近误差——Eckart–Young 下界。
 * 用于验证「NMF 因非负约束，误差只会 ≥ 无约束最优」。
 */
export function svdBestRankKError(V, k) {
  let A = V;
  let transposed = false;
  if (V.length < V[0].length) { A = transpose(V); transposed = true; }
  const { U, s } = svd(A);
  // 前 k 个奇异值保留，其余被截断 ⇒ 误差² = Σ_{i≥k} s_i²
  let drop = 0;
  for (let i = k; i < s.length; i++) drop += s[i] ** 2;
  void U; void transposed;
  return drop;
}

/**
 * 构造可精确分解的非负测试数据：V = W₀·H₀（严格非负，秩 ≤ k）。
 * @returns {{V:number[][], W0:number[][], H0:number[][]}}
 */
export function makeNMFData({ m = 8, n = 10, k = 3, seed = 42 } = {}) {
  const rng = mulberry32(seed);
  const W0 = [], H0 = [];
  for (let i = 0; i < m; i++) {
    const row = [];
    for (let a = 0; a < k; a++) row.push(0.2 + rng() * 1.8);
    W0.push(row);
  }
  for (let a = 0; a < k; a++) {
    const row = [];
    for (let j = 0; j < n; j++) row.push(0.2 + rng() * 1.8);
    H0.push(row);
  }
  const V = matmul(W0, H0);
  return { V, W0, H0 };
}

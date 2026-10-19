/**
 * 旗舰AI引擎 · 独立成分分析（FastICA，Hyvärinen & Oja 2000）
 * 站在：Comon 1994《Independent Component Analysis, A New Concept?》（可识别性:
 *       至多一个高斯源时盲源分离可辨识到置换与缩放）·
 *       Hyvärinen & Oja 2000《Independent Component Analysis: Algorithms and
 *       Applications》(Neural Networks 13(4-5))（FastICA 定点迭代）之上。
 *
 * 模型：观测 x = A·s（A 满秩混合矩阵，s 为相互独立的非高斯源），目标是恢复
 * 「 unmixing 」矩阵 W 使 ŝ = W·x 逼近 s（Comon: 在置换、符号与缩放下可辨识）。
 *
 * 流程（经典两步法）：
 *   ① 中心化 → ② 白化：协方差 C 特征分解 C = EΛEᵀ，白化矩阵 K = Λ^(−1/2)Eᵀ，
 *      白化数据 Z = Xc·Kᵀ 满足 Cov(Z) = I（精确到特征分解精度）；
 *   ③ FastICA 逐单位最大化非高斯性近似负熵 J(u) ≈ [E G(u) − E G(v)]²（v~N(0,1)），
 *      取 G(u) = log cosh u（对称、稳健），g(u) = tanh u：
 *        w⁺ ← E[z·g(wᵀz)] − E[g′(wᵀz)]·w，再 w⁺ ← (w⁺ − Σ_{j<p}(w⁺ᵀw_j)w_j)/‖·‖
 *      （Gram–Schmidt deflation 去相关 + 单位化），收敛判据 |w⁺ᵀw| ≈ 1；
 *   ④ unmixing：ŝ = (x − μ)·Kᵀ·Wᵀ。
 *
 * 不变量：
 *   ① 白化数据协方差 ≡ I（maxAbs(Cov(Z) − I) < 1e-8）
 *   ② W·Wᵀ ≡ I（deflation 保证各行单位范数且两两正交，误差 ≤ 1e-10）
 *   ③ Comon 1994 源恢复：混合 A·s 后 ICA 恢复的每个源与某真实源 |corr| > 0.99
 *   ④ 非高斯性增强：恢复源的平均 |超额峰度| > 原始混合信号（FastICA 的构造目标）
 *   ⑤ 恢复源近似去相关：Cov(ŝ) ≈ I（非对角元 → 0）
 *   ⑥ 确定性：同种子两次拟合的 W 逐位一致（算法无隐藏随机性）
 */

import { zeros, transpose, matmul, matvec, dot, norm, jacobiEig } from '../core/linalg.js';
import { mean, covarianceMatrix, mulberry32, gaussianSample } from '../core/stats.js';

/** 中心化：返回中心化数据与列均值 */
function icaCenter(X) {
  const mu = mean(X);
  const Xc = X.map((row) => row.map((v, j) => v - mu[j]));
  return { Xc, mu };
}

/**
 * 白化：对中心化数据协方差 C = EΛEᵀ 做特征分解，得 K = Λ^(−1/2)Eᵀ，
 * 使 Z = Xc·Kᵀ 的协方差为单位阵。特征值按相对下限 reg·λ_max 兜底（防数值奇异，
 * 对良态数据不触发、不影响任何精确恒等式）。
 * @returns {{Z:number[][], K:number[][], eigenvalues:number[], eigenvectors:number[][]}}
 */
function icaWhiten(Xc, { reg = 1e-12 } = {}) {
  const d = Xc[0].length;
  const C = covarianceMatrix(Xc);
  const { values, vectors } = jacobiEig(C); // 降序；vectors[k] 对应 values[k]
  let lamMax = 0;
  for (const l of values) if (l > lamMax) lamMax = l;
  if (!(lamMax > 0)) throw new Error('icaWhiten: 协方差退化（数据零方差），无法白化');
  const floor = reg * lamMax;
  const K = zeros(d, d);
  for (let k = 0; k < d; k++) {
    const inv = 1 / Math.sqrt(Math.max(values[k], floor));
    for (let i = 0; i < d; i++) K[k][i] = inv * vectors[k][i]; // K = Λ^(−1/2)·Eᵀ
  }
  const Z = matmul(Xc, transpose(K)); // Z = Xc·Kᵀ，Cov(Z) = I
  return { Z, K, eigenvalues: values, eigenvectors: vectors };
}

/** Gram–Schmidt deflation：v 减去其在已确定各行 W[0..p−1] 上的投影 */
function icaOrthogonalize(v, W, p) {
  const out = v.slice();
  for (let q = 0; q < p; q++) {
    const proj = dot(out, W[q]);
    for (let j = 0; j < out.length; j++) out[j] -= proj * W[q][j];
  }
  return out;
}

/** 确定性随机单位初始化：与已求分量正交；退化时退到标准基中最独立者 */
function icaUnitInit(rng, d, W, p) {
  for (let attempt = 0; attempt < 8; attempt++) {
    const w = new Array(d);
    for (let j = 0; j < d; j++) w[j] = gaussianSample(rng, 0, 1);
    const orth = icaOrthogonalize(w, W, p);
    const nw = norm(orth);
    if (nw > 1e-8) return orth.map((v) => v / nw);
  }
  let bestE = 0, bestNorm = -1;
  for (let e = 0; e < d; e++) {
    const w = new Array(d).fill(0);
    w[e] = 1;
    const nw = norm(icaOrthogonalize(w, W, p));
    if (nw > bestNorm) { bestNorm = nw; bestE = e; }
  }
  const w = new Array(d).fill(0);
  w[bestE] = 1;
  const nw = norm(icaOrthogonalize(w, W, p)) || 1;
  return icaOrthogonalize(w, W, p).map((v) => v / nw);
}

/**
 * FastICA deflation 主循环（Hyvärinen & Oja 2000 算法 1/2 的白化空间版）。
 * 每求出一个单位向量立即与已确定分量正交化，保证 W·Wᵀ = I。
 * @returns {{W:number[][], steps:number, converged:boolean}}
 */
function icaDeflate(Z, { components = Z[0].length, maxIter = 200, tol = 1e-12, seed = 42 } = {}) {
  const n = Z.length;
  const d = Z[0].length;
  const k = Math.min(components, d);
  const rng = mulberry32(seed);
  const W = zeros(k, d);
  let steps = 0;
  let converged = true;
  for (let p = 0; p < k; p++) {
    let w = icaUnitInit(rng, d, W, p);
    let compConv = false;
    for (let it = 0; it < maxIter && !compConv; it++) {
      steps++;
      // u_i = z_i·w；g(u) = tanh(u)；g′(u) = 1 − tanh²(u)
      const u = matvec(Z, w);
      let gPrimeMean = 0;
      const acc = new Array(d).fill(0);
      for (let i = 0; i < n; i++) {
        const t = Math.tanh(u[i]);
        gPrimeMean += 1 - t * t;
        for (let j = 0; j < d; j++) acc[j] += Z[i][j] * t;
      }
      for (let j = 0; j < d; j++) acc[j] /= n;
      gPrimeMean /= n;
      // w⁺ = E[z·g(wᵀz)] − E[g′(wᵀz)]·w（定点方程）
      const wRaw = new Array(d);
      for (let j = 0; j < d; j++) wRaw[j] = acc[j] - gPrimeMean * w[j];
      // deflation 去相关 + 单位化
      const orth = icaOrthogonalize(wRaw, W, p);
      const nw = norm(orth);
      if (nw < 1e-300) { compConv = false; break; }
      const wNew = orth.map((v) => v / nw);
      // 收敛判据：两次迭代方向一致（含符号翻转）
      compConv = Math.abs(1 - Math.abs(dot(wNew, w))) < tol;
      w = wNew;
    }
    converged = converged && compConv;
    for (let j = 0; j < d; j++) W[p][j] = w[j];
  }
  return { W, steps, converged };
}

/**
 * FastICA 拟合。
 * @param {number[][]} X 观测矩阵 n×d（行 = 样本，即混合信号）
 * @param {{components?:number, maxIter?:number, tol?:number, seed?:number, reg?:number}} [opts]
 *   components 要恢复的独立成分个数（默认 d）
 *   maxIter    每个成分的最大定点迭代次数（默认 200）
 *   tol        方向收敛阈值 |1 − |w⁺ᵀw||（默认 1e-12）
 *   seed       初始化的 PRNG 种子（默认 42，决定性与可复现）
 *   reg        白化特征值相对下限（默认 1e-12，仅防数值奇异）
 * @returns {{W:number[][], K:number[][],
 *            whitening:{mean:number[], eigenvalues:number[], eigenvectors:number[][]},
 *            components:number, steps:number, converged:boolean}}
 *   W         白化空间的 unmixing 矩阵（k×d，W·Wᵀ = I_k）
 *   K         白化矩阵（d×d）；whitening 记录中心化均值与协方差谱（icaTransform 复用）
 */
export function icaFit(X, opts = {}) {
  const { components, maxIter = 200, tol = 1e-12, seed = 42, reg = 1e-12 } = opts;
  if (X.length === 0) throw new Error('icaFit: 输入样本为空');
  const d = X[0].length;
  const { Xc, mu } = icaCenter(X);
  const { Z, K, eigenvalues, eigenvectors } = icaWhiten(Xc, { reg });
  const { W, steps, converged } = icaDeflate(Z, {
    components: components || d, maxIter, tol, seed,
  });
  return {
    W, K,
    whitening: { mean: mu, eigenvalues, eigenvectors },
    components: W.length,
    steps,
    converged,
  };
}

/**
 * 用拟合模型估计源信号：ŝ = (x − μ)·Kᵀ·Wᵀ（先白化、再 unmixing）。
 * @param {number[][]} X 观测矩阵 n×d
 * @param {{W:number[][], K:number[][], whitening:{mean:number[]}}} model icaFit 的产出
 * @returns {number[][]} 估计源矩阵 n×k（行 = 样本；列顺序/符号与真实源可差置换与符号）
 */
export function icaTransform(X, model) {
  const { K, W, whitening } = model;
  const mu = whitening.mean;
  const Xc = X.map((row) => row.map((v, j) => v - mu[j]));
  const Z = matmul(Xc, transpose(K)); // 白化
  return matmul(Z, transpose(W));     // unmixing
}

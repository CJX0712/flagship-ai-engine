/**
 * 旗舰AI引擎 · 高斯过程回归（Gaussian Process Regression）
 * 站在：Rasmussen & Williams 2006《Gaussian Processes for Machine Learning》
 *       （算法 2.1 预测；公式 5.8 边际似然；公式 5.9 超参数梯度）
 * 核：RBF / 平方指数 κ(x,z) = σ_f²·exp(−‖x−z‖²/(2ℓ²))，含噪声项 σ_n²·I。
 * 不变量：
 *   ① Cholesky 分解精确：LLᵀ ≡ K（核矩阵对称正定）
 *   ② 无噪声极限插值：σ_n → 0 时训练点后验均值 ≡ 观测 y（GP 的插值性质）
 *   ③ 后验方差 ≤ 先验方差，且远离训练点时 → σ_f²（不确定性随距离单调饱和）
 *   ④ 边际似然解析梯度 ≡ 中心差分（R&W 5.9：½tr[(ααᵀ − K⁻¹)∂K/∂θ]）
 *   ⑤ 单点闭式：n=1 时 μ(x*) = k(x*,x)·y/(k(x,x)+σ_n²)，σ² = σ_f² − k²/(k+σ_n²)
 */

import { zeros, transpose, matmul, solve, cholesky } from '../core/linalg.js';
import { rbfKernel } from './svm.js';

/**
 * GP 核矩阵：K_ij = σ_f²·exp(−‖x_i−x_j‖²/(2ℓ²)) + σ_n²·δ_ij
 * @param {number[][]} X
 * @param {number[][]} [Z] 缺省为 X
 */
export function gprKernelMatrix(X, Z, { lengthScale = 1, sigmaF = 1, noise = 1e-6 } = {}) {
  const Zz = Z || X;
  const n = X.length, m = Zz.length;
  const gamma = 1 / (2 * lengthScale * lengthScale);
  const K = zeros(n, m);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) {
      K[i][j] = sigmaF * sigmaF * rbfKernel(X[i], Zz[j], gamma);
    }
  }
  if (!Z || Z === X) {
    for (let i = 0; i < n; i++) K[i][i] += noise * noise;
  }
  return K;
}

/**
 * GP 拟合：Cholesky 分解 K = LLᵀ，α = K⁻¹y，并给出边际似然。
 * @returns {{X:number[][], y:number[], K:number[][], L:number[][], alpha:number[],
 *            logML:number, params:object}}
 */
export function gprFit(X, y, { lengthScale = 1, sigmaF = 1, noise = 1e-6 } = {}) {
  const K = gprKernelMatrix(X, null, { lengthScale, sigmaF, noise });
  const L = cholesky(K);
  // α = K⁻¹y：先解 L z = y，再解 Lᵀ α = z
  const z = solve(L, y);
  const alpha = solve(transpose(L), z);
  let quad = 0;
  for (let i = 0; i < y.length; i++) quad += y[i] * alpha[i];
  let logDet = 0;
  for (let i = 0; i < L.length; i++) logDet += Math.log(L[i][i]);
  const logML = -0.5 * quad - logDet - 0.5 * y.length * Math.log(2 * Math.PI);
  return { X, y, K, L, alpha, logML, params: { lengthScale, sigmaF, noise } };
}

/**
 * GP 预测：μ* = K_*ᵀα，σ²* = κ(x*,x*) − vᵀv（v = L⁻¹K_*，R&W 算法 2.1）
 * @returns {{mean:number[], variance:number[]}}
 */
export function gprPredict(model, Xs) {
  const { X, alpha, L, params } = model;
  const { lengthScale, sigmaF } = params;
  const Ks = gprKernelMatrix(X, Xs, { lengthScale, sigmaF }); // n × m（不含噪声）
  const m = Xs.length;
  const mean = new Array(m).fill(0);
  const variance = new Array(m).fill(0);
  for (let j = 0; j < m; j++) {
    // v = L⁻¹ k_*（下三角前代）
    const kstar = new Array(X.length);
    for (let i = 0; i < X.length; i++) kstar[i] = Ks[i][j];
    const v = solve(L, kstar);
    let mu = 0, vv = 0;
    for (let i = 0; i < kstar.length; i++) {
      mu += kstar[i] * alpha[i];
      vv += v[i] * v[i];
    }
    const prior = sigmaF * sigmaF; // κ(x*,x*) = σ_f²
    mean[j] = mu;
    variance[j] = Math.max(prior - vv, 0);
  }
  return { mean, variance };
}

/** 先验方差（κ(x*,x*) = σ_f²），供不确定性对照 */
export function gprPriorVariance(model) {
  return model.params.sigmaF * model.params.sigmaF;
}

/**
 * K⁻¹（经 Cholesky 逐列求解；n 小时开销可接受，仅供不变量对照）
 */
export function gprInverseK(model) {
  const n = model.K.length;
  const I = zeros(n, n);
  for (let i = 0; i < n; i++) I[i][i] = 1;
  const cols = [];
  for (let j = 0; j < n; j++) {
    const e = new Array(n).fill(0);
    e[j] = 1;
    cols.push(solve(model.K, e));
  }
  const inv = zeros(n, n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) inv[i][j] = cols[j][i];
  return inv;
}

/**
 * 边际似然对超参数的解析梯度（R&W 公式 5.9）：
 *   ∂/∂θ log p(y|X,θ) = ½·tr[(ααᵀ − K⁻¹)·∂K/∂θ]
 * @returns {{dLengthScale:number, dSigmaF:number, dNoise:number}}
 */
export function gprHyperGradAnalytic(model) {
  const { X, alpha, params, K } = model;
  const { lengthScale: ell, sigmaF, noise } = params;
  const n = X.length;
  const Kinv = gprInverseK(model);
  // W = ααᵀ − K⁻¹
  const W = zeros(n, n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) W[i][j] = alpha[i] * alpha[j] - Kinv[i][j];
  }
  const dist2 = zeros(n, n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let d = 0; d < X[i].length; d++) { const dd = X[i][d] - X[j][d]; s += dd * dd; }
      dist2[i][j] = s;
    }
  }
  const gamma = 1 / (2 * ell * ell);
  let gL = 0, gF = 0, gN = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const base = rbfKernel(X[i], X[j], gamma); // exp(−‖Δ‖²/2ℓ²)
      // ∂K/∂ℓ = σ_f²·base·‖Δ‖²/ℓ³
      const dKdl = sigmaF * sigmaF * base * dist2[i][j] / (ell * ell * ell);
      // ∂K/∂σ_f = 2σ_f·base
      const dKdf = 2 * sigmaF * base;
      gL += W[i][j] * dKdl;
      gF += W[i][j] * dKdf;
    }
    // ∂K/∂σ_n = 2σ_n·δ_ij
    gN += W[i][i] * 2 * noise;
  }
  return { dLengthScale: 0.5 * gL, dSigmaF: 0.5 * gF, dNoise: 0.5 * gN };
}

/** 边际似然对超参数的中心差分梯度（解析梯度的交叉验证标尺） */
export function gprHyperGradNumeric(model, { eps = 1e-6 } = {}) {
  const p = model.params;
  const f = (over) => gprFit(model.X, model.y, { ...p, ...over }).logML;
  const d = (key) => (f({ [key]: p[key] + eps }) - f({ [key]: p[key] - eps })) / (2 * eps);
  return {
    dLengthScale: d('lengthScale'),
    dSigmaF: d('sigmaF'),
    dNoise: d('noise'),
  };
}

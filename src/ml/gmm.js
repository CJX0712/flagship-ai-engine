/**
 * 旗舰AI引擎 · 高斯混合模型（GMM）EM 算法
 * 站在：Dempster, Laird & Rubin 1977《Maximum Likelihood from Incomplete Data
 *       via the EM Algorithm》(JRSSB, 39:1) · Bishop 2006《Pattern Recognition
 *       and Machine Learning》§9.2。
 *
 * 模型：p(x) = Σ_k π_k N(x | μ_k, Σ_k)，π_k ≥ 0，Σ π_k = 1。
 * EM（单调不降，Wu 1983 / McLachlan & Krishnan 2008 证明）：
 *   E 步：γ_{ik} = π_k N(x_i|μ_k,Σ_k) / Σ_j π_j N(·)，用 logsumexp 稳定计算。
 *   M 步：π_k = N_k/n，μ_k = Σ γ_ik x_i / N_k，Σ_k = Σ γ_ik (x_i−μ_k)(x_i−μ_k)ᵀ / N_k。
 *
 * 不变量：
 *   ① 每行 responsibilities 之和 ≡ 1（至 1e-9）；
 *   ② weights 之和 ≡ 1；
 *   ③ 每个 cov 对称且半正定（最小特征值 ≥ −1e-8）；
 *   ④ logLikTrace 单调不降（最大下降 < 1e-6，EM 收敛定理背书）；
 *   ⑤ 强分离双高斯（标签已知）硬指派准确率 ≥ 0.95。
 */

import { zeros, identity, cholesky, jacobiEig } from '../core/linalg.js';
import { logsumexp } from '../core/numerical.js';
import { mulberry32 } from '../core/stats.js';

/** 协方差加 reg·I（保证正定，避免 Cholesky 奇异；仅一次除法，不下三角二次除） */
function gmmAddReg(cov, reg) {
  const d = cov.length;
  const out = cov.map((row) => row.slice());
  for (let i = 0; i < d; i++) out[i][i] += reg;
  return out;
}

/**
 * 单分量对数高斯密度：给定 Cholesky 因子 L（Σ = L Lᵀ），
 *   log N(x|μ,Σ) = −½(d·log2π + log|Σ| + ‖L⁻¹(x−μ)‖²)，
 * 其中 log|Σ| = 2 Σ_i log L_ii，内积经前向回代求得。
 */
function gmmLogGauss(x, mu, L) {
  const d = x.length;
  const z = new Array(d);
  for (let i = 0; i < d; i++) z[i] = x[i] - mu[i];
  const y = new Array(d).fill(0);
  for (let i = 0; i < d; i++) {
    let s = z[i];
    for (let j = 0; j < i; j++) s -= L[i][j] * y[j];
    y[i] = s / L[i][i];
  }
  let maha = 0;
  for (let i = 0; i < d; i++) maha += y[i] * y[i];
  let logdet = 0;
  for (let i = 0; i < d; i++) logdet += Math.log(L[i][i]);
  logdet *= 2;
  return -0.5 * (d * Math.log(2 * Math.PI) + logdet + maha);
}

/** k-means 初始化：取 K 个互异随机点作种子，Lloyd 迭代收敛（EM 良好初值） */
function gmmKmeansInit(X, K, rng) {
  const n = X.length, d = X[0].length;
  const idx = [];
  let guard = 0;
  while (idx.length < K && guard < 200 * K) {
    const c = Math.floor(rng() * n);
    if (!idx.includes(c)) idx.push(c);
    guard++;
  }
  const centers = idx.map((i) => X[i].slice());
  for (let it = 0; it < 30; it++) {
    const assign = new Array(n).fill(0);
    for (let i = 0; i < n; i++) {
      const xi = X[i];
      let best = Infinity, bi = 0;
      for (let k = 0; k < K; k++) {
        const ck = centers[k];
        let s = 0;
        for (let j = 0; j < d; j++) { const e = xi[j] - ck[j]; s += e * e; }
        if (s < best) { best = s; bi = k; }
      }
      assign[i] = bi;
    }
    const sum = centers.map(() => new Array(d).fill(0));
    const cnt = new Array(K).fill(0);
    for (let i = 0; i < n; i++) {
      const k = assign[i]; cnt[k]++;
      const xi = X[i];
      for (let j = 0; j < d; j++) sum[k][j] += xi[j];
    }
    let moved = false;
    for (let k = 0; k < K; k++) {
      if (cnt[k] === 0) continue;
      for (let j = 0; j < d; j++) {
        const nv = sum[k][j] / cnt[k];
        if (Math.abs(nv - centers[k][j]) > 1e-9) moved = true;
        centers[k][j] = nv;
      }
    }
    if (!moved) break;
  }
  return centers;
}

/** E 步：返回 responsibilities (n×K) 与数据对数似然（logsumexp 稳定） */
function gmmEStep(X, weights, means, Ls) {
  const n = X.length, K = weights.length;
  const R = zeros(n, K);
  let logLik = 0;
  const logpks = new Array(K);
  for (let i = 0; i < n; i++) {
    const xi = X[i];
    for (let k = 0; k < K; k++) {
      logpks[k] = Math.log(weights[k] + 1e-300) + gmmLogGauss(xi, means[k], Ls[k]);
    }
    const ls = logsumexp(logpks);
    for (let k = 0; k < K; k++) R[i][k] = Math.exp(logpks[k] - ls);
    logLik += ls;
  }
  return { R, logLik };
}

/** M 步：更新 weights / means / covs（全量累加后单次除以 N_k） */
function gmmMStep(X, R, K, reg) {
  const n = X.length, d = X[0].length;
  const Nk = new Array(K).fill(0);
  for (let i = 0; i < n; i++) for (let k = 0; k < K; k++) Nk[k] += R[i][k];
  const weights = new Array(K);
  const means = [];
  for (let k = 0; k < K; k++) {
    weights[k] = Nk[k] / n;
    const mu = new Array(d).fill(0);
    for (let i = 0; i < n; i++) {
      const r = R[i][k];
      if (r === 0) continue;
      const xi = X[i];
      for (let j = 0; j < d; j++) mu[j] += r * xi[j];
    }
    for (let j = 0; j < d; j++) mu[j] /= (Nk[k] > 0 ? Nk[k] : 1);
    means.push(mu);
  }
  const covs = [];
  for (let k = 0; k < K; k++) {
    const C = zeros(d, d);
    const nk = Nk[k] > 1e-12 ? Nk[k] : 1e-12;
    const mu = means[k];
    for (let i = 0; i < n; i++) {
      const r = R[i][k];
      if (r === 0) continue;
      const xi = X[i];
      for (let a = 0; a < d; a++) {
        const ea = xi[a] - mu[a];
        for (let b = 0; b < d; b++) C[a][b] += r * ea * (xi[b] - mu[b]);
      }
    }
    for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) C[a][b] /= nk;
    covs.push(gmmAddReg(C, reg));
  }
  const Ls = covs.map((C) => cholesky(C));
  return { weights, means, covs, Ls };
}

/**
 * GMM-EM 拟合。
 * @param {number[][]} X 数据（n×d）
 * @param {number} K 分量个数
 * @param {object} [opts] { iters=100, reg=1e-6, seed=42 }
 * @returns {{weights:number[], means:number[][], covs:number[][][],
 *            responsibilities:number[][], logLik:number, logLikTrace:number[]}}
 */
export function gmmFit(X, K, { iters = 100, reg = 1e-6, seed = 42 } = {}) {
  const n = X.length, d = X[0].length;
  const rng = mulberry32(seed >>> 0);
  const initMeans = gmmKmeansInit(X, K, rng);
  const initCov = identity(d);
  let means = initMeans.map((m) => m.slice());
  let covs = new Array(K).fill(0).map(() => gmmAddReg(initCov, reg));
  let Ls = covs.map((C) => cholesky(C));
  let weights = new Array(K).fill(1 / K);

  const logLikTrace = [];
  {
    const { logLik } = gmmEStep(X, weights, means, Ls);
    logLikTrace.push(logLik);
  }
  let responsibilities = zeros(n, K);
  for (let iter = 0; iter < iters; iter++) {
    const es = gmmEStep(X, weights, means, Ls);
    const ms = gmmMStep(X, es.R, K, reg);
    weights = ms.weights; means = ms.means; covs = ms.covs; Ls = ms.Ls;
    const es2 = gmmEStep(X, weights, means, Ls);
    responsibilities = es2.R;
    logLikTrace.push(es2.logLik);
  }
  const logLik = logLikTrace[logLikTrace.length - 1];
  return { weights, means, covs, responsibilities, logLik, logLikTrace };
}

/**
 * GMM 预测：返回 responsibilities 与硬指派标签（每点 argmax 责任度）。
 * @param {number[][]} X 数据
 * @param {object} model gmmFit 的返回
 * @returns {{responsibilities:number[][], labels:number[]}}
 */
export function gmmPredict(X, model) {
  const { weights, means, covs } = model;
  const Ls = covs.map((C) => cholesky(C));
  const { R } = gmmEStep(X, weights, means, Ls);
  const labels = new Array(X.length);
  for (let i = 0; i < X.length; i++) {
    let bi = 0, bv = -Infinity;
    for (let k = 0; k < weights.length; k++) if (R[i][k] > bv) { bv = R[i][k]; bi = k; }
    labels[i] = bi;
  }
  return { responsibilities: R, labels };
}

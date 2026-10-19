/**
 * 旗舰AI引擎 · 概率统计
 * 对齐 Bishop《Pattern Recognition and Machine Learning》第 1–2 章。
 * 对数一律取自然底。
 */

import { zeros } from './linalg.js';

/** 高斯密度 N(x | μ, σ)（一维） */
export function gaussianPDF(x, mu = 0, sigma = 1) {
  const z = (x - mu) / sigma;
  return Math.exp(-0.5 * z * z) / (sigma * Math.sqrt(2 * Math.PI));
}

/** 高斯对数密度（一维），供朴素贝叶斯 / GMM 稳定求和 */
export function gaussianLogPDF(x, mu = 0, sigma = 1) {
  const z = (x - mu) / sigma;
  return -0.5 * z * z - Math.log(sigma) - 0.5 * Math.log(2 * Math.PI);
}

/** 熵 H(p) = −Σ p log p（约定 0·log 0 = 0） */
export function entropy(p) {
  let h = 0;
  for (const x of p) if (x > 0) h -= x * Math.log(x);
  return h;
}

/** 交叉熵 H(p, q) = −Σ p log q */
export function crossEntropy(p, q) {
  let h = 0;
  for (let i = 0; i < p.length; i++) if (p[i] > 0) h -= p[i] * Math.log(Math.max(q[i], 1e-300));
  return h;
}

/**
 * KL 散度 KL(p‖q) = Σ p log(p/q)。
 * 不变量：≥ 0 且 =0 ⟺ p = q（Gibbs 不等式）。
 */
export function klDivergence(p, q) {
  let kl = 0;
  for (let i = 0; i < p.length; i++) {
    if (p[i] === 0) continue;
    if (q[i] <= 0) return Infinity;
    kl += p[i] * Math.log(p[i] / q[i]);
  }
  return kl;
}

/** 均值（按列） */
export function mean(X) {
  const n = X.length, d = X[0].length;
  const mu = new Array(d).fill(0);
  for (const row of X) for (let j = 0; j < d; j++) mu[j] += row[j];
  return mu.map(v => v / n);
}

/**
 * 样本协方差矩阵（/(n−1)）：对称半正定。
 * 历史 bug 警示：曾用「上三角累加 → 逐行镜像 → 逐行除 n−1」写法，
 * 镜像发生在除法之前，导致下三角被二次相除（C[i][j] = C[j][i]/(n−1)，i>j），
 * 矩阵不再对称。现改为全量累加 + 单次除法，对称性由不变量套件钉死。
 */
export function covarianceMatrix(X) {
  const n = X.length, d = X[0].length;
  const mu = mean(X);
  const C = zeros(d, d);
  for (const row of X) {
    for (let i = 0; i < d; i++) {
      const xi = row[i] - mu[i];
      for (let j = 0; j < d; j++) C[i][j] += xi * (row[j] - mu[j]);
    }
  }
  for (let i = 0; i < d; i++) {
    for (let j = 0; j < d; j++) C[i][j] /= n - 1;
  }
  return C;
}

/** mulberry32：可复现 32 位 PRNG（种子化，全库随机性统一走它） */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Box–Muller 高斯采样 */
export function gaussianSample(rng, mu = 0, sigma = 1) {
  let u1 = rng();
  while (u1 <= 1e-12) u1 = rng();
  const u2 = rng();
  const r = Math.sqrt(-2 * Math.log(u1));
  return mu + sigma * r * Math.cos(2 * Math.PI * u2);
}

/**
 * 两月亮数据集（与 sklearn make_moons 同参数化）：
 * 外月 (cos t, sin t)、内月 (1 − cos t, 0.5 − sin t)，t ∈ [0, π]，叠加高斯噪声。
 * 两个交叠月牙仅靠归一化割/核方法可切分——k-means 等凸假设方法失效。
 * @returns {{X:number[][], y:number[]}} y ∈ {0,1}
 */
export function twoMoons({ n = 60, sigma = 0.12, seed = 42 } = {}) {
  const rng = mulberry32(seed);
  const X = [], y = [];
  const half = Math.floor(n / 2);
  for (let i = 0; i < half; i++) {
    const t = Math.PI * (i + 0.5) / half;
    X.push([gaussianSample(rng, Math.cos(t), sigma), gaussianSample(rng, Math.sin(t), sigma)]);
    y.push(0);
  }
  for (let i = 0; i < half; i++) {
    const t = Math.PI * (i + 0.5) / half;
    X.push([gaussianSample(rng, 1 - Math.cos(t), sigma), gaussianSample(rng, 0.5 - Math.sin(t), sigma)]);
    y.push(1);
  }
  return { X, y };
}

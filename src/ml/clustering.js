/**
 * 旗舰AI引擎 · 聚类与生成模型
 * k-means（k-means++ 初始化，Arthur & Vassilvitskii 2007）/ GMM-EM（Bishop《PRML》第 9 章）。
 * 不变量：Lloyd 迭代 SSE 单调不增；EM 对数似然单调不减；责任度行和 = 1。
 */

import { cholesky, dot, zeros } from '../core/linalg.js';
import { logsumexp } from '../core/numerical.js';
import { covarianceMatrix, mulberry32 } from '../core/stats.js';

/** 向量平方欧氏距离 */
function sqdist(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; }
  return s;
}

/** k-means++ 种子点选择 */
function kmeansPPCenters(X, k, rng) {
  const n = X.length;
  const centers = [X[Math.floor(rng() * n)].slice()];
  while (centers.length < k) {
    const d2 = X.map(x => Math.min(...centers.map(c => sqdist(x, c))));
    const total = d2.reduce((a, b) => a + b, 0);
    let r = rng() * total, idx = n - 1;
    for (let i = 0; i < n; i++) { r -= d2[i]; if (r <= 0) { idx = i; break; } }
    centers.push(X[idx].slice());
  }
  return centers;
}

/**
 * k-means（Lloyd 迭代）。
 * 返回 { labels, centers, sseHistory }；sseHistory 逐迭代严格记录，单调不增。
 */
export function kmeans(X, k, { iters = 100, seed = 42 } = {}) {
  const n = X.length, d = X[0].length;
  const rng = mulberry32(seed);
  let centers = kmeansPPCenters(X, k, rng);
  const labels = new Array(n).fill(0);
  const sseHistory = [];
  for (let it = 0; it < iters; it++) {
    // 分配步
    let moved = false;
    for (let i = 0; i < n; i++) {
      let best = 0, bd = Infinity;
      for (let c = 0; c < k; c++) {
        const dd = sqdist(X[i], centers[c]);
        if (dd < bd) { bd = dd; best = c; }
      }
      if (labels[i] !== best) { labels[i] = best; moved = true; }
    }
    // 更新步
    const sums = zeros(k, d);
    const cnt = new Array(k).fill(0);
    for (let i = 0; i < n; i++) {
      cnt[labels[i]]++;
      for (let j = 0; j < d; j++) sums[labels[i]][j] += X[i][j];
    }
    for (let c = 0; c < k; c++) {
      if (cnt[c] === 0) { // 空簇：换最远点
        let far = 0, fd = -1;
        for (let i = 0; i < n; i++) {
          const dd = sqdist(X[i], centers[labels[i]]);
          if (dd > fd) { fd = dd; far = i; }
        }
        centers[c] = X[far].slice();
      } else {
        for (let j = 0; j < d; j++) centers[c][j] = sums[c][j] / cnt[c];
      }
    }
    // 记录 SSE（按新中心）
    let sse = 0;
    for (let i = 0; i < n; i++) sse += sqdist(X[i], centers[labels[i]]);
    sseHistory.push(sse);
    if (!moved && it > 0 && sseHistory.length >= 2 &&
        Math.abs(sseHistory[sseHistory.length - 1] - sseHistory[sseHistory.length - 2]) < 1e-15) break;
  }
  return { labels, centers, sseHistory };
}

/** 多元高斯对数密度（经 Cholesky：log|Σ| 与马氏距离数值稳定） */
function logMVN(x, mu, cov) {
  const d = x.length;
  const L = cholesky(cov);
  const diff = x.map((v, i) => v - mu[i]);
  const y = new Array(d).fill(0);
  for (let i = 0; i < d; i++) {
    let s = diff[i];
    for (let j = 0; j < i; j++) s -= L[i][j] * y[j];
    y[i] = s / L[i][i];
  }
  let logdet = 0;
  for (let i = 0; i < d; i++) logdet += Math.log(L[i][i]);
  return -0.5 * (d * Math.log(2 * Math.PI) + 2 * logdet + dot(y, y));
}

/**
 * 高斯混合模型 EM（全协方差 + 1e-6 岭稳定）。
 * 返回 { means, covs, weights, resp, llHistory }。
 * 不变量：责任度每行 Σ=1；权重 Σ=1；协方差对称半正定；对数似然单调不减。
 */
export function gmm(X, k, { iters = 60, seed = 42, ridge = 1e-6 } = {}) {
  const n = X.length, d = X[0].length;
  const rng = mulberry32(seed);
  const { centers } = kmeans(X, k, { iters: 30, seed: seed * 31 + 7 });
  let means = centers.map(c => c.slice());
  const baseCov = covarianceMatrix(X);
  let covs = [];
  for (let c = 0; c < k; c++) {
    const C = baseCov.map(r => r.slice());
    for (let i = 0; i < d; i++) C[i][i] += ridge;
    covs.push(C);
  }
  let weights = new Array(k).fill(1 / k);
  const llHistory = [];
  let resp = zeros(n, k);
  for (let it = 0; it < iters; it++) {
    // E 步：对数域责任度 + 总对数似然
    let ll = 0;
    resp = zeros(n, k);
    for (let i = 0; i < n; i++) {
      const logk = new Array(k);
      for (let c = 0; c < k; c++) {
        logk[c] = Math.log(Math.max(weights[c], 1e-300)) + logMVN(X[i], means[c], covs[c]);
      }
      const lse = logsumexp(logk);
      ll += lse;
      for (let c = 0; c < k; c++) resp[i][c] = Math.exp(logk[c] - lse);
    }
    llHistory.push(ll);
    // M 步
    const Nk = new Array(k).fill(0);
    for (let i = 0; i < n; i++) for (let c = 0; c < k; c++) Nk[c] += resp[i][c];
    for (let c = 0; c < k; c++) {
      if (Nk[c] < 1e-10) { // 空分量：重置到随机样本
        means[c] = X[Math.floor(rng() * n)].slice();
        weights[c] = 1 / k;
        continue;
      }
      const mu = new Array(d).fill(0);
      for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) mu[j] += resp[i][c] * X[i][j];
      means[c] = mu.map(v => v / Nk[c]);
      const C = zeros(d, d);
      for (let i = 0; i < n; i++) {
        const w = resp[i][c];
        for (let a = 0; a < d; a++) {
          const da = X[i][a] - means[c][a];
          for (let b = 0; b < d; b++) C[a][b] += w * da * (X[i][b] - means[c][b]); // 双三角累加，天然对称
        }
      }
      for (let a = 0; a < d; a++) {
        for (let b = 0; b < d; b++) C[a][b] /= Nk[c];
        C[a][a] += ridge;
      }
      covs[c] = C;
      weights[c] = Nk[c] / n;
    }
  }
  return { means, covs, weights, resp, llHistory };
}

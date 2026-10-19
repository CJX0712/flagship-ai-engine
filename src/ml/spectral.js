/**
 * 旗舰AI引擎 · 谱聚类（Ng-Jordan-Weiss 算法）
 * 站在：Shi & Malik 2000（归一化割 Ncut）· Ng, Jordan & Weiss 2002（NJT/NJW）
 *       von Luxburg 2007《A Tutorial on Spectral Clustering》（谱界定理）
 * 不变量：
 *   ① L_sym 最小特征值 = 0，对应特征向量 ≡ √d/‖√d‖（图拉普拉斯基本定理）
 *   ② 谱界：λ(L_sym) ⊆ [0, 2]（归一化拉普拉斯谱定理）
 *   ③ 特征向量正交归一（Jacobobi 输出 QᵀQ = I）
 *   ④ 两月亮 ARI = 1.000000（非凸结构，谱方法精确恢复）
 *   ⑤ 对照：原始空间 k-means 同数据 ARI 显著更低（凸假设失效）
 *   ⑥ 图对称性与度矩阵一致性：W = Wᵀ、D = diag(W1)、L_sym 对称
 */

import { zeros, jacobiEig } from '../core/linalg.js';
import { kmeans } from './clustering.js';
import { rbfKernel } from './svm.js';

/**
 * 相似度图：k 近邻 RBF 图，对称化取并（j ∈ kNN(i) 或 i ∈ kNN(j) ⇒ W_ij = exp(−γ‖xi−xj‖²)）。
 * gamma 传 'auto' 时启用局部尺度（Zelnik-Manor & Perona 2004）：
 * W_ij = exp(−‖xi−xj‖²/(σ_i·σ_j))，σ_i = 点 i 到其第 k 近邻的距离——
 * 自适应带宽对密度不均/交叠结构（如两月亮）远优于全局 γ。
 * 自环清零；孤立点度为 0 时拉普拉斯对应行置 0（不影响其它点）。
 */
export function similarityGraph(X, { k = 7, gamma = 1 } = {}) {
  const n = X.length;
  const sq = (a, b) => {
    let s = 0;
    for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; }
    return s;
  };
  // 每点的 k 近邻（用于局部尺度 σ_i 与建边）
  const nbrs = [];
  for (let i = 0; i < n; i++) {
    const order = [];
    for (let j = 0; j < n; j++) if (j !== i) order.push([j, sq(X[i], X[j])]);
    order.sort((a, b) => a[1] - b[1]);
    nbrs.push(order.slice(0, Math.min(k, order.length)));
  }
  const local = gamma === 'auto';
  const sigma = local ? nbrs.map(nb => Math.sqrt(nb[nb.length - 1][1]) || 1e-12) : null;
  const weight = (d2, i, j) => (local ? Math.exp(-d2 / (sigma[i] * sigma[j])) : rbfKernel(X[i], X[j], gamma));

  const W = zeros(n, n);
  for (let i = 0; i < n; i++) {
    for (const [j, d2] of nbrs[i]) {
      const w = weight(d2, i, j);
      if (w > W[i][j]) { W[i][j] = w; W[j][i] = w; }
    }
  }
  return W;
}

/** 对称归一化拉普拉斯：L_sym = I − D^{−1/2} W D^{−1/2}，D = diag(W1) */
export function normalizedLaplacian(W) {
  const n = W.length;
  const degrees = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) degrees[i] += W[i][j];
  }
  const dis = degrees.map(v => (v > 0 ? 1 / Math.sqrt(v) : 0));
  const L = zeros(n, n);
  for (let i = 0; i < n; i++) {
    L[i][i] = degrees[i] > 0 ? 1 : 0;
    for (let j = 0; j < n; j++) L[i][j] -= dis[i] * W[i][j] * dis[j];
  }
  return { L, degrees };
}

/**
 * 谱嵌入：取 L_sym 最小 k 个特征值对应的特征向量构成 U（n×k），
 * 再按 NJW 规范做行归一化 T_ij = U_ij/‖U_i‖₂。
 */
export function spectralEmbedding(L, k) {
  const n = L.length;
  const eig = jacobiEig(L); // values 降序
  const kk = Math.min(k, n);
  const U = zeros(n, kk);
  for (let c = 0; c < kk; c++) {
    const v = eig.vectors[n - 1 - c]; // 最小特征值对应的特征向量
    for (let i = 0; i < n; i++) U[i][c] = v[i];
  }
  const T = zeros(n, kk);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let c = 0; c < kk; c++) s += U[i][c] * U[i][c];
    s = Math.sqrt(s);
    for (let c = 0; c < kk; c++) T[i][c] = s > 1e-300 ? U[i][c] / s : 0;
  }
  return { U, T, eigenvalues: eig.values };
}

/**
 * 谱聚类完整流程：相似度图 → 归一化拉普拉斯 → 谱嵌入 → k-means（复用 Lloyd）。
 * @returns {{labels:number[], W:number[][], L:number[][], degrees:number[],
 *            U:number[][], T:number[][], sse:number[]}}
 */
export function spectralCluster(X, k, { neighbors = 7, gamma = 1, seed = 42, iters = 100 } = {}) {
  const W = similarityGraph(X, { k: neighbors, gamma });
  const { L, degrees } = normalizedLaplacian(W);
  const { U, T } = spectralEmbedding(L, k);
  const km = kmeans(T, k, { seed, iters });
  return { labels: km.labels, W, L, degrees, U, T, sse: km.sse };
}

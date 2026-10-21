/**
 * 旗舰AI引擎 · 流形学习：经典 MDS + Isomap
 * 站在：Torgerson 1952《Multidimensional Scaling: I. Theory and Method》
 *       （经典 MDS = 双居中 + 特征分解，Young–Householder 1938 的完备化）
 *       · Tenenbaum, de Silva & Langford 2000 Science《A Global Geometric
 *       Framework for Nonlinear Dimensionality Reduction》（Isomap）
 *       · Kruskal 1964《Multidimensional Scaling by Optimizing Goodness of Fit
 *       to a Nonmetric Hypothesis》（应力 stress）
 *       · Cox & Cox《Multidimensional Scaling》§2（双居中矩阵 B = −½JD⁽²⁾J）。
 *
 * 经典 MDS：由距离矩阵 D 还原坐标——
 *   B = −½·J·D⁽²⁾·J（J = I − (1/n)11ᵀ 为投影掉均值的中心化矩阵，D⁽²⁾ 为逐元素平方）
 *   B 对称半正定；取前 k 大特征值 λ_i 与单位特征向量 v_i，则
 *   coords 的第 i 列 = √λ_i·v_i（特征向量符号任意，本模块按「最大绝对分量取正」规范化）
 * Isomap：欧氏 kNN 图 → 图上测地距离（Dijkstra）→ 对测地距离矩阵跑经典 MDS。
 *   测地距离 ≥ 欧氏距离（三角不等式，恒等式 ⑤）。
 *
 * 不变量（全部为可独立交叉验证的硬恒等式）：
 *   ① 经典 MDS ≡ PCA：欧氏距离矩阵上，MDS 坐标 ≡ 中心化 X 的主成分得分（允许符号翻转，maxAbs < 1e-8）；
 *   ② 双居中矩阵行和恒等式：B·1 = 0（maxAbs < 1e-12），B 对称（maxAbs < 1e-14）；
 *   ③ 满秩重构：k = n−1 时嵌入点间欧氏距离 ≡ 原始 D（maxAbs < 1e-8）；
 *   ④ Kruskal 应力单调：嵌入维数 k 增大 ⇒ 应力单调不增（包络口径，允许 1% 抖动）；
 *   ⑤ 测地距离 ≥ 欧氏距离（精确断言，违反计数 = 0）；
 *   ⑥ Isomap 展开质量：Swiss Roll 上嵌入空间的 kNN 重叠率显著高于 PCA 对照（差值 > 0.1）；
 *   ⑦ 同输入确定性：两次调用逐位一致（maxDiff = 0）。
 *
 * 复杂度提示：jacobiEig 为 O(n³)/sweep，Isomap 建议 n ≤ 300（浏览器端亦然）。
 */

import { zeros, jacobiEig } from '../core/linalg.js';
import { mulberry32, gaussianSample } from '../core/stats.js';

/**
 * 欧氏距离方阵（对称、对角为 0）。
 * @param {number[][]} X 样本（n×d）
 * @returns {number[][]} n×n 距离矩阵
 */
export function distanceMatrix(X) {
  const n = X.length;
  const d = X[0].length;
  const D = zeros(n, n);
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      let s = 0;
      for (let k = 0; k < d; k++) {
        const e = X[i][k] - X[j][k];
        s += e * e;
      }
      const v = Math.sqrt(s);
      D[i][j] = v;
      D[j][i] = v;
    }
  }
  return D;
}

/**
 * 双居中：B = −½·J·D⁽²⁾·J。
 * 等价的 O(n²) 写法（避免显式构造 n×n 的 J）：
 *   B_ij = −½·(D_ij² − rowMean_i − colMean_j + grandMean)
 * 行/列和恒为 0（不变量 ②）。
 * @param {number[][]} D 距离矩阵（对称、对角 0）
 * @returns {number[][]} B（对称半正定，行和 = 列和 = 0）
 */
export function doubleCenter(D) {
  const n = D.length;
  const D2 = new Array(n);
  for (let i = 0; i < n; i++) {
    D2[i] = new Array(n);
    for (let j = 0; j < n; j++) D2[i][j] = D[i][j] * D[i][j];
  }
  const rowMean = new Array(n).fill(0);
  const colMean = new Array(n).fill(0);
  let grand = 0;
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = 0; j < n; j++) s += D2[i][j];
    rowMean[i] = s / n;
    grand += s;
  }
  grand /= n * n;
  for (let j = 0; j < n; j++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += D2[i][j];
    colMean[j] = s / n;
  }
  const B = zeros(n, n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      B[i][j] = -0.5 * (D2[i][j] - rowMean[i] - colMean[j] + grand);
    }
  }
  return B;
}

/**
 * 经典 MDS：B = doubleCenter(D) 的前 k 大特征对 ⇒ 坐标。
 * @param {number[][]} D 距离矩阵
 * @param {number} k 目标维数（自动裁剪到 ≤ n）
 * @returns {{coords:number[][], eigenvalues:number[], B:number[][]}}
 */
export function classicalMDS(D, k) {
  const n = D.length;
  const kk = Math.max(0, Math.min(k, n));
  const B = doubleCenter(D);
  const { values, vectors } = jacobiEig(B);
  const coords = new Array(n);
  for (let i = 0; i < n; i++) coords[i] = new Array(kk).fill(0);
  const eigenvalues = new Array(kk);
  for (let c = 0; c < kk; c++) {
    const lam = values[c];
    eigenvalues[c] = lam;
    if (lam <= 0) continue; // 非正特征值（非欧氏/噪声）⇒ 该维坐标为 0
    const v = vectors[c];
    // 符号规范化：绝对值最大的分量取正（特征向量符号本任意，规范化后结果可复现）
    let idx = 0;
    let mx = -1;
    for (let i = 0; i < n; i++) {
      const a = Math.abs(v[i]);
      if (a > mx) { mx = a; idx = i; }
    }
    const sign = v[idx] >= 0 ? 1 : -1;
    const s = sign * Math.sqrt(lam);
    for (let i = 0; i < n; i++) coords[i][c] = s * v[i];
  }
  return { coords, eigenvalues, B };
}

/**
 * k-近邻图（无权、对称化）：i∈kNN(j) 或 j∈kNN(i) ⇒ 连边。
 * @param {number[][]} X 样本
 * @param {number} k 近邻数
 * @returns {number[][]} 邻接矩阵（0/1，对角为 0）
 */
export function knnGraph(X, k) {
  const n = X.length;
  const D = distanceMatrix(X);
  const adj = zeros(n, n);
  for (let i = 0; i < n; i++) {
    // 选 k 个最近（排除自身）：O(n) 选择，避免整行排序的 O(n log n)
    const chosen = new Array(k).fill(-1);
    const chosenD = new Array(k).fill(Infinity);
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      const dv = D[i][j];
      if (dv >= chosenD[k - 1]) continue;
      let p = k - 1;
      while (p > 0 && chosenD[p - 1] > dv) {
        chosenD[p] = chosenD[p - 1];
        chosen[p] = chosen[p - 1];
        p--;
      }
      chosenD[p] = dv;
      chosen[p] = j;
    }
    for (let c = 0; c < k; c++) {
      if (chosen[c] < 0) continue;
      adj[i][chosen[c]] = 1;
      adj[chosen[c]][i] = 1;
    }
  }
  return adj;
}

/**
 * 图上全源最短路径（Dijkstra，O(n³) 朴素实现；不连通对返回 Infinity）。
 * @param {number[][]} adj 邻接矩阵（0/1）
 * @param {number[][]} X 样本（提供边权 = 两端点欧氏距离）
 * @returns {number[][]} 测地距离矩阵
 */
export function graphGeodesic(adj, X) {
  const n = X.length;
  const d = X[0].length;
  const G = zeros(n, n);
  for (let s = 0; s < n; s++) {
    const dist = new Array(n).fill(Infinity);
    const done = new Array(n).fill(false);
    dist[s] = 0;
    for (let it = 0; it < n; it++) {
      let u = -1;
      let best = Infinity;
      for (let i = 0; i < n; i++) {
        if (!done[i] && dist[i] < best) { best = dist[i]; u = i; }
      }
      if (u < 0 || best === Infinity) break; // 剩余节点不可达
      done[u] = true;
      for (let v = 0; v < n; v++) {
        if (done[v] || adj[u][v] === 0) continue;
        let e = 0;
        for (let k = 0; k < d; k++) {
          const diff = X[u][k] - X[v][k];
          e += diff * diff;
        }
        e = Math.sqrt(e);
        const nd = dist[u] + e;
        if (nd < dist[v]) dist[v] = nd;
      }
    }
    for (let i = 0; i < n; i++) G[s][i] = dist[i];
  }
  return G;
}

/**
 * Isomap：kNN 图 → 测地距离 → 经典 MDS。
 * @param {number[][]} X 样本（n×d）
 * @param {{k?:number, dim?:number}} [opts]
 * @returns {{embedding:number[][], geodesic:number[][], eigenvalues:number[], adj:number[][]}}
 */
export function isomap(X, { k = 5, dim = 2 } = {}) {
  const adj = knnGraph(X, k);
  const geodesic = graphGeodesic(adj, X);
  const { coords, eigenvalues } = classicalMDS(geodesic, dim);
  return { embedding: coords, geodesic, eigenvalues, adj };
}

/**
 * Kruskal 应力（stress-1）：sqrt( Σ_{i<j}(d_ij − δ_ij)² / Σ_{i<j} δ_ij² )
 *   d_ij = 嵌入空间距离（拟合值），δ_ij = 原始距离矩阵 D（观测值）。
 * 分母取 **观测** 距离的平方和 ⇒ 与嵌入维数 k 无关（固定标尺），
 * 于是经典 MDS 逐维追加坐标只会降低残差 ⇒ 应力 monotonically 不增（不变量 ④）。
 * @param {number[][]} embedding 嵌入坐标
 * @param {number[][]} D 原始距离矩阵
 * @returns {number} 应力（越小越好，0 表示等距嵌入）
 */
export function stress(embedding, D) {
  const n = embedding.length;
  let num = 0;
  let den = 0;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      let s = 0;
      for (let k = 0; k < embedding[i].length; k++) {
        const e = embedding[i][k] - embedding[j][k];
        s += e * e;
      }
      const dE = Math.sqrt(s);
      const diff = dE - D[i][j];
      num += diff * diff;
      den += D[i][j] * D[i][j]; // 固定分母：观测距离平方和
    }
  }
  return den > 0 ? Math.sqrt(num / den) : 0;
}

/**
 * 两个空间中 kNN 集合的重叠率（Isomap 展开质量的度量，不变量 ⑥）。
 * @param {number[][]} A 空间 A 的坐标
 * @param {number[][]} B 空间 B 的坐标
 * @param {number} k 近邻数
 * @returns {number} 平均重叠率 ∈ [0, 1]
 */
export function mdsKnnOverlap(A, B, k) {
  const n = A.length;
  const nbrA = mdsKnnSets(A, k);
  const nbrB = mdsKnnSets(B, k);
  let total = 0;
  for (let i = 0; i < n; i++) {
    const setB = new Set(nbrB[i]);
    let hit = 0;
    for (const j of nbrA[i]) if (setB.has(j)) hit++;
    total += hit / k;
  }
  return n > 0 ? total / n : 0;
}

/** 各点的 k 近邻下标集合（排除自身） */
function mdsKnnSets(X, k) {
  const n = X.length;
  const d = X[0].length;
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    const chosen = new Array(k).fill(-1);
    const chosenD = new Array(k).fill(Infinity);
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      let s = 0;
      for (let c = 0; c < d; c++) {
        const e = X[i][c] - X[j][c];
        s += e * e;
      }
      if (s >= chosenD[k - 1]) continue;
      let p = k - 1;
      while (p > 0 && chosenD[p - 1] > s) {
        chosenD[p] = chosenD[p - 1];
        chosen[p] = chosen[p - 1];
        p--;
      }
      chosenD[p] = s;
      chosen[p] = j;
    }
    out[i] = chosen.filter((v) => v >= 0);
  }
  return out;
}

/**
 * Swiss Roll 数据集（Tenenbaum et al. 2000 的标准参数化）：
 *   u = 3π/2·(1 + 2t)，x = u·cos u，z = u·sin u，y = 21·h
 * 同时返回**内在坐标** [s(u), h]（s 为卷绕方向的弧长 ∫√(1+u²)du）——
 * 它是流形上真实测地距离的参考系（不变量 ⑥ 的对照基准）。
 * @param {{n?:number, seed?:number, noise?:number}} [opts]
 * @returns {{X:number[][], intrinsic:number[][]}}
 */
export function mdsSwissRoll({ n = 200, seed = 42, noise = 0 } = {}) {
  const rng = mulberry32(seed);
  const X = new Array(n);
  const intrinsic = new Array(n);
  for (let i = 0; i < n; i++) {
    const t = rng();
    const h = rng();
    const u = 1.5 * Math.PI * (1 + 2 * t);
    const arc = 0.5 * (u * Math.sqrt(1 + u * u) + Math.asinh(u)); // ∫√(1+u²)du
    X[i] = [
      u * Math.cos(u) + gaussianSample(rng, 0, noise),
      21 * h + gaussianSample(rng, 0, noise),
      u * Math.sin(u) + gaussianSample(rng, 0, noise),
    ];
    intrinsic[i] = [arc, 21 * h];
  }
  return { X, intrinsic };
}

/**
 * 旗舰AI引擎 · 密度聚类：DBSCAN（Ester et al. 1996）
 * 站在密度可达性之上：核心点邻域内点数 ≥ minPts；簇 = 核心点密度连通的最大集合。
 * 不变量：
 *   ① 可分团簇：两个分离团簇 → 恰好 2 簇、0 噪声，标签与真实划分完全一致（ARI = 1）
 *   ② 核心见证：每簇至少 1 个核心点；每个非噪声点在其标签簇内有一个 ε-邻域核心点
 *   ③ 离群稳健：追加孤立远点 → 必标 −1，且原有标签逐点不变
 *   ④ 密度连通传递性：标签 ≡ 核心点邻域图的连通分量（边界点归首个可达簇）
 */

/**
 * DBSCAN 聚类（确定性算法，O(n²) 暴力邻域）。
 * @param {number[][]} X 样本（行主序）
 * @param {{eps?:number, minPts?:number}} opts
 * @returns {{labels:number[], core:boolean[], nClusters:number, noise:number}}
 */
export function dbscan(X, { eps = 0.5, minPts = 5 } = {}) {
  const n = X.length;
  const d = X[0].length;
  const eps2 = eps * eps;

  // 预计算邻域（平方距离，免开方）
  const nbr = Array.from({ length: n }, () => []);
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      let s = 0;
      for (let k = 0; k < d; k++) {
        const diff = X[i][k] - X[j][k];
        s += diff * diff;
      }
      if (s <= eps2) { nbr[i].push(j); nbr[j].push(i); }
    }
  }

  const core = nbr.map((list) => list.length >= minPts);
  const labels = new Array(n).fill(-1); // −1 = 未访问（最终为噪声）
  let nClusters = 0;

  for (let i = 0; i < n; i++) {
    if (!core[i] || labels[i] !== -1) continue;
    // BFS 扩展：核心点出发，密度可达集合
    const c = nClusters++;
    labels[i] = c;
    const queue = [i];
    while (queue.length) {
      const p = queue.shift();
      for (const q of nbr[p]) {
        if (labels[q] === -1) {
          labels[q] = c; // 边界点一并归入（密度可达）
          if (core[q]) queue.push(q);
        }
      }
    }
  }

  // 未访问点 = 噪声（保持 −1）
  let noise = 0;
  for (let i = 0; i < n; i++) if (labels[i] === -1) noise++;
  return { labels, core, nClusters, noise };
}

/**
 * 调整互信息（ARI，Hubert & Arabie 1985）——簇划分与真实标注的一致性度量。
 * ARI = 1 ⟺ 完全一致；期望为 0（随机划分）。
 * @param {number[]} a 划分一（允许 −1 噪声标签）
 * @param {number[]} b 划分二
 * @returns {number} ARI ∈ (−∞, 1]
 */
export function adjustedRandIndex(a, b) {
  const n = a.length;
  const keyA = new Map(), keyB = new Map();
  const ia = a.map((v) => { if (!keyA.has(v)) keyA.set(v, keyA.size); return keyA.get(v); });
  const ib = b.map((v) => { if (!keyB.has(v)) keyB.set(v, keyB.size); return keyB.get(v); });
  const na = keyA.size, nb = keyB.size;
  // 列联表
  const ct = Array.from({ length: na }, () => new Array(nb).fill(0));
  for (let i = 0; i < n; i++) ct[ia[i]][ib[i]]++;
  const comb2 = (x) => (x * (x - 1)) / 2;
  let sumComb = 0, sumA = 0, sumB = 0;
  for (let i = 0; i < na; i++) {
    const rowSum = ct[i].reduce((x, y) => x + y, 0);
    sumA += comb2(rowSum);
    for (let j = 0; j < nb; j++) sumComb += comb2(ct[i][j]);
  }
  // 列和
  sumB = 0;
  for (let j = 0; j < nb; j++) {
    let colSum = 0;
    for (let i = 0; i < na; i++) colSum += ct[i][j];
    sumB += comb2(colSum);
  }
  const total = comb2(n);
  const expected = (sumA * sumB) / total;
  const maxIndex = (sumA + sumB) / 2;
  return (sumComb - expected) / (maxIndex - expected);
}

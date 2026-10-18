/**
 * 旗舰AI引擎 · CART 决策树（Breiman et al. 1984）
 * 二叉分裂 + Gini 不纯度 + 预剪枝。
 * 不变量：任意分裂的加权子节点 Gini ≤ 父节点 Gini（Gini 的凸性保证）；可分数据训练精度 = 1。
 */

/** Gini 不纯度：1 − Σ p²（按类别计数） */
export function giniImpurity(labels) {
  const n = labels.length;
  if (n === 0) return 0;
  const cnt = new Map();
  for (const l of labels) cnt.set(l, (cnt.get(l) || 0) + 1);
  let s = 0;
  for (const c of cnt.values()) s += (c / n) ** 2;
  return 1 - s;
}

export class DecisionTree {
  /** @param {{maxDepth?:number, minSamples?:number}} opts */
  constructor({ maxDepth = 4, minSamples = 2 } = {}) {
    this.maxDepth = maxDepth;
    this.minSamples = minSamples;
    this.splits = []; // { depth, feature, threshold, parentGini, childGini }
  }

  fit(X, y) {
    this.classes_ = [...new Set(y)].sort((a, b) => a - b);
    const idx = X.map((_, i) => i);
    this.root_ = this._build(X, y, idx, 0);
    return this;
  }

  _leaf(y, idx) {
    const cnt = new Map();
    for (const i of idx) cnt.set(y[i], (cnt.get(y[i]) || 0) + 1);
    const proba = new Map();
    for (const c of this.classes_) proba.set(c, (cnt.get(c) || 0) / idx.length);
    let best = this.classes_[0], bc = -1;
    for (const [c, p] of proba) if (p > bc) { bc = p; best = c; }
    return { leaf: true, label: best, proba };
  }

  _build(X, y, idx, depth) {
    const parentGini = giniImpurity(idx.map(i => y[i]));
    if (depth >= this.maxDepth || idx.length < this.minSamples || parentGini <= 1e-12) {
      return this._leaf(y, idx);
    }
    const d = X[0].length;
    let best = null; // { feature, threshold, childGini, left: idx, right: idx }
    for (let f = 0; f < d; f++) {
      const sorted = idx.slice().sort((a, b) => X[a][f] - X[b][f]);
      for (let t = 1; t < sorted.length; t++) {
        const v1 = X[sorted[t - 1]][f], v2 = X[sorted[t]][f];
        if (v2 - v1 < 1e-12) continue; // 同值不裂
        const thr = (v1 + v2) / 2;
        const left = [], right = [];
        for (const i of idx) (X[i][f] <= thr ? left : right).push(i);
        if (left.length === 0 || right.length === 0) continue;
        const cg = (left.length / idx.length) * giniImpurity(left.map(i => y[i])) +
                   (right.length / idx.length) * giniImpurity(right.map(i => y[i]));
        if (!best || cg < best.childGini) best = { feature: f, threshold: thr, childGini: cg, left, right };
      }
    }
    if (!best) return this._leaf(y, idx);
    this.splits.push({
      depth, feature: best.feature, threshold: best.threshold,
      parentGini, childGini: best.childGini,
    });
    return {
      leaf: false,
      feature: best.feature,
      threshold: best.threshold,
      left: this._build(X, y, best.left, depth + 1),
      right: this._build(X, y, best.right, depth + 1),
    };
  }

  predictProba(x) {
    let node = this.root_;
    while (!node.leaf) node = x[node.feature] <= node.threshold ? node.left : node.right;
    return { label: node.label, proba: node.proba };
  }

  predict(x) {
    return this.predictProba(x).label;
  }

  /** 训练集精度 */
  score(X, y) {
    let hit = 0;
    for (let i = 0; i < X.length; i++) if (this.predict(X[i]) === y[i]) hit++;
    return hit / X.length;
  }
}

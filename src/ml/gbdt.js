/**
 * 旗舰AI引擎 · 梯度提升回归树（GBDT / Gradient Boosting）
 * 站在：Friedman 2001《Greedy Function Approximation: A Gradient Boosting Machine》
 *       Friedman 1999（CART 回归树作为基学习器）· Chen & Guestrin 2016（XGBoost 的同源思想）
 * 核心：用「负梯度」替代 AdaBoost 的「样本权重重标定」——
 *       每轮 F_m = F_{m−1} + lr·h_m，其中 h_m 拟合残差 r_i = −∂L/∂F = y_i − F_{m−1}(x_i)。
 * 不变量：
 *   ① 训练 MSE 单调不增（贪心函数逼近的直接推论，学习率足够小时必然成立）
 *   ② 每棵树的拟合目标 ≡ 当前残差（负梯度），与残差相关系数 > 0（方向正确）
 *   ③ 学习率收缩极限：lr → 0 时预测 ≡ 初始常数 ȳ（收缩退化为均值模型）
 *   ④ 树桩闭式：maxDepth = 1 时每个叶子的值 ≡ 该叶残差均值（最小二乘解析解）
 *   ⑤ 迭代增益：末轮训练 MSE 显著低于单棵树与常数基线
 */

/**
 * CART 回归树：递归二分最小化平方误差 Σ(y−ȳ)²。
 * @returns {object} 节点 {leaf, value, feature, threshold, left, right, n}
 */
export function regressionTreeFit(X, y, {
  maxDepth = 3, minSamplesSplit = 2, minSamplesLeaf = 1,
} = {}) {
  const d = X[0].length;

  const sse = (ys) => {
    if (ys.length === 0) return 0;
    const mu = ys.reduce((a, b) => a + b, 0) / ys.length;
    let s = 0;
    for (const v of ys) s += (v - mu) ** 2;
    return s;
  };

  const build = (idx, depth) => {
    const ys = idx.map((i) => y[i]);
    const mu = ys.reduce((a, b) => a + b, 0) / ys.length;
    const base = sse(ys);
    if (depth >= maxDepth || idx.length < minSamplesSplit || base < 1e-15) {
      return { leaf: true, value: mu, n: idx.length };
    }
    let best = { gain: 0, feature: -1, threshold: 0 };
    for (let f = 0; f < d; f++) {
      // 候选阈值：该特征上相邻取值的中点
      const vals = [...new Set(idx.map((i) => X[i][f]))].sort((a, b) => a - b);
      for (let t = 0; t < vals.length - 1; t++) {
        const thr = (vals[t] + vals[t + 1]) / 2;
        const li = [], ri = [];
        for (const i of idx) (X[i][f] <= thr ? li : ri).push(i);
        if (li.length < minSamplesLeaf || ri.length < minSamplesLeaf) continue;
        const gain = base - sse(li.map((i) => y[i])) - sse(ri.map((i) => y[i]));
        if (gain > best.gain) best = { gain, feature: f, threshold: thr, li, ri };
      }
    }
    if (best.feature < 0 || best.gain <= 1e-15) {
      return { leaf: true, value: mu, n: idx.length };
    }
    return {
      leaf: false, feature: best.feature, threshold: best.threshold, n: idx.length,
      left: build(best.li, depth + 1),
      right: build(best.ri, depth + 1),
    };
  };

  return build(X.map((_, i) => i), 0);
}

/** 回归树单点预测 */
export function regressionTreePredict(node, x) {
  let cur = node;
  while (!cur.leaf) {
    cur = x[cur.feature] <= cur.threshold ? cur.left : cur.right;
  }
  return cur.value;
}

/** 回归树批量预测 */
export function regressionTreePredictAll(node, X) {
  return X.map((x) => regressionTreePredict(node, x));
}

/**
 * GBDT 拟合。
 * @returns {{trees:object[], lr:number, init:number, trainLoss:number[], residuals:number[][]}}
 */
export function gbdtFit(X, y, {
  nEstimators = 50, lr = 0.1, maxDepth = 3, minSamplesSplit = 2, minSamplesLeaf = 1,
} = {}) {
  const n = X.length;
  const init = y.reduce((a, b) => a + b, 0) / n;
  let F = new Array(n).fill(init);
  const trees = [];
  const trainLoss = [];
  const residuals = [];
  const mse = (pred) => {
    let s = 0;
    for (let i = 0; i < n; i++) s += (y[i] - pred[i]) ** 2;
    return s / n;
  };
  trainLoss.push(mse(F));
  for (let m = 0; m < nEstimators; m++) {
    // 负梯度（MSE 损失下即残差）
    const r = new Array(n);
    for (let i = 0; i < n; i++) r[i] = y[i] - F[i];
    residuals.push(r);
    const tree = regressionTreeFit(X, r, { maxDepth, minSamplesSplit, minSamplesLeaf });
    trees.push(tree);
    for (let i = 0; i < n; i++) F[i] += lr * regressionTreePredict(tree, X[i]);
    trainLoss.push(mse(F));
  }
  return { trees, lr, init, trainLoss, residuals, finalPred: F };
}

/** GBDT 预测（累加全部树的收缩输出） */
export function gbdtPredict(model, X) {
  const n = X.length;
  const out = new Array(n).fill(model.init);
  for (const tree of model.trees) {
    for (let i = 0; i < n; i++) out[i] += model.lr * regressionTreePredict(tree, X[i]);
  }
  return out;
}

/**
 * 分阶段预测序列（每轮一个快照），用于验证损失的单调下降轨迹。
 * @returns {number[][]} staged[m][i] = 第 m 轮后对 X[i] 的预测
 */
export function gbdtStagedPredict(model, X) {
  const n = X.length;
  const F = new Array(n).fill(model.init);
  const staged = [F.slice()];
  for (const tree of model.trees) {
    for (let i = 0; i < n; i++) F[i] += model.lr * regressionTreePredict(tree, X[i]);
    staged.push(F.slice());
  }
  return staged;
}

/** 皮尔逊相关系数（用于「树 ≡ 残差方向」的不变量） */
export function pearsonCorr(a, b) {
  const n = a.length;
  const ma = a.reduce((s, v) => s + v, 0) / n;
  const mb = b.reduce((s, v) => s + v, 0) / n;
  let num = 0, da = 0, db = 0;
  for (let i = 0; i < n; i++) {
    num += (a[i] - ma) * (b[i] - mb);
    da += (a[i] - ma) ** 2;
    db += (b[i] - mb) ** 2;
  }
  return num / Math.sqrt(da * db);
}

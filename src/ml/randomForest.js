/**
 * 旗舰AI引擎 · 随机森林（Random Forest，Breiman 2001）
 * 站在 Breiman 2001《Random Forests》。
 *
 * 设计要点（复用 tree.js，不重新实现树）：
 *   · 不纯度计算复用 tree.js 的 giniImpurity（分类）；
 *   · 单棵基准树（方差缩减对照）复用 tree.js 的 DecisionTree（见 rfSingleTree）；
 *   · 现有 DecisionTree 的分裂为「全特征」实现、无法注入「每节点 mtry 随机特征子集」，
 *     故本模块以轻量 CART 节点（rf* 前缀 helper）实现支持 per-node mtry 的树，
 *     不再重定义 DecisionTree / giniImpurity，仅复用二者。
 *
 * 算法：
 *   · 每棵树在 bootstrap（有放回自助采样）子集上训练；
 *   · 每节点分裂时随机选取 mtry 个特征（分类默认 round(√d)，回归 round(d/3)）；
 *   · 分类多数投票、回归均值；
 *   · OOB 误差：对每个样本用「未包含该样本」的树投票；
 *   · 特征重要性：汇总各树基于 Gini/方差不纯度下降的归一化重要性，使其和 ≡ 1。
 *
 * 导出：
 *   randomForestTrain(X, y, opts) → { trees, oobIndices, featureImportances, numTrees, classes?, isRegression }
 *   randomForestPredict(X, model) → { predictions, probabilities? }
 *   rfSingleTree(X, y, opts)      → 单棵未集成 DecisionTree 基准（对照/诊断）
 *
 * 不变量（见 verify/invariants.js · suiteRandomForest）：
 *   · featureImportances 之和 ≡ 1（≤1e-9）；
 *   · trees.length === numTrees；
 *   · twoMoons(n=200,sigma=0.05) 分类准确率 ≥ 0.95；
 *   · 方差缩减：RF 准确率 ≥ 单棵未集成树均值（且 ≥ 单棵 DecisionTree 基准）；
 *   · OOB 误差 < 0.1；
 *   · 预测类别 ≡ argmax(probabilities)。
 */

import { DecisionTree, giniImpurity } from './tree.js';
import { mulberry32 } from '../core/stats.js';

/** 回归不纯度：方差（均方误差） */
function rfVariance(values) {
  const n = values.length;
  if (n === 0) return 0;
  let m = 0;
  for (let i = 0; i < n; i++) m += values[i];
  m /= n;
  let s = 0;
  for (let i = 0; i < n; i++) { const d = values[i] - m; s += d * d; }
  return s / n;
}

/** 从 [0,d) 中无放回抽取 k 个特征下标（部分 Fisher–Yates 洗牌） */
function rfSampleFeatures(d, k, rng) {
  const pool = new Array(d);
  for (let i = 0; i < d; i++) pool[i] = i;
  for (let i = 0; i < k; i++) {
    const j = i + Math.floor(rng() * (d - i));
    const tmp = pool[i]; pool[i] = pool[j]; pool[j] = tmp;
  }
  return pool.slice(0, k);
}

/** 分类叶节点：多数类 + 类概率 Map */
function rfClassLeaf(y, idx) {
  const cnt = new Map();
  for (let i = 0; i < idx.length; i++) {
    const c = y[idx[i]];
    cnt.set(c, (cnt.get(c) || 0) + 1);
  }
  const proba = new Map();
  const total = idx.length;
  for (const [c, k] of cnt) proba.set(c, k / total);
  let bestC = null, bc = -Infinity;
  for (const [c, p] of proba) if (p > bc) { bc = p; bestC = c; }
  return { leaf: true, label: bestC, proba };
}

/** 回归叶节点：均值 */
function rfRegLeaf(y, idx) {
  let s = 0;
  for (let i = 0; i < idx.length; i++) s += y[idx[i]];
  return { leaf: true, value: idx.length ? s / idx.length : 0 };
}

/**
 * 递归构建一棵 CART（支持每节点 mtry 随机特征子集）。
 * @param {number[][]} X
 * @param {Array} y
 * @param {number[]} idx 该节点样本下标
 * @param {number} depth
 * @param {object} opts {maxDepth, minSamples, isRegression}
 * @param {object} ctx {rng, mtry, importance, nTotal}
 */
function rfBuild(X, y, idx, depth, opts, ctx) {
  const { maxDepth, minSamples, isRegression } = opts;
  const n = idx.length;
  if (depth >= maxDepth || n < minSamples) {
    return isRegression ? rfRegLeaf(y, idx) : rfClassLeaf(y, idx);
  }
  const parentImp = isRegression
    ? rfVariance(idx.map((i) => y[i]))
    : giniImpurity(idx.map((i) => y[i]));
  if (parentImp <= 1e-12) {
    return isRegression ? rfRegLeaf(y, idx) : rfClassLeaf(y, idx);
  }
  const d = X[0].length;
  const mtry = Math.min(ctx.mtry, d);
  const feats = rfSampleFeatures(d, mtry, ctx.rng);
  let best = null; // {feature, threshold, imp, left, right}
  for (let fi = 0; fi < feats.length; fi++) {
    const f = feats[fi];
    const sorted = idx.slice().sort((a, b) => X[a][f] - X[b][f]);
    for (let t = 1; t < sorted.length; t++) {
      const v1 = X[sorted[t - 1]][f];
      const v2 = X[sorted[t]][f];
      if (v2 - v1 < 1e-12) continue;
      const thr = 0.5 * (v1 + v2);
      const left = [], right = [];
      for (let i = 0; i < n; i++) (X[idx[i]][f] <= thr ? left : right).push(idx[i]);
      if (left.length === 0 || right.length === 0) continue;
      const lImp = isRegression ? rfVariance(left.map((i) => y[i])) : giniImpurity(left.map((i) => y[i]));
      const rImp = isRegression ? rfVariance(right.map((i) => y[i])) : giniImpurity(right.map((i) => y[i]));
      const cg = (left.length / n) * lImp + (right.length / n) * rImp;
      if (!best || cg < best.imp) best = { feature: f, threshold: thr, imp: cg, left, right };
    }
  }
  if (!best) {
    return isRegression ? rfRegLeaf(y, idx) : rfClassLeaf(y, idx);
  }
  // 特征重要性：不纯度下降按节点样本量加权（最后统一归一化到和 = 1）
  ctx.importance[best.feature] += (n / ctx.nTotal) * (parentImp - best.imp);
  return {
    leaf: false,
    feature: best.feature,
    threshold: best.threshold,
    left: rfBuild(X, y, best.left, depth + 1, opts, ctx),
    right: rfBuild(X, y, best.right, depth + 1, opts, ctx),
  };
}

/** 单样本遍历一棵树，返回叶节点 */
function rfPredictLeaf(node, x) {
  while (!node.leaf) node = (x[node.feature] <= node.threshold) ? node.left : node.right;
  return node;
}

/**
 * 训练随机森林。
 * @param {number[][]} X 特征矩阵 (n×d)
 * @param {Array} y 标签（分类为整数类标；回归为数值）
 * @param {object} [opts] {numTrees=100, maxDepth=10, minSamples=2, mtry=null, seed=42, isRegression=false}
 * @returns {{trees:object[], oobIndices:number[][], featureImportances:number[], numTrees:number, classes?:Array, isRegression:boolean}}
 */
export function randomForestTrain(X, y, opts = {}) {
  const numTrees = opts.numTrees != null ? opts.numTrees : 100;
  const maxDepth = opts.maxDepth != null ? opts.maxDepth : 10;
  const minSamples = opts.minSamples != null ? opts.minSamples : 2;
  const isRegression = !!opts.isRegression;
  const seed = opts.seed != null ? opts.seed : 42;
  const d = X[0].length;
  const N = X.length;
  const mtry = opts.mtry != null
    ? opts.mtry
    : (isRegression ? Math.max(1, Math.round(d / 3)) : Math.max(1, Math.round(Math.sqrt(d))));

  const classes = isRegression ? undefined : [...new Set(y)].sort((a, b) => a - b);
  const rng = mulberry32(seed >>> 0);
  const importance = new Array(d).fill(0);
  const trees = [];
  const oobIndices = [];

  for (let t = 0; t < numTrees; t++) {
    const bootIdx = new Array(N);
    const inBag = new Array(N).fill(false);
    for (let i = 0; i < N; i++) {
      const k = Math.floor(rng() * N);
      bootIdx[i] = k;
      inBag[k] = true;
    }
    const oob = [];
    for (let i = 0; i < N; i++) if (!inBag[i]) oob.push(i);
    oobIndices.push(oob);

    const ctx = { rng, mtry, importance, nTotal: N };
    const tree = rfBuild(X, y, bootIdx, 0, { maxDepth, minSamples, isRegression }, ctx);
    trees.push(tree);
  }

  // 归一化特征重要性 → 和 ≡ 1
  let s = 0;
  for (let f = 0; f < d; f++) s += importance[f];
  const featureImportances = new Array(d);
  if (s > 0) {
    for (let f = 0; f < d; f++) featureImportances[f] = importance[f] / s;
  } else {
    const u = 1 / d;
    for (let f = 0; f < d; f++) featureImportances[f] = u; // 退化保护：均匀分布
  }

  return { trees, oobIndices, featureImportances, numTrees, classes, isRegression };
}

/**
 * 预测。
 * @param {number[][]} X
 * @param {object} model randomForestTrain 的返回值
 * @returns {{predictions:Array, probabilities?:number[][]}}
 */
export function randomForestPredict(X, model) {
  const { trees, classes, isRegression } = model;
  const nTrees = trees.length;
  const predictions = [];
  const probabilities = isRegression ? undefined : [];

  for (let i = 0; i < X.length; i++) {
    const x = X[i];
    if (isRegression) {
      let s = 0;
      for (let t = 0; t < nTrees; t++) s += rfPredictLeaf(trees[t], x).value;
      predictions.push(s / nTrees);
    } else {
      const votes = new Array(classes.length).fill(0);
      for (let t = 0; t < nTrees; t++) {
        const leaf = rfPredictLeaf(trees[t], x);
        for (let k = 0; k < classes.length; k++) votes[k] += (leaf.proba.get(classes[k]) || 0);
      }
      let bestK = 0, bestV = -Infinity;
      for (let k = 0; k < classes.length; k++) if (votes[k] > bestV) { bestV = votes[k]; bestK = k; }
      predictions.push(classes[bestK]);
      const probs = new Array(classes.length);
      for (let k = 0; k < classes.length; k++) probs[k] = votes[k] / nTrees;
      probabilities.push(probs);
    }
  }
  return { predictions, probabilities };
}

/**
 * 单棵未集成基准树（复用 tree.js 的 DecisionTree），用于方差缩减对照/诊断。
 * @param {number[][]} X
 * @param {Array} y
 * @param {object} [opts] {maxDepth=10, minSamples=2}
 * @returns {DecisionTree}
 */
export function rfSingleTree(X, y, opts = {}) {
  const maxDepth = opts.maxDepth != null ? opts.maxDepth : 10;
  const minSamples = opts.minSamples != null ? opts.minSamples : 2;
  return new DecisionTree({ maxDepth, minSamples }).fit(X, y);
}

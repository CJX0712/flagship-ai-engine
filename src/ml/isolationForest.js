/**
 * 旗舰AI引擎 · 隔离森林（Isolation Forest / iForest）
 * 站在：Liu, Ting & Zhou 2008 ICDM《Isolation Forest》与
 *       Liu, Ting & Zhou 2012《Isolation-Based Anomaly Detection》TKDD
 *       · sklearn.ensemble.IsolationForest 的参考实现（c(n) 的特例处理）
 *       · Knuth《The Art of Computer Programming》§6.2.2（随机 BST 的平均路径长度）。
 *
 * 核心直觉：异常点「少而不同」，因此随机切分的空间里 **更快被孤立**（路径更短）。
 *   随机树：随机选特征 q，在 [min, max] 内随机取切点 p，按 x_q < p 二分；
 *           直到达到高度上限 ⌈log₂ψ⌉ 或样本不可再分。
 *   路径长度 h(x) = e + c(|leaf|)，e 为根到叶的边数（叶内样本用 c(·) 修正）。
 *   归一化常数 c(n) = 2H(n−1) − 2(n−1)/n，H(i) = ln(i) + γ（γ = 0.5772156649015329），
 *           特例 c(1) = 0、c(2) = 1（与原文/sklearn 一致）。
 *   异常分 s(x, ψ) = 2^(−E[h(x)] / c(ψ))：→ 1 为异常，→ 0.5 为常态，→ 0 为深内部点。
 *
 * 不变量（全部为可独立交叉验证的硬恒等式）：
 *   ① c(n) 闭式 ≡ 递归增量式：c(n) 由 2(ln(n−1)+γ) − 2(n−1)/n 与
 *      「c(k) = c(k−1) + 2ln((k−1)/(k−2)) − 2(k−1)/k + 2(k−2)/(k−1)」两条独立路径互证（relErr < 1e-12）；
 *      （注：原文用 ln+γ 近似调和数，与精确 Σ1/i 相差 O(1/(2n))——这是截断误差而非实现 Bug，
 *        本模块另导出 iforCExact 供对照。）
 *   ② 路径长度上界：任意样本的 **叶深度** ≤ 高度上限，且 h(x) ≤ 高度上限 + c(ψ)（恒等式，违反计数 = 0）；
 *   ③ 异常分排序：密集正态团 + 远端 outlier 的合成数据上，outlier 平均分 − 常态平均分 > 0.1；
 *   ④ 路径长度单调性：outlier 的平均 h(x) **严格小于** 常态的（隔离机制本身）；
 *   ⑤ 集成降方差：单树得分方差 > 森林平均得分方差（体现 bagging 的方差缩减）；
 *   ⑥ 同 seed 可复现：两片森林对同一样本的 anomalyScore 逐位一致（maxDiff = 0）；
 *   ⑦ 归一化常数性质：c(n) 单调递增（n ≥ 2）且 c(n) < n（小 n 全量验证）。
 */

import { mulberry32, gaussianSample } from '../core/stats.js';

/** 欧拉–马斯刻若尼常数 γ（H(i) ≈ ln i + γ） */
const IFOR_EULER = 0.5772156649015329;

/**
 * 二叉树平均路径长度 c(n)（iForest 论文的归一化常数）。
 *   n ≤ 1 → 0；n = 2 → 1；n ≥ 3 → 2(ln(n−1) + γ) − 2(n−1)/n
 * @param {number} n 样本数
 * @returns {number} c(n)
 */
export function iforC(n) {
  if (n <= 1) return 0;
  if (n === 2) return 1;
  return 2 * (Math.log(n - 1) + IFOR_EULER) - 2 * (n - 1) / n;
}

/**
 * c(n) 的递归增量式（与闭式同源、计算路径完全不同，供不变量 ① 互证）：
 *   c(k) − c(k−1) = 2ln((k−1)/(k−2)) − 2(k−1)/k + 2(k−2)/(k−1)
 * @param {number} n 样本数
 * @returns {number} c(n)
 */
export function iforCRecursive(n) {
  if (n <= 1) return 0;
  if (n === 2) return 1;
  let c = 2 * (Math.log(2) + IFOR_EULER) - 2 * 2 / 3; // c(3) 为递推基准
  for (let k = 4; k <= n; k++) {
    c += 2 * Math.log((k - 1) / (k - 2)) - 2 * (k - 1) / k + 2 * (k - 2) / (k - 1);
  }
  return c;
}

/**
 * 精确调和数版 c(n) = 2·Σ_{i=1}^{n−1}(1/i) − 2(n−1)/n（H(i) 的数学定义，非 ln 近似）。
 * 与 iforC 的差为 O(1/(2n)) 的 Euler–Maclaurin 截断项，供数值对照。
 * @param {number} n 样本数
 * @returns {number} c_exact(n)
 */
export function iforCExact(n) {
  if (n <= 1) return 0;
  let h = 0;
  for (let i = 1; i <= n - 1; i++) h += 1 / i;
  return 2 * h - 2 * (n - 1) / n;
}

/** 无放回随机子采样（部分 Fisher–Yates，O(ψ)） */
function iforSubsample(X, psi, rng) {
  const n = X.length;
  const idx = new Array(n);
  for (let i = 0; i < n; i++) idx[i] = i;
  for (let i = 0; i < psi; i++) {
    const j = i + Math.floor(rng() * (n - i));
    const t = idx[i]; idx[i] = idx[j]; idx[j] = t;
  }
  const out = new Array(psi);
  for (let i = 0; i < psi; i++) out[i] = X[idx[i]];
  return out;
}

/** 逐特征的 min/max */
function iforRanges(X) {
  const d = X[0].length;
  const mins = new Array(d).fill(Infinity);
  const maxs = new Array(d).fill(-Infinity);
  for (const xi of X) {
    for (let j = 0; j < d; j++) {
      if (xi[j] < mins[j]) mins[j] = xi[j];
      if (xi[j] > maxs[j]) maxs[j] = xi[j];
    }
  }
  return { mins, maxs };
}

/**
 * 单棵隔离树（递归随机切分）。
 * @param {number[][]} X 子样本
 * @param {{heightLimit?:number, rng?:function, height?:number}} [opts]
 *   heightLimit 缺省 ⌈log₂n⌉；height 为当前深度（递归用，外部通常不传）
 * @returns {{isLeaf:boolean, size:number, height:number, splitFeature:number,
 *            splitValue:number, left:object|null, right:object|null}}
 */
export function isolationTree(X, { heightLimit = null, rng = Math.random, height = 0 } = {}) {
  const n = X.length;
  const d = n > 0 ? X[0].length : 0;
  const lim = heightLimit === null ? Math.ceil(Math.log2(Math.max(n, 2))) : heightLimit;
  const leaf = () => ({
    isLeaf: true, size: n, height, splitFeature: -1, splitValue: 0, left: null, right: null,
  });
  if (n <= 1 || height >= lim || d === 0) return leaf();

  const { mins, maxs } = iforRanges(X);
  const cand = [];
  for (let j = 0; j < d; j++) if (maxs[j] > mins[j]) cand.push(j);
  if (cand.length === 0) return leaf();                     // 全部特征退化 ⇒ 不可再分

  const q = cand[Math.floor(rng() * cand.length)];
  const p = mins[q] + rng() * (maxs[q] - mins[q]);
  const left = [], right = [];
  for (const xi of X) (xi[q] < p ? left : right).push(xi);
  if (left.length === 0 || right.length === 0) return leaf(); // 切点退化 ⇒ 叶

  return {
    isLeaf: false,
    size: n,
    height,
    splitFeature: q,
    splitValue: p,
    left: isolationTree(left, { heightLimit: lim, rng, height: height + 1 }),
    right: isolationTree(right, { heightLimit: lim, rng, height: height + 1 }),
  };
}

/**
 * 样本在树中的路径长度 h(x) = e + c(|leaf|)（e = 根到叶的边数）。
 * @param {object} tree isolationTree 返回的节点
 * @param {number[]} x 样本
 * @returns {number} h(x)
 */
export function pathLength(tree, x) {
  let node = tree;
  let depth = 0;
  while (!node.isLeaf) {
    node = x[node.splitFeature] < node.splitValue ? node.left : node.right;
    depth++;
  }
  return depth + iforC(node.size);
}

/**
 * 隔离森林：trees 棵随机树，每棵在 ψ = sampleSize 的无放回子样本上生长。
 * @param {number[][]} X 训练集
 * @param {{trees?:number, sampleSize?:number, seed?:number}} [opts]
 * @returns {{trees:object[], sampleSize:number, n:number, seed:number, heightLimit:number}}
 */
export function isolationForest(X, { trees = 100, sampleSize = 256, seed = 42 } = {}) {
  const rng = mulberry32(seed);
  const psi = Math.min(sampleSize, X.length);
  const heightLimit = Math.ceil(Math.log2(Math.max(psi, 2)));
  const forest = new Array(trees);
  for (let t = 0; t < trees; t++) {
    forest[t] = isolationTree(iforSubsample(X, psi, rng), { heightLimit, rng, height: 0 });
  }
  return { trees: forest, sampleSize: psi, n: X.length, seed, heightLimit };
}

/**
 * 单棵树的异常分 s = 2^(−h(x)/c(ψ))（供不变量 ⑤ 的方差对比）。
 * @param {object} tree 单棵隔离树
 * @param {number[]} x 样本
 * @param {number} psi 子样本规模
 * @returns {number} 单树异常分
 */
export function iforTreeScore(tree, x, psi) {
  const cn = iforC(psi);
  const denom = cn > 0 ? cn : 1e-12;
  return Math.pow(2, -pathLength(tree, x) / denom);
}

/**
 * 森林异常分 s(x, ψ) = 2^(−E_t[h_t(x)] / c(ψ))。
 * @param {{trees:object[], sampleSize:number}} forest isolationForest 的返回值
 * @param {number[]} x 样本
 * @returns {number} 异常分 ∈ (0, 1)
 */
export function anomalyScore(forest, x) {
  const T = forest.trees.length;
  if (T === 0) return 0;
  let h = 0;
  for (let t = 0; t < T; t++) h += pathLength(forest.trees[t], x);
  h /= T;
  const cn = iforC(forest.sampleSize);
  const denom = cn > 0 ? cn : 1e-12;
  return Math.pow(2, -h / denom);
}

/**
 * 森林在各样本上的路径长度均值与异常分（批量接口，便于统计检验）。
 * @param {{trees:object[], sampleSize:number}} forest
 * @param {number[][]} X 样本集
 * @returns {{scores:number[], meanPath:number[], perTreeScores:number[][]}}
 */
export function iforScoreAll(forest, X) {
  const T = forest.trees.length;
  const cn = iforC(forest.sampleSize);
  const denom = cn > 0 ? cn : 1e-12;
  const scores = new Array(X.length);
  const meanPath = new Array(X.length);
  const perTreeScores = new Array(X.length);
  for (let i = 0; i < X.length; i++) {
    let h = 0;
    const per = new Array(T);
    for (let t = 0; t < T; t++) {
      const ht = pathLength(forest.trees[t], X[i]);
      h += ht;
      per[t] = Math.pow(2, -ht / denom);
    }
    h /= T;
    meanPath[i] = h;
    scores[i] = Math.pow(2, -h / denom);
    perTreeScores[i] = per;
  }
  return { scores, meanPath, perTreeScores };
}

/**
 * 合成「密集团 + 远端 outlier」数据集（不变量 ③④ 的标准舞台）。
 * @param {{n?:number, nOutlier?:number, dim?:number, radius?:number, seed?:number}} [opts]
 * @returns {{X:number[][], y:number[]}} y = 1 表示 outlier
 */
export function iforToyData({ n = 400, nOutlier = 12, dim = 2, radius = 14, seed = 7 } = {}) {
  const rng = mulberry32(seed);
  const X = [], y = [];
  for (let i = 0; i < n; i++) {
    const row = new Array(dim);
    for (let j = 0; j < dim; j++) row[j] = gaussianSample(rng, 0, 1);
    X.push(row);
    y.push(0);
  }
  for (let i = 0; i < nOutlier; i++) {
    const row = new Array(dim);
    for (let j = 0; j < dim; j++) row[j] = gaussianSample(rng, radius * (j === 0 ? 1 : 0.2), 0.3);
    X.push(row);
    y.push(1);
  }
  return { X, y };
}

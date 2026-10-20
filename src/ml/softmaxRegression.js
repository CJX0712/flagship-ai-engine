/**
 * 旗舰AI引擎 · 多类逻辑回归（Softmax Regression / Multinomial Logistic Regression）
 * 站在：Bishop《Pattern Recognition and Machine Learning》§4.3.4 ·
 *       Hastie, Tibshirani & Friedman《The Elements of Statistical Learning》§4.4 ·
 *       Bridle 1990《Probabilistic Interpretation of Feedforward Classification
 *       Network Outputs》（softmax 激活）。
 *
 * 模型：z_k = w_kᵀ x（含截距列），p_k = softmax(z)_k，
 *   损失 L = −(1/n) Σ_i log p_{y_i}（交叉熵），
 *   解析梯度 ∇_{w_k} = X_bᵀ (p_{·,k} − 1{y_i=k}) / n。
 * K=2 时退化为二分类逻辑回归（引擎 logisticRegression）：同一决策超平面、预测标签一致。
 *
 * 不变量：
 *   ① 解析梯度 ≡ 中心差分数值梯度（双判据：显著分量 maxRelErr < 1e-6）；
 *   ② K=2 退化：预测标签 ≡ logisticRegression 预测标签（同可分数据逐位一致）；
 *   ③ 概率归一：Σ_k p_{i,k} = 1（精确，softmax 定义）；
 *   ④ 交叉熵非负且训练单调不增；
 *   ⑤ 同 seed 两次训练 W 逐位一致。
 */

import { addIntercept, logisticRegression } from './regression.js';
import { softmax, relErr } from '../core/numerical.js';
import { mulberry32 } from '../core/stats.js';
import { dot } from '../core/linalg.js';

/** one-hot 编码：y 为类别下标数组 → n×K 矩阵（内部梯度构造用） */
function oneHot(y, K) {
  return y.map((c) => {
    const row = new Array(K).fill(0);
    row[c] = 1;
    return row;
  });
}

/**
 * 前向：给定权重列向量组 W（K 列，每列长度 d+1），返回概率矩阵 P 与平均交叉熵损失。
 * @param {number[][]} W 每列一个类别的权重向量
 * @param {number[][]} Xb 含截距的设计矩阵
 * @param {number[]} y 类别下标标签数组
 * @returns {{P:number[][], loss:number}}
 */
export function softmaxForward(W, Xb, y) {
  const n = Xb.length;
  const K = W.length;
  const P = Xb.map((row) => {
    const s = new Array(K);
    for (let k = 0; k < K; k++) s[k] = dot(row, W[k]);
    return softmax(s);
  });
  let loss = 0;
  for (let i = 0; i < n; i++) loss -= Math.log(Math.max(P[i][y[i]], 1e-12));
  return { P, loss: loss / n };
}

/**
 * 解析梯度：每列 G[k] = X_bᵀ (P_{:,k} − 1{y_i=k}) / n。
 * @param {number[]} y 类别下标标签数组
 */
export function softmaxGrad(W, Xb, y) {
  const n = Xb.length;
  const K = W.length;
  const d = W[0].length;
  const { P } = softmaxForward(W, Xb, y);
  const G = Array.from({ length: K }, () => new Array(d).fill(0));
  for (let i = 0; i < n; i++) {
    const row = Xb[i];
    for (let k = 0; k < K; k++) {
      const e = P[i][k] - (y[i] === k ? 1 : 0);
      const gk = G[k];
      for (let j = 0; j < d; j++) gk[j] += row[j] * e;
    }
  }
  for (let k = 0; k < K; k++) for (let j = 0; j < d; j++) G[k][j] /= n;
  return G;
}

/**
 * 梯度检验：解析梯度 vs 中心差分数值梯度（双判据 CS231n）。
 * @returns {{maxRelErr:number, maxAbsErr:number, significant:number}}
 */
export function softmaxGradientCheck(X, y, { eps = 1e-5, seed = 7 } = {}) {
  const Xb = addIntercept(X);
  const d = Xb[0].length;
  const K = Math.max(...y) + 1;
  const rng = mulberry32(seed);
  const W = Array.from({ length: K }, () => new Array(d).fill(0).map(() => (rng() - 0.5) * 0.5));
  const G = softmaxGrad(W, Xb, y);
  const lossAt = (Wp) => softmaxForward(Wp, Xb, y).loss;
  const pairs = [];
  for (let k = 0; k < K; k++) {
    for (let j = 0; j < d; j++) {
      const orig = W[k][j];
      W[k][j] = orig + eps; const lp = lossAt(W);
      W[k][j] = orig - eps; const lm = lossAt(W);
      W[k][j] = orig;
      pairs.push([G[k][j], (lp - lm) / (2 * eps)]);
    }
  }
  const maxAbsAna = Math.max(...pairs.map(([a]) => Math.abs(a)));
  const floor = 1e-3 * maxAbsAna;
  let maxRel = 0, maxAbs = 0, significant = 0;
  for (const [a, b] of pairs) {
    maxAbs = Math.max(maxAbs, Math.abs(a - b));
    if (Math.abs(a) >= floor) {
      significant++;
      maxRel = Math.max(maxRel, relErr(a, b));
    }
  }
  return { maxRelErr: maxRel, maxAbsErr: maxAbs, significant };
}

/**
 * 多类逻辑回归训练（全批量梯度下降）。
 * @returns {{W:number[][], K:number, losses:number[], predict:Function, predictProba:Function}}
 */
export function softmaxRegression(X, y, { lr = 0.5, iters = 400, seed = 1, K } = {}) {
  const Xb = addIntercept(X);
  const d = Xb[0].length;
  const KK = K != null ? K : Math.max(...y) + 1;
  const rng = mulberry32(seed);
  const W = Array.from({ length: KK }, () => new Array(d).fill(0).map(() => (rng() - 0.5) * 1e-3));
  const losses = [];
  for (let it = 0; it < iters; it++) {
    const G = softmaxGrad(W, Xb, y);
    losses.push(softmaxForward(W, Xb, y).loss);
    for (let k = 0; k < KK; k++) for (let j = 0; j < d; j++) W[k][j] -= lr * G[k][j];
  }
  losses.push(softmaxForward(W, Xb, y).loss);
  const predict = (X2) => {
    const Xb2 = addIntercept(X2);
    return Xb2.map((row) => {
      let best = 0;
      for (let k = 1; k < KK; k++) if (dot(row, W[k]) > dot(row, W[best])) best = k;
      return best;
    });
  };
  const predictProba = (X2) => addIntercept(X2).map((row) => {
    const s = new Array(KK);
    for (let k = 0; k < KK; k++) s[k] = dot(row, W[k]);
    return softmax(s);
  });
  return { W, K: KK, losses, predict, predictProba };
}

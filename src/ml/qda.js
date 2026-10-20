/**
 * 旗舰AI引擎 · 二次判别分析（QDA）
 * 站在：Fisher 1936《The Use of Multiple Measurements in Taxonomic Problems》·
 *       Bishop 2006《Pattern Recognition and Machine Learning》§4.3（各类协方差
 *       不同的高斯判别分析）。
 *
 * 判别三足鼎立：naiveBayes = 对角协方差（条件独立假设）、lda = 各类共享协方差
 * （线性边界）、qda = 各类完整协方差 Σ_c（二次边界）。对数判别式（Bishop 4.63）：
 *   δ_c(x) = log π_c − ½ log|Σ_c| − ½ (x−μ_c)ᵀ Σ_c⁻¹ (x−μ_c)
 * 后验 p(c|x) = softmax_c δ_c(x)（log 域计算、logsumexp 归一，数值稳定）。
 * 当各类共享同一 Σ 时二次项 −½xᵀΣ⁻¹x 对 argmax 无贡献 ⇒ 边界退化为线性（= LDA）。
 *
 * 不变量：
 *   ① classMeans ≡ 各类暴力均值（逐元素 < 1e-9）；
 *   ② 各类 covs 对称且 ≡ covarianceMatrix 无偏估计（/(n_c−1)，逐元素 < 1e-9）；
 *   ③ posterior 每行和 ≡ 1（< 1e-12）；
 *   ④ log-posterior ≡ 独立实现（solve 求逆 + 多元高斯对数密度直接算，< 1e-9）；
 *   ⑤ 协方差不同的可分合成数据：训练准确率 = 1.0；
 *   ⑥ 同协方差时 QDA 边界退化为线性：与 LDA + 最近类心预测标签完全一致。
 */

import { zeros, cholesky, inverse, matvec, dot } from '../core/linalg.js';
import { logsumexp } from '../core/numerical.js';

/**
 * 第 c 类样本协方差（/(n_c−1) 无偏估计）+ reg·I 保证正定。
 * 全量累加后单次除法（不得先镜像后除），对称性由构造保证。
 */
function qdaClassCovariance(rows, mu, reg) {
  const d = mu.length;
  const C = zeros(d, d);
  for (const r of rows) {
    for (let i = 0; i < d; i++) {
      const ei = r[i] - mu[i];
      for (let j = 0; j < d; j++) C[i][j] += ei * (r[j] - mu[j]);
    }
  }
  for (let i = 0; i < d; i++) {
    for (let j = 0; j < d; j++) C[i][j] /= rows.length - 1;
    C[i][i] += reg; // 岭式正则：保证 Cholesky 可分解、逆存在
  }
  return C;
}

/**
 * QDA 拟合：按类估计 μ_c、Σ_c（无偏 + reg·I）、Σ_c⁻¹、log|Σ_c|（Cholesky 对角线）、先验 π_c。
 * @param {number[][]} X 数据（n×d）
 * @param {number[]} y 类别标签（按首次出现序登记 classes）
 * @param {object} [opts] { reg=1e-6 }
 * @returns {{classMeans:number[][], covs:number[][], covsInv:number[][], logDets:number[], priors:number[], classes:*[]}}
 */
export function qdaFit(X, y, { reg = 1e-6 } = {}) {
  const classes = [];
  for (const yi of y) if (!classes.includes(yi)) classes.push(yi);
  const d = X[0].length;
  const classMeans = [];
  const covs = [];
  const covsInv = [];
  const logDets = [];
  const priors = [];
  for (const c of classes) {
    const rows = [];
    for (let i = 0; i < X.length; i++) if (y[i] === c) rows.push(X[i]);
    if (rows.length < 2) throw new Error('QDA：每个类别至少需要 2 个样本才能估计协方差');
    const mu = new Array(d).fill(0);
    for (const r of rows) for (let j = 0; j < d; j++) mu[j] += r[j];
    for (let j = 0; j < d; j++) mu[j] /= rows.length;
    const cov = qdaClassCovariance(rows, mu, reg);
    const L = cholesky(cov);
    let logDet = 0; // log|Σ| = 2·Σᵢ log Lᵢᵢ（Cholesky 对角线求和）
    for (let i = 0; i < d; i++) logDet += 2 * Math.log(L[i][i]);
    classMeans.push(mu);
    covs.push(cov);
    covsInv.push(inverse(cov));
    logDets.push(logDet);
    priors.push(rows.length / X.length);
  }
  return { classMeans, covs, covsInv, logDets, priors, classes };
}

/**
 * 单样本判别式向量 δ_c(x)（log 域，未归一）——供边界绘制与预测共用。
 * @param {number[]} x 单样本
 * @param {object} model qdaFit 的返回
 * @returns {number[]} 各类 δ 值
 */
export function qdaDiscriminant(x, model) {
  const { classMeans, covsInv, logDets, priors } = model;
  const d = x.length;
  return classMeans.map((mu, k) => {
    const diff = new Array(d);
    for (let j = 0; j < d; j++) diff[j] = x[j] - mu[j];
    const quad = dot(diff, matvec(covsInv[k], diff)); // (x−μ)ᵀΣ⁻¹(x−μ)
    return Math.log(priors[k]) - 0.5 * logDets[k] - 0.5 * quad;
  });
}

/**
 * QDA 预测：δ 经 logsumexp 归一得后验，argmax 得标签。
 * @param {number[][]} X 数据
 * @param {object} model qdaFit 的返回
 * @returns {{posteriors:number[][], labels:*[]}} labels 为 classes 中的类别标签
 */
export function qdaPredict(X, model) {
  const posteriors = X.map((xi) => {
    const delta = qdaDiscriminant(xi, model);
    const lse = logsumexp(delta);
    return delta.map((v) => Math.exp(v - lse));
  });
  const labels = posteriors.map((p) => {
    let bi = 0, bv = -Infinity;
    for (let k = 0; k < p.length; k++) if (p[k] > bv) { bv = p[k]; bi = k; }
    return model.classes[bi];
  });
  return { posteriors, labels };
}

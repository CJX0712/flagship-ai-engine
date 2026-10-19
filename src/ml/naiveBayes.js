/**
 * 旗舰AI引擎 · 高斯朴素贝叶斯（GNB）
 * 站在：Maron 1961 · Domingos & Pazzani 1997《On the Optimality of the Simple
 *       Bayesian Classifier under Zero-One Loss》（特征条件独立下最优）。
 *
 * 后验（对数域，数值稳定）：log p(c|x) ∝ log p(c) + Σ_j log N(x_j | μ_{cj}, σ²_{cj})，
 *   再以 softmax 回概率。条件独立假设把联合高斯拆成逐特征一维高斯之积。
 *
 * 不变量：
 *   ① priors 之和 ≡ 1；
 *   ② 每样本 posteriors 之和 ≡ 1（至 1e-9）；
 *   ③ 独立特征、类别可分合成数据准确率 = 1.0；
 *   ④ 单点对数项 ≡ 手算（gaussianLogPDF，差 < 1e-9）；
 *   ⑤ labels = argmax posteriors（逐样本核对）。
 */

import { softmax } from '../core/numerical.js';
import { gaussianLogPDF } from '../core/stats.js';

/**
 * GNB 拟合。
 * @param {number[][]} X 数据（n×d）
 * @param {number[]} y 类别标签
 * @param {object} [opts] { varSmooth=1e-9 }
 * @returns {{priors:number[], classMeans:number[][], classVars:number[][], classes:*[]}}
 */
export function nbFit(X, y, { varSmooth = 1e-9 } = {}) {
  const classes = [];
  for (const yi of y) if (!classes.includes(yi)) classes.push(yi);
  const d = X[0].length;
  const priors = [];
  const classMeans = [];
  const classVars = [];
  for (const c of classes) {
    const rows = [];
    for (let i = 0; i < X.length; i++) if (y[i] === c) rows.push(X[i]);
    const nC = rows.length;
    const mu = new Array(d).fill(0);
    for (const r of rows) for (let j = 0; j < d; j++) mu[j] += r[j];
    for (let j = 0; j < d; j++) mu[j] /= nC;
    const v = new Array(d).fill(0);
    for (const r of rows) for (let j = 0; j < d; j++) { const e = r[j] - mu[j]; v[j] += e * e; }
    // 全量累加后单次除以 nC（不下三角二次除），并加平滑避免 log(0)
    for (let j = 0; j < d; j++) v[j] = v[j] / nC + varSmooth;
    priors.push(nC / X.length);
    classMeans.push(mu);
    classVars.push(v);
  }
  return { priors, classMeans, classVars, classes };
}

/**
 * GNB 预测。
 * @param {number[][]} X 数据
 * @param {object} model nbFit 的返回
 * @returns {{posteriors:number[][], labels:number[]}}
 */
export function nbPredict(X, model) {
  const { priors, classMeans, classVars, classes } = model;
  const K = classes.length;
  const posteriors = [];
  for (let i = 0; i < X.length; i++) {
    const xi = X[i];
    const logp = new Array(K);
    for (let k = 0; k < K; k++) {
      const mu = classMeans[k];
      const va = classVars[k];
      let s = Math.log(priors[k] + 1e-300);
      for (let j = 0; j < xi.length; j++) s += gaussianLogPDF(xi[j], mu[j], Math.sqrt(va[j]));
      logp[k] = s;
    }
    posteriors.push(softmax(logp));
  }
  const labels = posteriors.map((p) => {
    let bi = 0, bv = -Infinity;
    for (let k = 0; k < p.length; k++) if (p[k] > bv) { bv = p[k]; bi = k; }
    return bi;
  });
  return { posteriors, labels };
}

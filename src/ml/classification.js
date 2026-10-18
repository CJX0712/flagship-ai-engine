/**
 * 旗舰AI引擎 · 分类模型
 * k-近邻（Cover & Hart 1967）/ 高斯朴素贝叶斯（拉普拉斯思想：方差平滑防零）。
 */

import { norm } from '../core/linalg.js';
import { gaussianLogPDF, mean } from '../core/stats.js';

/**
 * kNN 预测：k 近邻多数表决（平票取较小标签，保证确定性）。
 * 不变量：k=1 时训练集零误差（过拟合边界情形）。
 */
export function knnPredict(Xtrain, ytrain, x, k = 3) {
  const dists = Xtrain.map((row, i) => ({ d: norm(row.map((v, j) => v - x[j])), label: ytrain[i] }));
  dists.sort((a, b) => a.d - b.d);
  const votes = new Map();
  for (let i = 0; i < Math.min(k, dists.length); i++) {
    const l = dists[i].label;
    votes.set(l, (votes.get(l) || 0) + 1);
  }
  let best = null, bestCnt = -1;
  for (const [label, cnt] of [...votes.entries()].sort((a, b) => a[0] - b[0])) {
    if (cnt > bestCnt) { best = label; bestCnt = cnt; }
  }
  return best;
}

/**
 * 高斯朴素贝叶斯拟合：按类估计每维均值/方差（+1e-9 平滑）。
 */
export function naiveBayesFit(X, y) {
  const classes = [...new Set(y)].sort((a, b) => a - b);
  const byClass = new Map(classes.map(c => [c, []]));
  for (let i = 0; i < X.length; i++) byClass.get(y[i]).push(X[i]);
  const model = { classes, means: {}, vars: {}, priors: {} };
  const n = X.length;
  for (const c of classes) {
    const rows = byClass.get(c);
    const d = rows[0].length;
    const mu = mean(rows);
    const varr = new Array(d).fill(0);
    for (const row of rows) for (let j = 0; j < d; j++) varr[j] += (row[j] - mu[j]) ** 2;
    model.means[c] = mu;
    model.vars[c] = varr.map(v => v / rows.length + 1e-9);
    model.priors[c] = rows.length / n;
  }
  return model;
}

/** 朴素贝叶斯预测：后验归一化。不变量：各类后验之和 = 1。 */
export function naiveBayesPredict(model, x) {
  const scores = model.classes.map(c => {
    let s = Math.log(model.priors[c]);
    for (let j = 0; j < x.length; j++) s += gaussianLogPDF(x[j], model.means[c][j], Math.sqrt(model.vars[c][j]));
    return s;
  });
  const m = Math.max(...scores);
  const e = scores.map(s => Math.exp(s - m));
  const Z = e.reduce((a, b) => a + b, 0);
  const probs = e.map(v => v / Z);
  let best = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
  return { label: model.classes[best], probs };
}

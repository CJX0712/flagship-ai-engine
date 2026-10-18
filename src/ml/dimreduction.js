/**
 * 旗舰AI引擎 · 主成分分析
 * 经 Jacobi 特征分解（对称阵金标准），非幂迭代近似。
 * 不变量：主轴正交归一；全秩重构恒等；解释方差降序且和 = 总方差。
 */

import { jacobiEig } from '../core/linalg.js';
import { covarianceMatrix, mean } from '../core/stats.js';

/**
 * PCA：返回 { mu, components(k 条主轴), explained(k 个特征值), proj(n×k), reconstruct(n×d), totalVar }。
 * k = 数据维度时 reconstruct 与中心化数据逐元素相等（浮点容差内）。
 */
export function pca(X, k) {
  const n = X.length, d = X[0].length;
  const kk = Math.min(k, d);
  const mu = mean(X);
  const Xc = X.map(row => row.map((v, j) => v - mu[j]));
  const C = covarianceMatrix(X);
  const { values, vectors } = jacobiEig(C); // 降序
  const components = vectors.slice(0, kk);   // 每条为主轴（单位向量）
  const explained = values.slice(0, kk);
  const proj = Xc.map(row => components.map(v => row.reduce((s, x, j) => s + x * v[j], 0)));
  const reconstruct = proj.map(pr => {
    const back = new Array(d).fill(0);
    for (let t = 0; t < kk; t++) for (let j = 0; j < d; j++) back[j] += pr[t] * components[t][j];
    return back.map((v, j) => v + mu[j]);
  });
  const totalVar = values.reduce((a, b) => a + b, 0);
  return { mu, components, explained, proj, reconstruct, totalVar };
}

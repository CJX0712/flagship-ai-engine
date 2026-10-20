/**
 * 旗舰AI引擎 · 弹性网络回归（Elastic Net, Zou & Hastie 2005）
 * 站在：Zou & Hastie 2005《Regularization and Variable Selection via the
 *       Elastic Net》（解决 Lasso 在相关特征下的「最多选一个」缺陷，兼具 L1 稀疏与 L2 分组）·
 *       Friedman, Hastie & Tibshirani 2010《Regularization Paths for Generalized
 *       Linear Models via Coordinate Descent》（glmnet 循环坐标下降）。
 *
 * 目标（X 列中心化、y 中心化，截距单独估计、不受惩罚）：
 *   min_w ½‖ỹ − X̃w‖² + λ1‖w‖₁ + ½ λ2‖w‖₂²
 * 坐标闭式更新（软阈值 + L2 收缩）：
 *   wⱼ ← S(ρⱼ, λ1) / (zⱼ + λ2)，
 *   ρⱼ = Σᵢ x̃ᵢⱼ rᵢ⁽⁻ʲ⁾（残差增量维护），zⱼ = Σᵢ x̃ᵢⱼ²。
 * 退化恒等式：λ1=0 ≡ 岭回归（引擎 ridgeRegression）；λ2=0 ≡ Lasso（引擎 lassoFit）；
 *           λ1=λ2=0 ≡ 普通最小二乘（引擎 linearRegression）。
 *
 * 不变量：
 *   ① λ1=0 解 ≡ ridgeRegression（w 与 b 逐元素 < 1e-6）；
 *   ② λ2=0 解 ≡ lassoFit（w 与 b 逐元素 < 1e-7）；
 *   ③ λ1=λ2=0 解 ≡ linearRegression 斜率（< 1e-6）；
 *   ④ 正交设计闭式：X̃ᵀX̃ = I 时 wⱼ = S(β̂ⱼ, λ1)/(1 + λ2)（β̂=OLS，< 1e-9）；
 *   ⑤ KKT 证书：解处 ∀j |ρⱼ| ≤ λ1(1+1e-6) 且 wⱼ≠0 ⇒ ρⱼ = λ1·sign(wⱼ)（< 1e-7）；
 *   ⑥ 坐标下降目标单调不增。
 */

import { dot } from '../core/linalg.js';
import { linearRegression, ridgeRegression } from './regression.js';
import { lassoFit } from './lasso.js';

/** 软阈值算子 S(ρ,λ) = sign(ρ)·max(|ρ|−λ, 0)（λ=0 时为恒等） */
function enSoftThreshold(rho, lambda) {
  if (rho > lambda) return rho - lambda;
  if (rho < -lambda) return rho + lambda;
  return 0;
}

/**
 * 弹性网络坐标下降拟合。
 * @param {number[][]} X 数据（n×d）
 * @param {number[]} y 响应（n）
 * @param {object} [opts] { lambda1=1, lambda2=1, iters=2000, tol=1e-12 }
 * @returns {{w:number[], b:number, lambda1:number, lambda2:number, nIter:number,
 *            activeSet:number[], objective:number[]}}
 */
export function elasticNetFit(X, y, { lambda1 = 1, lambda2 = 1, iters = 2000, tol = 1e-12 } = {}) {
  const n = X.length;
  const d = X[0].length;
  const xbar = new Array(d).fill(0);
  for (const r of X) for (let j = 0; j < d; j++) xbar[j] += r[j];
  for (let j = 0; j < d; j++) xbar[j] /= n;
  let ybar = 0;
  for (const v of y) ybar += v;
  ybar /= n;
  const Xc = X.map((r) => {
    const t = new Array(d);
    for (let j = 0; j < d; j++) t[j] = r[j] - xbar[j];
    return t;
  });
  const yc = y.map((v) => v - ybar);
  const z = new Array(d).fill(0);
  for (let j = 0; j < d; j++) for (let i = 0; i < n; i++) z[j] += Xc[i][j] * Xc[i][j];
  const w = new Array(d).fill(0);
  const r = yc.slice(); // 残差 r = ỹ − X̃w（w=0 初值 ⇒ r=yc）
  const objective = [];
  let nIter = 0;
  for (let it = 0; it < iters; it++) {
    nIter = it + 1;
    let maxDelta = 0;
    for (let j = 0; j < d; j++) {
      if (z[j] === 0) continue; // 常数列无可学信息
      let rho = z[j] * w[j];
      for (let i = 0; i < n; i++) rho += Xc[i][j] * r[i];
      const wNew = enSoftThreshold(rho, lambda1) / (z[j] + lambda2);
      if (wNew !== w[j]) {
        const delta = wNew - w[j];
        for (let i = 0; i < n; i++) r[i] -= Xc[i][j] * delta; // 残差增量维护
        w[j] = wNew;
        const ad = Math.abs(delta);
        if (ad > maxDelta) maxDelta = ad;
      }
    }
    // 目标：½‖r‖² + λ1‖w‖₁ + ½λ2‖w‖₂²（r 为当前全残差）
    let rr = 0;
    for (let i = 0; i < n; i++) rr += r[i] * r[i];
    let l1 = 0, l2 = 0;
    for (let j = 0; j < d; j++) { l1 += Math.abs(w[j]); l2 += w[j] * w[j]; }
    objective.push(0.5 * rr + lambda1 * l1 + 0.5 * lambda2 * l2);
    if (maxDelta < tol) break;
  }
  let b = ybar;
  for (let j = 0; j < d; j++) b -= w[j] * xbar[j];
  const activeSet = [];
  for (let j = 0; j < d; j++) if (w[j] !== 0) activeSet.push(j);
  return { w, b, lambda1, lambda2, nIter, activeSet, objective };
}

/**
 * 弹性网络预测：ŷ = b + X·w。
 * @returns {number[]}
 */
export function elasticNetPredict(X, model) {
  const { w, b } = model;
  return X.map((xi) => b + dot(xi, w));
}

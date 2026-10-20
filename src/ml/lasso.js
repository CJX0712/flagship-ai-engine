/**
 * 旗舰AI引擎 · Lasso 稀疏线性回归（坐标下降）
 * 站在：Tibshirani 1996《Regression Shrinkage and Selection via the Lasso》·
 *       Friedman, Hastie & Tibshirani 2010《Regularization Paths for Generalized
 *       Linear Models via Coordinate Descent》（glmnet）· Efron et al. 2004 LARS。
 *
 * 目标函数：min_w ½‖ỹ − X̃w‖² + λ‖w‖₁
 *   X 按列中心化、y 中心化 ⇒ 截距 b = ȳ − wᵀx̄ 不进入惩罚项（glmnet 标准做法）。
 * 坐标闭式更新（软阈值，L1 近端显式解）：
 *   wⱼ ← softThreshold(ρⱼ, λ) / zⱼ，
 *   ρⱼ = Σᵢ x̃ᵢⱼ rᵢ⁽⁻ʲ⁾（r⁽⁻ʲ⁾ = ỹ − X̃w 除去第 j 分量，增量维护），
 *   zⱼ = Σᵢ x̃ᵢⱼ²（常数列 zⱼ=0 ⇒ wⱼ 恒 0）。
 * 循环至 ‖Δw‖∞ < tol（坐标下降 ⇒ 凸目标，必收敛到全局最优）。
 *
 * 不变量：
 *   ① KKT 最优性证书：解处 ∀j |ρⱼ| ≤ λ(1+1e-6) 且 wⱼ≠0 ⇒ ρⱼ = λ·sign(wⱼ)（<1e-7）；
 *   ② 正交设计闭式：X̃ᵀX̃ = n·I 时 w ≡ OLS 系数逐元素 soft-threshold（阈值 λ/n，<1e-10）；
 *   ③ λ=0 解 ≡ linearRegression（<1e-6）；
 *   ④ λ ≥ maxⱼ|Σᵢx̃ᵢⱼỹᵢ| ⇒ w 全 0（精确）；
 *   ⑤ 路径单调：λ↓ ⇒ ‖w‖₁ 非减；
 *   ⑥ 欠定 d>n ⇒ ‖w‖₀ ≤ min(n,d)。
 */

import { dot } from '../core/linalg.js';

/** soft-threshold 算子：S(ρ,λ) = sign(ρ)·max(|ρ|−λ, 0)（λ=0 时为恒等） */
function lassoSoftThreshold(rho, lambda) {
  if (rho > lambda) return rho - lambda;
  if (rho < -lambda) return rho + lambda;
  return 0;
}

/**
 * Lasso 坐标下降拟合。
 * @param {number[][]} X 数据（n×d）
 * @param {number[]} y 响应（n）
 * @param {object} [opts] { lambda=1, iters=2000, tol=1e-12 }
 * @returns {{w:number[], b:number, lambda:number, nIter:number, activeSet:number[]}}
 */
export function lassoFit(X, y, { lambda = 1, iters = 2000, tol = 1e-12 } = {}) {
  const n = X.length, d = X[0].length;
  // 列中心化与响应中心化（截距由此单独估计、不受惩罚）
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
  // zⱼ = Σᵢ x̃ᵢⱼ²（一次预计算，循环内复用）
  const z = new Array(d).fill(0);
  for (let j = 0; j < d; j++) for (let i = 0; i < n; i++) z[j] += Xc[i][j] * Xc[i][j];
  const w = new Array(d).fill(0);
  const r = yc.slice(); // 残差 r = ỹ − X̃w（w=0 初值 ⇒ r=yc）
  let nIter = 0;
  for (let it = 0; it < iters; it++) {
    nIter = it + 1;
    let maxDelta = 0;
    for (let j = 0; j < d; j++) {
      if (z[j] === 0) continue; // 常数列无可学信息
      // ρⱼ = x̃ⱼᵀr + zⱼ·wⱼ（把第 j 分量加回残差，避免重算全残差）
      let rho = z[j] * w[j];
      for (let i = 0; i < n; i++) rho += Xc[i][j] * r[i];
      const wNew = lassoSoftThreshold(rho, lambda) / z[j];
      if (wNew !== w[j]) {
        const delta = wNew - w[j];
        for (let i = 0; i < n; i++) r[i] -= Xc[i][j] * delta; // 残差增量维护
        w[j] = wNew;
        const ad = Math.abs(delta);
        if (ad > maxDelta) maxDelta = ad;
      }
    }
    if (maxDelta < tol) break; // ‖Δw‖∞ < tol ⇒ 收敛
  }
  // 截距：b = ȳ − Σⱼ wⱼ x̄ⱼ（不惩罚）
  let b = ybar;
  for (let j = 0; j < d; j++) b -= w[j] * xbar[j];
  const activeSet = [];
  for (let j = 0; j < d; j++) if (w[j] !== 0) activeSet.push(j);
  return { w, b, lambda, nIter, activeSet };
}

/**
 * Lasso 正则化路径：λ 从大到小求解（内部按降序排序，与传入顺序无关）。
 * @param {number[][]} X 数据
 * @param {number[]} y 响应
 * @param {number[]} lambdas λ 序列（自动按从大到小排序）
 * @param {object} [opts] 透传给 lassoFit（如 { iters }）
 * @returns {Array<object>} 与降序 λ 对齐的解数组
 */
export function lassoPath(X, y, lambdas, opts = {}) {
  const sorted = lambdas.slice().sort((a, b) => b - a); // λ 从大到小（正则化路径惯例）
  return sorted.map((lambda) => lassoFit(X, y, { ...opts, lambda }));
}

/**
 * Lasso 预测：ŷ = b + X·w。
 * @param {number[][]} X 数据
 * @param {object} model lassoFit 的返回
 * @returns {number[]}
 */
export function lassoPredict(X, model) {
  const { w, b } = model;
  return X.map((xi) => b + dot(xi, w));
}

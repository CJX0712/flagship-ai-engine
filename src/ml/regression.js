/**
 * 旗舰AI引擎 · 回归模型
 * 线性回归（正规方程）/ 岭回归 / 逻辑回归（含解析梯度与梯度检验）。
 * 梯度检验遵循 CS231n 惯例：中心差分 vs 解析梯度，maxRelErr ≈ 1e-8。
 */

import { matmul, matvec, transpose, solve, identity, zeros, dot, norm } from '../core/linalg.js';
import { sigmoid, relErr } from '../core/numerical.js';
import { mulberry32 } from '../core/stats.js';

/** 设计矩阵加截距列（首列全 1） */
export function addIntercept(X) {
  return X.map(row => [1, ...row]);
}

/**
 * 普通最小二乘：ŵ = argmin ‖X_b w − y‖²。
 * 不变量（正规方程）：X_bᵀ (y − X_b ŵ) = 0，即残差与设计矩阵正交。
 */
export function linearRegression(X, y) {
  const Xb = addIntercept(X);
  const Xt = transpose(Xb);
  const w = solve(matmul(Xt, Xb), matvec(Xt, y));
  return { w, predict: (X2) => matvec(addIntercept(X2), w) };
}

/**
 * 岭回归：ŵ = argmin ‖X_b w − y‖² + λ‖w_{1:}‖²（截距不正则）。
 * 不变量：系数收缩 ‖w_{1:}‖₂ ≤ ‖w_OLS,{1:}‖₂；λ→0 时退化为 OLS。
 */
export function ridgeRegression(X, y, lambda = 1) {
  const Xb = addIntercept(X);
  const Xt = transpose(Xb);
  const XtX = matmul(Xt, Xb);
  const d = XtX.length;
  const P = identity(d);
  P[0][0] = 0; // 截距不正则
  for (let i = 0; i < d; i++) XtX[i][i] += lambda * (P[i][i] ? 1 : 0);
  const w = solve(XtX, matvec(Xt, y));
  return { w, predict: (X2) => matvec(addIntercept(X2), w) };
}

/** 逻辑回归平均损失与解析梯度：L = −(1/n)Σ[y log p + (1−y)log(1−p)]，∇ = X_bᵀ(p − y)/n */
export function logisticLossGrad(w, Xb, y) {
  const n = Xb.length;
  let loss = 0;
  const p = new Array(n);
  for (let i = 0; i < n; i++) {
    p[i] = sigmoid(dot(Xb[i], w));
    const pc = Math.max(Math.min(p[i], 1 - 1e-12), 1e-12);
    loss -= y[i] * Math.log(pc) + (1 - y[i]) * Math.log(1 - pc);
  }
  const g = zeros(Xb[0].length, 1).map(r => r[0]); // 长度 d+1 的零向量
  for (let i = 0; i < n; i++) {
    const e = p[i] - y[i];
    for (let j = 0; j < g.length; j++) g[j] += Xb[i][j] * e;
  }
  return { loss: loss / n, grad: g.map(v => v / n), p };
}

/**
 * 逻辑回归（梯度下降）。
 * 返回 { w, losses, predict, predictProba }；losses 记录每轮平均损失（单调下降趋势）。
 */
export function logisticRegression(X, y, { lr = 0.5, iters = 400, seed = 1 } = {}) {
  const Xb = addIntercept(X);
  const d = Xb[0].length;
  const rng = mulberry32(seed);
  const w = new Array(d).fill(0).map(() => (rng() - 0.5) * 1e-3);
  const losses = [];
  for (let it = 0; it < iters; it++) {
    const { loss, grad } = logisticLossGrad(w, Xb, y);
    losses.push(loss);
    for (let j = 0; j < d; j++) w[j] -= lr * grad[j];
  }
  losses.push(logisticLossGrad(w, Xb, y).loss);
  return {
    w,
    losses,
    predictProba: (X2) => addIntercept(X2).map(row => sigmoid(dot(row, w))),
    predict: (X2) => addIntercept(X2).map(row => (sigmoid(dot(row, w)) >= 0.5 ? 1 : 0)),
  };
}

/**
 * 逻辑回归梯度检验：中心差分 vs 解析梯度。
 * 不变量：maxRelErr ≈ 1e-8（阈值 1e-6）。
 */
export function logisticGradientCheck(X, y, { eps = 1e-5, seed = 7 } = {}) {
  const Xb = addIntercept(X);
  const d = Xb[0].length;
  const rng = mulberry32(seed);
  const w = new Array(d).fill(0).map(() => (rng() - 0.5) * 0.5);
  const { grad } = logisticLossGrad(w, Xb, y);
  let maxErr = 0;
  for (let j = 0; j < d; j++) {
    const orig = w[j];
    w[j] = orig + eps;
    const lp = logisticLossGrad(w, Xb, y).loss;
    w[j] = orig - eps;
    const lm = logisticLossGrad(w, Xb, y).loss;
    w[j] = orig;
    const num = (lp - lm) / (2 * eps);
    maxErr = Math.max(maxErr, relErr(num, grad[j]));
  }
  return maxErr;
}

/** ‖v‖∞ 辅助（自检用） */
export function linf(v) {
  return norm(v, Infinity) || 0;
}

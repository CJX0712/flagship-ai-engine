/**
 * 旗舰AI引擎 · 核支持向量机（SVM, Cortes & Vapnik 1995；SMO 求解 Platt 1998）
 * 对偶问题：max_α Σαᵢ − ½ΣΣ αᵢαⱼyᵢyⱼK(xᵢ,xⱼ)，s.t. 0≤αᵢ≤C，Σαᵢyᵢ=0
 * 决策函数：f(x) = ΣαᵢyᵢK(xᵢ,x) + b
 * 不变量：
 *   ① 对偶等式约束 Σαᵢyᵢ = 0（SMO 成对更新恒保持）
 *   ② KKT 互补松弛：αᵢ=0 → yᵢf(xᵢ)≥1；0<αᵢ<C → yᵢf(xᵢ)=1；αᵢ=C → yᵢf(xᵢ)≤1
 *   ③ 可分数据上训练精度 = 1（最大间隔几何意义）
 */
import { mulberry32 } from '../core/stats.js';

/** RBF 核 K(x,z) = exp(−γ‖x−z‖²) */
export function rbfKernel(x, z, gamma) {
  let s = 0;
  for (let i = 0; i < x.length; i++) {
    const d = x[i] - z[i];
    s += d * d;
  }
  return Math.exp(-gamma * s);
}

/**
 * SMO 序列最小优化。
 * 工作集策略：对每个 KKT 违反点 i，按 |Eᵢ−Eⱼ| 降序扫描全部候选 j，
 * 取第一个能产生有效更新的对（避免单点退化导致停滞，Platt 原文的第二选择启发式）。
 * @param {number[][]} X 训练样本
 * @param {number[]} y ±1 标签
 * @param {{C?:number, gamma?:number, tol?:number, patience?:number, maxSweeps?:number}} opts
 * @returns {{alpha:number[], b:number, gamma:number, C:number, X:number[][], y:number[], sweeps:number}}
 */
export function svmTrain(X, y, { C = 1, gamma = 0.5, tol = 1e-3, patience = 10, maxSweeps = 200 } = {}) {
  const n = X.length;
  // 预计算核矩阵（Gram 矩阵，对称半正定）
  const K = Array.from({ length: n }, (_, i) => new Array(n));
  for (let i = 0; i < n; i++) {
    K[i][i] = 1;
    for (let j = i + 1; j < n; j++) {
      const v = rbfKernel(X[i], X[j], gamma);
      K[i][j] = v;
      K[j][i] = v;
    }
  }
  const alpha = new Array(n).fill(0);
  let b = 0;
  let sweeps = 0, idle = 0;

  const f = (i) => {
    let s = b;
    for (let j = 0; j < n; j++) if (alpha[j] !== 0) s += alpha[j] * y[j] * K[j][i];
    return s;
  };

  while (idle < patience && sweeps < maxSweeps) {
    sweeps++;
    let numChanged = 0;
    for (let i = 0; i < n; i++) {
      const Ei = f(i) - y[i];
      const ri = y[i] * Ei;
      // KKT 违反判定
      if (!((ri < -tol && alpha[i] < C - tol) || (ri > tol && alpha[i] > tol))) continue;
      // 候选 j 按 |Eᵢ−Eⱼ| 降序（确定性）
      const cand = [];
      for (let k = 0; k < n; k++) {
        if (k === i) continue;
        cand.push([k, f(k) - y[k]]);
      }
      cand.sort((p, q) => Math.abs(Ei - q[1]) - Math.abs(Ei - p[1]));
      for (const [j, Ej] of cand) {
        const aiOld = alpha[i], ajOld = alpha[j];
        let L, H;
        if (y[i] !== y[j]) { L = Math.max(0, ajOld - aiOld); H = Math.min(C, C + ajOld - aiOld); }
        else { L = Math.max(0, aiOld + ajOld - C); H = Math.min(C, aiOld + ajOld); }
        if (H - L < 1e-12) continue;                       // 盒约束无可行方向
        const eta = 2 * K[i][j] - K[i][i] - K[j][j];       // RBF 下恒 < 0
        if (eta >= -1e-12) continue;
        let aj = ajOld - y[j] * (Ei - Ej) / eta;
        aj = Math.min(H, Math.max(L, aj));
        if (Math.abs(aj - ajOld) < 1e-9) continue;         // 更新量不足
        alpha[j] = aj;
        alpha[i] += y[i] * y[j] * (ajOld - aj);            // 恒保持 Σαy = 0
        const b1 = b - Ei - y[i] * (alpha[i] - aiOld) * K[i][j] - y[j] * (aj - ajOld) * K[i][i];
        const b2 = b - Ej - y[i] * (alpha[i] - aiOld) * K[i][j] - y[j] * (aj - ajOld) * K[j][j];
        if (alpha[i] > tol && alpha[i] < C - tol) b = b1;
        else if (aj > tol && aj < C - tol) b = b2;
        else b = (b1 + b2) / 2;
        numChanged++;
        break;                                             // 每个 i 每轮只走一步
      }
    }
    idle = numChanged === 0 ? idle + 1 : 0;
  }
  return { alpha, b, gamma, C, X, y, sweeps };
}

/** 决策函数值 f(x) = ΣαᵢyᵢK(xᵢ,x) + b */
export function svmDecision(model, x) {
  let s = model.b;
  for (let i = 0; i < model.alpha.length; i++) {
    if (model.alpha[i] !== 0) s += model.alpha[i] * model.y[i] * rbfKernel(model.X[i], x, model.gamma);
  }
  return s;
}

/** 类别预测：sign(f(x)) */
export function svmPredict(model, x) {
  return svmDecision(model, x) >= 0 ? 1 : -1;
}

/** 对偶目标函数值（供诊断/可视化） */
export function svmDualObjective(model) {
  let s = 0;
  for (let i = 0; i < model.alpha.length; i++) {
    s += model.alpha[i];
    for (let j = 0; j < model.alpha.length; j++) {
      s -= 0.5 * model.alpha[i] * model.alpha[j] * model.y[i] * model.y[j]
        * rbfKernel(model.X[i], model.X[j], model.gamma);
    }
  }
  return s;
}

/** 支持向量索引（0 < αᵢ < C） */
export function svmSupportVectors(model) {
  const idx = [];
  for (let i = 0; i < model.alpha.length; i++) {
    if (model.alpha[i] > 1e-6 && model.alpha[i] < model.C - 1e-6) idx.push(i);
  }
  return idx;
}

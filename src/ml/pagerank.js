/**
 * 旗舰AI引擎 · PageRank（网页排名）
 * 站在 Page & Brin 1998《The PageRank Citation Ranking》之上：把 Web 看成有向图，
 * 用随机游走的平稳分布衡量「重要性」——被重要页面指向的页面更重要。
 *
 * 数学：Google 矩阵 G = d·S + (1−d)/n·J（S 为随机游走转移矩阵、d=0.85 阻尼、
 * J 全 1 矩阵即「远程跳转 / teleport」）。重要性向量 r 是 G 的特征向量：
 *   r = G·r，rⱼ = d·Σ_i S[j][i]·r_i + (1−d)/n，且 Σ_j r_j = 1。
 * 等价线性系统：(I − dS)·r = (1−d)/n·1（阻尼 d<1 确保可逆、唯一解）。
 * 悬挂节点（出度为 0）的质量按 1/n 均匀重分配，避免「质量泄漏」。
 *
 * 不变量：
 *   ① 秩和 ≡ 1（概率分布）。
 *   ② 平稳方程 ‖G·r − r‖∞ → 0（r 是 G 的左/右特征向量，特征值 1）。
 *   ③ 幂迭代结果与精确线性解 (I − dS)r = (1−d)/n·1 一致（差异 < 1e-9）。
 *   ④ 幂迭代残差 ‖r_{k+1}−r_k‖₁ 单调不增并收敛。
 *   ⑤ 正则有向图（环 / 完全图）秩均匀（最大偏差 < 1e-9）。
 *   ⑥ 含悬挂节点的图仍质量守恒（Σr=1）且收敛到合法平稳分布。
 */

import { zeros, solve } from '../core/linalg.js';

/** 由邻接矩阵 A（A[i][j]=1 表示有向边 i→j）构建随机游走转移矩阵 S（列随机）。 */
function buildStochastic(A) {
  const n = A.length;
  const S = zeros(n, n);
  for (let i = 0; i < n; i++) {
    let out = 0;
    for (let j = 0; j < n; j++) out += A[i][j];
    if (out === 0) {
      // 悬挂节点：质量按 1/n 均匀重分配
      for (let j = 0; j < n; j++) S[j][i] = 1 / n;
    } else {
      for (let j = 0; j < n; j++) S[j][i] = A[i][j] / out;
    }
  }
  return S;
}

/**
 * PageRank 幂迭代求解。
 * @param {number[][]} A 邻接矩阵（A[i][j]=1 表示边 i→j）
 * @returns {{ranks:number[], iterations:number, residuals:number[]}}
 */
export function pageRank(A, { damping = 0.85, iters = 300, tol = 1e-12 } = {}) {
  const n = A.length;
  const S = buildStochastic(A);
  let r = new Array(n).fill(1 / n);
  let iterations = 0;
  const residuals = [];
  for (let k = 0; k < iters; k++) {
    const Sr = new Array(n).fill(0);
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let i = 0; i < n; i++) s += S[j][i] * r[i];
      Sr[j] = s;
    }
    const rn = Sr.map((v) => damping * v + (1 - damping) / n);
    let diff = 0;
    for (let j = 0; j < n; j++) diff += Math.abs(rn[j] - r[j]);
    iterations = k + 1;
    residuals.push(diff);
    r = rn;
    if (diff < tol) break;
  }
  return { ranks: r, iterations, residuals };
}

/**
 * PageRank 精确解：(I − dS)·r = (1−d)/n·1（直接线性求解）。
 */
export function pagerankExact(A, { damping = 0.85 } = {}) {
  const n = A.length;
  const S = buildStochastic(A);
  const M = zeros(n, n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) M[i][j] = (i === j ? 1 : 0) - damping * S[i][j];
  }
  const b = new Array(n).fill((1 - damping) / n);
  return solve(M, b);
}

/** 平稳方程残差 ‖G·r − r‖∞（r 应为 G 的特征向量，特征值 1） */
export function pagerankStationary(ranks, A, { damping = 0.85 } = {}) {
  const n = A.length;
  const S = buildStochastic(A);
  let mx = 0;
  for (let j = 0; j < n; j++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += S[j][i] * ranks[i];
    const Gr = damping * s + (1 - damping) / n;
    mx = Math.max(mx, Math.abs(Gr - ranks[j]));
  }
  return mx;
}

/** 秩之和（应 ≡ 1） */
export function pagerankSum(ranks) {
  let s = 0;
  for (const v of ranks) s += v;
  return s;
}

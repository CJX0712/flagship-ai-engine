/**
 * 旗舰AI引擎 · 迭代线性求解（Jacobi / Gauss–Seidel / 共轭梯度）
 * 站在：Hestenes & Stiefel 1952《Methods of Conjugate Gradients for Solving
 *       Linear Systems》J. Research NBS 49（CG：n 步终止的 Krylov 子空间法）
 *       · Jacobi 1845 / Gauss–Seidel 1874（定常迭代：x = Bx + c 的不动点）
 *       · Golub & Van Loan《Matrix Computations》§10–11（收敛性、谱半径判据）
 *       · Trefethen & Bau《Numerical Linear Algebra》§38（CG 的 A-共轭与残差正交）。
 *
 * 约定：A 为 n×n（行主序 number[][]），b 为长度 n 的 number[]。
 *   Jacobi       x_i^{(k+1)} = (b_i − Σ_{j≠i} A_ij·x_j^{(k)}) / A_ii      （全旧值）
 *   Gauss–Seidel x_i^{(k+1)} = (b_i − Σ_{j<i} A_ij·x_j^{(k+1)} − Σ_{j>i} A_ij·x_j^{(k)}) / A_ii（就地取新值）
 *   CG（对称正定 A，Fletcher–Reeves β）
 *       α_k = r_kᵀr_k / (p_kᵀA p_k)； x_{k+1} = x_k + α_k p_k； r_{k+1} = r_k − α_k A p_k
 *       β_k = r_{k+1}ᵀr_{k+1} / (r_kᵀr_k)； p_{k+1} = r_{k+1} + β_k p_k
 * 三个求解器均返回逐轮残差 ‖b − Ax‖₂ 的历史（residuals[0] 为初始残差）。
 *
 * 不变量（全部为可独立交叉验证的硬恒等式）：
 *   ① CG ≡ 直接解：对称正定 A = MᵀM + n·I 上，CG 解与 solve(A,b) 逐分量一致（maxAbs < 1e-8）；
 *   ② CG 有限步终止：n 维系统至多 n 步收敛（iters ≤ n 且末残差 < tol）；
 *   ③ A-共轭性：搜索方向满足 p_iᵀA p_j = 0（i≠j，相对 A-范数 < 1e-8）；
 *   ④ 残差正交性：r_iᵀr_j = 0（i≠j，相对 < 1e-8）；
 *   ⑤ Jacobi / Gauss–Seidel 在**严格对角占优**矩阵上收敛到 solve(A,b)（maxAbs < 1e-6），
 *      且残差单调下降（包络口径：逐轮比值 ≤ 1）；
 *   ⑥ Jacobi 迭代矩阵 B = I − D⁻¹A 的谱半径 ρ(B) < 1（Gershgorin 上界 + 幂迭代双证）；
 *   ⑦ 同输入确定性：两次调用结果逐位一致（maxDiff = 0）。
 */

import { matvec, dot, norm, identity } from './linalg.js';

/** 残差向量 r = b − A·x */
function lsResidual(A, b, x) {
  const Ax = matvec(A, x);
  const r = new Array(b.length);
  for (let i = 0; i < b.length; i++) r[i] = b[i] - Ax[i];
  return r;
}

/** 残差 2-范数 ‖b − A·x‖₂ */
function lsResidualNorm(A, b, x) {
  return norm(lsResidual(A, b, x));
}

/** A-范数 ‖p‖_A = √(pᵀAp)（A 对称正定时有意义） */
function lsANorm(A, p) {
  return Math.sqrt(Math.abs(dot(p, matvec(A, p))));
}

/**
 * 雅可比迭代：x^{(k+1)} = D⁻¹(b − (L+U)x^{(k)})，须 A_ii ≠ 0。
 * @param {number[][]} A 系数矩阵
 * @param {number[]} b 右端项
 * @param {{x0?:number[], iters?:number, tol?:number}} [opts] 初值、最大轮数、残差阈值
 * @returns {{x:number[], residuals:number[], iters:number, converged:boolean}}
 */
export function jacobiIter(A, b, { x0 = null, iters = 200, tol = 1e-10 } = {}) {
  const n = b.length;
  let x = x0 ? x0.slice() : new Array(n).fill(0);
  const residuals = [lsResidualNorm(A, b, x)];
  let done = 0;
  let converged = residuals[0] < tol;
  while (!converged && done < iters) {
    const xn = new Array(n);
    for (let i = 0; i < n; i++) {
      const Ai = A[i];
      let s = b[i];
      for (let j = 0; j < n; j++) if (j !== i) s -= Ai[j] * x[j];
      xn[i] = s / Ai[i];
    }
    x = xn;
    done++;
    const rn = lsResidualNorm(A, b, x);
    residuals.push(rn);
    if (rn < tol) converged = true;
  }
  return { x, residuals, iters: done, converged };
}

/**
 * 高斯–赛德尔迭代：就地使用已更新的分量（收敛通常快于 Jacobi，且不额外占用向量）。
 * @param {number[][]} A 系数矩阵
 * @param {number[]} b 右端项
 * @param {{x0?:number[], iters?:number, tol?:number}} [opts]
 * @returns {{x:number[], residuals:number[], iters:number, converged:boolean}}
 */
export function gaussSeidelIter(A, b, { x0 = null, iters = 200, tol = 1e-10 } = {}) {
  const n = b.length;
  const x = x0 ? x0.slice() : new Array(n).fill(0);
  const residuals = [lsResidualNorm(A, b, x)];
  let done = 0;
  let converged = residuals[0] < tol;
  while (!converged && done < iters) {
    for (let i = 0; i < n; i++) {
      const Ai = A[i];
      let s = b[i];
      for (let j = 0; j < i; j++) s -= Ai[j] * x[j];       // 本轮已更新的新值
      for (let j = i + 1; j < n; j++) s -= Ai[j] * x[j];   // 上一轮的旧值
      x[i] = s / Ai[i];
    }
    done++;
    const rn = lsResidualNorm(A, b, x);
    residuals.push(rn);
    if (rn < tol) converged = true;
  }
  return { x, residuals, iters: done, converged };
}

/**
 * 共轭梯度法（对称正定系统，Fletcher–Reeves 公式）。
 * Krylov 子空间维数 ≤ n ⇒ 精确算术下至多 n 步终止（不变量 ②）。
 * @param {number[][]} A 对称正定矩阵
 * @param {number[]} b 右端项
 * @param {{x0?:number[], tol?:number, maxIter?:number}} [opts] maxIter 缺省取 n
 * @returns {{x:number[], residuals:number[], residualVectors:number[][],
 *            directions:number[][], iters:number, converged:boolean}}
 */
export function conjugateGradient(A, b, { x0 = null, tol = 1e-10, maxIter = null } = {}) {
  const n = b.length;
  const maxIt = maxIter === null ? n : maxIter;
  const x = x0 ? x0.slice() : new Array(n).fill(0);
  let r = lsResidual(A, b, x);
  const p = r.slice();
  const residuals = [norm(r)];
  const residualVectors = [r.slice()];
  const directions = [p.slice()];
  let rs = dot(r, r);
  let iters = 0;
  let converged = residuals[0] < tol;
  while (!converged && iters < maxIt) {
    const Ap = matvec(A, p);
    const denom = dot(p, Ap);
    if (denom === 0) break;                 // A 奇异/方向退化：安全退出
    const alpha = rs / denom;
    for (let i = 0; i < n; i++) {
      x[i] += alpha * p[i];
      r[i] -= alpha * Ap[i];
    }
    iters++;
    const rn = norm(r);
    residuals.push(rn);
    residualVectors.push(r.slice());
    if (rn < tol) { converged = true; break; }
    const rsNew = dot(r, r);
    const beta = rsNew / rs;
    for (let i = 0; i < n; i++) p[i] = r[i] + beta * p[i];
    directions.push(p.slice());
    rs = rsNew;
  }
  return { x, residuals, residualVectors, directions, iters, converged };
}

/**
 * Jacobi 迭代矩阵 B = I − D⁻¹A（谱半径 ρ(B) < 1 ⟺ 迭代收敛）。
 * @param {number[][]} A 系数矩阵
 * @returns {number[][]} B
 */
export function jacobiIterationMatrix(A) {
  const n = A.length;
  const B = identity(n);
  for (let i = 0; i < n; i++) {
    const inv = 1 / A[i][i];
    for (let j = 0; j < n; j++) B[i][j] -= inv * A[i][j];
  }
  return B;
}

/**
 * 谱半径估计（规范化幂迭代 + 对数平均的 Lyapunov 指数法）。
 * 每步归一化后累计 log‖Bv‖，取平均再取指数 ⇒ 对负主特征值/复共轭主对同样稳健
 * （普通 Rayleigh 商在非对称阵上不可靠）。
 * @param {number[][]} B 方阵
 * @param {{iters?:number, burnIn?:number}} [opts]
 * @returns {number} ρ(B) 的估计值
 */
export function spectralRadius(B, { iters = 800, burnIn = 60 } = {}) {
  const n = B.length;
  let v = new Array(n);
  for (let i = 0; i < n; i++) v[i] = Math.sin(i * 2.39996323 + 1.0); // 确定性初值（无 rng 依赖）
  let nv = norm(v);
  if (nv === 0) return 0;
  for (let i = 0; i < n; i++) v[i] /= nv;
  let logSum = 0;
  let count = 0;
  for (let k = 0; k < iters; k++) {
    const w = matvec(B, v);
    const nw = norm(w);
    if (nw === 0) break;
    if (k >= burnIn) { logSum += Math.log(nw); count++; }
    for (let i = 0; i < n; i++) v[i] = w[i] / nw;
  }
  return count > 0 ? Math.exp(logSum / count) : 0;
}

/**
 * 行对角占优比 r_i = Σ_{j≠i}|A_ij| / |A_ii|（Gershgorin ⇒ ρ(B) ≤ max_i r_i）。
 * @param {number[][]} A 系数矩阵
 * @returns {{max:number, ratios:number[], strictlyDominant:boolean}}
 */
export function diagonalDominance(A) {
  const n = A.length;
  const ratios = new Array(n);
  let max = 0;
  let strictly = true;
  for (let i = 0; i < n; i++) {
    let off = 0;
    for (let j = 0; j < n; j++) if (j !== i) off += Math.abs(A[i][j]);
    const r = off / Math.abs(A[i][i]);
    ratios[i] = r;
    if (r > max) max = r;
    if (r >= 1) strictly = false;
  }
  return { max, ratios, strictlyDominant: strictly };
}

/**
 * 搜索方向的 A-共轭度：max_{i≠j} |p_iᵀA p_j| / (‖p_i‖_A·‖p_j‖_A)。
 * CG 的构造保证该量为 0（不变量 ③）。
 * @param {number[][]} A 对称正定矩阵
 * @param {number[][]} directions 搜索方向序列
 * @returns {number} 最大相对共轭误差
 */
export function conjugateResidual(A, directions) {
  const m = directions.length;
  let worst = 0;
  for (let i = 0; i < m; i++) {
    const Api = matvec(A, directions[i]);
    const ni = lsANorm(A, directions[i]);
    for (let j = i + 1; j < m; j++) {
      const cross = dot(directions[j], Api);
      const denom = ni * lsANorm(A, directions[j]);
      if (denom > 0) worst = Math.max(worst, Math.abs(cross) / denom);
    }
  }
  return worst;
}

/**
 * 残差序列的正交度：max_{i≠j} |r_iᵀr_j| / (‖r_i‖·‖r_j‖)。
 * CG 的构造保证该量为 0（不变量 ④）。
 * 数值说明：收敛末步的残差已落到机器精度地板（方向纯为舍入噪声），
 * 其「相对内积」无意义——故按 ‖r‖ < 1e-10·‖r₀‖ 剔除后再比较。
 * @param {number[][]} residualVectors 残差向量序列
 * @param {number} [floor] 噪声地板（相对 ‖r₀‖ 的比例），默认 1e-10
 * @returns {number} 最大相对内积
 */
export function residualOrthogonality(residualVectors, floor = 1e-10) {
  const m = residualVectors.length;
  const nrm = residualVectors.map((v) => norm(v));
  let r0 = 0;
  for (const v of nrm) if (v > r0) r0 = v;
  const thr = floor * r0;
  let worst = 0;
  for (let i = 0; i < m; i++) {
    if (nrm[i] <= thr) continue;
    for (let j = i + 1; j < m; j++) {
      const denom = nrm[i] * nrm[j];
      if (nrm[j] <= thr || denom <= 0) continue;
      worst = Math.max(worst, Math.abs(dot(residualVectors[i], residualVectors[j])) / denom);
    }
  }
  return worst;
}

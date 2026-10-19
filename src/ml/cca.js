/**
 * 旗舰AI引擎 · 典型相关分析（CCA，Hotelling 1936）
 * 站在：Hotelling 1936《Relations Between Two Sets of Variates》
 *       （Biometrika 28(3/4)，多元统计分析的原点之一）之上。
 *
 * 模型：给定两组变量 x∈ℝ^p、y∈ℝ^q，找线性组合 u = aᵀx、v = bᵀy 使
 * corr(u, v) 最大。前 d = min(p, q) 个「典型对」(a_k, b_k) 依次最大化
 * 与已得典型变量正交的约束下的相关系数 r_k（典型相关系数，降序）。
 *
 * 求解（eig 路线）：典型方向满足广义特征问题
 *   (Σxy·Σyy⁻¹·Σyx)·a_k = ρ_k²·Σxx·a_k   （ρ_k² = 典型相关系数²）
 * 因 Σxx⁻¹ΣxyΣyy⁻¹Σyx 是两个对称阵之积、本身非对称，实现将其归约为对称
 * 半正定矩阵 H = Σxx^(−1/2)·(ΣxyΣyy⁻¹Σyx)·Σxx^(−1/2)（同一特征值，Jacobi
 * 可解），再 a_k = Σxx^(−1/2)·u_k、b_k ∝ Σyy⁻¹Σyx·a_k，最后按
 * a_kᵀΣxx a_k = 1、b_kᵀΣyy b_k = 1 归一化，使 Cov(U_k) = Cov(V_k) = 1、
 * Cov(U_k, V_k) = r_k。协方差经特征值相对下限 reg·λ_max 兜底（仅防数值奇异）。
 *
 * 不变量：
 *   ① 相关系数 r_k ∈ [0, 1] 且非增（r_1 ≥ r_2 ≥ …，Jacobi 输出按特征值降序）
 *   ② 典型变量：Cov(U) = Cov(V) = I，Cov(U, V) = diag(r)（交叉协方差对角）
 *   ③ 完美相关 ⇒ r_1 = 1：Y = X·M（M 满秩可逆）时全部 r_k ≡ 1（到 1e-6）
 *   ④ 独立 X, Y ⇒ 各 r_k ≈ 0（< 1e-2；样本噪声尺度 ~ 1/√n）
 *   ⑤ 交叉验证：白化 + 交叉协方差 SVD（独立实现的一条路线）与 ccaFit 一致（1e-8）
 *   ⑥ 归一化约定：AᵀΣxxA = I、BᵀΣyyB = I（典型变量单位方差）
 */

import { zeros, transpose, matmul, matvec, jacobiEig } from '../core/linalg.js';
import { covarianceMatrix, mean } from '../core/stats.js';

/** 二次型 vᵀMv */
function ccaQuad(M, v) {
  let s = 0;
  for (let i = 0; i < M.length; i++) {
    let t = 0;
    for (let j = 0; j < M[0].length; j++) t += M[i][j] * v[j];
    s += v[i] * t;
  }
  return s;
}

/**
 * 谱白化因子 W = Λ^(−1/2)·Eᵀ（对称矩阵 M = EΛEᵀ 的左白化因子）：
 * 满足 W·M·Wᵀ = I。特征值按下限 max(λ, reg·λ_max) 截断——对良态矩阵是
 * 恒等操作（reg 默认 1e-6，λ_min ≫ reg·λ_max 时不触发），对奇异协方差
 * 仍给出数值稳定的白化。
 * @returns {{half:number[][]}} half = W = Λ^(−1/2)·Eᵀ
 */
function ccaSpectrumHalf(values, vectors, reg) {
  const d = values.length;
  let lamMax = 0;
  for (const l of values) if (l > lamMax) lamMax = l;
  if (!(lamMax > 0)) throw new Error('ccaFit: 协方差退化（零方差数据）');
  const floor = reg * lamMax;
  const half = zeros(d, d);
  for (let k = 0; k < d; k++) {
    const c = 1 / Math.sqrt(Math.max(values[k], floor));
    for (let i = 0; i < d; i++) half[k][i] = c * vectors[k][i]; // half = Λ^(−1/2)·Eᵀ
  }
  return { half };
}

/**
 * 谱逆：对称矩阵 M = EΛEᵀ 的特征值按下限 max(λ, reg·λ_max) 截断后求逆。
 */
function ccaSpectrumInverse(values, vectors, reg) {
  const d = values.length;
  let lamMax = 0;
  for (const l of values) if (l > lamMax) lamMax = l;
  if (!(lamMax > 0)) throw new Error('ccaFit: 协方差退化（零方差数据）');
  const floor = reg * lamMax;
  const inv = zeros(d, d);
  for (let k = 0; k < d; k++) {
    const c = 1 / Math.max(values[k], floor);
    for (let i = 0; i < d; i++) {
      for (let j = 0; j < d; j++) inv[i][j] += c * vectors[k][i] * vectors[k][j];
    }
  }
  return inv;
}

/**
 * CCA 拟合（eig 路线：Σxx⁻¹ΣxyΣyy⁻¹Σyx 的特征对）。
 * @param {number[][]} X 第一组变量 n×p
 * @param {number[][]} Y 第二组变量 n×q
 * @param {{reg?:number}} [opts] reg 协方差特征值相对下限（默认 1e-6，良态数据不触发）
 * @returns {{A:number[][], B:number[][], correlations:number[],
 *            xMeans:number[], yMeans:number[], reg:number, dimensions:{p:number,q:number,k:number}}}
 *   A (p×k)、B (q×k)：典型方向（列）；correlations：典型相关系数降序
 */
export function ccaFit(X, Y, { reg = 1e-6 } = {}) {
  const n = X.length;
  if (n === 0) throw new Error('ccaFit: 输入样本为空');
  if (Y.length !== n) throw new Error('ccaFit: X 与 Y 样本数不一致');
  const p = X[0].length, q = Y[0].length;
  const xMeans = mean(X), yMeans = mean(Y);
  const Xc = X.map((row) => row.map((v, j) => v - xMeans[j]));
  const Yc = Y.map((row) => row.map((v, j) => v - yMeans[j]));
  const Sxx = covarianceMatrix(X);
  const Syy = covarianceMatrix(Y);
  // 交叉协方差（全量累加、单次除 n−1，与引擎协方差铁律一致）
  const Sxy = zeros(p, q);
  for (let i = 0; i < p; i++) {
    for (let j = 0; j < q; j++) {
      let s = 0;
      for (let t = 0; t < n; t++) s += Xc[t][i] * Yc[t][j];
      Sxy[i][j] = s / (n - 1);
    }
  }
  const Syx = transpose(Sxy);

  const { values: lx, vectors: ex } = jacobiEig(Sxx);
  const { values: ly, vectors: ey } = jacobiEig(Syy);
  const Wx = ccaSpectrumHalf(lx, ex, reg).half;       // 白化因子：Wx·Σxx·Wxᵀ = I
  const SyyInv = ccaSpectrumInverse(ly, ey, reg);    // Σyy⁻¹（特征值截断式）

  // 对称约化：Σxx⁻¹ΣxyΣyy⁻¹Σyx 本身非对称（两个对称阵之积），但作白化代换
  // a = Wxᵀ·u 后广义特征问题 (ΣxyΣyy⁻¹Σyx)·a = ρ²·Σxx·a 化为对称半正定问题
  //   H·u = ρ²·u，  H = Wx·(ΣxyΣyy⁻¹Σyx)·Wxᵀ
  // H 与「Σxx⁻¹ΣxyΣyy⁻¹Σyx 的特征对」特征值相同，且 a = Wxᵀ·u 天然满足
  // a_iᵀΣxx a_j = δ_ij，归一化与对角化恒等式精确成立。
  const P = matmul(matmul(Sxy, SyyInv), Syx);          // ΣxyΣyy⁻¹Σyx（p×p 对称半正定）
  const H = matmul(matmul(Wx, P), transpose(Wx));      // Wx·P·Wxᵀ，对称半正定
  const { values: mus, vectors: evecs } = jacobiEig(H); // 降序；特征值 = ρ²

  const k = Math.min(p, q);
  const A = zeros(p, k);
  const B = zeros(q, k);
  const correlations = [];
  for (let idx = 0; idx < k; idx++) {
    const rho2 = Math.max(0, mus[idx]);
    const r = Math.min(1, Math.sqrt(rho2)); // 数值防冲：ρ² 可能为 1+ε
    correlations.push(r);
    // a_k = Wxᵀ·u_k（u 已单位化 ⇒ aᵀΣxx a = 1 精确成立）
    const a = matvec(transpose(Wx), evecs[idx]);
    // b_k ∝ Σyy⁻¹Σyx·a_k（Hotelling 标准关系 b = Σyy⁻¹Σyx a / ρ 的方向）
    const bDir = matvec(SyyInv, matvec(Syx, a));
    const nb = Math.sqrt(ccaQuad(Syy, bDir));
    for (let j = 0; j < p; j++) A[j][idx] = a[j];
    if (nb > 1e-300) {
      for (let j = 0; j < q; j++) B[j][idx] = bDir[j] / nb;
    } else {
      // 病态兜底：r_k ≈ 0 时 b 方向无意义，取确定性基向量（不影响 r_k）
      B[0][idx] = 1;
    }
  }
  return {
    A, B, correlations, xMeans, yMeans, reg,
    dimensions: { p, q, k },
  };
}

/**
 * CCA 投影：按训练均值中心化后 U = (X − xMeans)·A、V = (Y − yMeans)·B。
 * @param {number[][]} X n×p
 * @param {number[][]} Y n×q
 * @param {{A:number[][], B:number[][], xMeans:number[], yMeans:number[]}} model ccaFit 的产出
 * @returns {{U:number[][], V:number[][]}} 典型变量矩阵（n×k，列 = 各阶典型变量）
 */
export function ccaProject(X, Y, model) {
  const { A, B, xMeans, yMeans } = model;
  const kA = A[0].length, kB = B[0].length;
  const U = X.map((row) => {
    const xc = row.map((v, j) => v - xMeans[j]);
    const out = new Array(kA).fill(0);
    for (let c = 0; c < kA; c++) {
      let s = 0;
      for (let j = 0; j < xc.length; j++) s += xc[j] * A[j][c];
      out[c] = s;
    }
    return out;
  });
  const V = Y.map((row) => {
    const yc = row.map((v, j) => v - yMeans[j]);
    const out = new Array(kB).fill(0);
    for (let c = 0; c < kB; c++) {
      let s = 0;
      for (let j = 0; j < yc.length; j++) s += yc[j] * B[j][c];
      out[c] = s;
    }
    return out;
  });
  return { U, V };
}

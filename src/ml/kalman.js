/**
 * 旗舰AI引擎 · 卡尔曼滤波（Kalman Filter）
 * 站在：Kalman 1960《A New Approach to Linear Filtering and Prediction Problems》
 *       Grewal & Andrews《Kalman Filtering: Theory and Practice》（Joseph 协方差更新式）
 *       Anderson & Moore《Optimal Filtering》（离散代数 Riccati 方程 DARE）
 * 模型（线性高斯状态空间）：x_t = F·x_{t−1} + w_t (w~N(0,Q))；z_t = H·x_t + v_t (v~N(0,R))
 * 协方差更新采用 **Joseph 形式** P = (I−KH)P⁻(I−KH)ᵀ + KRKᵀ——
 * 相比朴素 (I−KH)P⁻ 在浮点下强制保持对称半正定（Grewal & Andrews 的工业标准做法）。
 * 不变量：
 *   ① 卡尔曼增益闭式 K = P⁻Hᵀ(HP⁻Hᵀ+R)⁻¹（最优性的必要条件）
 *   ② 新息协方差一致：新息经验协方差 ≈ S = HP⁻Hᵀ+R（滤波器自洽性的统计检验）
 *   ③ 新息白噪声：滞后−1 自相关 ≈ 0（新息序列必须是白噪声，否则滤波器非最优）
 *   ④ 稳态收敛：P⁻ 迭代收敛到离散代数 Riccati 方程（DARE）的解析解
 *   ⑤ 协方差恒为对称半正定（Joseph 形式的直接保障）
 */

import { zeros, identity, matmul, transpose, matvec, solve } from '../core/linalg.js';

/** 矩阵减法 A − B */
function msub(A, B) {
  return A.map((row, i) => row.map((v, j) => v - B[i][j]));
}

/**
 * 卡尔曼滤波主循环。
 * @param {{F:number[][], H:number[][], Q:number[][], R:number[][], x0:number[], P0:number[][]}} model
 * @param {number[][]} observations 每步观测 z_t（长度 m 的列向量）
 * @returns {{states:number[][], covariances:number[][][], predicted:number[][],
 *            predictedCov:number[][][], gains:number[][][], innovations:number[][],
 *            innovationCov:number[][][], logLik:number[]}}
 */
export function kalmanFilter(model, observations) {
  const { F, H, Q, R } = model;
  const n = F.length;             // 状态维
  const m = H.length;             // 观测维
  let x = [...model.x0];
  let P = model.P0.map((r) => [...r]);
  const Ft = transpose(F), Ht = transpose(H);
  const I = identity(n);

  const states = [], covariances = [], predicted = [], predictedCov = [];
  const gains = [], innovations = [], innovationCov = [], logLik = [];

  for (const z of observations) {
    // —— 预测步 ——
    const xMinus = matvec(F, x);
    const PMinus = matmul(matmul(F, P), Ft).map((row, i) => row.map((v, j) => v + Q[i][j]));
    predicted.push(xMinus);
    predictedCov.push(PMinus);

    // —— 更新步 ——
    const S = matmul(matmul(H, PMinus), Ht).map((row, i) => row.map((v, j) => v + R[i][j]));
    // K = P⁻HᵀS⁻¹：逐列解 S·kᵀ = (P⁻Hᵀ)ᵀ
    const PHt = matmul(PMinus, Ht);
    const K = zeros(n, m);
    for (let c = 0; c < m; c++) {
      const rhs = new Array(m);
      for (let r = 0; r < m; r++) rhs[r] = PHt[r][c]; // S 对称，列/行同
      const sol = solve(S, rhs);
      for (let r = 0; r < n; r++) K[r][c] = sol[r];
    }
    const innovation = [];
    for (let r = 0; r < m; r++) {
      let hv = 0;
      for (let c = 0; c < n; c++) hv += H[r][c] * xMinus[c];
      innovation.push(z[r] - hv);
    }
    // x = x⁻ + K·innovation
    const Kinn = matvec(K, innovation);
    const xNew = xMinus.map((v, i) => v + Kinn[i]);
    // Joseph 形式：P = (I−KH)P⁻(I−KH)ᵀ + KRKᵀ
    const KH = matmul(K, H);
    const IKH = msub(I, KH);
    const P1 = matmul(matmul(IKH, PMinus), transpose(IKH));
    const KRKt = matmul(matmul(K, R), transpose(K));
    const PNew = P1.map((row, i) => row.map((v, j) => v + KRKt[i][j]));

    states.push(xNew);
    covariances.push(PNew);
    gains.push(K);
    innovations.push(innovation);
    innovationCov.push(S);
    // 高斯对数似然增量（滤波器一致性的副产品）
    const Sinv = solve(S, innovation);
    let q = 0;
    for (let r = 0; r < m; r++) q += innovation[r] * Sinv[r];
    let logDetS = 0;
    const Ls = cholDiag(S);
    for (const d of Ls) logDetS += Math.log(d);
    logLik.push(-0.5 * (q + m * Math.log(2 * Math.PI) + logDetS));

    x = xNew;
    P = PNew;
  }
  return {
    states, covariances, predicted, predictedCov,
    gains, innovations, innovationCov, logLik,
  };
}

/** 对称正定阵的 Cholesky 对角元（仅取对数行列式用） */
function cholDiag(A) {
  const n = A.length;
  const L = zeros(n, n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = A[i][j];
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      L[i][j] = i === j ? Math.sqrt(Math.max(s, 1e-300)) : s / L[j][j];
    }
  }
  return L.map((row, i) => row[i]);
}

/**
 * 标量系统的离散代数 Riccati 方程（DARE）解析解。
 * 稳态预测方差 P⁻ 满足：h²P⁻² + [r(1−f²) − qh²]P⁻ − qr = 0，取正根。
 */
export function kalmanScalarSteadyState(f, h, q, r) {
  const A = h * h;
  const B = r * (1 - f * f) - q * h * h;
  const C = -q * r;
  const disc = Math.sqrt(Math.max(B * B - 4 * A * C, 0));
  const Pminus = (-B + disc) / (2 * A);
  const Kss = Pminus * h / (Pminus * h * h + r);
  return { Pminus, Kss };
}

/**
 * 迭代求解标量 DARE 的不动点（预测方差递推），用于与解析解互证。
 */
export function kalmanScalarIterate(f, h, q, r, iters = 5000) {
  let Pm = q; // 从 Q 起迭代（也可从 0 起，收敛到同一不动点）
  for (let k = 0; k < iters; k++) {
    const K = Pm * h / (Pm * h * h + r);
    const P = (1 - K * h) * Pm;
    Pm = f * f * P + q;
  }
  return Pm;
}

/** 新息序列的滞后−k 自相关（白噪声检验） */
export function innovationAutocorr(series, lag = 1) {
  const n = series.length;
  const mu = series.reduce((s, v) => s + v, 0) / n;
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) den += (series[i] - mu) ** 2;
  for (let i = lag; i < n; i++) num += (series[i] - mu) * (series[i - lag] - mu);
  return den > 0 ? num / den : 0;
}

/** 序列经验方差 */
export function empiricalVariance(series) {
  const n = series.length;
  const mu = series.reduce((s, v) => s + v, 0) / n;
  let s = 0;
  for (const v of series) s += (v - mu) ** 2;
  return s / n;
}

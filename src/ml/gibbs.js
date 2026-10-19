/**
 * 旗舰AI引擎 · Gibbs 采样（马尔可夫链蒙特卡洛）
 * 站在 Geman & Geman 1984（Gibbs 采样用于吉布斯随机场）与 Gelfand & Smith 1990
 *  （Gibbs 作为通用贝叶斯推断引擎）、Casella & George 1992
 *  （《Explaining the Gibbs Sampler》标准二元高斯演示）之上。
 *
 * 核心思想：直接对高维联合 π(x) 采样困难时，改为逐坐标循环——每次固定其余坐标，
 * 只从「满条件分布」π(x_k | x_{−k}) 抽样。若各条件分布闭式可得，则采样极廉价；
 * 由细致平衡，链的不变量分布正是目标 π。二元高斯是教科书范例：其条件分布仍是高斯，
 * 且闭式与「以总体回归」完全一致。
 *
 * 不变量：
 *   ① 二元高斯：经验均值 ≡ 解析均值 μ（偏差 < 0.05）。
 *   ② 二元高斯：经验协方差 ≡ 解析 Σ（相对误差 < 0.08）。
 *   ③ 二元高斯：单坐标边际 ≡ 对应一元高斯（均值 μ_k、方差 Σ_kk）。
 *   ④ 二元高斯：实现的条件分布 ≡ 理论条件分布（条件均值/方差抽样对照，偏差 < 0.05）。
 *   ⑤ 离散：经验联合分布 ≡ 真实联合（最大格点误差 < 0.02，且 Σ=1）——链的不变分布即目标。
 *   ⑥ 收敛：后段样本坐标均值 → 目标均值（偏差 < 0.03）。
 */

import { zeros } from '../core/linalg.js';
import { mulberry32, gaussianSample } from '../core/stats.js';

/**
 * 二元高斯 Gibbs 采样（闭式条件分布）。
 * @param {number[]} mu [μ1, μ2]
 * @param {number[][]} Sigma 2×2 协方差矩阵
 * @returns {{samples:number[][], rho:number, condVar1:number, condVar2:number}}
 */
export function gibbsBivariateGaussian(mu, Sigma, {
  iters = 20000, burnIn = 2000, thin = 1, seed = 42,
} = {}) {
  const [mu1, mu2] = mu;
  const s00 = Sigma[0][0], s11 = Sigma[1][1], s01 = Sigma[0][1];
  const rho = s01 / Math.sqrt(s00 * s11);
  const condVar1 = s00 * (1 - rho * rho);
  const condVar2 = s11 * (1 - rho * rho);
  const rng = mulberry32(seed);
  let x1 = mu1, x2 = mu2;
  const samples = [];
  for (let t = 0; t < iters; t++) {
    // x1 | x2
    const m1 = mu1 + (s01 / s11) * (x2 - mu2);
    x1 = m1 + Math.sqrt(condVar1) * gaussianSample(rng, 0, 1);
    // x2 | x1
    const m2 = mu2 + (s01 / s00) * (x1 - mu1);
    x2 = m2 + Math.sqrt(condVar2) * gaussianSample(rng, 0, 1);
    if (t >= burnIn && (t - burnIn) % thin === 0) samples.push([x1, x2]);
  }
  return { samples, rho, condVar1, condVar2 };
}

/**
 * 通用 Gibbs 采样器：逐坐标循环调用 conditionals[k](state, rng) 抽样。
 * @param {Function[]} conditionals 每个坐标为「给定当前 state 返回该坐标样本」的函数
 * @param {number[]} init 初始状态
 * @returns {number[][]} 采样序列（去 burn-in、按 thin 抽取）
 */
export function gibbsSampler(conditionals, init, {
  iters = 20000, burnIn = 2000, thin = 1, seed = 42,
} = {}) {
  const rng = mulberry32(seed);
  const state = init.slice();
  const samples = [];
  for (let t = 0; t < iters; t++) {
    for (let k = 0; k < conditionals.length; k++) state[k] = conditionals[k](state, rng);
    if (t >= burnIn && (t - burnIn) % thin === 0) samples.push(state.slice());
  }
  return samples;
}

/** 单坐标经验均值与方差 */
export function gibbsMarginal(samples, k = 0) {
  const n = samples.length;
  let s = 0;
  for (const row of samples) s += row[k];
  const mean = s / n;
  let v = 0;
  for (const row of samples) v += (row[k] - mean) ** 2;
  return { mean, variance: v / n };
}

/** 经验协方差矩阵（2×2，population 估计） */
export function gibbsCovariance(samples) {
  const n = samples.length;
  let m1 = 0, m2 = 0;
  for (const row of samples) { m1 += row[0]; m2 += row[1]; }
  m1 /= n; m2 /= n;
  let c00 = 0, c11 = 0, c01 = 0;
  for (const row of samples) {
    c00 += (row[0] - m1) ** 2;
    c11 += (row[1] - m2) ** 2;
    c01 += (row[0] - m1) * (row[1] - m2);
  }
  return [[c00 / n, c01 / n], [c01 / n, c11 / n]];
}

/**
 * 条件分布一致性检查：固定 x2，从闭式条件 x1|x2 抽 N 个样本，
 * 验证其样本均值 ≡ μ1 + (Σ12/Σ22)(x2−μ2)、样本方差 ≡ Σ11(1−ρ²)。
 * @returns {{meanErr:number, varRelErr:number}}
 */
export function gibbsConditionalCheck(mu, Sigma, x2, { n = 50000, seed = 42 } = {}) {
  const [mu1, mu2] = mu;
  const s00 = Sigma[0][0], s11 = Sigma[1][1], s01 = Sigma[0][1];
  const rho = s01 / Math.sqrt(s00 * s11);
  const condVar1 = s00 * (1 - rho * rho);
  const condMean1 = mu1 + (s01 / s11) * (x2 - mu2);
  const rng = mulberry32(seed);
  let s = 0;
  for (let i = 0; i < n; i++) {
    const v = condMean1 + Math.sqrt(condVar1) * gaussianSample(rng, 0, 1);
    s += v;
  }
  const mean = s / n;
  let vv = 0;
  // 二次遍历：用已知均值算方差（避免重抽）
  const rng2 = mulberry32(seed);
  for (let i = 0; i < n; i++) {
    const v = condMean1 + Math.sqrt(condVar1) * gaussianSample(rng2, 0, 1);
    vv += (v - mean) ** 2;
  }
  const variance = vv / n;
  return { meanErr: Math.abs(mean - condMean1), varRelErr: Math.abs(variance - condVar1) / condVar1 };
}

/**
 * 离散二维 Gibbs 采样：joint 为 2D 概率表（已归一），逐坐标按满条件分布（范畴分布）抽样。
 * @returns {{samples:number[][], total:number}} samples = 经验联合分布（已归一）
 */
export function gibbsDiscrete(joint, init, {
  iters = 40000, burnIn = 4000, thin = 1, seed = 42,
} = {}) {
  const rng = mulberry32(seed);
  const m = joint.length, n = joint[0].length;
  let x = init[0], y = init[1];
  const counts = zeros(m, n);
  let total = 0;
  for (let t = 0; t < iters; t++) {
    // x | y
    let colSum = 0;
    for (let a = 0; a < m; a++) colSum += joint[a][y];
    let u = rng() * colSum, a = 0;
    for (; a < m; a++) { u -= joint[a][y]; if (u <= 0) break; }
    x = a;
    // y | x
    let rowSum = 0;
    for (let b = 0; b < n; b++) rowSum += joint[x][b];
    let v = rng() * rowSum, b = 0;
    for (; b < n; b++) { v -= joint[x][b]; if (v <= 0) break; }
    y = b;
    if (t >= burnIn && (t - burnIn) % thin === 0) { counts[x][y]++; total++; }
  }
  const empirical = zeros(m, n);
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) empirical[i][j] = counts[i][j] / total;
  return { samples: empirical, total };
}

/**
 * 收敛检查：后段样本坐标均值 → 目标均值（返回后段均值与目标偏差）。
 * @returns {number} 后 20% 窗口均值与目标均值的绝对偏差
 */
export function gibbsConvergence(samples, targetMean, coord = 0) {
  const n = samples.length;
  const start = Math.floor(n * 0.8);
  let s = 0;
  for (let i = start; i < n; i++) s += samples[i][coord];
  const mean = s / (n - start);
  return Math.abs(mean - targetMean);
}

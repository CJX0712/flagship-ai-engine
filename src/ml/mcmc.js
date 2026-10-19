/**
 * 旗舰AI引擎 · 马尔可夫链蒙特卡洛（MCMC）
 * 站在：Metropolis et al. 1953 · Hastings 1970（MH 算法）
 *       Roberts, Gelman & Gilks 1997（随机游走 MH 的最优接受率 0.234）
 *       Geyer 1992（burn-in / thin 的实用准则）· Robert & Casella《Monte Carlo Statistical Methods》
 * 核心机制：
 *   ① MH 只需**未归一化密度** π̃ ∝ π——归一化常数在比值中相消，这是贝叶斯推断
 *      能处理任意后验的根本原因。
 *   ② 对称提议（随机游走 q(x'|x) = q(x|x')）⇒ 接受率退化为 min(1, π(x')/π(x))。
 *   ③ 细致平衡 π(x)T(x→x') = π(x')T(x'→x) 保证 π 是链的平稳分布。
 * 不变量：
 *   ① 目标为标准正态时，采样经验矩 ≡ 解析矩（均值 0、方差 1、四阶矩 3、P(|x|<1)=0.6827）
 *   ② 未归一化密度与归一化密度给出**同一**平稳分布（归一化常数无关性）
 *   ③ 接受率落在有效区间（0.2–0.6，理论最优 0.234），且随提议尺度单调下降
 *   ④ 细致平衡：对称提议下 π(x)q(x'|x) 与 π(x')q(x|x') 逐对相消
 *   ⑤ 重要性采样估计 ≡ 解析期望，且有效样本量 ESS 合理（权重非退化）
 */

import { mulberry32, gaussianSample } from '../core/stats.js';

/**
 * Metropolis-Hastings（对称随机游走提议，支持多维状态）。
 * @param {Function} logTarget 未归一化对数密度 log π̃(x)
 * @param {number[]} x0 初值
 * @returns {{samples:number[][], acceptanceRate:number, nAccept:number,
 *            logPiTrace:number[], effectiveSize:number}}
 */
export function metropolisHastings(logTarget, x0, {
  iters = 30000, proposalSigma = 1, burnIn = 3000, thin = 2, seed = 42,
} = {}) {
  const rng = mulberry32(seed);
  const d = x0.length;
  let x = [...x0];
  let logPi = logTarget(x);
  const samples = [];
  const logPiTrace = [];
  let nAccept = 0;
  for (let t = 0; t < iters; t++) {
    const prop = new Array(d);
    for (let i = 0; i < d; i++) prop[i] = x[i] + proposalSigma * gaussianSample(rng, 0, 1);
    const logPiProp = logTarget(prop);
    // 对称提议 ⇒ 比值中 q 相消
    const logAlpha = logPiProp - logPi;
    const accept = Math.log(rng() || 1e-300) < logAlpha;
    if (accept) { x = prop; logPi = logPiProp; nAccept++; }
    if (t >= burnIn && (t - burnIn) % thin === 0) {
      samples.push([...x]);
      logPiTrace.push(logPi);
    }
  }
  return {
    samples,
    acceptanceRate: nAccept / iters,
    nAccept,
    logPiTrace,
    effectiveSize: samples.length,
  };
}

/** MH 接受概率 min(1, π(x')/π(x))（对数域实现，数值稳定） */
export function mhAcceptProb(logPiCur, logPiProp) {
  return Math.min(1, Math.exp(Math.min(logPiProp - logPiCur, 0)));
}

/**
 * 细致平衡核对：对称高斯提议下 q(x'|x) = q(x|x')，
 * 故 π(x)T(x→x') 与 π(x')T(x'→x) 应当逐对相等（相对误差 ≈ 0）。
 * @returns {number} 最大相对偏差
 */
export function detailedBalanceResidual(logPiA, logPiB, xA, xB, proposalSigma) {
  const logq = (from, to) => {
    let s = 0;
    for (let i = 0; i < from.length; i++) s += (to[i] - from[i]) ** 2;
    return -0.5 * s / (proposalSigma ** 2);
  };
  // T(x→x') = q(x'|x)·α(x→x')；α = min(1, π(x')/π(x))
  const logT_AB = logq(xA, xB) + Math.min(0, logPiB - logPiA);
  const logT_BA = logq(xB, xA) + Math.min(0, logPiA - logPiB);
  const lhs = logPiA + logT_AB;
  const rhs = logPiB + logT_BA;
  return Math.abs(lhs - rhs) / (Math.abs(lhs) + Math.abs(rhs) + 1e-12);
}

/**
 * 重要性采样：E_π[f] ≈ Σ w_i f(x_i) / Σ w_i，w_i = π̃(x_i)/q(x_i)。
 * @param {Function} logTarget 未归一化对数目标
 * @param {Function} logProposal 对数提议密度（须已归一化或同样未归一化，此处按已归一化用）
 * @param {Function} sampleProposal rng → 样本
 * @param {Function} f 被积函数
 * @returns {{estimate:number, ess:number, weights:number[], n:number}}
 */
export function importanceSampling(logTarget, logProposal, sampleProposal, f, n = 200000, seed = 42) {
  const rng = mulberry32(seed);
  const logw = new Array(n);
  const vals = new Array(n);
  for (let i = 0; i < n; i++) {
    const x = sampleProposal(rng);
    logw[i] = logTarget(x) - logProposal(x);
    vals[i] = f(x);
  }
  // max-shift 防溢出（复用数值稳定思想）；用循环求最大值——
  // Math.max(...logw) 在 n ≥ 10⁵ 时会因参数展开超出调用栈上限
  let mx = -Infinity;
  for (const v of logw) if (v > mx) mx = v;
  const w = logw.map((v) => Math.exp(v - mx));
  const sw = w.reduce((a, b) => a + b, 0);
  let est = 0;
  for (let i = 0; i < n; i++) est += (w[i] / sw) * vals[i];
  let sw2 = 0;
  for (const v of w) sw2 += v * v;
  const ess = (sw * sw) / sw2;
  return { estimate: est, ess, weights: w.map((v) => v / sw), n };
}

/**
 * 采样序列的经验矩（一维）
 * @returns {{mean:number, variance:number, m4:number, pWithin1:number, n:number}}
 */
export function mcmcEmpiricalMoments(series) {
  const n = series.length;
  let s1 = 0, s2 = 0, s4 = 0, within = 0;
  for (const v of series) {
    s1 += v; s2 += v * v; s4 += v ** 4;
    if (Math.abs(v) < 1) within++;
  }
  const mean = s1 / n;
  const variance = s2 / n - mean * mean;
  return {
    mean,
    variance,
    m4: s4 / n,
    pWithin1: within / n,
    n,
  };
}

/** 提取多维样本的第 k 个坐标（边缘序列） */
export function marginal(series, k = 0) {
  return series.map((v) => v[k]);
}

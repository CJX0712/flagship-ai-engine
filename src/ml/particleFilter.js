/**
 * 旗舰AI引擎 · 粒子滤波（SIR / 序贯重要性重采样）
 * 站在：Gordon, Salmond & Smith 1993《Novel approach to nonlinear/non-Gaussian
 *       Bayesian state estimation》（Bootstrap 滤波 / SIR）
 *       · Doucet, de Freitas & Gordon 2001《Sequential Monte Carlo in Practice》
 *       · Arulampalam et al. 2002《A Tutorial on Particle Filters》
 *       · Carpenter, Clifford & Fearnhead 1999（系统重采样 systematic resampling）
 *
 * 用 N 个带权粒子 {x⁽ⁱ⁾, w⁽ⁱ⁾} 逼近后验 p(x_k | y_{1:k})：
 *   初始化 x⁽ⁱ⁾_0 ~ p(x_0)                     —— init(rng)，缺省退化为点质量 x0
 *   传播   x⁽ⁱ⁾_k ~ p(x_k | x⁽ⁱ⁾_{k−1})        —— 用户提供的 transition(x, rng)
 *   权重   w⁽ⁱ⁾ ∝ w⁽ⁱ⁾_{k−1}·p(y_k | x⁽ⁱ⁾_k)   —— 用户提供的 likelihood(y, x)（可未归一化）
 *   估计   x̂_k = Σᵢ w⁽ⁱ⁾x⁽ⁱ⁾（最小均方估计）
 *   重采样 当 ESS = 1/Σw² < 阈值（默认 N/2）或每步都做（resampleEvery）
 * 权重一律在 **log 域** 累加 + logsumexp 归一化（下溢/上溢双免疫）。
 * 卡尔曼滤波是线性高斯情形的解析解——粒子滤波在 N → ∞ 时必须收敛到它：
 * 对照实验的 **先验也必须一致**（先验方差 P₀ 用 init 采样器表达），
 * 否则前几步的卡尔曼增益系统性偏小，形成与蒙特卡洛无关的偏差。
 *
 * 不变量：
 *   ① 权重归一化：每步 Σw = 1（1e-12）；
 *   ② 线性高斯 ⇒ 粒子滤波估计收敛到卡尔曼解析解（N 足够大）；
 *   ③ ESS ≡ 1/Σw² 重算一致，且恒 ∈ [1, N]；
 *   ④ 系统重采样后粒子数 = N、权重精确均匀 1/N（均匀权重下重采样 ≡ 恒等置换）；
 *   ⑤ 退化情形：全部粒子相同 ⇒ 估计 ≡ 该粒子且 ESS ≡ N；
 *   ⑥ 同 seed 两次运行逐位一致（随机数流完全确定）。
 */

import { logsumexp } from '../core/numerical.js';
import { mulberry32 } from '../core/stats.js';

/**
 * 有效样本量 ESS = 1/Σᵢ wᵢ²（权重需已归一化）。
 * 退化界：1 ≤ ESS ≤ N（全权重集中在一个粒子时 = 1，均匀时 = N）。
 * @param {number[]} weights 归一化权重
 * @returns {number} 有效样本量
 */
export function effectiveSampleSize(weights) {
  let s2 = 0;
  for (const w of weights) s2 += w * w;
  return s2 > 0 ? 1 / s2 : 0;
}

/**
 * 系统重采样（systematic resampling）。
 * 位置 uᵢ = (i + u₀)/N，u₀ ~ U[0, 1/N)：单次随机数、方差最小、O(N) 时间。
 * 恒等式：权重均匀 wᵢ ≡ 1/N 时，每个粒子恰好被选中一次 ⇒ indices ≡ 0..N−1。
 * @param {number[][]} particles 粒子（每条为状态向量）
 * @param {number[]} weights 归一化权重
 * @param {function} rng 随机数发生器
 * @returns {{particles:number[][], indices:number[], weights:number[]}} 重采样后的粒子、来源下标、均匀权重
 */
export function systematicResample(particles, weights, rng) {
  const N = particles.length;
  const cum = new Array(N);
  let acc = 0;
  for (let i = 0; i < N; i++) { acc += weights[i]; cum[i] = acc; }
  // 尾部归一化补偿（浮点累加误差），保证最后一个区间严格闭合到 1
  cum[N - 1] = 1;
  const u0 = rng() / N;
  const out = new Array(N);
  const indices = new Array(N);
  let i = 0;
  for (let k = 0; k < N; k++) {
    const u = u0 + k / N;
    while (i < N - 1 && u > cum[i]) i++;
    indices[k] = i;
    out[k] = particles[i].slice();
  }
  const uniform = new Array(N).fill(1 / N);
  return { particles: out, indices, weights: uniform };
}

/** 加权均值估计 x̂ = Σᵢ wᵢx⁽ⁱ⁾（Σw = 1） */
function pfWeightedMean(particles, weights) {
  const d = particles[0].length;
  const out = new Array(d).fill(0);
  for (let i = 0; i < particles.length; i++) {
    const w = weights[i];
    const p = particles[i];
    for (let j = 0; j < d; j++) out[j] += w * p[j];
  }
  return out;
}

/**
 * 粒子滤波主循环（SIR / Bootstrap 的通用化：任意转移与似然均可接入）。
 * @param {{x0:number[], init?:function, transition:function, likelihood:function,
 *          N?:number, resampleEvery?:boolean, essThreshold?:number, rng?:function,
 *          seed?:number}} model
 *   transition(x, rng) → 下一状态；likelihood(y, x) → **未归一化** 的似然（可为密度值）
 *   init(rng) → 初始粒子采样（缺省：全部粒子取 x0，即先验退化为点质量）
 * @param {number[][]} observations 观测序列 y_1..y_T
 * @returns {{estimates:number[][], particles:number[][], weights:number[],
 *            weightsHistory:number[][], essHistory:number[], resampled:boolean[], N:number}}
 */
export function particleFilter(model, observations) {
  const x0 = model.x0;
  const transition = model.transition;
  const likelihood = model.likelihood;
  const N = model.N === undefined ? 2000 : model.N;
  const resampleEvery = model.resampleEvery !== false;
  const essThreshold = model.essThreshold === undefined ? N / 2 : model.essThreshold;
  const rng = model.rng || mulberry32(model.seed === undefined ? 42 : model.seed);
  const init = model.init || (() => x0.slice());

  // 初始化：从先验 p(x_0) 采样 N 个粒子，权重均匀
  let particles = new Array(N);
  for (let i = 0; i < N; i++) particles[i] = init(rng).slice();
  const logUniform = -Math.log(N);
  let logW = new Array(N).fill(logUniform);

  const estimates = [];
  const weightsHistory = [];
  const essHistory = [];
  const resampled = [];

  for (const y of observations) {
    // —— 传播：x⁽ⁱ⁾ ~ p(x_k | x⁽ⁱ⁾_{k−1}) ——
    for (let i = 0; i < N; i++) particles[i] = transition(particles[i], rng);

    // —— 权重：log w += log p(y | x)（用户可给未归一化似然，常数项在归一化中相消）——
    for (let i = 0; i < N; i++) {
      const ll = Math.log(Math.max(likelihood(y, particles[i]), 1e-300));
      logW[i] += Number.isFinite(ll) ? ll : -Infinity;
    }

    // —— log 域归一化（logsumexp：下溢/上溢双免疫）——
    const lse = logsumexp(logW);
    let w;
    if (!Number.isFinite(lse)) {
      // 全部粒子似然为 0（退化）：回落为均匀分布，保持滤波器存活
      w = new Array(N).fill(1 / N);
      logW = new Array(N).fill(logUniform);
    } else {
      w = new Array(N);
      for (let i = 0; i < N; i++) {
        w[i] = Math.exp(logW[i] - lse);
        logW[i] = Math.log(Math.max(w[i], 1e-300)); // 归一后继续作为下一轮先验权重
      }
    }

    // —— 估计与诊断 ——
    const ess = effectiveSampleSize(w);
    estimates.push(pfWeightedMean(particles, w));
    weightsHistory.push(w);
    essHistory.push(ess);

    // —— 重采样：每步 或 ESS 跌破阈值 ——
    if (resampleEvery || ess < essThreshold) {
      const r = systematicResample(particles, w, rng);
      particles = r.particles;
      logW = new Array(N).fill(logUniform);
      resampled.push(true);
    } else {
      resampled.push(false);
    }
  }

  const weights = new Array(N).fill(1 / N);
  return { estimates, particles, weights, weightsHistory, essHistory, resampled, N };
}

/**
 * 线性高斯模型的解析参考（卡尔曼滤波）与粒子滤波的对照入口：
 * 返回逐步最大绝对偏差，用于「无噪声极限下 SIR ⇒ 卡尔曼」的交叉验证。
 * @param {number[][]} pfEstimates 粒子滤波估计序列
 * @param {number[][]} kfStates 卡尔曼滤波状态序列
 * @returns {{maxDiff:number, meanDiff:number, stepDiff:number[]}}
 */
export function pfCompareWithKalman(pfEstimates, kfStates) {
  const T = Math.min(pfEstimates.length, kfStates.length);
  const stepDiff = new Array(T);
  let maxDiff = 0, sumDiff = 0, cnt = 0;
  for (let t = 0; t < T; t++) {
    let worst = 0;
    for (let j = 0; j < pfEstimates[t].length; j++) {
      worst = Math.max(worst, Math.abs(pfEstimates[t][j] - kfStates[t][j]));
    }
    stepDiff[t] = worst;
    if (worst > maxDiff) maxDiff = worst;
    sumDiff += worst;
    cnt++;
  }
  return { maxDiff, meanDiff: cnt > 0 ? sumDiff / cnt : 0, stepDiff };
}

/**
 * 旗舰AI引擎 · 隐马尔可夫模型（HMM, Baum & Petrie 1966）
 * 前向/后向 + Baum-Welch 期望最大化（EM 定理保证对数似然单调不减）。
 * 缩放前向（Rabiner 1989）：α̂_1 = π⊙B(·O₁)，c_t 归一，logP(O) = −Σln c_t
 * 不变量：
 *   ① 缩放前向 logP(O) ≡ 全路径暴力枚举 logsumexp（独立交叉验证）
 *   ② 平滑后验 γ_t 每行 Σ=1
 *   ③ ξ_t(i,j) 归一后 Σ=1，且 Σⱼξ_t(i,j) = γ_t(i)（前后向一致性）
 *   ④ Baum-Welch 对数似然单调不减（EM 定理）
 */
import { logsumexp } from '../core/numerical.js';

/**
 * 缩放前向算法。
 * @param {number[][]} A N×N 转移矩阵
 * @param {number[][]} B N×M 发射矩阵
 * @param {number[]} pi 初始分布
 * @param {number[]} obs 观测序列（符号 0..M−1）
 * @returns {{alphaHat:number[][], c:number[], logLik:number}}
 */
export function hmmForward(A, B, pi, obs) {
  const N = A.length, T = obs.length;
  const alphaHat = [], c = [];
  let a = pi.map((p, i) => p * B[i][obs[0]]);
  let ct = a.reduce((s, v) => s + v, 0);
  if (ct <= 0) return { alphaHat, c: [ct], logLik: -Infinity };
  c.push(ct);
  alphaHat.push(a.map((v) => v / ct));
  for (let t = 1; t < T; t++) {
    const prev = alphaHat[t - 1];
    const at = new Array(N);
    for (let j = 0; j < N; j++) {
      let s = 0;
      for (let i = 0; i < N; i++) s += prev[i] * A[i][j];
      at[j] = s * B[j][obs[t]];
    }
    ct = at.reduce((s, v) => s + v, 0);
    if (ct <= 0) return { alphaHat, c, logLik: -Infinity };
    c.push(ct);
    alphaHat.push(at.map((v) => v / ct));
  }
  // P(O) = Π_t c_t ⇒ logP(O) = Σ_t ln c_t
  const logLik = c.reduce((s, v) => s + Math.log(v), 0);
  return { alphaHat, c, logLik };
}

/**
 * 缩放后向算法：β̂_T = 1，β̂_t(i) = [ΣⱼAᵢⱼBⱼ(O_{t+1})β̂_{t+1}(j)] / c_{t+1}
 * @returns {number[][]} betaHat [T][N]
 */
export function hmmBackward(A, B, obs, c) {
  const N = A.length, T = obs.length;
  const betaHat = new Array(T);
  betaHat[T - 1] = new Array(N).fill(1);
  for (let t = T - 2; t >= 0; t--) {
    const bt = new Array(N);
    for (let i = 0; i < N; i++) {
      let s = 0;
      for (let j = 0; j < N; j++) s += A[i][j] * B[j][obs[t + 1]] * betaHat[t + 1][j];
      bt[i] = s / c[t + 1];
    }
    betaHat[t] = bt;
  }
  return betaHat;
}

/** 平滑后验 γ_t(i) = α̂_t(i)·β̂_t(i)（每行 Σ=1） */
export function hmmGamma(alphaHat, betaHat) {
  return alphaHat.map((a, t) => a.map((v, i) => v * betaHat[t][i]));
}

/**
 * 期望充分统计 ξ_t(i,j) = α̂_t(i)·Aᵢⱼ·Bⱼ(O_{t+1})·β̂_{t+1}(j) / c_{t+1}（t = 0..T−2）
 * 归一后 Σ_{ij}ξ_t = 1，且 Σⱼξ_t(i,j) = γ_t(i)（前后向一致性）
 */
export function hmmXi(A, B, alphaHat, betaHat, obs, c) {
  const N = A.length, T = obs.length;
  const xi = [];
  for (let t = 0; t < T - 1; t++) {
    const m = Array.from({ length: N }, () => new Array(N));
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        m[i][j] = alphaHat[t][i] * A[i][j] * B[j][obs[t + 1]] * betaHat[t + 1][j] / c[t + 1];
      }
    }
    xi.push(m);
  }
  return xi;
}

/**
 * Baum-Welch 参数重估（EM）。
 * @returns {{A:number[][], B:number[][], pi:number[], logLikHistory:number[]}}
 */
export function hmmBaumWelch(A, B, pi, obs, { iters = 40, tol = 1e-10 } = {}) {
  const N = A.length, M = B[0].length, T = obs.length;
  let A0 = A.map((r) => r.slice());
  let B0 = B.map((r) => r.slice());
  let pi0 = pi.slice();
  const logLikHistory = [];
  for (let it = 0; it < iters; it++) {
    const { alphaHat, c, logLik } = hmmForward(A0, B0, pi0, obs);
    if (!isFinite(logLik)) break;
    logLikHistory.push(logLik);
    const betaHat = hmmBackward(A0, B0, obs, c);
    const gamma = hmmGamma(alphaHat, betaHat);
    const xi = hmmXi(A0, B0, alphaHat, betaHat, obs, c);
    // M 步：π、A、B
    const piN = gamma[0].slice();
    const AN = Array.from({ length: N }, () => new Array(N).fill(0));
    const BN = Array.from({ length: N }, () => new Array(M).fill(0));
    for (let t = 0; t < T - 1; t++) {
      for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) AN[i][j] += xi[t][i][j];
      }
    }
    for (let i = 0; i < N; i++) {
      let rowSum = 0;
      for (let j = 0; j < N; j++) rowSum += AN[i][j];
      if (rowSum > 0) for (let j = 0; j < N; j++) AN[i][j] /= rowSum;
      else AN[i] = A0[i].slice();
    }
    for (let t = 0; t < T; t++) {
      for (let i = 0; i < N; i++) BN[i][obs[t]] += gamma[t][i];
    }
    for (let i = 0; i < N; i++) {
      let rowSum = 0;
      for (let k = 0; k < M; k++) rowSum += BN[i][k];
      if (rowSum > 0) for (let k = 0; k < M; k++) BN[i][k] /= rowSum;
      else BN[i] = B0[i].slice();
    }
    const piSum = piN.reduce((s, v) => s + v, 0);
    if (piSum > 0) for (let i = 0; i < N; i++) piN[i] /= piSum;
    A0 = AN; B0 = BN; pi0 = piN;
    if (logLikHistory.length >= 2
      && Math.abs(logLikHistory[logLikHistory.length - 1] - logLikHistory[logLikHistory.length - 2]) < tol) {
      const last = hmmForward(A0, B0, pi0, obs).logLik;
      if (isFinite(last)) logLikHistory.push(last);
      break;
    }
  }
  return { A: A0, B: B0, pi: pi0, logLikHistory };
}

/**
 * 暴力枚举全部 N^T 条状态路径求 logP(O)（logsumexp 稳定累加）。
 * 与 hmmForward 互为独立实现，用于交叉验证。
 */
export function hmmBruteForceLogLik(A, B, pi, obs) {
  const N = A.length, T = obs.length;
  const logs = [];
  const path = new Array(T);
  const rec = (t) => {
    if (t === T) {
      let lp = Math.log(pi[path[0]]) + Math.log(B[path[0]][obs[0]]);
      for (let s = 1; s < T; s++) {
        lp += Math.log(A[path[s - 1]][path[s]]) + Math.log(B[path[s]][obs[s]]);
      }
      logs.push(lp);
      return;
    }
    for (let i = 0; i < N; i++) { path[t] = i; rec(t + 1); }
  };
  rec(0);
  return logsumexp(logs);
}

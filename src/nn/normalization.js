/**
 * 旗舰AI引擎 · 归一化层（Normalization Layers）
 * 站在 Ioffe & Szegedy 2015（Batch Normalization，把每 mini-batch 的激活
 *   拉到零均值单位方差，解除内部协变量偏移）与 Ba et al. 2016
 *  （Layer Normalization，对单样本最后一维做归一，序列/Transformer 标配）之上。
 *
 * 设计要点：
 *   ① 训练期对每个特征通道（BatchNorm）/ 每个样本（LayerNorm）求均值方差，
 *      标准化后再做仿射 y = γ·x̂ + β —— γ、β 是可学习参数。
 *   ② 推理期 BatchNorm 复用训练期累计的 runningMean / runningVar（指数滑动平均）。
 *   ③ 反向传播严格按 CS231n 的解析公式，配合中心差分梯度检验钉死正确性。
 *
 * 不变量：
 *   ① BatchNorm(γ=1,β=0) 后每个通道均值 ≡ 0、方差 ≡ 1（浮点残差 < 1e-9）。
 *   ② 仿射可任意恢复：输出通道均值 ≡ β、标准差 ≡ |γ|。
 *   ③ 解析梯度 ≡ 中心差分（maxRel < 1e-6）。
 *   ④ 训练期 running stats 指数滑动平均收敛到真实均值/方差。
 *   ⑤ LayerNorm 同③：单样本归一后均值 0、方差 1，且对归一维度置换等变。
 */

import { relErr } from '../core/numerical.js';
import { mulberry32 } from '../core/stats.js';

/* ───────────────────────── Batch Normalization ───────────────────────── */

/**
 * BatchNorm 前向（训练期 / 推理期统一入口）。
 * @param {number[][]} X n×d 输入（n 个样本、d 个特征通道）
 * @param {number[]} gamma 长度 d 的缩放参数
 * @param {number[]} beta  长度 d 的平移参数
 * @returns {{Y:number[][], cache:object}}
 */
export function batchNormForward(X, gamma, beta, {
  eps = 1e-5, training = true, runningMean = null, runningVar = null, momentum = 0.1,
} = {}) {
  const n = X.length, d = X[0].length;
  const mean = new Array(d).fill(0), varr = new Array(d).fill(0);
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) mean[j] += X[i][j];
  for (let j = 0; j < d; j++) mean[j] /= n;
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) {
    const diff = X[i][j] - mean[j];
    varr[j] += diff * diff;
  }
  for (let j = 0; j < d; j++) varr[j] /= n;

  const muUse = training ? mean : runningMean;
  const varUse = training ? varr : runningVar;
  const std = varUse.map((v) => Math.sqrt(v + eps));
  const xhat = X.map((row) => row.map((v, j) => (v - muUse[j]) / std[j]));
  const Y = xhat.map((row) => row.map((v, j) => gamma[j] * v + beta[j]));

  // 训练期累计 running stats（指数滑动平均）
  if (training && runningMean && runningVar) {
    for (let j = 0; j < d; j++) {
      runningMean[j] = (1 - momentum) * runningMean[j] + momentum * mean[j];
      runningVar[j] = (1 - momentum) * runningVar[j] + momentum * varr[j];
    }
  }
  const cache = { X, mean, varr, std, xhat, gamma, beta, eps, n, training };
  return { Y, cache };
}

/**
 * BatchNorm 反向传播（CS231n 解析公式）。
 * @param {number[][]} dY 上游梯度 ∂L/∂Y（n×d）
 * @returns {{dx:number[][], dgamma:number[], dbeta:number[]}}
 */
export function batchNormBackward(dY, cache) {
  const { X, mean, varr, std, xhat, gamma, eps, n } = cache;
  const d = X[0].length;

  const dgamma = new Array(d).fill(0), dbeta = new Array(d).fill(0);
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) {
    dgamma[j] += dY[i][j] * xhat[i][j];
    dbeta[j] += dY[i][j];
  }
  const dxhat = dY.map((row, i) => row.map((v, j) => v * gamma[j]));

  const dvar = new Array(d).fill(0);
  for (let j = 0; j < d; j++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += dxhat[i][j] * (X[i][j] - mean[j]);
    dvar[j] = s * (-0.5) * Math.pow(varr[j] + eps, -1.5);
  }
  const dmean = new Array(d).fill(0);
  for (let j = 0; j < d; j++) {
    let s1 = 0, s2 = 0;
    for (let i = 0; i < n; i++) { s1 += dxhat[i][j]; s2 += (X[i][j] - mean[j]); }
    dmean[j] = s1 * (-1 / std[j]) + dvar[j] * (-2 / n) * s2;
  }
  const dx = X.map((row, i) => {
    const out = new Array(d);
    for (let j = 0; j < d; j++) {
      out[j] = dxhat[i][j] / std[j] + dvar[j] * 2 * (X[i][j] - mean[j]) / n + dmean[j] / n;
    }
    return out;
  });
  return { dx, dgamma, dbeta };
}

/** 推理期 BatchNorm（使用 running stats，不做归一化统计） */
export function batchNormInfer(X, gamma, beta, runningMean, runningVar, { eps = 1e-5 } = {}) {
  const Y = X.map((row) => row.map((v, j) => {
    const std = Math.sqrt(runningVar[j] + eps);
    return gamma[j] * (v - runningMean[j]) / std + beta[j];
  }));
  return Y;
}

/**
 * BatchNorm 梯度检验：解析梯度 ≡ 中心差分（对 X / γ / β 三路）。
 * @returns {{maxRel:number, maxAbs:number}}
 */
export function batchNormGradCheck(X, gamma, beta, { eps = 1e-5, seed = 42 } = {}) {
  const d = X[0].length;
  const rng = mulberry32(seed);
  const dY = X.map(() => new Array(d).fill(0).map(() => rng() * 2 - 1));
  const loss = () => {
    const { Y } = batchNormForward(X, gamma, beta, { training: true });
    let s = 0;
    for (let i = 0; i < X.length; i++) for (let j = 0; j < d; j++) s += Y[i][j] * dY[i][j];
    return s;
  };
  const { dx, dgamma, dbeta } = batchNormBackward(dY, batchNormForward(X, gamma, beta, { training: true }).cache);
  let maxRel = 0, maxAbs = 0;
  for (let i = 0; i < X.length; i++) for (let j = 0; j < d; j++) {
    const o = X[i][j];
    X[i][j] = o + eps; const lp = loss();
    X[i][j] = o - eps; const lm = loss();
    X[i][j] = o;
    const g = (lp - lm) / (2 * eps);
    maxRel = Math.max(maxRel, relErr(g, dx[i][j]));
    maxAbs = Math.max(maxAbs, Math.abs(g - dx[i][j]));
  }
  for (let j = 0; j < d; j++) {
    const og = gamma[j];
    gamma[j] = og + eps; const lpg = loss();
    gamma[j] = og - eps; const lmg = loss();
    gamma[j] = og;
    const gg = (lpg - lmg) / (2 * eps);
    maxRel = Math.max(maxRel, relErr(gg, dgamma[j]));
    maxAbs = Math.max(maxAbs, Math.abs(gg - dgamma[j]));
    const ob = beta[j];
    beta[j] = ob + eps; const lpb = loss();
    beta[j] = ob - eps; const lmb = loss();
    beta[j] = ob;
    const gb = (lpb - lmb) / (2 * eps);
    maxRel = Math.max(maxRel, relErr(gb, dbeta[j]));
    maxAbs = Math.max(maxAbs, Math.abs(gb - dbeta[j]));
  }
  return { maxRel, maxAbs };
}

/* ───────────────────────── Layer Normalization ───────────────────────── */

/**
 * 层归一化（行式，对最后一维统计）：(x − μ)/√(σ²+ε)·γ + β。
 * 与 transformer.js 的 layerNorm（单向量）呼应，此处提供批量 + 仿射 + 缓存（供反向）。
 * @returns {{Y:number[][], cache:object}}
 */
export function layerNorm2d(X, { eps = 1e-5, gamma = null, beta = null } = {}) {
  const d = X[0].length;
  const g = gamma || new Array(d).fill(1);
  const b = beta || new Array(d).fill(0);
  const Y = X.map((row) => {
    let mu = 0;
    for (const v of row) mu += v;
    mu /= d;
    let varr = 0;
    for (const v of row) varr += (v - mu) * (v - mu);
    varr /= d;
    const std = Math.sqrt(varr + eps);
    return row.map((v, j) => (v - mu) / std * g[j] + b[j]);
  });
  return { Y, cache: { X, eps, g, b, d } };
}

/**
 * LayerNorm 反向传播（每样本独立，对归一维 d 求和）。
 * @returns {{dx:number[][], dgamma:number[], dbeta:number[]}}
 */
export function layerNormBackward(dY, cache) {
  const { X, eps, g, d } = cache;
  const n = X.length;
  const dgamma = new Array(d).fill(0), dbeta = new Array(d).fill(0);
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) {
    let mu = 0;
    for (const v of X[i]) mu += v;
    mu /= d;
    let varr = 0;
    for (const v of X[i]) varr += (v - mu) * (v - mu);
    varr /= d;
    const std = Math.sqrt(varr + eps);
    const xhat = (X[i][j] - mu) / std;
    dgamma[j] += dY[i][j] * xhat;
    dbeta[j] += dY[i][j];
  }
  const dx = X.map((row, i) => {
    let mu = 0;
    for (const v of row) mu += v;
    mu /= d;
    let varr = 0;
    for (const v of row) varr += (v - mu) * (v - mu);
    varr /= d;
    const std = Math.sqrt(varr + eps);
    const dxhat = row.map((v, j) => dY[i][j] * g[j]);
    let dvar = 0;
    for (let j = 0; j < d; j++) dvar += dxhat[j] * (row[j] - mu) * (-0.5) * Math.pow(varr + eps, -1.5);
    let dmean = 0;
    let s2 = 0;
    for (let j = 0; j < d; j++) { dmean += dxhat[j] * (-1 / std); s2 += (row[j] - mu); }
    dmean += dvar * (-2 / d) * s2;
    const out = new Array(d);
    for (let j = 0; j < d; j++) out[j] = dxhat[j] / std + dvar * 2 * (row[j] - mu) / d + dmean / d;
    return out;
  });
  return { dx, dgamma, dbeta };
}

/**
 * LayerNorm 梯度检验：解析梯度 ≡ 中心差分。
 * @returns {{maxRel:number, maxAbs:number}}
 */
export function layerNormGradCheck(X, { eps = 1e-5, seed = 42 } = {}) {
  const d = X[0].length;
  const rng = mulberry32(seed);
  const gamma = new Array(d).fill(0).map(() => rng() * 2 - 1);
  const beta = new Array(d).fill(0).map(() => rng() * 2 - 1);
  const dY = X.map(() => new Array(d).fill(0).map(() => rng() * 2 - 1));
  const loss = () => {
    const { Y } = layerNorm2d(X, { gamma, beta });
    let s = 0;
    for (let i = 0; i < X.length; i++) for (let j = 0; j < d; j++) s += Y[i][j] * dY[i][j];
    return s;
  };
  const { dx, dgamma, dbeta } = layerNormBackward(dY, layerNorm2d(X, { gamma, beta }).cache);
  let maxRel = 0, maxAbs = 0;
  for (let i = 0; i < X.length; i++) for (let j = 0; j < d; j++) {
    const o = X[i][j];
    X[i][j] = o + eps; const lp = loss();
    X[i][j] = o - eps; const lm = loss();
    X[i][j] = o;
    const g = (lp - lm) / (2 * eps);
    maxRel = Math.max(maxRel, relErr(g, dx[i][j]));
    maxAbs = Math.max(maxAbs, Math.abs(g - dx[i][j]));
  }
  for (let j = 0; j < d; j++) {
    const og = gamma[j];
    gamma[j] = og + eps; const lpg = loss();
    gamma[j] = og - eps; const lmg = loss();
    gamma[j] = og;
    const gg = (lpg - lmg) / (2 * eps);
    maxRel = Math.max(maxRel, relErr(gg, dgamma[j]));
    maxAbs = Math.max(maxAbs, Math.abs(gg - dgamma[j]));
    const ob = beta[j];
    beta[j] = ob + eps; const lpb = loss();
    beta[j] = ob - eps; const lmb = loss();
    beta[j] = ob;
    const gb = (lpb - lmb) / (2 * eps);
    maxRel = Math.max(maxRel, relErr(gb, dbeta[j]));
    maxAbs = Math.max(maxAbs, Math.abs(gb - dbeta[j]));
  }
  return { maxRel, maxAbs };
}

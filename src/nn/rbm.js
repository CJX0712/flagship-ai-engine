/**
 * 旗舰AI引擎 · 受限玻尔兹曼机（RBM，CD-k 训练）
 * 站在：Smolensky 1986（Harmonium 信息处理模型）· Hinton 2002《Training Products
 *       of Experts by Minimizing Contrastive Divergence》· Hinton & Salakhutdinov
 *       2006 Science《Reducing the Dimensionality of Data with Neural Networks》。
 *
 * 二部图能量：E(v,h) = −vᵀWh − bᵀv − cᵀh；
 *   条件概率 P(hⱼ=1|v) = σ((Wᵀv+c)ⱼ)，P(vᵢ=1|h) = σ((Wh+b)ᵢ)（二部结构 ⇒ 条件独立）。
 * 自由能（配分函数恒等式）：FE(v) = −bᵀv − Σⱼ log(1+e^{cⱼ+(Wv)ⱼ}) ≡ −log Σ_h e^{−E(v,h)}。
 * CD-k 训练：正相 v⁰→h⁰ 采样得数据期望；负相 k 步 Gibbs 得模型期望；
 *   ΔW = lr(E_data[vhᵀ] − E_model[vhᵀ])，b、c 同理（批梯度，E 为批内均值）。
 *
 * 不变量：
 *   ① P(h|v) ≡ 独立 σ(Wᵀv+c)（<1e-15）；
 *   ② 自由能-配分函数恒等式：枚举全部 2^v 可见态，softmax(−FE) ≡ 联合分布直接枚举归一（<1e-10）；
 *   ③ CD 训练后训练集平均重构误差 < 训练前（固定 seed，留足 margin）；
 *   ④ 训练后训练样本 FE < 随机样本 FE（能量间隔 > 0）；
 *   ⑤ CD 更新恒等式：单步 ΔW ≡ lr(E_data[vhᵀ]−E_model[vhᵀ])（重算正负相一致 <1e-12）；
 *   ⑥ softmax(−FE) 在全状态上概率和 = 1（<1e-12）。
 */

import { zeros, randomMatrix } from '../core/linalg.js';
import { sigmoid } from '../core/numerical.js';
import { mulberry32 } from '../core/stats.js';

/** 隐藏激活概率 P(h=1|v) = σ(Wᵀv + c)（按隐藏单元序） */
function rbmHiddenProbsFrom(W, c, v) {
  const h = c.length;
  const p = new Array(h);
  for (let j = 0; j < h; j++) {
    let s = c[j];
    for (let i = 0; i < v.length; i++) s += W[i][j] * v[i];
    p[j] = sigmoid(s);
  }
  return p;
}

/** 可见激活概率 P(v=1|h) = σ(Wh + b)（按可见单元序） */
function rbmVisibleProbs(W, b, h) {
  const v = b.length;
  const p = new Array(v);
  for (let i = 0; i < v; i++) {
    let s = b[i];
    for (let j = 0; j < h.length; j++) s += W[i][j] * h[j];
    p[i] = sigmoid(s);
  }
  return p;
}

/** 逐单元伯努利采样（按向量序消耗 rng，保证可复现） */
function rbmBernoulliVec(probs, rng) {
  const out = new Array(probs.length);
  for (let i = 0; i < probs.length; i++) out[i] = rng() < probs[i] ? 1 : 0;
  return out;
}

/**
 * RBM 训练（批 CD-k）。
 * @param {number[][]} X 0/1 数据矩阵（n×visible）
 * @param {object} [opts] { hidden=4, cdK=1, lr=0.1, iters=400, seed=42 }
 * @returns {{W:number[][], b:number[], c:number[], hidden:number, visible:number}}
 */
export function rbmTrain(X, { hidden = 4, cdK = 1, lr = 0.1, iters = 400, seed = 42 } = {}) {
  const n = X.length;
  const visible = X[0].length;
  const rng = mulberry32(seed); // 全库统一 PRNG：初始化 + 所有采样同源
  const W = randomMatrix(visible, hidden, rng, -0.1, 0.1); // 小随机初始化（打破对称）
  const b = new Array(visible).fill(0);
  const c = new Array(hidden).fill(0);
  for (let it = 0; it < iters; it++) {
    // 正/负相统计量（全量累加，循环末单次乘 lr/n）
    const dW = zeros(visible, hidden), db = new Array(visible).fill(0), dc = new Array(hidden).fill(0);
    const mW = zeros(visible, hidden), mb = new Array(visible).fill(0), mc = new Array(hidden).fill(0);
    for (let s = 0; s < n; s++) {
      const v0 = X[s];
      // ── 正相：v⁰ → h⁰（采样），数据期望 E_data[v⁰h⁰ᵀ] ──
      const h0 = rbmBernoulliVec(rbmHiddenProbsFrom(W, c, v0), rng);
      for (let i = 0; i < visible; i++) {
        db[i] += v0[i];
        for (let j = 0; j < hidden; j++) dW[i][j] += v0[i] * h0[j];
      }
      for (let j = 0; j < hidden; j++) dc[j] += h0[j];
      // ── 负相：从 h⁰ 出发 k 步 Gibbs，模型期望 E_model[vᵏhᵏᵀ] ──
      let v = v0, h = h0;
      for (let t = 0; t < cdK; t++) {
        v = rbmBernoulliVec(rbmVisibleProbs(W, b, h), rng);
        h = rbmBernoulliVec(rbmHiddenProbsFrom(W, c, v), rng);
      }
      for (let i = 0; i < visible; i++) {
        mb[i] += v[i];
        for (let j = 0; j < hidden; j++) mW[i][j] += v[i] * h[j];
      }
      for (let j = 0; j < hidden; j++) mc[j] += h[j];
    }
    // Δθ = lr·(E_data − E_model)，E 为批内均值（除 n）
    const scale = lr / n;
    for (let i = 0; i < visible; i++) {
      b[i] += scale * (db[i] - mb[i]);
      for (let j = 0; j < hidden; j++) W[i][j] += scale * (dW[i][j] - mW[i][j]);
    }
    for (let j = 0; j < hidden; j++) c[j] += scale * (dc[j] - mc[j]);
  }
  return { W, b, c, hidden, visible };
}

/**
 * 隐藏单元条件概率 P(h=1|v)。
 * @param {object} model rbmTrain 的返回
 * @param {number[]} v 可见状态（0/1 向量）
 * @returns {number[]}
 */
export function rbmHiddenProbs(model, v) {
  return rbmHiddenProbsFrom(model.W, model.c, v);
}

/**
 * 一步重构：v⁰ → h⁰（概率）→ v¹（概率）。
 * @param {object} model rbmTrain 的返回
 * @param {number[]} v 可见状态
 * @returns {{vProb:number[], hProb:number[]}}
 */
export function rbmReconstruct(model, v) {
  const hProb = rbmHiddenProbsFrom(model.W, model.c, v);
  const vProb = rbmVisibleProbs(model.W, model.b, hProb);
  return { vProb, hProb };
}

/**
 * 自由能 FE(v) = −bᵀv − Σⱼ log(1+e^{cⱼ+(Wv)ⱼ})（log(1+e^x) 用双支式防溢出）。
 * @param {object} model rbmTrain 的返回
 * @param {number[]} v 可见状态
 * @returns {number}
 */
export function rbmFreeEnergy(model, v) {
  const { W, b, c } = model;
  let fe = 0;
  for (let i = 0; i < v.length; i++) fe -= b[i] * v[i];
  for (let j = 0; j < c.length; j++) {
    let x = c[j];
    for (let i = 0; i < v.length; i++) x += W[i][j] * v[i];
    // log(1+e^x) = max(x,0) + log(1+e^{−|x|})
    fe -= (x > 0 ? x : 0) + Math.log(1 + Math.exp(-Math.abs(x)));
  }
  return fe;
}

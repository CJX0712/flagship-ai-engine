/**
 * 旗舰AI引擎 · Word2Vec：Skip-gram with Negative Sampling（SGNS）
 * 站在：Mikolov et al. 2013《Distributed Representations of Words and Phrases
 *       and their Compositionality》(NIPS) / 《Efficient Estimation of Word
 *       Representations in Vector Space》(ICLR Workshop)
 *       Goldberg & Levy 2014（word2vec 目标的负采样推导）
 * 目标函数（每个中心词 c、上下文词 o、K 个负样本 n_k）：
 *   J = −log σ(v_c·u_o) − Σ_k log σ(−v_c·u_{n_k})
 * 梯度（σ′(x) = σ(x)(1−σ(x)) 的化简）：
 *   ∂J/∂v_c = (σ(s_o) − 1)·u_o + Σ_k σ(s_k)·u_{n_k}
 *   ∂J/∂u_o = (σ(s_o) − 1)·v_c ；  ∂J/∂u_{n_k} = σ(s_k)·v_c
 * 不变量：
 *   ① 解析梯度 ≡ 中心差分（v_c / u_o / u_neg 三路）
 *   ② σ(−x) = 1 − σ(x)（负采样损失的对偶恒等，bit 级）
 *   ③ 损失闭式 ≡ 逐项 sigmoid 参考实现（双实现互证）
 *   ④ 训练有效性：训练后共现词对的余弦相似度 > 非共现对（表示学习把共现结构编码进向量）
 *   ⑤ 训练损失（滑动平均）整体下降
 */

import { sigmoid } from '../core/numerical.js';
import { mulberry32 } from '../core/stats.js';
import { norm, dot } from '../core/linalg.js';

/**
 * SGNS 单步损失：−log σ(v_c·u_o) − Σ_k log σ(−v_c·u_k)
 */
export function sgnsLoss(vc, uo, negatives) {
  const so = dot(vc, uo);
  // 数值稳定：log σ(x) = −log(1+e^{−x}) = −softplus(−x)
  let loss = -Math.log(sigmoid(so) + 1e-300);
  for (const uk of negatives) {
    const sk = dot(vc, uk);
    loss += -Math.log(sigmoid(-sk) + 1e-300);
  }
  return loss;
}

/**
 * SGNS 解析梯度。
 * @returns {{loss:number, dvc:number[], duo:number[], dneg:number[][]}}
 */
export function sgnsGradients(vc, uo, negatives) {
  const d = vc.length;
  const dvc = new Array(d).fill(0);
  const duo = new Array(d).fill(0);
  const dneg = negatives.map(() => new Array(d).fill(0));
  let loss = 0;
  const so = dot(vc, uo);
  const po = sigmoid(so);
  loss += -Math.log(po + 1e-300);
  for (let i = 0; i < d; i++) {
    dvc[i] += (po - 1) * uo[i];
    duo[i] += (po - 1) * vc[i];
  }
  for (let k = 0; k < negatives.length; k++) {
    const uk = negatives[k];
    const sk = dot(vc, uk);
    const pk = sigmoid(sk);
    loss += -Math.log(1 - pk + 1e-300); // log σ(−s) = log(1−σ(s))
    for (let i = 0; i < d; i++) {
      dvc[i] += pk * uk[i];
      dneg[k][i] += pk * vc[i];
    }
  }
  return { loss, dvc, duo, dneg };
}

/**
 * 损失闭式的参考实现（逐项累加，用于与向量化实现互证）。
 */
export function sgnsLossReference(vc, uo, negatives) {
  let loss = 0;
  let so = 0;
  for (let i = 0; i < vc.length; i++) so += vc[i] * uo[i];
  loss += -Math.log(1 / (1 + Math.exp(-so)));
  for (const uk of negatives) {
    let sk = 0;
    for (let i = 0; i < vc.length; i++) sk += vc[i] * uk[i];
    loss += -Math.log(1 / (1 + Math.exp(sk)));
  }
  return loss;
}

/**
 * SGNS 梯度检验：解析梯度 ≡ 中心差分。
 * @returns {{maxRel:number, maxAbs:number}}
 */
export function sgnsGradientCheck(vc, uo, negatives, { eps = 1e-5 } = {}) {
  const g = sgnsGradients(vc, uo, negatives);
  let maxRel = 0, maxAbs = 0;
  const cmp = (num, ana) => {
    maxRel = Math.max(maxRel, Math.abs(num - ana) / (Math.abs(num) + Math.abs(ana) + 1e-8));
    maxAbs = Math.max(maxAbs, Math.abs(num - ana));
  };
  const probeVec = (v, gv, rebuild) => {
    for (let i = 0; i < v.length; i++) {
      const orig = v[i];
      v[i] = orig + eps; const lp = rebuild();
      v[i] = orig - eps; const lm = rebuild();
      v[i] = orig;
      cmp((lp - lm) / (2 * eps), gv[i]);
    }
  };
  probeVec(vc, g.dvc, () => sgnsLoss(vc, uo, negatives));
  probeVec(uo, g.duo, () => sgnsLoss(vc, uo, negatives));
  for (let k = 0; k < negatives.length; k++) {
    probeVec(negatives[k], g.dneg[k], () => sgnsLoss(vc, uo, negatives));
  }
  return { maxRel, maxAbs };
}

/**
 * 训练 SGNS（纯 SGD，固定学习率）。
 * @param {number[][]} pairs [center, context] 索引对
 * @param {number} vocabSize 词表大小
 * @param {{dim?:number, negatives?:number, lr?:number, epochs?:number, seed?:number}} opts
 * @returns {{Win:number[][], Wout:number[][], losses:number[]}}
 */
export function trainSGNS(pairs, vocabSize, {
  dim = 10, negatives = 5, lr = 0.05, epochs = 60, seed = 42,
} = {}) {
  const rng = mulberry32(seed);
  const Win = [], Wout = [];
  for (let i = 0; i < vocabSize; i++) {
    const a = [], b = [];
    for (let d = 0; d < dim; d++) {
      a.push((rng() - 0.5) * 0.4);
      b.push((rng() - 0.5) * 0.1);
    }
    Win.push(a); Wout.push(b);
  }
  const losses = [];
  for (let ep = 0; ep < epochs; ep++) {
    let total = 0;
    for (const [c, o] of pairs) {
      const negIdx = [];
      for (let k = 0; k < negatives; k++) {
        let n;
        do { n = Math.floor(rng() * vocabSize); } while (n === o || n === c);
        negIdx.push(n);
      }
      const negs = negIdx.map((i) => Wout[i]);
      const g = sgnsGradients(Win[c], Wout[o], negs);
      total += g.loss;
      for (let i = 0; i < dim; i++) {
        Win[c][i] -= lr * g.dvc[i];
        Wout[o][i] -= lr * g.duo[i];
        for (let k = 0; k < negatives; k++) Wout[negIdx[k]][i] -= lr * g.dneg[k][i];
      }
    }
    losses.push(total / pairs.length);
  }
  return { Win, Wout, losses };
}

/** 余弦相似度（词向量的语义标尺） */
export function wordSimilarity(a, b) {
  const na = norm(a), nb = norm(b);
  return na > 0 && nb > 0 ? dot(a, b) / (na * nb) : 0;
}

/**
 * 构造确定性的人工语料：两个"语义团簇"（团内高频共现，团间无共现）。
 * @returns {{pairs:number[][], vocabSize:number, cooc:number[][], noncooc:number[][]}}
 */
export function makeToyCorpus({ clusterSize = 4, windowSize = 2, repeats = 12, seed = 42 } = {}) {
  const rng = mulberry32(seed);
  const vocabSize = clusterSize * 2;
  const clusterA = [];
  for (let i = 0; i < clusterSize; i++) clusterA.push(i);
  const clusterB = [];
  for (let i = 0; i < clusterSize; i++) clusterB.push(clusterSize + i);
  const pairs = [];
  const emit = (cl) => {
    for (let r = 0; r < repeats; r++) {
      const seq = cl.slice();
      for (let i = seq.length - 1; i > 0; i--) {
        const j = Math.floor(rng() * (i + 1));
        [seq[i], seq[j]] = [seq[j], seq[i]];
      }
      for (let i = 0; i < seq.length; i++) {
        for (let w = 1; w <= windowSize; w++) {
          if (i + w < seq.length) pairs.push([seq[i], seq[i + w]]);
        }
      }
    }
  };
  emit(clusterA);
  emit(clusterB);
  const cooc = [], noncooc = [];
  for (const a of clusterA) {
    for (const b of clusterA) if (a !== b) cooc.push([a, b]);
  }
  for (const a of clusterB) {
    for (const b of clusterB) if (a !== b) cooc.push([a, b]);
  }
  for (const a of clusterA) {
    for (const b of clusterB) { noncooc.push([a, b]); noncooc.push([b, a]); }
  }
  return { pairs, vocabSize, cooc, noncooc };
}

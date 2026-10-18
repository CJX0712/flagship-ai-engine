/**
 * 旗舰AI引擎 · 注意力机制
 * 《Attention Is All You Need》(Vaswani et al. 2017) 的可验证实现。
 * 不变量：注意力行和 = 1；对损失 ΣO 的手推梯度与中心差分一致（≈1e-8）；
 *        单头 + 单位投影 ≡ 缩放点积注意力。
 */

import { matmul, transpose, zeros, randomMatrix } from '../core/linalg.js';
import { softmax, relErr } from '../core/numerical.js';
import { mulberry32 } from '../core/stats.js';

/**
 * 缩放点积注意力：A = softmax(QKᵀ/√d_k)（逐行），O = A·V。
 * 返回 { A, O, scores }。
 */
export function scaledDotProductAttention(Q, K, V) {
  const dk = Q[0].length;
  const scale = 1 / Math.sqrt(dk);
  const scores = matmul(Q, transpose(K)).map(row => row.map(v => v * scale));
  const A = scores.map(row => softmax(row));
  const O = matmul(A, V);
  return { A, O, scores };
}

/**
 * 手推反向（损失 L = Σ O，即 dO 全 1）：
 *   dV = Aᵀ·dO
 *   dA = dO·Vᵀ
 *   dS_i = A_i ⊙ (dA_i − ⟨dA_i, A_i⟩)   （softmax 雅可比）
 *   dQ = dS·K/√d_k，dK = dSᵀ·Q/√d_k
 */
export function attentionGradients(Q, K, V) {
  const { A } = scaledDotProductAttention(Q, K, V);
  const nq = Q.length, dv = V[0].length;
  const dO = zeros(nq, dv).map(r => r.map(() => 1));
  const dV = matmul(transpose(A), dO);
  const dA = matmul(dO, transpose(V));
  const dS = zeros(nq, K.length);
  for (let i = 0; i < nq; i++) {
    let dotGA = 0;
    for (let j = 0; j < K.length; j++) dotGA += dA[i][j] * A[i][j];
    for (let j = 0; j < K.length; j++) dS[i][j] = A[i][j] * (dA[i][j] - dotGA);
  }
  const scale = 1 / Math.sqrt(Q[0].length);
  const dQ = matmul(dS, K).map(row => row.map(v => v * scale));
  const dK = matmul(transpose(dS), Q).map(row => row.map(v => v * scale));
  return { dQ, dK, dV, A };
}

/** 注意力梯度检验：maxRelErr ≈ 1e-8（阈值 1e-6） */
export function attentionGradientCheck(Q, K, V, { eps = 1e-5 } = {}) {
  const { dQ, dK, dV } = attentionGradients(Q, K, V);
  const lossOf = () => {
    const { O } = scaledDotProductAttention(Q, K, V);
    let s = 0;
    for (const row of O) for (const v of row) s += v;
    return s;
  };
  let maxErr = 0;
  const scan = (M, G) => {
    for (let i = 0; i < M.length; i++) {
      for (let j = 0; j < M[i].length; j++) {
        const orig = M[i][j];
        M[i][j] = orig + eps;
        const lp = lossOf();
        M[i][j] = orig - eps;
        const lm = lossOf();
        M[i][j] = orig;
        maxErr = Math.max(maxErr, relErr((lp - lm) / (2 * eps), G[i][j]));
      }
    }
  };
  scan(Q, dQ);
  scan(K, dK);
  scan(V, dV);
  return maxErr;
}

/**
 * 多头注意力：投影 → 分头 → 各头缩放点积 → 拼接 → 输出投影。
 * 不变量：heads=1 且 Wq=Wk=Wv=Wo=I 时与 scaledDotProductAttention 逐元素相等。
 */
export function multiHeadAttention(Q, K, V, { heads = 1, Wq = null, Wk = null, Wv = null, Wo = null }) {
  const d = Q[0].length;
  if (d % heads !== 0) throw new Error('d 必须被 heads 整除');
  const dh = d / heads;
  const I = zeros(d, d).map((r, i) => r.map((v, j) => (i === j ? 1 : 0)));
  const Qp = matmul(Q, Wq || I);
  const Kp = matmul(K, Wk || I);
  const Vp = matmul(V, Wv || I);
  const concat = zeros(Q.length, d);
  let A0 = null;
  for (let t = 0; t < heads; t++) {
    const slice = (M) => M.map(row => row.slice(t * dh, (t + 1) * dh));
    const { A, O } = scaledDotProductAttention(slice(Qp), slice(Kp), slice(Vp));
    if (t === 0) A0 = A;
    for (let i = 0; i < Q.length; i++) for (let j = 0; j < dh; j++) concat[i][t * dh + j] = O[i][j];
  }
  const O = matmul(concat, Wo || I);
  return { O, A: A0 };
}

/** 生成注意力检验用随机张量 */
export function randomAttentionTensors(nq, nk, d, seed = 42) {
  const rng = mulberry32(seed);
  return {
    Q: randomMatrix(nq, d, rng),
    K: randomMatrix(nk, d, rng),
    V: randomMatrix(nk, d, rng),
  };
}

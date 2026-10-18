/**
 * 旗舰AI引擎 · Transformer 基础件
 * LayerNorm / 正弦位置编码 / Transformer 块（前处理 LN 残差结构，GPT 风格）。
 */

import { matmul, transpose, zeros, randomMatrix } from '../core/linalg.js';
import { relu } from '../core/numerical.js';
import { mulberry32 } from '../core/stats.js';
import { multiHeadAttention } from './attention.js';

/**
 * 层归一化（Ba et al. 2016）：(x − μ)/√(σ² + ε) · γ + β，按最后维统计。
 */
export function layerNorm(x, { eps = 1e-5, gamma = null, beta = null } = {}) {
  const d = x.length;
  let mu = 0;
  for (const v of x) mu += v;
  mu /= d;
  let varr = 0;
  for (const v of x) varr += (v - mu) ** 2;
  varr /= d;
  return x.map((v, i) => (v - mu) / Math.sqrt(varr + eps) * (gamma ? gamma[i] : 1) + (beta ? beta[i] : 0));
}

/** 逐行层归一化（序列 × 特征） */
export function layerNormRows(X, opts = {}) {
  return X.map(row => layerNorm(row, opts));
}

/** 正弦位置编码：PE(pos,2i)=sin(pos/10000^{2i/d})，PE(pos,2i+1)=cos(·) */
export function positionalEncoding(seqLen, d) {
  const PE = zeros(seqLen, d);
  for (let pos = 0; pos < seqLen; pos++) {
    for (let i = 0; i < d; i += 2) {
      const denom = Math.pow(10000, i / d);
      PE[pos][i] = Math.sin(pos / denom);
      if (i + 1 < d) PE[pos][i + 1] = Math.cos(pos / denom);
    }
  }
  return PE;
}

/**
 * Transformer 块（pre-LN 残差）：
 *   X → LN → MHA → ⊕ → LN → FFN(ReLU, 4×) → ⊕ → out
 * 返回 { out, attn }；attn 为第一头注意力矩阵（行和 = 1）。
 */
export function transformerBlock(X, { heads = 1, seed = 42, eps = 1e-5 } = {}) {
  const d = X[0].length;
  const rng = mulberry32(seed);
  const scale = 1 / Math.sqrt(d);
  const Wq = randomMatrix(d, d, rng, -1, 1).map(r => r.map(v => v * scale));
  const Wk = randomMatrix(d, d, rng, -1, 1).map(r => r.map(v => v * scale));
  const Wv = randomMatrix(d, d, rng, -1, 1).map(r => r.map(v => v * scale));
  const Wo = randomMatrix(d, d, rng, -1, 1).map(r => r.map(v => v * scale));
  const W1 = randomMatrix(4 * d, d, rng, -1, 1).map(r => r.map(v => v * Math.sqrt(2 / d)));
  const W2 = randomMatrix(d, 4 * d, rng, -1, 1).map(r => r.map(v => v * Math.sqrt(2 / (4 * d))));
  const b1 = new Array(4 * d).fill(0);
  const b2 = new Array(d).fill(0);

  // 子层 1：注意力
  const H = layerNormRows(X, { eps });
  const { O, A: attn } = multiHeadAttention(H, H, H, { heads, Wq, Wk, Wv, Wo });
  const X2 = X.map((row, i) => row.map((v, j) => v + O[i][j]));
  // 子层 2：前馈
  const H2 = layerNormRows(X2, { eps });
  const hidden = matmul(H2, transpose(W1)).map((row, i) => row.map((v, j) => relu(v + b1[j])));
  const F = matmul(hidden, transpose(W2)).map((row, i) => row.map((v, j) => v + b2[j]));
  const out = X2.map((row, i) => row.map((v, j) => v + F[i][j]));
  return { out, attn };
}

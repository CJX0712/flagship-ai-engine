/**
 * 旗舰AI引擎 · 非线性降维：t-SNE（van der Maaten & Hinton 2008）
 * 站在信息几何之上：高维近邻用高斯条件分布 P 建模，低维用 Student-t 联合分布 Q 建模，
 * 梯度下降最小化 KL(P‖Q)。
 * 不变量：
 *   ① P/Q 均为对称联合分布：P = Pᵀ、Q = Qᵀ、Σ_{ij} = 1（概率质量守恒）
 *   ② 困惑度达成：二分搜索使每行 2^{H(p_i)} ≈ perplexity（条件分布正确归一）
 *   ③ 学习有效：KL 代价随迭代单调不增（最终 < 初值）
 *   ④ 结构保持：可分团簇的 k-NN 邻域重叠率 ≥ 0.85（局部结构保真）
 */
import { mulberry32, gaussianSample } from '../core/stats.js';

/** 成对平方距离矩阵 */
function pairwiseD2(X) {
  const n = X.length;
  const D2 = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      let s = 0;
      for (let k = 0; k < X[i].length; k++) {
        const diff = X[i][k] - X[j][k];
        s += diff * diff;
      }
      D2[i][j] = s;
      D2[j][i] = s;
    }
  }
  return D2;
}

/**
 * 为每个点二分搜索 beta，使条件分布困惑度 2^H 达成目标。
 * @param {number[][]} D2 平方距离矩阵
 * @param {number} perplexity 目标困惑度
 * @returns {{Pcond:number[][], betas:number[], entropies:number[]}}
 */
export function searchBetas(D2, perplexity) {
  const n = D2.length;
  const targetH = Math.log2(perplexity);
  const Pcond = Array.from({ length: n }, () => new Array(n).fill(0));
  const betas = new Array(n).fill(1);
  const entropies = new Array(n).fill(0);
  const log2 = Math.log2;

  for (let i = 0; i < n; i++) {
    let betaMin = 0, betaMax = 1e8;
    let pRow = new Array(n).fill(0);
    let H = 0;
    for (let iter = 0; iter < 60; iter++) {
      const beta = iter === 0 ? 1 : (betaMin + betaMax) / 2;
      // 数值稳定：max-shift 后归一
      const exps = new Array(n).fill(0);
      for (let j = 0; j < n; j++) if (j !== i) exps[j] = Math.exp(-beta * D2[i][j]);
      const sum = exps.reduce((a, b) => a + b, 0);
      H = 0;
      if (sum > 0) {
        for (let j = 0; j < n; j++) {
          if (j !== i && exps[j] > 0) {
            const p = exps[j] / sum;
            pRow[j] = p;
            H -= p * log2(p);
          } else pRow[j] = 0;
        }
      }
      if (H > targetH) { betaMin = beta; } // 熵偏高 → beta 太小 → 提高下界
      else { betaMax = beta; }             // 熵偏低 → beta 太大 → 降低上界
      if (Math.abs(H - targetH) < 1e-5 || (betaMax - betaMin) < 1e-10) break;
    }
    betas[i] = (betaMin + betaMax) / 2 || 1;
    entropies[i] = H;
    Pcond[i] = pRow;
  }
  return { Pcond, betas, entropies };
}

/**
 * t-SNE 主流程。
 * @param {number[][]} X 高维数据（行主序）
 * @param {{dims?:number, perplexity?:number, iters?:number, lr?:number, seed?:number}} opts
 * @returns {{Y:number[][], P:number[][], Q:number[][], costs:number[]}}
 */
export function tsne(X, {
  dims = 2, perplexity = 10, iters = 400, lr = 100, seed = 42,
} = {}) {
  const n = X.length;
  const D2 = pairwiseD2(X);

  // ① 高维联合分布 P（对称，ΣP = 1）
  const { Pcond, entropies } = searchBetas(D2, perplexity);
  const P = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      P[i][j] = (Pcond[i][j] + Pcond[j][i]) / (2 * n);
    }
  }

  // ② 低维初始化：N(0, 1e-4)（种子化，可复现）
  const rng = mulberry32(seed);
  const Y = Array.from({ length: n }, () =>
    Array.from({ length: dims }, () => gaussianSample(rng, 0, 1e-4)));

  const costs = [];
  const sampleEvery = Math.max(1, Math.floor(iters / 20));

  for (let iter = 0; iter < iters; iter++) {
    // ③ 低维联合分布 Q（Student-t）
    const W = Array.from({ length: n }, () => new Array(n).fill(0));
    let Z = 0;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        let d2 = 0;
        for (let k = 0; k < dims; k++) {
          const diff = Y[i][k] - Y[j][k];
          d2 += diff * diff;
        }
        const w = 1 / (1 + d2);
        W[i][j] = w;
        W[j][i] = w;
        Z += 2 * w;
      }
    }
    const Q = Array.from({ length: n }, () => new Array(n).fill(0));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) if (i !== j) Q[i][j] = W[i][j] / Z;
    }

    // ④ 梯度：dC/dy_i = 4 Σ_j (P_ij − Q_ij)·W_ij·(y_i − y_j)
    const grad = Array.from({ length: n }, () => new Array(dims).fill(0));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (i === j) continue;
        const coeff = 4 * (P[i][j] - Q[i][j]) * W[i][j];
        for (let k = 0; k < dims; k++) grad[i][k] += coeff * (Y[i][k] - Y[j][k]);
      }
    }

    if (iter % sampleEvery === 0) {
      let cost = 0;
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          if (i !== j && P[i][j] > 0) cost += P[i][j] * Math.log(P[i][j] / Math.max(Q[i][j], 1e-300));
        }
      }
      costs.push(cost);
    }

    for (let i = 0; i < n; i++) {
      for (let k = 0; k < dims; k++) Y[i][k] -= lr * grad[i][k];
    }
  }

  // 末态 Q 与代价（与循环内采样口径一致）
  const Q = Array.from({ length: n }, () => new Array(n).fill(0));
  {
    const W = Array.from({ length: n }, () => new Array(n).fill(0));
    let Z = 0;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        let d2 = 0;
        for (let k = 0; k < dims; k++) {
          const diff = Y[i][k] - Y[j][k];
          d2 += diff * diff;
        }
        const w = 1 / (1 + d2);
        W[i][j] = w; W[j][i] = w; Z += 2 * w;
      }
    }
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (i !== j) Q[i][j] = W[i][j] / Z;
  }
  return { Y, P, Q, costs, entropies, perplexity };
}

/**
 * k-NN 邻域重叠率：嵌入 Y 与原始 X 的局部结构保持度量。
 * @param {number[][]} X 原始数据
 * @param {number[][]} Y 嵌入
 * @param {number} [k=5]
 * @returns {number} 平均重叠率 ∈ [0,1]
 */
export function knnOverlap(X, Y, k = 5) {
  const topk = (M, i) => {
    const dists = M.map((row, j) => {
      if (j === i) return [j, Infinity];
      let s = 0;
      for (let d = 0; d < row.length; d++) s += (row[d] - M[i][d]) ** 2;
      return [j, s];
    }).filter(([, v]) => v !== Infinity);
    dists.sort((a, b) => a[1] - b[1]);
    return new Set(dists.slice(0, k).map(([j]) => j));
  };
  let total = 0;
  for (let i = 0; i < X.length; i++) {
    const sx = topk(X, i), sy = topk(Y, i);
    let inter = 0;
    for (const j of sx) if (sy.has(j)) inter++;
    total += inter / k;
  }
  return total / X.length;
}

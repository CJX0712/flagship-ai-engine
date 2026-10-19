/**
 * 旗舰AI引擎 · 核主成分分析（Kernel PCA）
 * 站在：Schölkopf, Smola & Müller 1998《Nonlinear Component Analysis as a
 *       Kernel Eigenvalue Problem》(Neural Computation 10(5))
 * 不变量：
 *   ① 线性核 KPCA ≡ 经典 PCA（主成分子空间一致，符号/尺度无关）
 *   ② 中心化核矩阵 K̃ 行和 ≈ 0 且 1ᵀK̃1 ≈ 0（特征空间均值为零）
 *   ③ K̃ 对称半正定（特征值 ≥ −ε·λ_max）
 *   ④ 谱定理：K̃ = VΛVᵀ 完全重构误差 ≈ 0
 *   ⑤ 样本外变换 kpcaTransform(X_train) ≡ 训练 scores（自洽互证）
 *   ⑥ RBF 核下两月亮在嵌入空间线性可分（留一 1-NN 纯度 → 1）
 */

import { rbfKernel } from './svm.js';
import { zeros, jacobiEig } from '../core/linalg.js';

/** 线性核 K(x,z) = x·z */
export function linearKernel(x, z) {
  let s = 0;
  for (let i = 0; i < x.length; i++) s += x[i] * z[i];
  return s;
}

/** 核矩阵 K[i][j] = k(X_i, Z_j) */
export function kernelMatrix(X, Z, kernelFn) {
  const K = zeros(X.length, Z.length);
  for (let i = 0; i < X.length; i++) {
    for (let j = 0; j < Z.length; j++) K[i][j] = kernelFn(X[i], Z[j]);
  }
  return K;
}

/**
 * 特征空间中心化：K̃ = K − 1ₙK − K1ₙ + 1ₙK1ₙ（Schölkopf et al. 1998 §II）。
 * 中心化后 K̃ 行和 ≈ 0、1ᵀK̃1 ≈ 0——即映射后样本在特征空间的均值为零。
 */
export function centerKernel(K) {
  const n = K.length;
  const rowMean = new Array(n).fill(0);
  const colMean = new Array(n).fill(0);
  let total = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      rowMean[i] += K[i][j];
      colMean[j] += K[i][j];
      total += K[i][j];
    }
  }
  for (let i = 0; i < n; i++) { rowMean[i] /= n; colMean[i] /= n; }
  total /= n * n;
  const Kc = zeros(n, n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) Kc[i][j] = K[i][j] - rowMean[i] - colMean[j] + total;
  }
  return Kc;
}

/**
 * 核 PCA 主过程：中心化核矩阵 → Jacobi 特征分解 → 前 dims 个主分量。
 * scores[i][k] = α_k(i)·√λ_k（训练点在特征空间主方向上的投影）。
 * @returns {{K:number[][], Kc:number[][], eigenvalues:number[], alphas:number[][],
 *            scores:number[][], eig:{values:number[],vectors:number[][]},
 *            gamma:number, kernel:string, X:number[][], kfn:Function, dims:number}}
 */
export function kpca(X, { kernel = 'rbf', gamma = 0.1, dims = 2 } = {}) {
  const kfn = kernel === 'linear' ? linearKernel : (a, b) => rbfKernel(a, b, gamma);
  const K = kernelMatrix(X, X, kfn);
  const Kc = centerKernel(K);
  const eig = jacobiEig(Kc); // values 降序；vectors[k] = 第 k 大特征值对应的单位特征向量
  const n = X.length;
  const d = Math.min(dims, n);
  const alphas = [];
  const scores = zeros(n, d);
  for (let k = 0; k < d; k++) {
    const lam = Math.max(eig.values[k], 0);
    alphas.push(eig.vectors[k]);
    for (let i = 0; i < n; i++) scores[i][k] = eig.vectors[k][i] * Math.sqrt(lam);
  }
  return {
    K, Kc, eigenvalues: eig.values.slice(0, d), alphas, scores,
    eig, gamma, kernel, X, kfn, dims: d,
  };
}

/**
 * 样本外投影：y(x*) = (1/√λ_k)·Σ_j α_k(j)·k̃(x*, x_j)。
 * 其中 k̃ 为中心化交叉核：k(x*,x_j) − mean_j k(x*,·) − mean_i k(·,x_j) + mean k(·,·)。
 */
export function kpcaTransform(model, Xnew) {
  const { X: Xtr, kfn, K, alphas, eigenvalues, dims } = model;
  const n = Xtr.length;
  // 训练核的列均值与总均值（与待投影点无关，预计算一次）
  const colMean = new Array(n).fill(0);
  let total = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) { colMean[j] += K[i][j]; total += K[i][j]; }
  }
  for (let j = 0; j < n; j++) colMean[j] /= n;
  total /= n * n;

  const out = zeros(Xnew.length, dims);
  for (let i = 0; i < Xnew.length; i++) {
    const row = new Array(n);
    let rowSum = 0;
    for (let j = 0; j < n; j++) { row[j] = kfn(Xnew[i], Xtr[j]); rowSum += row[j]; }
    rowSum /= n;
    for (let k = 0; k < dims; k++) {
      const lam = eigenvalues[k];
      if (lam <= 0) { out[i][k] = 0; continue; }
      let s = 0;
      for (let j = 0; j < n; j++) {
        const kc = row[j] - rowSum - colMean[j] + total;
        s += alphas[k][j] * kc;
      }
      out[i][k] = s / Math.sqrt(lam);
    }
  }
  return out;
}

/** 谱定理重构误差：‖K̃ − VΛVᵀ‖_F（取完整特征分解，应为机器精度级） */
export function kpcaReconstructionError(Kc, eig) {
  const n = Kc.length;
  const { values, vectors } = eig;
  let err = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let v = 0;
      for (let k = 0; k < n; k++) v += values[k] * vectors[k][i] * vectors[k][j];
      err += (v - Kc[i][j]) ** 2;
    }
  }
  return Math.sqrt(err);
}

/** 嵌入空间留一 1-NN 分类纯度（核方法把非线性结构线性化的直接度量） */
export function knnPurity(embedding, labels) {
  const n = embedding.length;
  let correct = 0;
  for (let i = 0; i < n; i++) {
    let best = -1, bd = Infinity;
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      let s = 0;
      for (let d = 0; d < embedding[j].length; d++) s += (embedding[i][d] - embedding[j][d]) ** 2;
      if (s < bd) { bd = s; best = j; }
    }
    if (labels[best] === labels[i]) correct++;
  }
  return correct / n;
}

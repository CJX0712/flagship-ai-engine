/**
 * 旗舰AI引擎 · 全连接神经网络
 * 前向（ReLU 隐层 + softmax 输出）/ 反向传播 / 中心差分梯度检验。
 * 梯度检验是不变量之母：maxRelErr ≈ 1e-8 ⇒ 反向传播数学正确。
 */

import { matmul, transpose, zeros, randomMatrix } from '../core/linalg.js';
import { softmax, relu, relErr } from '../core/numerical.js';
import { mulberry32 } from '../core/stats.js';

export class DenseNet {
  /**
   * @param {number[]} sizes 各层宽度，如 [3, 6, 3]
   * He 初始化（He et al. 2015），零偏置。
   */
  constructor(sizes, { seed = 42 } = {}) {
    this.sizes = sizes.slice();
    this.L = sizes.length - 1;
    const rng = mulberry32(seed);
    this.W = [];
    this.b = [];
    for (let l = 0; l < this.L; l++) {
      const fanIn = sizes[l];
      const scale = Math.sqrt(2 / fanIn);
      this.W.push(randomMatrix(sizes[l + 1], fanIn, rng, -1, 1).map(row => row.map(x => x * scale)));
      this.b.push(new Array(sizes[l + 1]).fill(0));
    }
  }

  /** 前向：返回 { acts(各层激活，acts[0]=X), zs(各层线性输出), P(softmax 概率) } */
  forward(X) {
    const acts = [X];
    const zs = [];
    let cur = X;
    for (let l = 0; l < this.L; l++) {
      const z = matmul(cur, transpose(this.W[l])).map((row, i) =>
        row.map((v, j) => v + this.b[l][j])
      );
      zs.push(z);
      cur = (l === this.L - 1)
        ? z.map(row => softmax(row))
        : z.map(row => row.map(relu));
      acts.push(cur);
    }
    return { acts, zs, P: cur };
  }

  /** 平均交叉熵损失 */
  loss(X, y) {
    const { P } = this.forward(X);
    let s = 0;
    for (let i = 0; i < P.length; i++) {
      s -= Math.log(Math.max(P[i][y[i]], 1e-300));
    }
    return s / P.length;
  }

  /** 反向传播：返回 { dW, db, P }（对平均损失的梯度） */
  backward(X, y) {
    const n = X.length;
    const { acts, zs, P } = this.forward(X);
    // softmax + CE 联合求导：dZ_L = (P − Y)/n
    let delta = P.map((row, i) => row.map((p, j) => (p - (y[i] === j ? 1 : 0)) / n));
    const dW = new Array(this.L);
    const db = new Array(this.L);
    for (let l = this.L - 1; l >= 0; l--) {
      dW[l] = matmul(transpose(delta), acts[l]); // (out×in)
      const gb = zeros(1, delta[0].length)[0];
      for (const row of delta) for (let j = 0; j < row.length; j++) gb[j] += row[j];
      db[l] = gb;
      if (l > 0) {
        const dA = matmul(delta, this.W[l]); // n × sizes[l]
        const Zprev = zs[l - 1];
        delta = dA.map((row, i) => row.map((v, j) => (Zprev[i][j] > 0 ? v : 0)));
      }
    }
    return { dW, db, P };
  }
}

/**
 * 梯度检验：逐参数中心差分 (L(θ+ε)−L(θ−ε))/(2ε) vs 解析梯度。
 * 不变量：maxRelErr ≈ 1e-8（阈值 1e-6）。
 */
export function denseGradientCheck(net, X, y, { eps = 1e-5 } = {}) {
  const { dW, db } = net.backward(X, y);
  let maxErr = 0;
  for (let l = 0; l < net.L; l++) {
    for (let i = 0; i < net.W[l].length; i++) {
      for (let j = 0; j < net.W[l][i].length; j++) {
        const orig = net.W[l][i][j];
        net.W[l][i][j] = orig + eps;
        const lp = net.loss(X, y);
        net.W[l][i][j] = orig - eps;
        const lm = net.loss(X, y);
        net.W[l][i][j] = orig;
        maxErr = Math.max(maxErr, relErr((lp - lm) / (2 * eps), dW[l][i][j]));
      }
    }
    for (let i = 0; i < net.b[l].length; i++) {
      const orig = net.b[l][i];
      net.b[l][i] = orig + eps;
      const lp = net.loss(X, y);
      net.b[l][i] = orig - eps;
      const lm = net.loss(X, y);
      net.b[l][i] = orig;
      maxErr = Math.max(maxErr, relErr((lp - lm) / (2 * eps), db[l][i]));
    }
  }
  return maxErr;
}

/** 普通梯度下降训练，返回损失历史 */
export function trainDense(net, X, y, { lr = 0.5, iters = 300 } = {}) {
  const losses = [];
  for (let it = 0; it <= iters; it++) {
    losses.push(net.loss(X, y));
    if (it === iters) break;
    const { dW, db } = net.backward(X, y);
    for (let l = 0; l < net.L; l++) {
      for (let i = 0; i < net.W[l].length; i++) {
        for (let j = 0; j < net.W[l][i].length; j++) net.W[l][i][j] -= lr * dW[l][i][j];
      }
      for (let i = 0; i < net.b[l].length; i++) net.b[l][i] -= lr * db[l][i];
    }
  }
  return losses;
}

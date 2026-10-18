/**
 * 旗舰AI引擎 · 循环神经网络（Elman RNN）+ 时间反向传播（BPTT）
 * 站在 Werbos 1990 / Williams & Zipser 1995 的经典结论之上：
 *   h_t = tanh(W_xh·x_t + W_hh·h_{t−1} + b_h)，y_t = W_hy·h_t + b_y
 * 不变量：解析梯度（BPTT） ≡ 中心差分数值梯度（maxRelErr < 1e-6）
 */
import { zeros, randomMatrix } from '../core/linalg.js';
import { mulberry32 } from '../core/stats.js';
import { relErr } from '../core/numerical.js';

export class RNN {
  /**
   * @param {{inputSize:number, hiddenSize:number, outputSize:number, seed?:number}} cfg
   */
  constructor({ inputSize, hiddenSize, outputSize, seed = 42 }) {
    const rng = mulberry32(seed);
    const sX = 1 / Math.sqrt(Math.max(1, inputSize));   // Xavier 量级初始化，防 tanh 饱和
    const sH = 1 / Math.sqrt(Math.max(1, hiddenSize));
    this.d = inputSize;
    this.h = hiddenSize;
    this.o = outputSize;
    this.Wxh = randomMatrix(hiddenSize, inputSize, rng, -sX, sX);
    this.Whh = randomMatrix(hiddenSize, hiddenSize, rng, -sH, sH);
    this.Why = randomMatrix(outputSize, hiddenSize, rng, -sH, sH);
    this.bh = new Array(hiddenSize).fill(0);
    this.by = new Array(outputSize).fill(0);
  }

  /**
   * 前向：X 为 [T][d] 序列。
   * @returns {{T:number, X:number[][], Z:number[][], H:number[][], Y:number[][]}}
   *   H[t] = h_t（隐藏态），Y[t] = y_t（输出 logits），Z[t] = 隐藏预激活
   */
  forward(X) {
    const T = X.length;
    const Z = [], H = [], Y = [];
    let hPrev = new Array(this.h).fill(0);
    for (let t = 0; t < T; t++) {
      const z = new Array(this.h);
      for (let i = 0; i < this.h; i++) {
        let s = this.bh[i];
        for (let j = 0; j < this.d; j++) s += this.Wxh[i][j] * X[t][j];
        for (let j = 0; j < this.h; j++) s += this.Whh[i][j] * hPrev[j];
        z[i] = s;
      }
      const ht = z.map(Math.tanh);
      const yt = new Array(this.o);
      for (let i = 0; i < this.o; i++) {
        let s = this.by[i];
        for (let j = 0; j < this.h; j++) s += this.Why[i][j] * ht[j];
        yt[i] = s;
      }
      Z.push(z); H.push(ht); Y.push(yt);
      hPrev = ht;
    }
    return { T, X, Z, H, Y };
  }

  /** MSE 损失 L = ½·Σ_t‖y_t − target_t‖² */
  loss(cache, targets) {
    let L = 0;
    for (let t = 0; t < cache.T; t++) {
      for (let i = 0; i < this.o; i++) L += 0.5 * (cache.Y[t][i] - targets[t][i]) ** 2;
    }
    return L;
  }

  /**
   * 时间反向传播（BPTT）。
   * 递推：dh_t = W_hyᵀ·dY_t + W_hhᵀ·dz_{t+1}，dz_t = dh_t ⊙ (1 − h_t²)
   * @returns {{Wxh:number[][], Whh:number[][], Why:number[][], bh:number[], by:number[]}}
   */
  backward(cache, targets) {
    const { T, X, H } = cache;
    const g = {
      Wxh: zeros(this.h, this.d),
      Whh: zeros(this.h, this.h),
      Why: zeros(this.o, this.h),
      bh: new Array(this.h).fill(0),
      by: new Array(this.o).fill(0),
    };
    let dzNext = new Array(this.h).fill(0); // dz_{t+1}
    for (let t = T - 1; t >= 0; t--) {
      const dY = new Array(this.o);
      for (let i = 0; i < this.o; i++) {
        dY[i] = cache.Y[t][i] - targets[t][i];
        g.by[i] += dY[i];
        for (let j = 0; j < this.h; j++) g.Why[i][j] += dY[i] * H[t][j];
      }
      // dh_t = W_hyᵀ·dY_t + W_hhᵀ·dz_{t+1}
      const dh = new Array(this.h);
      for (let j = 0; j < this.h; j++) {
        let s = 0;
        for (let i = 0; i < this.o; i++) s += this.Why[i][j] * dY[i];
        for (let k = 0; k < this.h; k++) s += this.Whh[k][j] * dzNext[k];
        dh[j] = s;
      }
      const dz = dh.map((v, j) => v * (1 - H[t][j] * H[t][j]));
      const hPrev = t > 0 ? H[t - 1] : new Array(this.h).fill(0);
      for (let j = 0; j < this.h; j++) {
        g.bh[j] += dz[j];
        for (let k = 0; k < this.d; k++) g.Wxh[j][k] += dz[j] * X[t][k];
        for (let k = 0; k < this.h; k++) g.Whh[j][k] += dz[j] * hPrev[k];
      }
      dzNext = dz;
    }
    return g;
  }
}

/**
 * RNN 梯度检验：BPTT 解析梯度 vs 中心差分数值梯度。
 * 双判据（CS231n 规范）：显著分量（|g| ≥ 1e-3·max|g|）的 maxRelErr + 全分量 maxAbsErr。
 * 后者防御浮点抵消：极小分量上中心差分的相对误差被 ulp 噪声主导，并非梯度错误。
 * @returns {{maxRelErr:number, maxAbsErr:number, significant:number}}
 */
export function rnnGradientCheck(net, X, targets, { eps = 1e-5 } = {}) {
  const cache = net.forward(X);
  const g = net.backward(cache, targets);
  const mats = [['Wxh', net.Wxh, g.Wxh], ['Whh', net.Whh, g.Whh], ['Why', net.Why, g.Why]];
  const vecs = [['bh', net.bh, g.bh], ['by', net.by, g.by]];
  const pairs = []; // [analytic, numerical]
  for (const [, P, G] of mats) {
    for (let i = 0; i < P.length; i++) {
      for (let j = 0; j < P[i].length; j++) {
        const orig = P[i][j];
        P[i][j] = orig + eps;
        const lp = net.loss(net.forward(X), targets);
        P[i][j] = orig - eps;
        const lm = net.loss(net.forward(X), targets);
        P[i][j] = orig;
        pairs.push([G[i][j], (lp - lm) / (2 * eps)]);
      }
    }
  }
  for (const [, p, gr] of vecs) {
    for (let i = 0; i < p.length; i++) {
      const orig = p[i];
      p[i] = orig + eps;
      const lp = net.loss(net.forward(X), targets);
      p[i] = orig - eps;
      const lm = net.loss(net.forward(X), targets);
      p[i] = orig;
      pairs.push([gr[i], (lp - lm) / (2 * eps)]);
    }
  }
  const maxAbsAna = Math.max(...pairs.map(([a]) => Math.abs(a)));
  const floor = 1e-3 * maxAbsAna;
  let maxRel = 0, maxAbs = 0, significant = 0;
  for (const [a, b] of pairs) {
    maxAbs = Math.max(maxAbs, Math.abs(a - b));
    if (Math.abs(a) >= floor) {
      significant++;
      maxRel = Math.max(maxRel, relErr(a, b));
    }
  }
  return { maxRelErr: maxRel, maxAbsErr: maxAbs, significant };
}

/**
 * 序列回归训练（MSE，全批量梯度下降）。
 * @returns {{losses:number[]}} 每轮损失
 */
export function trainRNN(net, X, targets, { lr = 0.1, iters = 200 } = {}) {
  const losses = [];
  for (let it = 0; it < iters; it++) {
    const cache = net.forward(X);
    losses.push(net.loss(cache, targets));
    const g = net.backward(cache, targets);
    for (let i = 0; i < net.h; i++) {
      for (let j = 0; j < net.d; j++) net.Wxh[i][j] -= lr * g.Wxh[i][j];
      for (let j = 0; j < net.h; j++) net.Whh[i][j] -= lr * g.Whh[i][j];
      net.bh[i] -= lr * g.bh[i];
    }
    for (let i = 0; i < net.o; i++) {
      for (let j = 0; j < net.h; j++) net.Why[i][j] -= lr * g.Why[i][j];
      net.by[i] -= lr * g.by[i];
    }
  }
  return { losses };
}

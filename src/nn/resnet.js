/**
 * 旗舰AI引擎 · 残差网络（Residual Network）
 * 站在 He et al. 2015《Deep Residual Learning for Image Recognition》之上：
 * 通过恒等快捷连接 y = F(x, {Wᵢ}) + x 消解极深网络的退化（梯度消失/爆炸），
 * 使优化目标退化成「学残差≈0」而非「学恒等映射」，从而数百层仍可训练。
 *
 * 实现选择：以「残差 MLP 块」为载体（分支 = Linear→ReLU→Linear，out = x + 分支），
 * 它完整具备残差机制的全部数学性质，且可全反向传播 + 中心差分梯度检验。
 * 同一套代码通过 useSkip 开关即可退化成「普通堆叠」（无快捷连接）用于对照实验。
 *
 * 不变量：
 *   ① 恒等映射：分支权重零化后 forward(x) ≡ x（跳过连接保底）。
 *   ② 分支零化 ⇒ 单样本雅可比 ∂out/∂x = I（跳过连接是精确恒等）。
 *   ③ 全参数解析梯度 ≡ 中心差分（maxRel < 1e-6）。
 *   ④ 训练收敛：ResNet 在 two-moons 上最终 CE 显著下降且准确率 > 0.9。
 *   ⑤ 梯度高速公路（精确）：深度零化分支网络上，残差回传输入梯度范数 = √d，
 *      普通堆叠回传输入梯度范数 = 0 —— 跳过连接消除梯度消失，这是残差网络的根基。
 */

import { matmul, transpose, zeros, randomMatrix } from '../core/linalg.js';
import { softmax, relu, relErr } from '../core/numerical.js';
import { mulberry32, twoMoons } from '../core/stats.js';

/** He 初始化随机矩阵（均匀 [−1,1] × scale） */
function zeroCentered(rng, m, n, scale) {
  return randomMatrix(m, n, rng, -1, 1).map((row) => row.map((v) => v * scale));
}

/** 每行加偏置 */
function addBias(X, b) {
  return X.map((row) => row.map((v, j) => v + b[j]));
}

/** 矩阵深拷贝 */
function cloneM(A) { return A.map((row) => row.slice()); }

/**
 * 残差块：out = (skip ? x : 0) + F(x)，F = Linear(hidden)→ReLU→Linear(d)。
 */
export class ResidualBlock {
  constructor(inputSize, hiddenSize, { seed = 42, skip = true } = {}) {
    const rng = mulberry32(seed);
    const s1 = Math.sqrt(2 / inputSize);
    const s2 = Math.sqrt(2 / hiddenSize);
    this.d = inputSize;
    this.h = hiddenSize;
    this.skip = skip;
    this.W1 = zeroCentered(rng, inputSize, hiddenSize, s1); // d×h
    this.b1 = new Array(hiddenSize).fill(0);
    this.W2 = zeroCentered(rng, hiddenSize, inputSize, s2); // h×d
    this.b2 = new Array(inputSize).fill(0);
  }

  forward(X) {
    const z1 = addBias(matmul(X, this.W1), this.b1);
    const a1 = z1.map((row) => row.map(relu));
    const z2 = addBias(matmul(a1, this.W2), this.b2);
    const out = this.skip
      ? X.map((row, i) => row.map((v, j) => v + z2[i][j]))
      : z2;
    return { out, cache: { X, z1, a1, skip: this.skip } };
  }

  backward(dout, cache) {
    const { X, z1, a1, skip } = cache;
    const n = X.length, d = this.d, h = this.h;
    const dz2 = dout;
    const dW2 = matmul(transpose(a1), dz2); // h×d
    const db2 = new Array(d).fill(0);
    for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) db2[j] += dz2[i][j];
    const da1 = matmul(dz2, transpose(this.W2)); // n×h
    const dz1 = da1.map((row, i) => row.map((v, j) => (z1[i][j] > 0 ? v : 0)));
    const dW1 = matmul(transpose(X), dz1); // d×h
    const db1 = new Array(h).fill(0);
    for (let i = 0; i < n; i++) for (let j = 0; j < h; j++) db1[j] += dz1[i][j];
    const dxBranch = matmul(dz1, transpose(this.W1)); // n×d
    const dx = dxBranch.map((row, i) => row.map((v, j) => (skip ? v + dout[i][j] : v)));
    return { dx, dW1, db1, dW2, db2 };
  }
}

/**
 * 残差堆叠（ResNet）：numBlocks 个残差块 + 末层线性分类头（softmax）。
 */
export class ResNet {
  constructor({ inputDim, hiddenDim, numBlocks = 2, numClasses = 2, seed = 42, useSkip = true } = {}) {
    this.d = inputDim;
    this.h = hiddenDim;
    this.C = numClasses;
    this.blocks = [];
    for (let b = 0; b < numBlocks; b++) {
      this.blocks.push(new ResidualBlock(inputDim, hiddenDim, { seed: seed + b * 17, skip: useSkip }));
    }
    const rng = mulberry32(seed + 999);
    const s = Math.sqrt(2 / inputDim);
    this.Wout = zeroCentered(rng, numClasses, inputDim, s);
    this.bout = new Array(numClasses).fill(0);
  }

  forward(X) {
    const caches = [];
    let h = X;
    for (const blk of this.blocks) {
      const r = blk.forward(h);
      caches.push(r.cache);
      h = r.out;
    }
    const logits = addBias(matmul(h, transpose(this.Wout)), this.bout);
    const P = logits.map((row) => softmax(row));
    return { P, h, caches, logits };
  }

  loss(X, y) {
    const { P } = this.forward(X);
    let s = 0;
    for (let i = 0; i < P.length; i++) s -= Math.log(Math.max(P[i][y[i]], 1e-300));
    return s / P.length;
  }

  backward(X, y) {
    const { P, h, caches } = this.forward(X);
    const n = X.length;
    const delta = P.map((row, i) => row.map((p, c) => (p - (y[i] === c ? 1 : 0)) / n));
    const dWout = matmul(transpose(delta), h); // C×d：dWout[c][j] = Σ_i δ[i][c]·h[i][j]
    const dbout = new Array(this.C).fill(0);
    for (let i = 0; i < n; i++) for (let c = 0; c < this.C; c++) dbout[c] += delta[i][c];
    let dh = matmul(delta, this.Wout); // n×d
    const grads = { dWout, dbout };
    for (let b = this.blocks.length - 1; b >= 0; b--) {
      const g = this.blocks[b].backward(dh, caches[b]);
      dh = g.dx;
      grads['blk' + b] = g;
    }
    return { P, grads };
  }
}

/** ResNet 准确率 */
export function resnetAccuracy(net, X, y) {
  const { P } = net.forward(X);
  let ok = 0;
  for (let i = 0; i < X.length; i++) {
    let best = 0;
    for (let c = 1; c < net.C; c++) if (P[i][c] > P[i][best]) best = c;
    if (best === y[i]) ok++;
  }
  return ok / X.length;
}

/**
 * SGD 训练 ResNet（全参数）。返回损失历史。
 */
export function trainResNet(net, X, y, { lr = 0.1, iters = 400 } = {}) {
  const losses = [];
  for (let it = 0; it <= iters; it++) {
    losses.push(net.loss(X, y));
    if (it === iters) break;
    const { grads } = net.backward(X, y);
    for (let c = 0; c < net.C; c++) {
      for (let j = 0; j < net.d; j++) net.Wout[c][j] -= lr * grads.dWout[c][j];
      net.bout[c] -= lr * grads.dbout[c];
    }
    for (let b = 0; b < net.blocks.length; b++) {
      const g = grads['blk' + b];
      const blk = net.blocks[b];
      for (let c = 0; c < net.d; c++) for (let a = 0; a < net.h; a++) blk.W1[c][a] -= lr * g.dW1[c][a];
      for (let a = 0; a < net.h; a++) blk.b1[a] -= lr * g.db1[a];
      for (let a = 0; a < net.h; a++) for (let j = 0; j < net.d; j++) blk.W2[a][j] -= lr * g.dW2[a][j];
      for (let a = 0; a < net.d; a++) blk.b2[a] -= lr * g.db2[a];
    }
  }
  return losses;
}

/* ───────────────────────── 不变量辅助 ───────────────────────── */

/** 恒等映射检查：零化分支 ⇒ forward(X) ≡ X（最大逐元素差） */
export function residualIdentityCheck(inputDim, hiddenDim, seed = 42) {
  const blk = new ResidualBlock(inputDim, hiddenDim, { seed, skip: true });
  blk.W1 = blk.W1.map((r) => r.map(() => 0));
  blk.b1 = blk.b1.map(() => 0);
  blk.W2 = blk.W2.map((r) => r.map(() => 0));
  blk.b2 = blk.b2.map(() => 0);
  const rng = mulberry32(7);
  const X = [randomMatrix(1, inputDim, rng, -1, 1)[0]];
  const { out } = blk.forward(X);
  let mx = 0;
  for (let j = 0; j < inputDim; j++) mx = Math.max(mx, Math.abs(out[0][j] - X[0][j]));
  return mx;
}

/** 零化分支 ⇒ 单样本雅可比 ∂out/∂x = I（有限差分），返回最大偏离 */
export function residualJacobianIdentity(inputDim, hiddenDim, seed = 42) {
  const blk = new ResidualBlock(inputDim, hiddenDim, { seed, skip: true });
  blk.W1 = blk.W1.map((r) => r.map(() => 0));
  blk.b1 = blk.b1.map(() => 0);
  blk.W2 = blk.W2.map((r) => r.map(() => 0));
  blk.b2 = blk.b2.map(() => 0);
  const rng = mulberry32(13);
  const x = randomMatrix(1, inputDim, rng, -1, 1)[0];
  const eps = 1e-6;
  const J = zeros(inputDim, inputDim);
  for (let j = 0; j < inputDim; j++) {
    const xp = x.slice(); xp[j] += eps;
    const xm = x.slice(); xm[j] -= eps;
    const op = blk.forward([xp]).out[0];
    const om = blk.forward([xm]).out[0];
    for (let i = 0; i < inputDim; i++) J[i][j] = (op[i] - om[i]) / (2 * eps);
  }
  let mx = 0;
  for (let i = 0; i < inputDim; i++) for (let j = 0; j < inputDim; j++) {
    mx = Math.max(mx, Math.abs(J[i][j] - (i === j ? 1 : 0)));
  }
  return mx;
}

/**
 * ResNet 全参数梯度检验：解析梯度 ≡ 中心差分。返回 { maxRel, maxAbs }。
 */
export function residualGradientCheck(net, X, y, { eps = 1e-5 } = {}) {
  const { grads } = net.backward(X, y);
  const loss = () => net.loss(X, y);
  let maxRel = 0, maxAbs = 0;
  // 末层
  for (let c = 0; c < net.C; c++) {
    for (let j = 0; j < net.d; j++) {
      const o = net.Wout[c][j];
      net.Wout[c][j] = o + eps; const lp = loss();
      net.Wout[c][j] = o - eps; const lm = loss();
      net.Wout[c][j] = o;
      const g = (lp - lm) / (2 * eps);
      maxRel = Math.max(maxRel, relErr(g, grads.dWout[c][j]));
      maxAbs = Math.max(maxAbs, Math.abs(g - grads.dWout[c][j]));
    }
    const ob = net.bout[c];
    net.bout[c] = ob + eps; const lpb = loss();
    net.bout[c] = ob - eps; const lmb = loss();
    net.bout[c] = ob;
    const gb = (lpb - lmb) / (2 * eps);
    maxRel = Math.max(maxRel, relErr(gb, grads.dbout[c]));
    maxAbs = Math.max(maxAbs, Math.abs(gb - grads.dbout[c]));
  }
  for (let b = 0; b < net.blocks.length; b++) {
    const blk = net.blocks[b];
    const g = grads['blk' + b];
    for (let c = 0; c < net.d; c++) for (let a = 0; a < net.h; a++) {
      const o = blk.W1[c][a];
      blk.W1[c][a] = o + eps; const lp = loss();
      blk.W1[c][a] = o - eps; const lm = loss();
      blk.W1[c][a] = o;
      const gg = (lp - lm) / (2 * eps);
      maxRel = Math.max(maxRel, relErr(gg, g.dW1[c][a]));
      maxAbs = Math.max(maxAbs, Math.abs(gg - g.dW1[c][a]));
    }
    for (let a = 0; a < net.h; a++) {
      const o = blk.b1[a];
      blk.b1[a] = o + eps; const lp = loss();
      blk.b1[a] = o - eps; const lm = loss();
      blk.b1[a] = o;
      const gg = (lp - lm) / (2 * eps);
      maxRel = Math.max(maxRel, relErr(gg, g.db1[a]));
      maxAbs = Math.max(maxAbs, Math.abs(gg - g.db1[a]));
    }
    for (let a = 0; a < net.h; a++) for (let j = 0; j < net.d; j++) {
      const o = blk.W2[a][j];
      blk.W2[a][j] = o + eps; const lp = loss();
      blk.W2[a][j] = o - eps; const lm = loss();
      blk.W2[a][j] = o;
      const gg = (lp - lm) / (2 * eps);
      maxRel = Math.max(maxRel, relErr(gg, g.dW2[a][j]));
      maxAbs = Math.max(maxAbs, Math.abs(gg - g.dW2[a][j]));
    }
    for (let a = 0; a < net.d; a++) {
      const o = blk.b2[a];
      blk.b2[a] = o + eps; const lp = loss();
      blk.b2[a] = o - eps; const lm = loss();
      blk.b2[a] = o;
      const gg = (lp - lm) / (2 * eps);
      maxRel = Math.max(maxRel, relErr(gg, g.db2[a]));
      maxAbs = Math.max(maxAbs, Math.abs(gg - g.db2[a]));
    }
  }
  return { maxRel, maxAbs };
}

/**
 * 梯度高速公路（精确）：深度零化分支网络上，给定上游梯度 dout = 1，
 * 残差回传的输入梯度范数 = √d（跳过连接保底），普通堆叠 = 0（梯度消失）。
 * @returns {{residualGradNorm:number, plainGradNorm:number, d:number}}
 */
export function residualGradientHighway({ inputDim, hiddenDim = 16, numBlocks = 8, seed = 42, dout = null } = {}) {
  const up = dout || new Array(inputDim).fill(1);
  const makeNet = (useSkip) => {
    const net = new ResNet({ inputDim, hiddenDim, numBlocks, numClasses: inputDim, seed, useSkip });
    // 零化所有分支，仅保留快捷连接（残差）或退化映射（普通）
    for (const blk of net.blocks) {
      blk.W1 = blk.W1.map((r) => r.map(() => 0));
      blk.b1 = blk.b1.map(() => 0);
      blk.W2 = blk.W2.map((r) => r.map(() => 0));
      blk.b2 = blk.b2.map(() => 0);
    }
    net.Wout = net.Wout.map((r) => r.map(() => 0));
    net.bout = net.bout.map(() => 0);
    return net;
  };
  const computeInputGrad = (net, X) => {
    // 一次前向 + 反向，取首样本输入梯度
    const { caches } = net.forward(X);
    const n = X.length;
    // 末层零化 ⇒ dh = dout·Wout = 0；故直接以 up 作为首块上游
    let dh = X.map((row) => up.slice());
    let dhAccum = zeros(n, inputDim);
    for (let b = 0; b < net.blocks.length; b++) {
      const g = net.blocks[b].backward(dh, caches[b]);
      dh = g.dx;
    }
    // dh 即输入梯度（n×d）
    let s = 0;
    for (const row of dh) for (const v of row) s += v * v;
    return Math.sqrt(s);
  };
  const residual = makeNet(true);
  const plain = makeNet(false);
  const rng = mulberry32(seed);
  const X = [randomMatrix(1, inputDim, rng, -1, 1)[0]];
  const residualGradNorm = computeInputGrad(residual, X);
  const plainGradNorm = computeInputGrad(plain, X);
  return { residualGradNorm, plainGradNorm, d: inputDim };
}

/**
 * 残差 vs 普通堆叠（同种子同超参训练对比）。
 * 返回 { residual:{loss,acc}, plain:{loss,acc} }。
 */
export function residualVsPlain({
  inputDim, hiddenDim = 12, numBlocks = 4, numClasses = 2,
  X, y, lr = 0.12, iters = 500, seed = 42,
} = {}) {
  const residual = new ResNet({ inputDim, hiddenDim, numBlocks, numClasses, seed, useSkip: true });
  const plain = new ResNet({ inputDim, hiddenDim, numBlocks, numClasses, seed, useSkip: false });
  const rLoss = trainResNet(residual, X, y, { lr, iters });
  const pLoss = trainResNet(plain, X, y, { lr, iters });
  return {
    residual: { loss: rLoss[rLoss.length - 1], acc: resnetAccuracy(residual, X, y) },
    plain: { loss: pLoss[pLoss.length - 1], acc: resnetAccuracy(plain, X, y) },
  };
}

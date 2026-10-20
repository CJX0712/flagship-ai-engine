/**
 * 旗舰AI引擎 · 门控循环单元（GRU, Cho et al. 2014）
 *   r_t = σ(W_xr·x_t + W_hr·h_{t−1} + b_r)              重置门
 *   z_t = σ(W_xz·x_t + W_hz·h_{t−1} + b_z)              更新门
 *   h̃_t = tanh(W_xh·x_t + r_t ⊙ (W_hh·h_{t−1}) + b_h)  候选隐态
 *   h_t = (1 − z_t) ⊙ h_{t−1} + z_t ⊙ h̃_t             隐态（z 控制记忆留存）
 *   y_t = W_hy·h_t + b_y
 * 站在：Cho, Van Merriënboer, Bahdanau & Bengio 2014《Learning Phrase Representations
 *       using RNN Encoder–Decoder》（GRU 原始论文）· Chung et al. 2014《Empirical
 *       Evaluation of Gated Recurrent Neural Networks》（GRU vs LSTM）·
 *       Sutton & Barto《Reinforcement Learning》— 同前作 LSTM 的 BPTT 梯度检验范式。
 *
 * 不变量：
 *   ① BPTT 解析梯度 ≡ 中心差分数值梯度（双判据：显著分量 maxRelErr < 1e-6）；
 *   ② 门控闭合恒等式：z_j≡1 ⇒ h_t=h̃_t；z_j≡0 ⇒ h_t=h_{t−1}（精确 0 偏差）；
 *   ③ 单步闭合式：给定权重，手算 h_t 与 forward 首步逐位一致（<1e-12）；
 *   ④ 重置门隔离：r≡0 时候选 h̃ 与 h_{t−1} 无关（改 h_{t−1} 不改 h̃）；
 *   ⑤ 训练收敛：教师信号（同网络前向目标）下 MSE 单调不增并趋零；
 *   ⑥ 同 seed 初始化一致：两次构造权重逐位相同。
 */

import { zeros, randomMatrix, matvec, dot } from '../core/linalg.js';
import { mulberry32 } from '../core/stats.js';
import { relErr, sigmoid } from '../core/numerical.js';

export class GRU {
  /**
   * @param {{inputSize:number, hiddenSize:number, outputSize:number, seed?:number}} cfg
   */
  constructor({ inputSize, hiddenSize, outputSize, seed = 42 }) {
    const rng = mulberry32(seed);
    const sX = 1 / Math.sqrt(Math.max(1, inputSize));
    const sH = 1 / Math.sqrt(Math.max(1, hiddenSize));
    this.d = inputSize;
    this.h = hiddenSize;
    this.o = outputSize;
    const g = (name, rows, cols, s) => { this[name] = randomMatrix(rows, cols, rng, -s, s); };
    g('Wxr', hiddenSize, inputSize, sX); g('Wxz', hiddenSize, inputSize, sX);
    g('Wxh', hiddenSize, inputSize, sX);
    g('Whr', hiddenSize, hiddenSize, sH); g('Whz', hiddenSize, hiddenSize, sH);
    g('Whh', hiddenSize, hiddenSize, sH);
    this.br = new Array(hiddenSize).fill(0);
    this.bz = new Array(hiddenSize).fill(0);
    this.bh = new Array(hiddenSize).fill(0);
    this.Why = randomMatrix(outputSize, hiddenSize, rng, -sH, sH);
    this.by = new Array(outputSize).fill(0);
  }

  /**
   * 前向：X 为 [T][d]。返回门控/隐态/输出按时间步存。
   * @returns {{T:number, X:number[][], R:number[][], Z:number[][], HT:number[][],
   *           H:number[][], Y:number[][]}}
   */
  forward(X) {
    const T = X.length;
    const R = [], Z = [], HT = [], H = [], Y = [];
    let hPrev = new Array(this.h).fill(0);
    for (let t = 0; t < T; t++) {
      const rT = [], zT = [], htT = [], hT = [];
      for (let j = 0; j < this.h; j++) {
        let sr = this.br[j], sz = this.bz[j], sh = this.bh[j];
        for (let k = 0; k < this.d; k++) {
          sr += this.Wxr[j][k] * X[t][k];
          sz += this.Wxz[j][k] * X[t][k];
          sh += this.Wxh[j][k] * X[t][k];
        }
        let hr = 0, hz = 0;
        for (let k = 0; k < this.h; k++) { hr += this.Whr[j][k] * hPrev[k]; hz += this.Whz[j][k] * hPrev[k]; }
        sr += hr; sz += hz; // 门 r/z 各自经 hPrev 循环项（W_hr vs W_hz）
        const rv = sigmoid(sr), zv = sigmoid(sz);
        let hh = 0;
        for (let k = 0; k < this.h; k++) hh += this.Whh[j][k] * hPrev[k];
        sh += rv * hh; // 重置门调制循环分量
        const htv = Math.tanh(sh);
        rT.push(rv); zT.push(zv); htT.push(htv);
        hT.push((1 - zv) * hPrev[j] + zv * htv);
      }
      const yT = new Array(this.o);
      for (let i = 0; i < this.o; i++) yT[i] = this.by[i] + matvec(this.Why, hT)[i];
      R.push(rT); Z.push(zT); HT.push(htT); H.push(hT); Y.push(yT);
      hPrev = hT;
    }
    return { T, X, R, Z, HT, H, Y };
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
   * GRU 时间反向传播（BPTT）。
   * 关键路径：h_t = (1−z)h_{t−1} + z·h̃；候选 h̃ 经 tanh 出，且 h̃ 通过重置门 r 与循环项 r⊙(W_hh h_{t−1}) 入。
   * 推导（逐 j）：
   *   dL/dh̃_j = dh_j·z_j；  dL/dz_j = dh_j·(h̃_j − h_{t−1,j})
   *   dL/dsh_j = (dL/dh̃_j)·(1−h̃_j²)；  dL/dsr_j = (dL/dr_j)·r_j(1−r_j)，其中 dL/dr_j = dL/dsh_j·H_j
   *   （H_j = Σ_m W_hh[j][m]·h_{t−1,m}，候选循环分量）
   * 向 t−1 传播（dhNext_j）：直接项 (1−z_j) + 经 z 的 W_hz + 经 r/sh 的 [W_hr·r_j(1−r_j)·H_j + r_j·W_hh] 分量。
   * @returns {object} 全部参数梯度
   */
  backward(cache, targets) {
    const { T, X, R, Z, HT, H } = cache;
    const g = {};
    for (const n of ['Wxr', 'Wxz', 'Wxh']) g[n] = zeros(this.h, this.d);
    for (const n of ['Whr', 'Whz', 'Whh']) g[n] = zeros(this.h, this.h);
    for (const n of ['br', 'bz', 'bh']) g[n] = new Array(this.h).fill(0);
    g.Why = zeros(this.o, this.h);
    g.by = new Array(this.o).fill(0);

    let dhNext = new Array(this.h).fill(0);
    for (let t = T - 1; t >= 0; t--) {
      const dY = new Array(this.o);
      for (let i = 0; i < this.o; i++) {
        dY[i] = cache.Y[t][i] - targets[t][i];
        g.by[i] += dY[i];
        for (let j = 0; j < this.h; j++) g.Why[i][j] += dY[i] * H[t][j];
      }
      const hPrev = t > 0 ? H[t - 1] : new Array(this.h).fill(0);
      const dh = new Array(this.h), dHT = new Array(this.h), dZ = new Array(this.h), dSH = new Array(this.h), dhNextNext = new Array(this.h).fill(0);
      for (let j = 0; j < this.h; j++) {
        dh[j] = 0;
        for (let i = 0; i < this.o; i++) dh[j] += this.Why[i][j] * dY[i];
        dh[j] += dhNext[j];
        dHT[j] = dh[j] * Z[t][j];                       // dL/dh̃_j
        dZ[j] = dh[j] * (HT[t][j] - hPrev[j]);           // dL/dz_j
        dSH[j] = dHT[j] * (1 - HT[t][j] * HT[t][j]);     // dL/dsh_j
      }
      // 候选循环分量 H_j = Σ_m W_hh[j][m]·hPrev[m]
      const Hj = new Array(this.h);
      for (let j = 0; j < this.h; j++) {
        let s = 0;
        for (let m = 0; m < this.h; m++) s += this.Whh[j][m] * hPrev[m];
        Hj[j] = s;
      }
      const dSR = new Array(this.h);
      for (let j = 0; j < this.h; j++) {
        const dRj = dSH[j] * Hj[j];                      // dL/dr_j
        dSR[j] = dRj * R[t][j] * (1 - R[t][j]);          // dL/dsr_j
        // 参数梯度（门 r）
        g.br[j] += dSR[j];
        for (let k = 0; k < this.d; k++) g.Wxr[j][k] += dSR[j] * X[t][k];
        for (let m = 0; m < this.h; m++) g.Whr[j][m] += dSR[j] * hPrev[m];
        // 参数梯度（门 z）
        const dzP = dZ[j] * Z[t][j] * (1 - Z[t][j]);
        g.bz[j] += dzP;
        for (let k = 0; k < this.d; k++) g.Wxz[j][k] += dzP * X[t][k];
        for (let m = 0; m < this.h; m++) g.Whz[j][m] += dzP * hPrev[m];
        // 参数梯度（候选 h̃）
        g.bh[j] += dSH[j];
        for (let k = 0; k < this.d; k++) g.Wxh[j][k] += dSH[j] * X[t][k];
        for (let m = 0; m < this.h; m++) g.Whh[j][m] += dSH[j] * R[t][j] * hPrev[m];
        // 向 t−1 传播
        for (let m = 0; m < this.h; m++) {
          if (m === j) dhNextNext[m] += dh[j] * (1 - Z[t][j]);                      // 直接项（仅 h_{t−1}[j]）
          dhNextNext[m] += dzP * this.Whz[j][m];                                    // 经 z
          dhNextNext[m] += dSH[j] * (this.Whr[j][m] * R[t][j] * (1 - R[t][j]) * Hj[j] + R[t][j] * this.Whh[j][m]); // 经 r/sh
        }
      }
      dhNext = dhNextNext;
    }
    return g;
  }
}

/**
 * GRU 梯度检验：BPTT 解析梯度 vs 中心差分数值梯度（双判据 CS231n）。
 * @returns {{maxRelErr:number, maxAbsErr:number, significant:number}}
 */
export function gruGradientCheck(net, X, targets, { eps = 1e-5 } = {}) {
    const cache = net.forward(X);
    const g = net.backward(cache, targets);
    const pairs = [];
    const probeM = (P, G) => {
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
    };
    const probeV = (p, gr) => {
      for (let i = 0; i < p.length; i++) {
        const orig = p[i];
        p[i] = orig + eps;
        const lp = net.loss(net.forward(X), targets);
        p[i] = orig - eps;
        const lm = net.loss(net.forward(X), targets);
        p[i] = orig;
        pairs.push([gr[i], (lp - lm) / (2 * eps)]);
      }
    };
    for (const n of ['Wxr', 'Wxz', 'Wxh', 'Whr', 'Whz', 'Whh', 'Why']) probeM(net[n], g[n]);
    for (const n of ['br', 'bz', 'bh', 'by']) probeV(net[n], g[n]);
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
 * 序列回归训练（MSE，全批量梯度下降）。返回 { losses }。
 * @returns {{losses:number[]}}
 */
export function trainGRU(net, X, targets, { lr = 0.05, iters = 200 } = {}) {
    const mats = ['Wxr', 'Wxz', 'Wxh', 'Whr', 'Whz', 'Whh', 'Why'];
    const vecs = ['br', 'bz', 'bh', 'by'];
    const losses = [];
    for (let it = 0; it < iters; it++) {
      const cache = net.forward(X);
      losses.push(net.loss(cache, targets));
      const g = net.backward(cache, targets);
      for (const n of mats) {
        const P = net[n], G = g[n];
        for (let i = 0; i < P.length; i++) for (let j = 0; j < P[i].length; j++) P[i][j] -= lr * G[i][j];
      }
      for (const n of vecs) {
        const p = net[n], gr = g[n];
        for (let i = 0; i < p.length; i++) p[i] -= lr * gr[i];
      }
    }
    return { losses };
  }

/**
 * 在给定权重下手工计算单步隐态（闭式），用于不变量 ②③ 的交叉验证。
 * 直接复现 forward 的逐 j 公式，避免复用缓存。
 */
export function gruStepClosed(net, x, hPrev) {
    const h = net.h;
    const rT = [], zT = [], htT = [], hT = [];
    for (let j = 0; j < h; j++) {
      let sr = net.br[j], sz = net.bz[j], sh = net.bh[j];
      for (let k = 0; k < net.d; k++) { sr += net.Wxr[j][k] * x[k]; sz += net.Wxz[j][k] * x[k]; sh += net.Wxh[j][k] * x[k]; }
      let hr = 0, hz = 0; for (let k = 0; k < h; k++) { hr += net.Whr[j][k] * hPrev[k]; hz += net.Whz[j][k] * hPrev[k]; }
      sr += hr; sz += hz;
      const rv = sigmoid(sr), zv = sigmoid(sz);
      let hh = 0; for (let k = 0; k < h; k++) hh += net.Whh[j][k] * hPrev[k];
      sh += rv * hh;
      rT.push(rv); zT.push(zv); htT.push(Math.tanh(sh));
      hT.push((1 - zv) * hPrev[j] + zv * Math.tanh(sh));
    }
    return { rT, zT, htT, hT };
  }

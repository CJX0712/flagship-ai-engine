/**
 * 旗舰AI引擎 · 长短期记忆网络（LSTM, Hochreiter & Schmidhuber 1997）
 *   i_t = σ(W_xi·x_t + W_hi·h_{t−1} + b_i)      输入门
 *   f_t = σ(W_xf·x_t + W_hf·h_{t−1} + b_f)      遗忘门
 *   g_t = tanh(W_xg·x_t + W_hg·h_{t−1} + b_g)   候选细胞态
 *   o_t = σ(W_xo·x_t + W_ho·h_{t−1} + b_o)      输出门
 *   c_t = f_t ⊙ c_{t−1} + i_t ⊙ g_t            细胞态（线性记忆通道）
 *   h_t = o_t ⊙ tanh(c_t)
 *   y_t = W_hy·h_t + b_y
 * 不变量：BPTT 解析梯度 ≡ 中心差分数值梯度（maxRelErr < 1e-6）
 */
import { zeros, randomMatrix } from '../core/linalg.js';
import { mulberry32 } from '../core/stats.js';
import { relErr, sigmoid } from '../core/numerical.js';

export class LSTM {
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
    g('Wxi', hiddenSize, inputSize, sX); g('Wxf', hiddenSize, inputSize, sX);
    g('Wxg', hiddenSize, inputSize, sX); g('Wxo', hiddenSize, inputSize, sX);
    g('Whi', hiddenSize, hiddenSize, sH); g('Whf', hiddenSize, hiddenSize, sH);
    g('Whg', hiddenSize, hiddenSize, sH); g('Who', hiddenSize, hiddenSize, sH);
    this.bi = new Array(hiddenSize).fill(0);
    this.bf = new Array(hiddenSize).fill(0);
    this.bg = new Array(hiddenSize).fill(0);
    this.bo = new Array(hiddenSize).fill(0);
    this.Why = randomMatrix(outputSize, hiddenSize, rng, -sH, sH);
    this.by = new Array(outputSize).fill(0);
  }

  /**
   * 前向：X 为 [T][d]。
   * @returns {{T:number, X:number[][], I:number[][], F:number[][], G:number[][], O:number[][],
   *           C:number[][], H:number[][], Y:number[][]}} 门控/细胞态/输出均按时间步存
   */
  forward(X) {
    const T = X.length;
    const I = [], F = [], G = [], O = [], C = [], H = [], Y = [];
    let hPrev = new Array(this.h).fill(0);
    let cPrev = new Array(this.h).fill(0);
    for (let t = 0; t < T; t++) {
      const iT = [], fT = [], gT = [], oT = [], cT = [], hT = [];
      for (let j = 0; j < this.h; j++) {
        let si = this.bi[j], sf = this.bf[j], sg = this.bg[j], so = this.bo[j];
        for (let k = 0; k < this.d; k++) {
          si += this.Wxi[j][k] * X[t][k];
          sf += this.Wxf[j][k] * X[t][k];
          sg += this.Wxg[j][k] * X[t][k];
          so += this.Wxo[j][k] * X[t][k];
        }
        for (let k = 0; k < this.h; k++) {
          si += this.Whi[j][k] * hPrev[k];
          sf += this.Whf[j][k] * hPrev[k];
          sg += this.Whg[j][k] * hPrev[k];
          so += this.Who[j][k] * hPrev[k];
        }
        const iv = sigmoid(si), fv = sigmoid(sf), gv = Math.tanh(sg), ov = sigmoid(so);
        const cv = fv * cPrev[j] + iv * gv;
        iT.push(iv); fT.push(fv); gT.push(gv); oT.push(ov);
        cT.push(cv);
        hT.push(ov * Math.tanh(cv));
      }
      const yT = new Array(this.o);
      for (let i = 0; i < this.o; i++) {
        let s = this.by[i];
        for (let j = 0; j < this.h; j++) s += this.Why[i][j] * hT[j];
        yT[i] = s;
      }
      I.push(iT); F.push(fT); G.push(gT); O.push(oT);
      C.push(cT); H.push(hT); Y.push(yT);
      hPrev = hT; cPrev = cT;
    }
    return { T, X, I, F, G, O, C, H, Y };
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
   * LSTM 时间反向传播。
   * 关键路径：h_t 经 o_t 出；c_t 经 f_t 线性回流（梯度高速公路，不衰减）。
   * @returns {object} 全部参数梯度（键名与参数一致）
   */
  backward(cache, targets) {
    const { T, X, I, F, G, O, C, H } = cache;
    const g = {};
    for (const n of ['Wxi', 'Wxf', 'Wxg', 'Wxo']) g[n] = zeros(this.h, this.d);
    for (const n of ['Whi', 'Whf', 'Whg', 'Who']) g[n] = zeros(this.h, this.h);
    for (const n of ['bi', 'bf', 'bg', 'bo']) g[n] = new Array(this.h).fill(0);
    g.Why = zeros(this.o, this.h);
    g.by = new Array(this.o).fill(0);

    let dhNext = new Array(this.h).fill(0);
    let dcNext = new Array(this.h).fill(0);
    for (let t = T - 1; t >= 0; t--) {
      const dY = new Array(this.o);
      for (let i = 0; i < this.o; i++) {
        dY[i] = cache.Y[t][i] - targets[t][i];
        g.by[i] += dY[i];
        for (let j = 0; j < this.h; j++) g.Why[i][j] += dY[i] * H[t][j];
      }
      const cPrev = t > 0 ? C[t - 1] : new Array(this.h).fill(0);
      const hPrev = t > 0 ? H[t - 1] : new Array(this.h).fill(0);
      const dh = new Array(this.h), dc = new Array(this.h);
      for (let j = 0; j < this.h; j++) {
        dh[j] = 0;
        for (let i = 0; i < this.o; i++) dh[j] += this.Why[i][j] * dY[i];
        dh[j] += dhNext[j];
        const tanhC = Math.tanh(C[t][j]);
        dc[j] = dcNext[j] + dh[j] * O[t][j] * (1 - tanhC * tanhC);
      }
      // 门控梯度（链式法则穿过激活函数）
      const dI = [], dF = [], dG = [], dO = [];
      for (let j = 0; j < this.h; j++) {
        dO.push(dh[j] * Math.tanh(C[t][j]));
        dF.push(dc[j] * cPrev[j]);
        dI.push(dc[j] * G[t][j]);
        dG.push(dc[j] * I[t][j]);
      }
      const diP = dI.map((v, j) => v * I[t][j] * (1 - I[t][j]));
      const dfP = dF.map((v, j) => v * F[t][j] * (1 - F[t][j]));
      const dgP = dG.map((v, j) => v * (1 - G[t][j] * G[t][j]));
      const doP = dO.map((v, j) => v * O[t][j] * (1 - O[t][j]));
      for (let j = 0; j < this.h; j++) {
        g.bi[j] += diP[j]; g.bf[j] += dfP[j]; g.bg[j] += dgP[j]; g.bo[j] += doP[j];
        for (let k = 0; k < this.d; k++) {
          g.Wxi[j][k] += diP[j] * X[t][k];
          g.Wxf[j][k] += dfP[j] * X[t][k];
          g.Wxg[j][k] += dgP[j] * X[t][k];
          g.Wxo[j][k] += doP[j] * X[t][k];
        }
        for (let k = 0; k < this.h; k++) {
          g.Whi[j][k] += diP[j] * hPrev[k];
          g.Whf[j][k] += dfP[j] * hPrev[k];
          g.Whg[j][k] += dgP[j] * hPrev[k];
          g.Who[j][k] += doP[j] * hPrev[k];
        }
      }
      // 向 t−1 传播：h_{t−1} 经四个门入；c_{t−1} 经遗忘门入
      for (let j = 0; j < this.h; j++) {
        let s = 0;
        for (let k = 0; k < this.h; k++) {
          s += this.Whi[k][j] * diP[k] + this.Whf[k][j] * dfP[k]
             + this.Whg[k][j] * dgP[k] + this.Who[k][j] * doP[k];
        }
        dhNext[j] = s;
        dcNext[j] = dc[j] * F[t][j];
      }
    }
    return g;
  }
}

/**
 * LSTM 梯度检验：BPTT 解析梯度 vs 中心差分数值梯度。
 * 双判据（CS231n 规范）：显著分量（|g| ≥ 1e-3·max|g|）的 maxRelErr + 全分量 maxAbsErr。
 * @returns {{maxRelErr:number, maxAbsErr:number, significant:number}}
 */
export function lstmGradientCheck(net, X, targets, { eps = 1e-5 } = {}) {
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
  for (const n of ['Wxi', 'Wxf', 'Wxg', 'Wxo', 'Whi', 'Whf', 'Whg', 'Who', 'Why']) probeM(net[n], g[n]);
  for (const n of ['bi', 'bf', 'bg', 'bo', 'by']) probeV(net[n], g[n]);
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
 * @returns {{losses:number[]}}
 */
export function trainLSTM(net, X, targets, { lr = 0.05, iters = 200 } = {}) {
  const mats = ['Wxi', 'Wxf', 'Wxg', 'Wxo', 'Whi', 'Whf', 'Whg', 'Who', 'Why'];
  const vecs = ['bi', 'bf', 'bg', 'bo', 'by'];
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

/**
 * 旗舰AI引擎 · 卷积神经网络（CNN）
 * 站在：LeCun et al. 1998《Gradient-Based Learning Applied to Document Recognition》
 *       Chellapilla et al. 2006 / Jia et al. 2014（im2col + GEMM 高性能卷积）
 *       Goodfellow et al. 2016《Deep Learning》Ch.9（卷积的等变性与参数共享）
 * 不变量：
 *   ① 输出尺寸闭式 oH = ⌊(H + 2p − k)/s⌋ + 1（精确整数恒等）
 *   ② im2col+GEMM 实现 ≡ 直接滑动窗口求和（双实现互证）
 *   ③ 平移等变性：conv(shift(X)) = shift(conv(X))（卷积层的结构性保证）
 *   ④ 反向传播解析梯度 ≡ 中心差分（dX / dW / db 三路）
 *   ⑤ 最大池化梯度路由：仅最大值位置回传，其余为 0
 * 张量约定：X 为 [C][H][W]（通道优先），核 W 为 [OC][C][kH][kW]，偏置 b 为 [OC]。
 */

import { zeros } from '../core/linalg.js';
import { relErr } from '../core/numerical.js';

/** 卷积输出尺寸闭式：⌊(n + 2p − k)/s⌋ + 1 */
export function convOutputSize(n, kernel, stride = 1, pad = 0) {
  return Math.floor((n + 2 * pad - kernel) / stride) + 1;
}

/**
 * im2col：把 [C][H][W] 输入展开为 (oH·oW) × (C·kH·kW) 的补丁矩阵。
 * 这是把卷积化为一次 GEMM 的标准技巧（Jia et al. 2014，Caffe 同款）。
 * @returns {{cols:number[][], oH:number, oW:number, C:number, H:number, W:number, kH:number, kW:number, stride:number, pad:number}}
 */
export function im2col(X, kH, kW, { stride = 1, pad = 0 } = {}) {
  const C = X.length, H = X[0].length, W0 = X[0][0].length;
  const oH = convOutputSize(H, kH, stride, pad);
  const oW = convOutputSize(W0, kW, stride, pad);
  const cols = zeros(oH * oW, C * kH * kW);
  for (let oh = 0; oh < oH; oh++) {
    for (let ow = 0; ow < oW; ow++) {
      const row = oh * oW + ow;
      let col = 0;
      for (let c = 0; c < C; c++) {
        for (let i = 0; i < kH; i++) {
          const r = oh * stride - pad + i;
          for (let j = 0; j < kW; j++) {
            const cc = ow * stride - pad + j;
            const inside = r >= 0 && r < H && cc >= 0 && cc < W0;
            cols[row][col++] = inside ? X[c][r][cc] : 0; // 零填充
          }
        }
      }
    }
  }
  return { cols, oH, oW, C, H, W: W0, kH, kW, stride, pad };
}

/**
 * 卷积前向：out[oc][oh][ow] = Σ_c Σ_i Σ_j W[oc][c][i][j]·X_patch + b[oc]
 * @returns {{out:number[][][], cache:object}}
 */
export function conv2dForward(X, W, b, { stride = 1, pad = 0 } = {}) {
  const OC = W.length, C = W[0].length, kH = W[0][0].length, kW = W[0][0][0].length;
  const geo = im2col(X, kH, kW, { stride, pad });
  const { cols, oH, oW } = geo;
  const out = [];
  for (let oc = 0; oc < OC; oc++) {
    const plane = zeros(oH, oW);
    // 核展平为向量：Wflat[c*kH*kW + i*kW + j]
    const wflat = new Array(C * kH * kW);
    let t = 0;
    for (let c = 0; c < C; c++) for (let i = 0; i < kH; i++) for (let j = 0; j < kW; j++) wflat[t++] = W[oc][c][i][j];
    for (let row = 0; row < cols.length; row++) {
      let s = b[oc];
      const line = cols[row];
      for (let q = 0; q < wflat.length; q++) s += line[q] * wflat[q];
      plane[Math.floor(row / oW)][row % oW] = s;
    }
    out.push(plane);
  }
  return { out, cache: { X, W, b, geo } };
}

/**
 * 卷积反向：dW / db 由 dOut 与输入补丁相关；dX 通过核的转置散射回原位。
 * @returns {{dX:number[][][], dW:number[][][][], db:number[]}}
 */
export function conv2dBackward(cache, dOut) {
  const { X, W, geo } = cache;
  const { cols, oH, oW, C, H, W: W0, kH, kW, stride, pad } = geo;
  const OC = W.length;
  const dW = [];
  const db = new Array(OC).fill(0);
  const dCols = zeros(cols.length, C * kH * kW);

  for (let oc = 0; oc < OC; oc++) {
    const dWoc = [];
    for (let c = 0; c < C; c++) {
      const plane = zeros(kH, kW);
      dWoc.push(plane);
    }
    for (let row = 0; row < cols.length; row++) {
      const oh = Math.floor(row / oW), ow = row % oW;
      const g = dOut[oc][oh][ow];
      db[oc] += g;
      let q = 0;
      for (let c = 0; c < C; c++) {
        for (let i = 0; i < kH; i++) {
          for (let j = 0; j < kW; j++, q++) {
            dWoc[c][i][j] += g * cols[row][q];
            dCols[row][q] += g * W[oc][c][i][j];
          }
        }
      }
    }
    dW.push(dWoc);
  }

  // dCols → dX（col2im 散射累加）
  const dX = [];
  for (let c = 0; c < C; c++) dX.push(zeros(H, W0));
  for (let row = 0; row < dCols.length; row++) {
    const oh = Math.floor(row / oW), ow = row % oW;
    let q = 0;
    for (let c = 0; c < C; c++) {
      for (let i = 0; i < kH; i++) {
        const r = oh * stride - pad + i;
        for (let j = 0; j < kW; j++, q++) {
          const cc = ow * stride - pad + j;
          if (r >= 0 && r < H && cc >= 0 && cc < W0) dX[c][r][cc] += dCols[row][q];
        }
      }
    }
  }
  return { dX, dW, db };
}

/**
 * 最大池化：返回输出与 argmax 路由表（用于反向精确回传）。
 * @returns {{out:number[][][], argmax:number[][][], size:number, stride:number, C:number, H:number, W:number}}
 */
export function maxPool2d(X, size, stride = size) {
  const C = X.length, H = X[0].length, W0 = X[0][0].length;
  const oH = Math.floor((H - size) / stride) + 1;
  const oW = Math.floor((W0 - size) / stride) + 1;
  const out = [], argmax = [];
  for (let c = 0; c < C; c++) {
    const po = zeros(oH, oW), pa = zeros(oH, oW);
    for (let oh = 0; oh < oH; oh++) {
      for (let ow = 0; ow < oW; ow++) {
        let best = -Infinity, bi = -1;
        for (let i = 0; i < size; i++) {
          for (let j = 0; j < size; j++) {
            const r = oh * stride + i, cc = ow * stride + j;
            if (X[c][r][cc] > best) { best = X[c][r][cc]; bi = r * W0 + cc; }
          }
        }
        po[oh][ow] = best;
        pa[oh][ow] = bi;
      }
    }
    out.push(po); argmax.push(pa);
  }
  return { out, argmax, size, stride, C, H, W: W0 };
}

/** 最大池化反向：梯度只经 argmax 位置回传，其余为 0 */
export function maxPoolBackward(poolCache, dOut) {
  const { argmax, C, H, W: W0 } = poolCache;
  const oH = dOut[0].length, oW = dOut[0][0].length;
  const dX = [];
  for (let c = 0; c < C; c++) {
    const plane = zeros(H, W0);
    for (let oh = 0; oh < oH; oh++) {
      for (let ow = 0; ow < oW; ow++) {
        const idx = argmax[c][oh][ow];
        plane[Math.floor(idx / W0)][idx % W0] += dOut[c][oh][ow];
      }
    }
    dX.push(plane);
  }
  return dX;
}

/**
 * 直接滑动窗口卷积（参考实现，用于与 im2col+GEMM 互证）
 */
export function conv2dNaive(X, W, b, { stride = 1, pad = 0 } = {}) {
  const C = X.length, H = X[0].length, W0 = X[0][0].length;
  const OC = W.length, kH = W[0][0].length, kW = W[0][0][0].length;
  const oH = convOutputSize(H, kH, stride, pad);
  const oW = convOutputSize(W0, kW, stride, pad);
  const out = [];
  for (let oc = 0; oc < OC; oc++) {
    const plane = zeros(oH, oW);
    for (let oh = 0; oh < oH; oh++) {
      for (let ow = 0; ow < oW; ow++) {
        let s = b[oc];
        for (let c = 0; c < C; c++) {
          for (let i = 0; i < kH; i++) {
            const r = oh * stride - pad + i;
            if (r < 0 || r >= H) continue;
            for (let j = 0; j < kW; j++) {
              const cc = ow * stride - pad + j;
              if (cc < 0 || cc >= W0) continue;
              s += W[oc][c][i][j] * X[c][r][cc];
            }
          }
        }
        plane[oh][ow] = s;
      }
    }
    out.push(plane);
  }
  return out;
}

/**
 * 卷积梯度检验：定义标量损失 L = Σ out ⊙ dOut（dOut 为固定上游梯度），
 * 逐一扰动 X / W / b 的每个元素，中心差分 vs 解析梯度。
 * @returns {{maxRel:number, maxAbs:number}}
 */
export function convGradientCheck(X, W, b, dOut, { stride = 1, pad = 0, eps = 1e-5 } = {}) {
  const loss = (XX, WW, bb) => {
    const { out } = conv2dForward(XX, WW, bb, { stride, pad });
    let s = 0;
    for (let oc = 0; oc < out.length; oc++) {
      for (let r = 0; r < out[oc].length; r++) {
        for (let c = 0; c < out[oc][r].length; c++) s += out[oc][r][c] * dOut[oc][r][c];
      }
    }
    return s;
  };
  const { out, cache } = conv2dForward(X, W, b, { stride, pad });
  const g = conv2dBackward(cache, dOut);
  let maxRel = 0, maxAbs = 0;
  const probe = (get, set, analytic) => {
    const orig = get();
    set(orig + eps);
    const lp = loss(X, W, b);
    set(orig - eps);
    const lm = loss(X, W, b);
    set(orig);
    const num = (lp - lm) / (2 * eps);
    maxRel = Math.max(maxRel, relErr(num, analytic));
    maxAbs = Math.max(maxAbs, Math.abs(num - analytic));
  };
  // X
  for (let c = 0; c < X.length; c++) {
    for (let r = 0; r < X[c].length; r++) {
      for (let c2 = 0; c2 < X[c][r].length; c2++) {
        probe(() => X[c][r][c2], (v) => { X[c][r][c2] = v; }, g.dX[c][r][c2]);
      }
    }
  }
  // W
  for (let oc = 0; oc < W.length; oc++) {
    for (let c = 0; c < W[oc].length; c++) {
      for (let i = 0; i < W[oc][c].length; i++) {
        for (let j = 0; j < W[oc][c][i].length; j++) {
          probe(() => W[oc][c][i][j], (v) => { W[oc][c][i][j] = v; }, g.dW[oc][c][i][j]);
        }
      }
    }
  }
  // b
  for (let oc = 0; oc < b.length; oc++) {
    probe(() => b[oc], (v) => { b[oc] = v; }, g.db[oc]);
  }
  return { maxRel, maxAbs, out };
}

/** 生成确定性随机张量（卷积演示/检验用） */
export function randomConvTensors(C, H, W, OC, kH, kW, rng) {
  const X = [];
  for (let c = 0; c < C; c++) {
    const plane = zeros(H, W);
    for (let r = 0; r < H; r++) for (let cc = 0; cc < W; cc++) plane[r][cc] = rng() * 2 - 1;
    X.push(plane);
  }
  const Wt = [];
  for (let oc = 0; oc < OC; oc++) {
    const chan = [];
    for (let c = 0; c < C; c++) {
      const plane = zeros(kH, kW);
      for (let r = 0; r < kH; r++) for (let cc = 0; cc < kW; cc++) plane[r][cc] = rng() * 2 - 1;
      chan.push(plane);
    }
    Wt.push(chan);
  }
  const b = new Array(OC);
  for (let oc = 0; oc < OC; oc++) b[oc] = rng() * 0.2 - 0.1;
  return { X, W: Wt, b };
}

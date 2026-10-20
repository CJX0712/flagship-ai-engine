/**
 * 旗舰AI引擎 · 自组织映射（Self-Organizing Map, Kohonen 1982）
 * 站在：Kohonen 1982《Self-Organized Formation of Topologically Correct
 *       Feature Maps》（SOM 原始论文）· Kohonen 1990/2001《Self-Organizing Maps》
 *       （邻域核 + 衰减律的标准训练框架）· Ultsch & Siemon 1990（SOM 与 CCA 对照）。
 *
 * 训练（在线 somTrain：随机样本序；批量 somBatchTrain：全数据同步邻域加权质心）：
 *   BMU b(x) = argmin_c ‖x − m_c‖₂（最近码本）；
 *   邻域核 h_{cb} = exp(−‖r_c − r_b‖² / (2σ²))（r 为格点坐标，链/网统一）；
 *   在线：m_c ← m_c + α·h_{cb}·(x − m_c)，σ、α 按 Kohonen 指数律同步衰减；
 *   批量：m_c = Σ_x h_{c,b(x)}·x / Σ_x h_{c,b(x)}（确定性，无采样噪声）。
 *
 * 不变量：
 *   ① BMU 恒等式：somBMU ≡ 展开式暴力 argmin（‖x‖²−2x·m+‖m‖² 口径，逐位一致）；
 *   ② 邻域核闭式：h = exp(−d²/(2σ²))，d=0 ⇒ h=1（精确），σ→∞ ⇒ h→1（<1e-15）；
 *   ③ 批量 SOM 量化误差逐扫描下降（包络口径：每轮 ≤ 前轮×1.1，末值 < 初值 5%）；
 *   ④ 拓扑保持：1D 链在 1D 数据上码本按格点序单调排列（升/降其一零乱序）；
 *   ⑤ σ→∞ 极限：h≡1 ⇒ 在线单步后码本离散度精确收缩为 (1−α) 倍（relErr<1e-12）；
 *   ⑥ 同 seed 两次训练码本逐位一致（maxDiff=0）。
 */

import { mulberry32 } from '../core/stats.js';

/** 高斯邻域核：h = exp(−d²/(2σ²))；d=0 ⇒ 1 */
export function somNeighborhood(dist2, sigma) {
  return Math.exp(-dist2 / (2 * sigma * sigma));
}

/**
 * 最佳匹配单元（BMU）：欧氏距离最小的码本格点（行主序确定性决胜）。
 * @returns {{r:number, c:number, dist2:number}}
 */
export function somBMU(codebook, x) {
  const rows = codebook.length, cols = codebook[0].length;
  let best = Infinity, br = 0, bc = 0;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      let d2 = 0;
      for (let k = 0; k < x.length; k++) {
        const dv = x[k] - codebook[r][c][k];
        d2 += dv * dv;
      }
      if (d2 < best) { best = d2; br = r; bc = c; }
    }
  }
  return { r: br, c: bc, dist2: best };
}

/** 量化误差：全样本到各自 BMU 的平均平方距离 */
export function somQuantError(codebook, X) {
  let s = 0;
  for (const x of X) s += somBMU(codebook, x).dist2;
  return s / X.length;
}

/**
 * 单步在线更新（纯函数：返回新码本，不修改入参）。
 * @param {number[][][]} codebook rows×cols×d
 * @param {number[]} x 输入样本
 * @param {{r:number,c:number}} bmu 最佳匹配单元
 */
export function somStep(codebook, x, bmu, alpha, sigma) {
  const rows = codebook.length, cols = codebook[0].length;
  return codebook.map((row, r) => row.map((m, c) => {
    const dist2 = (r - bmu.r) * (r - bmu.r) + (c - bmu.c) * (c - bmu.c);
    const h = somNeighborhood(dist2, sigma);
    return m.map((v, k) => v + alpha * h * (x[k] - v));
  }));
}

/**
 * SOM 训练（在线、随机样本序、码本由数据样本随机初始化）。
 * @returns {{codebook:number[][][], rows:number, cols:number, bmus:Array, quantErrors:number[]}}
 */
export function somTrain(X, {
  rows = 1, cols = 10, sigma0 = null, alpha0 = 0.5, iters = 2000, seed = 42,
} = {}) {
  const rng = mulberry32(seed);
  const d = X[0].length;
  const codebook = Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => X[Math.floor(rng() * X.length)].slice()));
  const s0 = sigma0 == null ? Math.max(rows, cols) / 2 : sigma0;
  const tau = Math.max(1, iters / 3); // Kohonen 指数衰减时间常数：末端衰减至 e^{−3}
  const every = Math.max(1, Math.floor(iters / 40));
  const quantErrors = [];
  let cur = codebook;
  for (let it = 0; it < iters; it++) {
    const decay = Math.exp(-it / tau);
    const sigma = s0 * decay;
    const alpha = alpha0 * decay;
    const x = X[Math.floor(rng() * X.length)];
    const bmu = somBMU(cur, x);
    cur = somStep(cur, x, bmu, alpha, sigma);
    if ((it + 1) % every === 0) quantErrors.push(somQuantError(cur, X));
  }
  const bmus = X.map((x) => somBMU(cur, x));
  return { codebook: cur, rows, cols, bmus, quantErrors };
}

/** 逐样本 BMU 扁平索引（行主序：r·cols + c） */
export function somLabels(X, model) {
  return model.bmus.map((b) => b.r * model.cols + b.c);
}

/**
 * 批量 SOM（Kohonen 批处理 MAP）：每轮用全数据同步更新码本
 *   m_c = Σ_x h_{c,b(x)}·x / Σ_x h_{c,b(x)}（邻域加权质心，确定性、无采样噪声）
 * @returns {{codebook:number[][][], rows:number, cols:number, bmus:Array, quantErrors:number[]}}
 */
export function somBatchTrain(X, {
  rows = 6, cols = 6, sigma0 = null, iters = 80, seed = 42,
} = {}) {
  const rng = mulberry32(seed);
  const d = X[0].length;
  const codebook = Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => X[Math.floor(rng() * X.length)].slice()));
  const s0 = sigma0 == null ? Math.max(rows, cols) / 2 : sigma0;
  const tau = Math.max(1, iters / 2); // σ 末端衰减至 e^{−2}
  const quantErrors = [];
  let cur = codebook;
  for (let it = 0; it < iters; it++) {
    const sigma = s0 * Math.exp(-it / tau);
    const bmus = X.map((x) => somBMU(cur, x));
    const num = Array.from({ length: rows }, () => Array.from({ length: cols }, () => new Array(d).fill(0)));
    const den = Array.from({ length: rows }, () => new Array(cols).fill(0));
    for (let i = 0; i < X.length; i++) {
      const b = bmus[i];
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const d2 = (r - b.r) * (r - b.r) + (c - b.c) * (c - b.c);
          const h = somNeighborhood(d2, sigma);
          den[r][c] += h;
          for (let k = 0; k < d; k++) num[r][c][k] += h * X[i][k];
        }
      }
    }
    cur = cur.map((row, r) => row.map((m, c) =>
      (den[r][c] > 0 ? m.map((_, k) => num[r][c][k] / den[r][c]) : m.slice())));
    quantErrors.push(somQuantError(cur, X));
  }
  const bmus = X.map((x) => somBMU(cur, x));
  return { codebook: cur, rows, cols, bmus, quantErrors };
}

/**
 * 旗舰AI引擎 · Mean Shift 众数搜索（均值漂移 / 众数追寻）
 * 站在：Fukunaga & Hostetler 1975 IEEE Trans. Information Theory
 *       《The estimation of the gradient of a density function, with applications
 *       in pattern recognition》· Cheng 1995 IEEE PAMI《Mean shift, mode seeking,
 *       and clustering》· Comaniciu & Meer 2002《Mean shift: a robust approach
 *       toward feature space analysis》。
 *
 * 高斯核 K(u) = exp(−‖u‖²/(2h²))，核密度估计（省略与 x 无关的归一化常数）：
 *   f̂(x) = Σᵢ K(xᵢ − x) = Σᵢ exp(−‖xᵢ − x‖²/(2h²))
 * 一步均值漂移（加权质心）：
 *   m(x) = Σᵢ wᵢxᵢ / Σᵢ wᵢ，wᵢ = K(xᵢ − x)
 * 梯度恒等式（本模块一切正确性的源头）：
 *   ∇f̂(x) = (f̂(x)/h²)·(m(x) − x)   ⟺   m(x) − x = h²·∇f̂(x)/f̂(x)
 * 即 mean shift 是「步长自适应（h²/f̂(x)）的密度梯度上升」，
 * 故 Cheng 1995 定理成立：沿迭代轨迹 f̂ 单调不降 ⇒ 收敛到局部众数（不动点 m(x*) = x*）。
 *
 * 不变量：
 *   ① 梯度恒等式：m(x) − x ≡ h²∇f̂(x)/f̂(x)（中心差分 vs 闭式，相对误差 1e-6）；
 *   ② h → ∞ 一步到质心：单次更新 ≡ 数据均值（1e-12）；
 *   ③ 单调上升：沿轨迹 f̂ 非降（Cheng 1995 收敛定理，全步检查）；
 *   ④ 收敛处 ‖m(x) − x‖ < tol（不动点条件）；
 *   ⑤ 单峰高斯团：任意起点收敛到样本均值附近；
 *   ⑥ 两团分离数据：众数合并后簇数 = 2，标签 ≡ k-means 结果（ARI = 1）。
 */

/** 平方欧氏距离（循环累加，避免大数组展开参数） */
function msSqDist(a, b) {
  let s = 0;
  for (let j = 0; j < a.length; j++) {
    const e = a[j] - b[j];
    s += e * e;
  }
  return s;
}

/** 欧氏距离 */
function msDist(a, b) {
  return Math.sqrt(msSqDist(a, b));
}

/**
 * 核密度估计 f̂(x) = Σᵢ exp(−‖xᵢ − x‖²/(2h²))。
 * 真正的概率密度需再除以 n·h^d·(2π)^{d/2}；本模块统一使用未归一化形式——
 * 归一化常数与 x 无关，不影响众数位置、不影响梯度恒等式的方向。
 * @param {number[][]} X 数据（n×d）
 * @param {number[]} x 查询点
 * @param {number} h 带宽
 * @returns {number} f̂(x)
 */
export function meanShiftKDE(X, x, h) {
  const inv2h2 = 1 / (2 * h * h);
  let s = 0;
  for (const xi of X) s += Math.exp(-msSqDist(xi, x) * inv2h2);
  return s;
}

/**
 * 一步均值漂移：返回新位置 m(x) = Σᵢwᵢxᵢ / Σᵢwᵢ（即下一次迭代的查询点）。
 * 漂移向量 = m(x) − x = h²∇f̂(x)/f̂(x)。
 * @param {number[][]} X 数据
 * @param {number[]} x 当前位置
 * @param {number} h 带宽
 * @returns {number[]} 新位置 m(x)
 */
export function meanShiftStep(X, x, h) {
  const d = x.length;
  const inv2h2 = 1 / (2 * h * h);
  const out = new Array(d).fill(0);
  let sum = 0;
  for (const xi of X) {
    const w = Math.exp(-msSqDist(xi, x) * inv2h2);
    sum += w;
    for (let j = 0; j < d; j++) out[j] += w * xi[j];
  }
  for (let j = 0; j < d; j++) out[j] /= sum; // sum ≥ exp(−maxDist²/2h²) > 0
  return out;
}

/**
 * 迭代均值漂移至局部众数（不动点）。
 * @param {number[][]} X 数据
 * @param {number[]} x0 起点
 * @param {number} h 带宽
 * @param {{maxIter?:number, tol?:number}} [opts] 最大迭代数与位移阈值
 * @returns {{mode:number[], steps:number, path:number[][], kdeTrace:number[], shift:number}}
 *   path[0] = 起点，path[k] 为第 k 次迭代后的位置；kdeTrace 与之逐点对应。
 */
export function meanShiftMode(X, x0, h, { maxIter = 200, tol = 1e-10 } = {}) {
  const path = [x0.slice()];
  const kdeTrace = [meanShiftKDE(X, x0, h)];
  let x = x0.slice();
  let steps = 0;
  let shift = Infinity;
  for (let it = 0; it < maxIter; it++) {
    const nx = meanShiftStep(X, x, h);
    steps++;
    shift = msDist(nx, x);
    x = nx;
    path.push(x.slice());
    kdeTrace.push(meanShiftKDE(X, x, h));
    if (shift < tol) break; // 不动点：‖m(x) − x‖ < tol
  }
  return { mode: x, steps, path, kdeTrace, shift };
}

/**
 * Mean Shift 聚类：每个数据点作为起点各跑一次众数搜索，
 * 收敛点（众数）按距离 < mergeTol 合并为同一簇（无需预先指定簇数 k）。
 * @param {number[][]} X 数据
 * @param {number} h 带宽
 * @param {{mergeTol?:number, maxIter?:number, tol?:number}} [opts]
 * @returns {{labels:number[], modes:number[][], centers:number[][], k:number}}
 */
export function meanShiftCluster(X, h, { mergeTol = 1e-4, maxIter = 200, tol = 1e-10 } = {}) {
  const n = X.length;
  const modes = new Array(n);
  for (let i = 0; i < n; i++) modes[i] = meanShiftMode(X, X[i], h, { maxIter, tol }).mode;
  const labels = new Array(n).fill(0);
  const centers = [];
  for (let i = 0; i < n; i++) {
    let best = -1;
    let bd = Infinity;
    for (let c = 0; c < centers.length; c++) {
      const dd = msDist(modes[i], centers[c]);
      if (dd < bd) { bd = dd; best = c; }
    }
    if (best < 0 || bd > mergeTol) {
      centers.push(modes[i].slice());
      labels[i] = centers.length - 1;
    } else {
      labels[i] = best;
    }
  }
  return { labels, modes, centers, k: centers.length };
}

/**
 * 均值漂移向量的闭式（加权质心）与「h²∇f̂/f̂」的中心差分两条独立路径互证。
 * 返回逐维相对误差，供梯度恒等式检验。
 * @param {number[][]} X 数据
 * @param {number[]} x 查询点
 * @param {number} h 带宽
 * @param {number} [delta] 中心差分步长
 * @returns {{shift:number[], gradient:number[], maxRelErr:number}}
 */
export function meanShiftGradientCheck(X, x, h, delta = 1e-5) {
  const d = x.length;
  const f0 = meanShiftKDE(X, x, h);
  const m = meanShiftStep(X, x, h);
  const shift = m.map((v, j) => v - x[j]);
  const gradient = new Array(d);
  let maxRelErr = 0;
  for (let j = 0; j < d; j++) {
    const xp = x.slice(); xp[j] += delta;
    const xm = x.slice(); xm[j] -= delta;
    const partial = (meanShiftKDE(X, xp, h) - meanShiftKDE(X, xm, h)) / (2 * delta);
    gradient[j] = partial;
    const want = h * h * partial / f0; // h²∇f̂/f̂
    const denom = Math.max(1e-8, Math.abs(want) + Math.abs(shift[j]));
    maxRelErr = Math.max(maxRelErr, Math.abs(want - shift[j]) / denom);
  }
  return { shift, gradient, maxRelErr };
}

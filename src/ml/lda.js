/**
 * 旗舰AI引擎 · 线性判别分析 / Fisher（LDA）
 * 站在：Fisher 1936《The Use of Multiple Measurements in Taxonomic Problems》、
 *       Rao 1948（判别式）· Bishop 2006《Pattern Recognition and Machine Learning》§4.1。
 *
 * 二分类闭式（Fisher 判别定理）：w ∝ Sw⁻¹(μ1 − μ2) 使 Rayleigh 商
 *   J(w) = wᵀSb w / wᵀSw w 取极大值；等价地 Sw·w ∥ (μ1 − μ2)。
 * 多分类：解广义特征问题 Sw⁻¹Sb w = λ w（经 Sw = L Lᵀ 变换为对称 B = L⁻¹Sb L⁻ᵀ 后分解）。
 *
 * 不变量：
 *   ① 各类 classMeans ≡ 暴力均值（逐元素 < 1e-9）；
 *   ② Sw、Sb 对称（max 非对称项 < 1e-9）；
 *   ③ Fisher 方向恒等式：Sw·w ∥ (μ1−μ2)（|cos| ≈ 1，至 1e-6）；
 *   ④ Fisher 比 J(w) ≥ 任意随机方向 J（w 的 J 最大）；
 *   ⑤ 线性可分合成数据：投影后类心间距 > 类内散布，且最近类心分类准确率 = 1.0。
 */

import { zeros, inverse, matmul, matvec, dot, transpose, jacobiEig, cholesky } from '../core/linalg.js';

/** 类内散度 Sw = Σ_c Σ_{x∈c} (x−μ_c)(x−μ_c)ᵀ（全量累加后单次成形，天然对称） */
function ldaScatterWithin(X, y, classMeans, classes) {
  const d = X[0].length;
  const Sw = zeros(d, d);
  for (let i = 0; i < X.length; i++) {
    const c = classes.indexOf(y[i]);
    const mu = classMeans[c];
    const xi = X[i];
    for (let a = 0; a < d; a++) {
      const ea = xi[a] - mu[a];
      for (let b = 0; b < d; b++) Sw[a][b] += ea * (xi[b] - mu[b]);
    }
  }
  return Sw;
}

/** 类间散度 Sb = Σ_c n_c (μ_c − μ)(μ_c − μ)ᵀ（μ 为总体均值） */
function ldaScatterBetween(X, y, classMeans, classes) {
  const d = X[0].length;
  const overall = new Array(d).fill(0);
  for (const xi of X) for (let j = 0; j < d; j++) overall[j] += xi[j];
  for (let j = 0; j < d; j++) overall[j] /= X.length;
  const counts = new Array(classes.length).fill(0);
  for (const yi of y) counts[classes.indexOf(yi)]++;
  const Sb = zeros(d, d);
  for (let c = 0; c < classes.length; c++) {
    const mu = classMeans[c];
    for (let a = 0; a < d; a++) {
      const da = mu[a] - overall[a];
      for (let b = 0; b < d; b++) Sb[a][b] += counts[c] * da * (mu[b] - overall[b]);
    }
  }
  return Sb;
}

/** Sw + reg·I 后求逆（保证 Sw 数值正定） */
function ldaInverse(A, reg) {
  const d = A.length;
  const Are = A.map((row, i) => row.map((v, j) => (i === j ? v + reg : v)));
  return inverse(Are);
}

/** 矩阵对称化（消除浮点非对称，供 Jacobi 分解） */
function ldaSymmetrize(A) {
  const d = A.length;
  for (let i = 0; i < d; i++) for (let j = i + 1; j < d; j++) {
    const avg = 0.5 * (A[i][j] + A[j][i]);
    A[i][j] = avg; A[j][i] = avg;
  }
}

/**
 * LDA 拟合。
 * 二分类直接用闭式 w = Sw⁻¹(μ1−μ2)；多分类用 Cholesky 变换后的对称广义特征分解。
 * @returns {{classMeans:number[][], Sw:number[][], Sb:number[][], w:number[], Wproj:number[][], classes:*[]}}
 */
export function ldaFit(X, y, { reg = 1e-6 } = {}) {
  const classes = [];
  for (const yi of y) if (!classes.includes(yi)) classes.push(yi);
  const d = X[0].length;
  const classMeans = classes.map((c) => {
    const rows = [];
    for (let i = 0; i < X.length; i++) if (y[i] === c) rows.push(X[i]);
    const mu = new Array(d).fill(0);
    for (const r of rows) for (let j = 0; j < d; j++) mu[j] += r[j];
    for (let j = 0; j < d; j++) mu[j] /= rows.length;
    return mu;
  });
  const Sw = ldaScatterWithin(X, y, classMeans, classes);
  const Sb = ldaScatterBetween(X, y, classMeans, classes);
  const SwInv = ldaInverse(Sw, reg);

  let Wproj;
  if (classes.length === 2) {
    // 闭式 Fisher 方向：w = Sw⁻¹(μ1 − μ2)（精确，避免非对称矩阵分解）
    const diff = new Array(d);
    for (let j = 0; j < d; j++) diff[j] = classMeans[1][j] - classMeans[0][j];
    let w = matvec(SwInv, diff);
    if (dot(w, diff) < 0) for (let j = 0; j < d; j++) w[j] = -w[j];
    Wproj = [w];
  } else {
    // 广义特征：Sw = L Lᵀ ⇒ B = L⁻¹ Sb L⁻ᵀ（对称），分解后 w = L⁻ᵀ v
    const L = cholesky((function () { const M = Sw.map((r, i) => r.map((v, j) => (i === j ? v + reg : v))); return M; })());
    const Linv = inverse(L);
    const LTinv = transpose(Linv);
    const B = matmul(matmul(Linv, Sb), LTinv);
    ldaSymmetrize(B);
    const { values, vectors } = jacobiEig(B);
    const order = values.map((v, k) => k).sort((a, b) => Math.abs(values[b]) - Math.abs(values[a]));
    Wproj = order.slice(0, classes.length - 1).map((k) => matvec(LTinv, vectors[k]));
  }
  const w = Wproj[0].slice();
  return { classMeans, Sw, Sb, w, Wproj, classes };
}

/**
 * LDA 投影。
 * 二分类 → 1-D 标量数组（每点 w·x）；多分类 → 坐标矩阵（n×(C−1)）。
 * @returns {number[]|number[][]}
 */
export function ldaProject(X, model) {
  const { Wproj, classes } = model;
  if (classes.length === 2) {
    const w = Wproj[0];
    return X.map((xi) => dot(w, xi));
  }
  return X.map((xi) => Wproj.map((row) => dot(row, xi)));
}

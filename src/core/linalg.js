/**
 * 旗舰AI引擎 · 核心线性代数
 * 站在 Golub & Van Loan《Matrix Computations》与 Numerical Recipes 之上。
 * 约定：矩阵 = number[][]（行主序），向量 = number[]。
 */

export const EPS = Number.EPSILON; // ≈ 2.22e-16

/** m×n 零矩阵 */
export function zeros(m, n) {
  const M = new Array(m);
  for (let i = 0; i < m; i++) M[i] = new Array(n).fill(0);
  return M;
}

/** n×n 单位阵 */
export function identity(n) {
  const I = zeros(n, n);
  for (let i = 0; i < n; i++) I[i][i] = 1;
  return I;
}

/** 转置 */
export function transpose(A) {
  const m = A.length, n = A[0].length;
  const T = zeros(n, m);
  for (let i = 0; i < m; i++) {
    const Ai = A[i];
    for (let j = 0; j < n; j++) T[j][i] = Ai[j];
  }
  return T;
}

/** 矩阵乘 C = A·B（i-k-j 内积顺序，缓存友好） */
export function matmul(A, B) {
  const m = A.length, kk = B.length, n = B[0].length;
  const C = zeros(m, n);
  for (let i = 0; i < m; i++) {
    const Ai = A[i], Ci = C[i];
    for (let k = 0; k < kk; k++) {
      const a = Ai[k];
      if (a === 0) continue;
      const Bk = B[k];
      for (let j = 0; j < n; j++) Ci[j] += a * Bk[j];
    }
  }
  return C;
}

/** 矩阵·向量 y = A·v */
export function matvec(A, v) {
  const m = A.length, n = A[0].length;
  const y = new Array(m).fill(0);
  for (let i = 0; i < m; i++) {
    let s = 0;
    const Ai = A[i];
    for (let j = 0; j < n; j++) s += Ai[j] * v[j];
    y[i] = s;
  }
  return y;
}

/** 点积 */
export function dot(u, v) {
  let s = 0;
  for (let i = 0; i < u.length; i++) s += u[i] * v[i];
  return s;
}

/** 向量 p-范数（默认 2-范数） */
export function norm(v, p = 2) {
  if (p === 2) {
    let s = 0;
    for (let i = 0; i < v.length; i++) s += v[i] * v[i];
    return Math.sqrt(s);
  }
  let s = 0;
  for (let i = 0; i < v.length; i++) s += Math.pow(Math.abs(v[i]), p);
  return Math.pow(s, 1 / p);
}

/** 矩阵 Frobenius 范数 */
export function frobeniusNorm(A) {
  let s = 0;
  for (const row of A) for (const x of row) s += x * x;
  return Math.sqrt(s);
}

/** 矩阵最大绝对元（∞-诱导近似，用于容差） */
export function maxAbs(A) {
  let mx = 0;
  for (const row of A) for (const x of row) mx = Math.max(mx, Math.abs(x));
  return mx;
}

/**
 * 线性方程组求解 A·x = b —— 部分主元高斯消元（Golub & Van Loan 算法 3.4.1）。
 * A 必须非奇异方阵。
 */
export function solve(A, b) {
  const n = A.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let i = col + 1; i < n; i++) {
      if (Math.abs(M[i][col]) > Math.abs(M[piv][col])) piv = i;
    }
    if (Math.abs(M[piv][col]) < 1e-300) throw new Error('矩阵奇异，无法求解');
    if (piv !== col) { const t = M[col]; M[col] = M[piv]; M[piv] = t; }
    const inv = 1 / M[col][col];
    for (let i = col + 1; i < n; i++) {
      const f = M[i][col] * inv;
      if (f === 0) continue;
      for (let j = col; j <= n; j++) M[i][j] -= f * M[col][j];
    }
  }
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let s = M[i][n];
    for (let j = i + 1; j < n; j++) s -= M[i][j] * x[j];
    x[i] = s / M[i][i];
  }
  return x;
}

/** 逆矩阵 A⁻¹（逐单位列求解） */
export function inverse(A) {
  const n = A.length;
  const cols = [];
  for (let j = 0; j < n; j++) {
    const e = new Array(n).fill(0);
    e[j] = 1;
    cols.push(solve(A, e));
  }
  const inv = zeros(n, n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) inv[i][j] = cols[j][i];
  return inv;
}

/**
 * Cholesky 分解 A = L·Lᵀ（A 对称正定）。
 * 不变量：L 下三角、L·Lᵀ = A。
 */
export function cholesky(A) {
  const n = A.length;
  const L = zeros(n, n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = A[i][j];
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      if (i === j) {
        if (s <= 0) throw new Error('矩阵非正定，Cholesky 分解失败');
        L[i][j] = Math.sqrt(s);
      } else {
        L[i][j] = s / L[j][j];
      }
    }
  }
  return L;
}

/**
 * QR 分解（Householder 反射，数值稳定）。
 * 返回 { Q, R }：Q 正交（m×m），R 上三角，A = Q·R。
 */
export function qr(A) {
  const m = A.length, n = A[0].length;
  const R = A.map(r => r.slice());
  let Q = identity(m);
  const steps = Math.min(n, m - 1);
  for (let k = 0; k < steps; k++) {
    const len = m - k;
    const x = new Array(len);
    for (let i = 0; i < len; i++) x[i] = R[k + i][k];
    const nx = norm(x);
    if (nx < 1e-300) continue;
    const alpha = x[0] > 0 ? -nx : nx; // 选号避免抵消
    const v = x.slice();
    v[0] -= alpha;
    const nv = norm(v);
    if (nv < 1e-300) continue;
    for (let i = 0; i < len; i++) v[i] /= nv;
    // R ← H·R
    for (let j = k; j < n; j++) {
      let d = 0;
      for (let i = 0; i < len; i++) d += v[i] * R[k + i][j];
      const two = 2 * d;
      for (let i = 0; i < len; i++) R[k + i][j] -= two * v[i];
    }
    // Q ← Q·H（右乘累积，A = Q·R）
    for (let i = 0; i < m; i++) {
      let d = 0;
      for (let t = 0; t < len; t++) d += Q[i][k + t] * v[t];
      const two = 2 * d;
      for (let t = 0; t < len; t++) Q[i][k + t] -= two * v[t];
    }
  }
  return { Q, R };
}

/**
 * 对称矩阵 Jacobi 特征分解（Numerical Recipes 经典旋转）。
 * 返回 { values, vectors }：特征值降序；vectors[k] 为 values[k] 对应的单位特征向量。
 * 不变量：VᵀV = I，‖A·v − λ·v‖ → O(EPS·‖A‖)。
 */
export function jacobiEig(A, { maxSweeps = 100, tol = 1e-13 } = {}) {
  const n = A.length;
  const a = A.map(r => r.slice());
  const V = identity(n);
  const scale = Math.max(1, frobeniusNorm(a));
  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p][q] * a[p][q];
    if (Math.sqrt(off) <= tol * scale) break;
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = a[p][q];
        if (Math.abs(apq) < 1e-300) continue;
        const theta = (a[q][q] - a[p][p]) / (2 * apq);
        const sgn = theta >= 0 ? 1 : -1;
        const t = sgn / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        const tau = s / (1 + c);
        const app = a[p][p], aqq = a[q][q];
        a[p][p] = app - t * apq;
        a[q][q] = aqq + t * apq;
        a[p][q] = a[q][p] = 0;
        for (let i = 0; i < n; i++) {
          if (i === p || i === q) continue;
          const aip = a[i][p], aiq = a[i][q];
          a[i][p] = a[p][i] = aip - s * (aiq + tau * aip);
          a[i][q] = a[q][i] = aiq + s * (aip - tau * aiq);
        }
        for (let i = 0; i < n; i++) {
          const vip = V[i][p], viq = V[i][q];
          V[i][p] = vip - s * (viq + tau * vip);
          V[i][q] = viq + s * (vip - tau * viq);
        }
      }
    }
  }
  const pairs = [];
  for (let i = 0; i < n; i++) pairs.push([a[i][i], i]);
  pairs.sort((x, y) => y[0] - x[0]);
  const values = pairs.map(p => p[0]);
  const vectors = pairs.map(p => V.map(row => row[p[1]]));
  return { values, vectors };
}

/**
 * 奇异值分解 A = U·Σ·Vᵀ（经 AᵀA 的 Jacobi 特征分解，要求行数 ≥ 列数、满秩最佳）。
 * 返回 { U(m×n 列正交), s(奇异值降序), V(n×n 正交) }。
 * 不变量：UᵀU = I，VᵀV = I，‖A − U·diag(s)·Vᵀ‖_F ≈ 0；
 *        秩-k 截断误差 = √(Σ_{i>k} sᵢ²)（Eckart–Young）。
 */
export function svd(A) {
  const m = A.length, n = A[0].length;
  if (m < n) throw new Error('svd 当前要求行数 ≥ 列数');
  const AtA = matmul(transpose(A), A);
  const { values: lam, vectors } = jacobiEig(AtA);
  const s = lam.map(l => Math.sqrt(Math.max(l, 0)));
  const sMax = s[0] || 0;
  const U = zeros(m, n); // 行主序 m×n，第 k 列 = u_k
  const cols = [];       // 已求得的正交列
  const computed = new Array(n).fill(false);
  const cutoff = 1e-12 * (sMax || 1);
  for (let k = 0; k < n; k++) {
    if (s[k] > cutoff) {
      const u = matvec(A, vectors[k]);
      const nu = norm(u);
      if (nu > 1e-300) {
        for (let i = 0; i < m; i++) U[i][k] = u[i] / nu;
        cols.push(U.map(row => row[k]));
        computed[k] = true;
      }
    }
  }
  // 秩亏时用标准基做 Gram-Schmidt 补全 U 的正交列
  for (let k = 0; k < n; k++) {
    if (computed[k]) continue;
    for (let e = 0; e < m && !computed[k]; e++) {
      let w = new Array(m).fill(0);
      w[e] = 1;
      for (const u of cols) {
        const d = dot(w, u);
        w = w.map((x, i) => x - d * u[i]);
      }
      const nw = norm(w);
      if (nw > 1e-6) {
        w = w.map(x => x / nw);
        for (let i = 0; i < m; i++) U[i][k] = w[i];
        cols.push(w);
        computed[k] = true;
      }
    }
  }
  const V = transpose(vectors); // 行主序 n×n，列 = 特征向量
  return { U, s, V };
}

/**
 * 幂迭代求主特征对（要求存在明显主导特征值；对称阵最稳）。
 * 返回 { value, vector, residual }；value 为 Rayleigh 商 vᵀAv。
 */
export function powerIteration(A, { iters = 500, rng = Math.random } = {}) {
  const n = A.length;
  let v = new Array(n);
  for (let i = 0; i < n; i++) v[i] = rng() - 0.5;
  v = v.map(x => x / (norm(v) || 1));
  let value = 0;
  for (let it = 0; it < iters; it++) {
    const w = matvec(A, v);
    const nw = norm(w);
    if (nw < 1e-300) break;
    v = w.map(x => x / nw);
    value = dot(v, matvec(A, v)); // Rayleigh 商
  }
  const Av = matvec(A, v);
  const residual = norm(Av.map((x, i) => x - value * v[i]));
  return { value, vector: v, residual };
}

/** 可复现随机矩阵（均匀 [lo, hi)） */
export function randomMatrix(m, n, rng, lo = -1, hi = 1) {
  const A = zeros(m, n);
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) A[i][j] = lo + (hi - lo) * rng();
  return A;
}

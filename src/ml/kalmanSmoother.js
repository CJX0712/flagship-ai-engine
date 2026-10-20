/**
 * 旗舰AI引擎 · RTS 平滑器（Rauch–Tung–Striebel 1965）
 * 站在：Rauch, Tung & Striebel 1965《Maximum Likelihood Estimates of Linear
 *       Dynamic Systems》· Anderson & Moore《Optimal Filtering》§7.3
 *       · Särkkä《Bayesian Filtering and Smoothing》§8.2
 *       · Shumway & Stoffer《Time Series Analysis》§6.2（EM 中的平滑协方差）
 *
 * 卡尔曼滤波只用「到当前时刻为止」的数据：x̂_{k|k} = E[x_k | y_1..y_k]；
 * RTS 平滑器再用「未来」的数据做一次反向修正：x̂_{k|N} = E[x_k | y_1..y_N]。
 * 反向递推（k = N−1 → 0）：
 *   增益   J_k   = P_{k|k}·Fᵀ·P_{k+1|k}⁻¹
 *   均值   x̂_{k|N} = x̂_{k|k} + J_k·(x̂_{k+1|N} − x̂_{k+1|k})
 *   协方差 P_{k|N} = P_{k|k} + J_k·(P_{k+1|N} − P_{k+1|k})·J_kᵀ
 * 端点闭合：x̂_{N|N} ≡ 滤波终点（平滑器最后一步不做任何修正）。
 *
 * 索引约定（与 kalman.js 完全兼容）：滤波输出的第 k 项对应「第 k+1 个时刻」——
 *   states[k] = x̂_{k+1|k+1}，covariances[k] = P_{k+1|k+1}；
 *   predicted[k] = x̂_{k+1|k}，predictedCov[k] = P_{k+1|k}。
 * 故平滑增益 J_k（k = 0..N−2）由 covariances[k] 与 predictedCov[k+1] 构成。
 *
 * 不变量：
 *   ① 终点等式：x̂_{N|N}^{平滑} ≡ 滤波终点（误差精确为 0）；
 *   ② 协方差序：P_{k|k} − P_{k|N} 半正定（平滑必降低不确定性，最小特征值 ≥ −1e-10）；
 *   ③ 增益恒等式：J_k ≡ P_{k|k}FᵀP_{k+1|k}⁻¹ 独立重算（1e-12）；
 *   ④ 常值状态闭式：F=1, Q=0 时平滑 x̂_{0|N} ≡ 全部观测的样本均值（1e-10）；
 *   ⑤ 反向递推自洽：x̂_{k|N} ≡ x̂_{k|k} + J_k(x̂_{k+1|N} − x̂_{k+1|k}) 重算（1e-12）；
 *   ⑥ 均方误差序：蒙特卡洛聚合 MSE(平滑) < MSE(滤波)（最优平滑 = 全数据后验均值）。
 */

import { zeros, matmul, transpose, inverse, jacobiEig } from '../core/linalg.js';
import { kalmanFilter } from './kalman.js';

/** 矩阵减法 A − B */
function rtsMsub(A, B) {
  return A.map((row, i) => row.map((v, j) => v - B[i][j]));
}

/** 矩阵加法 A + B */
function rtsMadd(A, B) {
  return A.map((row, i) => row.map((v, j) => v + B[i][j]));
}

/** 就地对称化 A ← (A + Aᵀ)/2（消除浮点非对称，协方差类的必需收尾） */
function rtsSymmetrize(A) {
  const n = A.length;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const avg = 0.5 * (A[i][j] + A[j][i]);
      A[i][j] = avg;
      A[j][i] = avg;
    }
  }
  return A;
}

/**
 * 平滑增益 J = P_{k|k}·Fᵀ·P_{k+1|k}⁻¹。
 * 等价写法：Jᵀ = P_{k+1|k}⁻¹·(P_{k|k}Fᵀ)ᵀ，此处直接求逆后右乘（n 很小，稳定可控）。
 * @param {number[][]} Pfilt 滤波协方差 P_{k|k}
 * @param {number[][]} F 状态转移矩阵
 * @param {number[][]} Ppred 预测协方差 P_{k+1|k}
 * @returns {number[][]} 增益 J_k
 */
function rtsGain(Pfilt, F, Ppred) {
  const M = matmul(Pfilt, transpose(F));
  return matmul(M, inverse(Ppred));
}

/**
 * 对称矩阵最小特征值（Jacobi 旋转分解，供半正定判定）。
 * @param {number[][]} A 对称矩阵
 * @returns {number} 最小特征值
 */
function rtsMinEigen(A) {
  const { values } = jacobiEig(A);
  let lo = Infinity;
  for (const v of values) if (v < lo) lo = v;
  return lo;
}

/**
 * RTS 平滑（Rauch–Tung–Striebel 反向递推）。
 * 内部先跑一遍 kalmanFilter 拿到 {states, covariances, predicted, predictedCov}，
 * 再从终点向起点反向传播未来信息。
 *
 * @param {{F:number[][], H:number[][], Q:number[][], R:number[][],
 *          x0:number[], P0:number[][]}} model 与 kalmanFilter 完全同构
 * @param {number[][]} observations 观测序列 z_1..z_N
 * @returns {{smoothed:number[][], smoothedCovs:number[][][], gains:number[][][],
 *            filtered:number[][], filteredCovs:number[][][],
 *            predicted:number[][], predictedCov:number[][][]}}
 *   gains 长度为 N−1（J_k 仅定义在 k = 0..N−2）；N=0 时全部为空数组。
 */
export function rtsSmooth(model, observations) {
  const { F } = model;
  const kf = kalmanFilter(model, observations);
  const N = kf.states.length;
  const smoothed = new Array(N);
  const smoothedCovs = new Array(N);
  const gains = new Array(Math.max(N - 1, 0));
  if (N === 0) {
    return {
      smoothed, smoothedCovs, gains,
      filtered: kf.states, filteredCovs: kf.covariances,
      predicted: kf.predicted, predictedCov: kf.predictedCov,
    };
  }

  // 端点闭合：平滑终点 ≡ 滤波终点（不做任何修正）
  let xS = kf.states[N - 1].slice();
  let PS = kf.covariances[N - 1].map((row) => row.slice());
  smoothed[N - 1] = xS;
  smoothedCovs[N - 1] = PS;

  // 反向递推 k = N−2 → 0
  for (let k = N - 2; k >= 0; k--) {
    const Pfilt = kf.covariances[k];
    const Ppred = kf.predictedCov[k + 1];
    const J = rtsGain(Pfilt, F, Ppred);
    gains[k] = J;

    // 均值：x̂_{k|N} = x̂_{k|k} + J_k(x̂_{k+1|N} − x̂_{k+1|k})
    const innov = xS.map((v, i) => v - kf.predicted[k + 1][i]);
    xS = kf.states[k].map((v, i) => {
      let s = v;
      for (let j = 0; j < innov.length; j++) s += J[i][j] * innov[j];
      return s;
    });

    // 协方差：P_{k|N} = P_{k|k} + J_k(P_{k+1|N} − P_{k+1|k})J_kᵀ
    const diff = rtsMsub(PS, Ppred);
    PS = rtsSymmetrize(rtsMadd(Pfilt, matmul(matmul(J, diff), transpose(J))));

    smoothed[k] = xS;
    smoothedCovs[k] = PS;
  }

  return {
    smoothed, smoothedCovs, gains,
    filtered: kf.states, filteredCovs: kf.covariances,
    predicted: kf.predicted, predictedCov: kf.predictedCov,
  };
}

/**
 * 平滑相对滤波的不确定性下降量：逐步返回 P_{k|k} − P_{k|N} 的最小特征值。
 * 定理（Anderson & Moore）：该差值恒为半正定 ⇒ 返回值非负。
 * @param {number[][][]} filteredCovs 滤波协方差序列
 * @param {number[][][]} smoothedCovs 平滑协方差序列
 * @returns {number[]} 每步的最小特征值
 */
export function rtsCovarianceGapEigen(filteredCovs, smoothedCovs) {
  const out = new Array(smoothedCovs.length);
  for (let k = 0; k < smoothedCovs.length; k++) {
    out[k] = rtsMinEigen(rtsMsub(filteredCovs[k], smoothedCovs[k]));
  }
  return out;
}

/**
 * 无模型的 RTS 诊断：估计序列相对真值序列的 RMSE。
 * @param {number[][]} estimates 估计序列（每步一个状态向量）
 * @param {number[][]} truth 真值序列
 * @param {number} [dim] 参与比较的维（默认全部维）
 * @returns {number} RMSE
 */
export function rtsRMSE(estimates, truth, dim = -1) {
  const n = Math.min(estimates.length, truth.length);
  const d = dim >= 0 ? 1 : estimates[0].length;
  let se = 0, cnt = 0;
  for (let t = 0; t < n; t++) {
    for (let j = 0; j < estimates[t].length; j++) {
      if (dim >= 0 && j !== dim) continue;
      se += (estimates[t][j] - truth[t][j]) ** 2;
      cnt++;
    }
  }
  return cnt > 0 ? Math.sqrt(se / cnt) : 0;
}

/** 零矩阵（供外部构造增益数组时使用，保持与 linalg 同约定） */
export function rtsZeros(m, n) {
  return zeros(m, n);
}

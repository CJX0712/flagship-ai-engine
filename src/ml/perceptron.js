/**
 * 旗舰AI引擎 · 感知机（Perceptron）
 * 站在：Rosenblatt 1958《The Perceptron: A Probabilistic Model for Information
 *       Storage and Organization in the Brain》(Psychological Review 70(6)) ·
 *       Novikoff 1962《On Convergence Proofs on Perceptrons》（有限步收敛定理）之上。
 *
 * 模型：f(x) = sign(w·x + b)，标签 y ∈ {−1, +1}。Rosenblatt 误分点更新规则：
 *   若 y_i·(w·x_i + b) ≤ 0（误分或压边界）⇒ w ← w + lr·y_i·x_i，b ← b + lr·y_i。
 * 一个完整遍历（pass）零误分 ⇒ 收敛。Novikoff 定理给出硬保证：线性可分数据
 * （几何间隔 γ > 0、最大样本范数 R）上，总误分次数 ≤ (R/γ)²——感知机是少数
 * 携带严格收敛界证明的线性模型。不可分数据（如 XOR）不收敛：达到 maxPasses 后
 * 有界停止，仅报告 mistakes，不抛异常。
 * 训练按固定样本顺序、无内嵌随机性（shuffle 默认关闭）⇒ 逐位确定、可复现。
 *
 * 不变量：
 *   ① 线性可分数据：收敛且 min_i y_i(w·x_i + b) > 0（训练误差 0）
 *   ② 收敛后 sign(w·x + b) ≡ y（mismatch = 0）
 *   ③ Novikoff 1962：mistakes ≤ (R/γ)²（由已知分隔面取 γ 的下界，上界仍合法）
 *   ④ 单步更新确定性：perceptronStep ≡ 手算 w+lr·y·x / b+lr·y（误差 0）
 *   ⑤ perceptronTrain(maxPasses=1) ≡ 用 perceptronStep 逐点模拟一整轮（误差 0）
 *   ⑥ 不可分数据（XOR）：maxPasses 内有界停止、mistakes > 0、不抛异常
 */

import { dot } from '../core/linalg.js';
import { mulberry32 } from '../core/stats.js';

/**
 * 感知机单步更新（Rosenblatt 1958 规则）。
 * @param {number[]} w 当前权重（不被修改）
 * @param {number} b 当前偏置
 * @param {number[]} x 触发更新的误分样本
 * @param {number} y 样本标签，须为 +1 或 −1
 * @param {number} [lr=1] 学习率
 * @returns {{w:number[], b:number}} 更新后的 (w, b)（全新数组，入参零改动）
 */
export function perceptronStep(w, b, x, y, lr = 1) {
  if (y !== 1 && y !== -1) {
    throw new Error(`perceptronStep: 标签必须为 +1/−1，收到 ${y}`);
  }
  const wNew = w.slice();
  for (let j = 0; j < wNew.length; j++) wNew[j] += lr * y * x[j];
  return { w: wNew, b: b + lr * y };
}

/**
 * 感知机训练（按样本顺序逐个判定，收敛于首个零误分完整遍历）。
 * @param {number[][]} X 样本矩阵 n×d（行主序）
 * @param {number[]} y 标签数组，y_i ∈ {−1, +1}
 * @param {{lr?:number, maxPasses?:number, seed?:number, shuffle?:boolean}} [opts]
 *   lr       学习率（默认 1，Rosenblatt 原始取值）
 *   maxPasses 最大遍历轮数（默认 100；不可分数据靠它有界停止）
 *   seed     仅当 shuffle=true 时用于遍历顺序（默认 0）
 *   shuffle  是否每轮遍历前打乱样本顺序（默认 false，保证逐点确定性）
 * @returns {{w:number[], b:number, mistakes:number, updates:number, converged:boolean}}
 *   w, b       学到的参数
 *   mistakes   误分遭遇总次数（每次触发一次更新，即 Novikoff 定理中的 mistake 数）
 *   updates    实际执行的完整遍历轮数
 *   converged  是否存在一整轮零误分（可分数据上由 Novikoff 定理保证发生）
 */
export function perceptronTrain(X, y, { lr = 1, maxPasses = 100, seed = 0, shuffle = false } = {}) {
  const n = X.length;
  if (n === 0) throw new Error('perceptronTrain: 输入样本为空');
  const d = X[0].length;
  if (y.length !== n) throw new Error('perceptronTrain: X 与 y 长度不一致');
  let w = new Array(d).fill(0);
  let b = 0;
  let mistakes = 0;
  let updates = 0;
  let converged = false;
  const rng = mulberry32(seed);
  const order = Array.from({ length: n }, (_, i) => i);
  for (let pass = 0; pass < maxPasses; pass++) {
    if (shuffle) {
      // Fisher–Yates 洗牌（仅改变遍历顺序，不影响收敛性与 Novikoff 计数语义）
      for (let i = n - 1; i > 0; i--) {
        const j = Math.floor(rng() * (i + 1));
        const t = order[i]; order[i] = order[j]; order[j] = t;
      }
    }
    let passMistakes = 0;
    for (const i of order) {
      // 误分判据：y_i·(w·x_i + b) ≤ 0
      if (y[i] * (dot(w, X[i]) + b) <= 0) {
        const step = perceptronStep(w, b, X[i], y[i], lr);
        w = step.w;
        b = step.b;
        mistakes++;
        passMistakes++;
      }
    }
    updates++;
    if (passMistakes === 0) { converged = true; break; }
  }
  return { w, b, mistakes, updates, converged };
}

/**
 * 感知机预测：sign(w·x + b)。约定 w·x + b = 0 时判 +1（边界归正类）。
 * @param {number[][]} X 样本矩阵 n×d
 * @param {number[]} w 权重
 * @param {number} b 偏置
 * @returns {number[]} 预测标签 ∈ {−1, +1}
 */
export function perceptronPredict(X, w, b) {
  return X.map((x) => (dot(w, x) + b >= 0 ? 1 : -1));
}

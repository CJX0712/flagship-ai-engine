/**
 * 旗舰AI引擎 · Hopfield 联想记忆
 * 站在：Hopfield 1982 PNAS《Neural networks and physical systems with emergent
 *       collective computational abilities》· Amit, Gutfreund & Sompolinsky 1985
 *       《Storing Infinite Numbers of Patterns in a Spin-Glass Model of Neural
 *       Networks》（容量 c ≈ 0.138n）。
 *
 * Hebbian 存储：W = (1/n) Σμ pμ pμᵀ（对角线置 0——无自反馈），偏置 b 默认 0。
 * 能量（Lyapunov 函数）：E(s) = −½ sᵀWs − bᵀs，s ∈ {−1,+1}ⁿ。
 * 异步单单元更新 sᵢ ← sign(fieldᵢ)，fieldᵢ = Σⱼ Wᵢⱼsⱼ + bᵢ；
 *   fieldᵢ = 0 时保持原值——此时 ΔE = −fieldᵢ·Δsᵢ ≤ 0 精确成立，能量单调不增。
 * 随机 ±1 模式在 m ≤ 0.138n 时是网络的稳定不动点（少量比特噪声可收敛回原模式）。
 *
 * 不变量：
 *   ① Lyapunov 能量不增：随机轨迹每步 ΔE ≤ 0（1e-12 容差，全步检查）；
 *   ② Hebbian 存储的模式是不动点（从 pμ 出发收敛回 pμ）；
 *   ③ W 对角元 = 0（精确）；
 *   ④ W 对称（精确）；
 *   ⑤ 25% 比特噪声探针仍收敛回原模式（m ≤ 0.138n）；
 *   ⑥ 存储模式能量 ≤ 其全部单比特翻转邻居能量（局部极小，ΔE = 2pᵢ(Wp)ᵢ）。
 */

import { zeros, matvec, dot } from '../core/linalg.js';
import { mulberry32 } from '../core/stats.js';

/** 单元 i 的局域场 fieldᵢ = Σⱼ Wᵢⱼsⱼ + bᵢ */
function hopfieldField(W, b, s, i) {
  let f = b[i];
  const Wi = W[i];
  for (let j = 0; j < s.length; j++) f += Wi[j] * s[j];
  return f;
}

/**
 * Hebbian 存储：W = (1/n)Σμ pμpμᵀ，可选对角线置 0。
 * @param {number[][]} patterns 模式矩阵（m×n，取值 ±1）
 * @param {object} [opts] { zeroDiag=true }
 * @returns {{W:number[][], b:number[], n:number}}
 */
export function hopfieldTrain(patterns, { zeroDiag = true } = {}) {
  const n = patterns[0].length;
  const W = zeros(n, n);
  for (const p of patterns) {
    for (let i = 0; i < n; i++) {
      const pi = p[i];
      for (let j = 0; j < n; j++) W[i][j] += pi * p[j];
    }
  }
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) W[i][j] /= n; // 1/n 归一
  if (zeroDiag) for (let i = 0; i < n; i++) W[i][i] = 0; // 自反馈禁登
  const b = new Array(n).fill(0);
  return { W, b, n };
}

/**
 * Hopfield 能量 E(s) = −½ sᵀWs − bᵀs。
 * @param {number[][]} W 权重矩阵
 * @param {number[]} b 偏置
 * @param {number[]} s 状态（±1 向量）
 * @returns {number}
 */
export function hopfieldEnergy(W, b, s) {
  return -0.5 * dot(s, matvec(W, s)) - dot(b, s);
}

/**
 * 异步单单元更新：sᵢ ← sign(fieldᵢ)，field=0 保持原值（保证 ΔE ≤ 0）。
 * @param {number[][]} W 权重矩阵
 * @param {number[]} b 偏置
 * @param {number[]} s 当前状态
 * @param {number} i 待更新单元
 * @returns {number[]} 新状态（不改输入）
 */
export function hopfieldUpdateUnit(W, b, s, i) {
  const f = hopfieldField(W, b, s, i);
  const ns = s.slice();
  ns[i] = f > 0 ? 1 : f < 0 ? -1 : s[i];
  return ns;
}

/**
 * 异步随机序召回：每轮以种子化随机序逐单元更新，直至整轮无变化或达 maxSteps 轮。
 * @param {number[][]} W 权重矩阵
 * @param {number[]} b 偏置
 * @param {number[]} s0 初始状态（探针）
 * @param {object} [opts] { maxSteps=100, seed=42 }
 * @returns {{state:number[], steps:number, energyTrace:number[], converged:boolean}}
 */
export function hopfieldRecall(W, b, s0, { maxSteps = 100, seed = 42 } = {}) {
  const n = s0.length;
  const rng = mulberry32(seed); // Fisher–Yates 洗牌源：同输入 ⇒ 同轨迹
  let s = s0.slice();
  const energyTrace = [hopfieldEnergy(W, b, s)]; // 每步一次记录，供 ΔE≤0 全步检查
  let converged = false;
  let steps = 0;
  const order = new Array(n);
  for (let sweep = 0; sweep < maxSteps && !converged; sweep++) {
    for (let i = 0; i < n; i++) order[i] = i;
    for (let i = n - 1; i > 0; i--) { // Fisher–Yates 随机序
      const k = Math.floor(rng() * (i + 1));
      const t = order[i]; order[i] = order[k]; order[k] = t;
    }
    let changed = false;
    for (const i of order) {
      const ns = hopfieldUpdateUnit(W, b, s, i);
      if (ns[i] !== s[i]) changed = true;
      s = ns;
      energyTrace.push(hopfieldEnergy(W, b, s));
    }
    steps = sweep + 1;
    if (!changed) converged = true; // 整轮无翻转 ⇒ 到达不动点
  }
  return { state: s, steps, energyTrace, converged };
}

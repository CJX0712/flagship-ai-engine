/**
 * 旗舰AI引擎 · 强化学习：Q-learning（Watkins 1989，免模型时序差分控制）
 * 站在 Bellman 最优方程之上：Q*(s,a) = Σ p(s'|s,a)[ r + γ·max_a' Q*(s',a') ]
 * 不变量：
 *   ① TD 自一致性：收敛后 Q ≡ T_Q(Q)（Bellman 残差 → 0，不动点定理）
 *   ② 双算法互证：Q-learning 的 Q ≡ 由值迭代 V* 反推的 Q*（链式 MDP 解析闭式 γ^(n−2−s)）
 *   ③ 贪心策略 π_Q ≡ 值迭代最优策略 π*（网格世界）
 *   ④ 学习改善：后期回合平均回报 > 前期（探索 → 利用）
 */
import { mulberry32 } from '../core/stats.js';

/**
 * 贪心动作（确定性 tie-break：取最小下标，保证可复现）。
 * @param {number[][]} Q 状态-动作值表
 * @param {number} s 状态
 * @returns {number} 动作下标
 */
export function greedyAction(Q, s) {
  let best = 0;
  for (let a = 1; a < Q[s].length; a++) if (Q[s][a] > Q[s][best]) best = a;
  return best;
}

/**
 * 由值迭代 V* 反推 Q*：Q*(s,a) = Σ_k p_k·(r_k + γ·V*(ns_k))。
 * 作为免模型 Q-learning 的独立基准（双算法互证）。
 * @param {number[][][]} P P[s][a] = [[ns, p], ...]
 * @param {number[][][]} R R[s][a] = 与 P[s][a] 对齐的奖励
 * @param {number} gamma 折扣因子
 * @param {number[]} V 值迭代收敛的 V*
 * @returns {number[][]} Q*
 */
export function qFromV(P, R, gamma, V) {
  const S = P.length, A = P[0].length;
  const Q = Array.from({ length: S }, () => new Array(A).fill(0));
  for (let s = 0; s < S; s++) {
    for (let a = 0; a < A; a++) {
      let q = 0;
      const list = P[s][a];
      for (let k = 0; k < list.length; k++) {
        q += list[k][1] * (R[s][a][k] + gamma * V[list[k][0]]);
      }
      Q[s][a] = q;
    }
  }
  return Q;
}

/**
 * Q-learning 主循环（ε-贪心探索 + 时序差分更新）。
 * @param {number[][][]} P 转移核
 * @param {number[][][]} R 奖励
 * @param {{episodes?:number, alpha?:number, epsStart?:number, epsEnd?:number,
 *          gamma?:number, maxSteps?:number, seed?:number}} opts
 * @returns {{Q:number[][], policy:number[], episodeReturns:number[], gamma:number}}
 */
export function qLearning(P, R, {
  episodes = 20000, alpha = 0.2, epsStart = 1.0, epsEnd = 0.05,
  gamma = 0.9, maxSteps = 80, seed = 42,
} = {}) {
  const S = P.length, A = P[0].length;
  const rng = mulberry32(seed);
  const Q = Array.from({ length: S }, () => new Array(A).fill(0));
  const episodeReturns = [];

  for (let ep = 0; ep < episodes; ep++) {
    const eps = epsStart + (epsEnd - epsStart) * (ep / episodes); // 线性衰减
    const s0 = Math.floor(rng() * S);
    let s = s0;
    let total = 0;
    for (let t = 0; t < maxSteps; t++) {
      // ε-贪心
      const a = rng() < eps ? Math.floor(rng() * A) : greedyAction(Q, s);
      // 依 P[s][a] 采样下一状态与即时奖励
      const list = P[s][a];
      let u = rng(), acc = 0, k = 0;
      for (; k < list.length; k++) {
        acc += list[k][1];
        if (u <= acc) break;
      }
      if (k >= list.length) k = list.length - 1;
      const ns = list[k][0];
      const r = R[s][a][k];
      // 时序差分更新
      let qMax = 0;
      for (let a2 = 0; a2 < A; a2++) qMax = Math.max(qMax, Q[ns][a2]);
      Q[s][a] += alpha * (r + gamma * qMax - Q[s][a]);
      total += r;
      s = ns;
    }
    episodeReturns.push(total);
  }

  const policy = Array.from({ length: S }, (_, s) => greedyAction(Q, s));
  return { Q, policy, episodeReturns, gamma };
}

/**
 * Bellman 最优残差：max |Q(s,a) − Σ_k p_k(r_k + γ·max_a' Q(ns_k, a'))|。
 * 收敛解满足残差 → 0（压缩映射不动点，γ < 1）。
 * @param {number[][][]} P 转移核
 * @param {number[][][]} R 奖励
 * @param {number} gamma 折扣因子
 * @param {number[][]} Q 待检验的 Q 表
 * @returns {number} 最大绝对残差
 */
export function bellmanResidualQ(P, R, gamma, Q) {
  const S = P.length, A = P[0].length;
  let worst = 0;
  for (let s = 0; s < S; s++) {
    for (let a = 0; a < A; a++) {
      let tq = 0;
      const list = P[s][a];
      for (let k = 0; k < list.length; k++) {
        const ns = list[k][0];
        let qMax = 0;
        for (let a2 = 0; a2 < A; a2++) qMax = Math.max(qMax, Q[ns][a2]);
        tq += list[k][1] * (R[s][a][k] + gamma * qMax);
      }
      worst = Math.max(worst, Math.abs(Q[s][a] - tq));
    }
  }
  return worst;
}

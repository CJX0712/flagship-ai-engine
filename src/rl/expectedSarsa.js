/**
 * 旗舰AI引擎 · Expected SARSA（van Seijen et al. 2009）
 * 站在：van Seijen, Mahmood, Pilarski, Machado & Sutton 2009
 *       《A New Q-Learning Variant for On-Policy or Off-Policy? Expected SARSA》·
 *       Sutton & Barto《Reinforcement Learning: An Introduction》§6.5
 *       （Expected SARSA 介于 Q-learning 与 SARSA 之间：对行为策略求期望，方差更小）。
 *
 * 与 SARSA 的分水岭在 TD 目标：
 *   SARSA        : r + γ·Q(s', a')            （a' 由 π 真实采样）
 *   Expected SARSA: r + γ·Σ_{a'} π(a'|s')·Q(s', a')   （对 π 取期望，解析）
 * ε=0（确定性贪心策略）时 Σ_{a'} π(·|s') Q = max_{a'} Q ⇒ Expected SARSA 单步
 *   ≡ Q-learning 贪婪目标 ≡ SARSA（a' 取贪婪）单步（不变量①）。
 *
 * 复用：qLearning.greedyAction / qFromV；sarsa.sarsaEpsilonProbs；
 *       valueIteration.valueIteration / chainMDP。
 * P/R 约定 P[s][a] = [[ns,p],…] 与前述 RL 层完全一致。
 *
 * 不变量：
 *   ① 退化恒等式：ε=0 ⇒ 单步 TD 目标 ≡ Q-learning 贪婪目标（1e-15）；
 *   ② 期望目标代数：r + γ·Σ π Q ≡ r + γ·dot(π, Q[s'])（精确）；
 *   ③ GLIE 收敛：chainMDP 上收敛 Q ≡ qFromV(V*)（1e-3）；
 *   ④ 同策略期望 Bellman 残差 ‖Q − B^π_{exp} Q‖ → 0（1e-3）；
 *   ⑤ 学习改善：后期回合平均回报 > 前期（探索 → 利用）；
 *   ⑥ 同 seed 两次训练 Q 逐位一致。
 */

import { mulberry32 } from '../core/stats.js';
import { greedyAction, qFromV } from './qLearning.js';
import { sarsaEpsilonProbs } from './sarsa.js';
import { valueIteration, chainMDP } from './valueIteration.js';

/**
 * 期望下一状态动作值：Σ_{a'} π(a'|s')·Q(s', a')。
 */
export function expectedValue(Q, s2, probs) {
  let v = 0;
  for (let a = 0; a < Q[s2].length; a++) v += probs[a] * Q[s2][a];
  return v;
}

/**
 * Expected SARSA 单步更新（原地修改 Q）。
 *   δ = r + γ·Σ_{a'} π(a'|s')Q(s',a') − Q(s,a)；  Q(s,a) ← Q(s,a) + α·δ
 * @returns {{Q:number[][], delta:number, expected:number}}
 */
export function expectedSarsaStep(Q, s, a, r, s2, { alpha, gamma, eps }) {
  const probs = sarsaEpsilonProbs(Q, s2, eps);
  const exp = expectedValue(Q, s2, probs);
  const delta = r + gamma * exp - Q[s][a];
  Q[s][a] += alpha * delta;
  return { Q, delta, expected: exp };
}

/** 依 P[s][a] 采样下一状态与即时奖励（与 sarsa.js 同一采样序，保证 rng 流一致） */
function expSampleTransition(P, R, s, a, rng) {
  const list = P[s][a];
  let u = rng();
  let acc = 0;
  let k = 0;
  for (; k < list.length; k++) {
    acc += list[k][1];
    if (u <= acc) break;
  }
  if (k >= list.length) k = list.length - 1;
  return { ns: list[k][0], r: R[s][a][k] };
}

/**
 * Expected SARSA 主循环（ε 线性衰减至 epsEnd，留出纯利用阶段满足 GLIE 收敛条件）。
 * @returns {{Q:number[][], policy:number[], episodeReturns:number[], gamma:number, eps:number}}
 */
export function expectedSarsa(P, R, {
  gamma = 0.9, alpha = 0.2, eps = 0.1, epsStart = null, epsEnd = null,
  episodes = 5000, stepsPerEpisode = 40, seed = 42, epsDecayFrac = 1,
} = {}) {
  const S = P.length;
  const A = P[0].length;
  const rng = mulberry32(seed);
  const Q = Array.from({ length: S }, () => new Array(A).fill(0));
  const episodeReturns = [];
  const span = Math.max(1, Math.floor(episodes * epsDecayFrac) - 1);
  for (let ep = 0; ep < episodes; ep++) {
    const epsNow = (epsStart == null)
      ? eps
      : epsStart + ((epsEnd == null ? eps : epsEnd) - epsStart) * Math.min(1, ep / span);
    let s = Math.floor(rng() * S);
    let a = (rng() < epsNow) ? Math.floor(rng() * A) : greedyAction(Q, s);
    let total = 0;
    for (let step = 0; step < stepsPerEpisode; step++) {
      const { ns, r } = expSampleTransition(P, R, s, a, rng);
      const a2 = (rng() < epsNow) ? Math.floor(rng() * A) : greedyAction(Q, ns);
      expectedSarsaStep(Q, s, a, r, ns, { alpha, gamma, eps: epsNow });
      total += r;
      s = ns;
      a = a2;
    }
    episodeReturns.push(total);
  }
  const policy = Array.from({ length: S }, (_, s) => greedyAction(Q, s));
  return { Q, policy, episodeReturns, gamma, eps };
}

/**
 * 同策略期望 Bellman 残差：max_{s,a} |Q(s,a) − Σ_k p_k(r_k + γ·Σ_{a'}π Q(s'_k,a'))|。
 * 其中 π 为 Q 导出的 ε-贪心策略，Σ_{a'}πQ 即期望目标。Expected SARSA 收敛 ⇒ 残差 → 0。
 * @returns {number} 最大绝对残差
 */
export function expectedBellmanResidual(P, R, gamma, Q, eps) {
  const S = P.length;
  let worst = 0;
  for (let s = 0; s < S; s++) {
    for (let a = 0; a < P[s].length; a++) {
      let bq = 0;
      const list = P[s][a];
      for (let k = 0; k < list.length; k++) {
        const ns = list[k][0];
        const probs = sarsaEpsilonProbs(Q, ns, eps);
        let v = 0;
        for (let b = 0; b < Q[ns].length; b++) v += probs[b] * Q[ns][b];
        bq += list[k][1] * (R[s][a][k] + gamma * v);
      }
      worst = Math.max(worst, Math.abs(Q[s][a] - bq));
    }
  }
  return worst;
}

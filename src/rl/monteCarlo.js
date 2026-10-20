/**
 * 旗舰AI引擎 · 蒙特卡洛强化学习（Monte Carlo RL）
 * 站在：Sutton & Barto《Reinforcement Learning: An Introduction》第 5 章
 *       （免模型 MC 预测 / ε-软 MC 控制，探索起点）·
 *       Singh & Sutton 1996《Reinforcement Learning with Replacing Eligibility Traces》
 *       （MC vs TD 偏差-方差权衡的奠基对照）·
 *       本模块只用完整回合回报、绝不 bootstrap——这是与 TD 家族（Q-learning/SARSA）的分水岭。
 *
 * 回合表示：{ states:[s_0…s_T], actions:[a_0…a_T], rewards:[r_1…r_{T+1}] }，
 * rewards[t] 为离开 states[t] 后得到的即时奖励；回报 G_t = Σ_{k≥0} γ^k·rewards[t+k]。
 *
 * 复用：valueIteration.policyEvaluation / valueIteration / chainMDP；
 *       qLearning.greedyAction / qFromV。P/R 约定 P[s][a] = [[ns,p],…] 与 rl 层一致。
 *
 * 不变量：
 *   ① 回报恒等式：G_t = Σ γ^k r_{t+k} 与逐项展开手算逐位一致（精确）；
 *   ② 确定性轨迹：first-visit ≡ every-visit（每状态每回合仅一次访问，maxDiff=0）；
 *   ③ MC 预测收敛：固定策略采样 ⇒ 首访均值 ≡ policyEvaluation(P,R,γ,π)（<1e-9，两算法互证）；
 *   ④ MC 控制 GLIE：ε-软 + 探索起点 ⇒ 贪心策略 ≡ valueIteration 最优策略且 Q ≈ qFromV(V*)；
 *   ⑤ ε=0 + 探索起点：Q̂(s,a) ≡ qFromV(V*)（贪心 MC-ES 收敛到最优动作值）；
 *   ⑥ 同 seed 两次训练 Q 逐位一致（maxDiff=0）。
 */

import { mulberry32 } from '../core/stats.js';
import { greedyAction, qFromV } from './qLearning.js';
import { valueIteration, policyEvaluation } from './valueIteration.js';

/**
 * 从 t 起的折扣回报：G_t = Σ_{k=0}^{T−t} γ^k·rewards[t+k]。
 * @param {number[]} rewards rewards[t] 为离开 states[t] 后的即时奖励
 * @param {number} gamma 折扣因子
 * @param {number} t 起始时刻
 */
export function mcReturn(rewards, gamma, t) {
  let g = 0, disc = 1;
  for (let k = t; k < rewards.length; k++) {
    g += disc * rewards[k];
    disc *= gamma;
  }
  return g;
}

/** 依 P[s][a] 采样下一状态与即时奖励（与 sarsa.js 同一采样序，保证 rng 流一致） */
function mcSampleTransition(P, R, s, a, rng) {
  const list = P[s][a];
  const u = rng();
  let acc = 0, k = 0;
  for (; k < list.length; k++) {
    acc += list[k][1];
    if (u <= acc) break;
  }
  if (k >= list.length) k = list.length - 1;
  return { ns: list[k][0], r: R[s][a][k] };
}

/**
 * 采集一个回合（至多 maxSteps 步）。
 * 行为策略四选一（优先级递减）：probs（S×A 概率矩阵）> policy（固定动作数组）> Q（ε-贪心）> 均匀随机；
 * startState 为探索起点。
 * @returns {{states:number[], actions:number[], rewards:number[]}}
 */
export function mcEpisode(P, R, {
  gamma = 0.9, eps = 0, rng, policy = null, probs = null, Q = null, startState = null, maxSteps = 200,
} = {}) {
  const S = P.length, A = P[0].length;
  const gen = rng || mulberry32(42);
  const states = [], actions = [], rewards = [];
  let s = startState == null ? Math.floor(gen() * S) : startState;
  for (let step = 0; step < maxSteps; step++) {
    let a;
    if (probs) {
      const u = gen();
      let acc = 0;
      a = A - 1;
      for (let k = 0; k < A; k++) { acc += probs[s][k]; if (u <= acc) { a = k; break; } }
    } else if (policy) a = policy[s];
    else if (gen() < eps) a = Math.floor(gen() * A);
    else if (Q) a = greedyAction(Q, s);
    else a = Math.floor(gen() * A);
    const { ns, r } = mcSampleTransition(P, R, s, a, gen);
    states.push(s); actions.push(a); rewards.push(r);
    s = ns;
  }
  return { states, actions, rewards };
}

/** 从回合集合推断状态数 */
function inferStates(episodes) {
  let S = 0;
  for (const ep of episodes) for (const s of ep.states) if (s + 1 > S) S = s + 1;
  return S;
}

/**
 * 首访蒙特卡洛预测：每回合中每状态仅首次访问计入回报均值。
 * @returns {number[]} V̂
 */
export function mcPredictionFirstVisit(episodes, gamma, { nStates = 0 } = {}) {
  const S = nStates || inferStates(episodes);
  const sums = new Array(S).fill(0), cnts = new Array(S).fill(0);
  for (const ep of episodes) {
    const seen = new Set();
    for (let t = 0; t < ep.states.length; t++) {
      const s = ep.states[t];
      if (seen.has(s)) continue;
      seen.add(s);
      sums[s] += mcReturn(ep.rewards, gamma, t);
      cnts[s]++;
    }
  }
  return sums.map((v, i) => (cnts[i] ? v / cnts[i] : 0));
}

/**
 * 每访蒙特卡洛预测：每次访问都计入（有偏但方差更小）。
 * @returns {number[]} V̂
 */
export function mcPredictionEveryVisit(episodes, gamma, { nStates = 0 } = {}) {
  const S = nStates || inferStates(episodes);
  const sums = new Array(S).fill(0), cnts = new Array(S).fill(0);
  for (const ep of episodes) {
    for (let t = 0; t < ep.states.length; t++) {
      const s = ep.states[t];
      sums[s] += mcReturn(ep.rewards, gamma, t);
      cnts[s]++;
    }
  }
  return sums.map((v, i) => (cnts[i] ? v / cnts[i] : 0));
}

/**
 * 在策略蒙特卡洛控制（ε-软 + 探索起点 + 首访回报均值）。
 * GLIE：epsStart→epsEnd 线性衰减（epsDecayFrac 控制衰减窗口），末端留纯利用阶段。
 * @param {number[][]} [opts.Q0] 初始动作值（默认全零）；灌入 Q* 可验证不动点性质
 * @returns {{Q:number[][], policy:number[], episodeReturns:number[]}}
 */
export function mcControl(P, R, {
  gamma = 0.9, eps = 0.1, epsStart = null, epsEnd = null, epsDecayFrac = 1,
  episodes = 4000, maxSteps = 200, seed = 42, Q0 = null,
} = {}) {
  const S = P.length, A = P[0].length;
  const rng = mulberry32(seed);
  const Q = Q0 ? Q0.map((row) => row.slice()) : Array.from({ length: S }, () => new Array(A).fill(0));
  const sums = Array.from({ length: S }, () => new Array(A).fill(0));
  const cnts = Array.from({ length: S }, () => new Array(A).fill(0));
  const episodeReturns = [];
  const span = Math.max(1, Math.floor(episodes * epsDecayFrac) - 1);
  for (let ep = 0; ep < episodes; ep++) {
    const epsNow = (epsStart == null)
      ? eps
      : epsStart + ((epsEnd == null ? eps : epsEnd) - epsStart) * Math.min(1, ep / span);
    const { states, actions, rewards } = mcEpisode(P, R, {
      gamma, eps: epsNow, rng, Q, maxSteps, startState: Math.floor(rng() * S),
    });
    episodeReturns.push(rewards.reduce((a, b) => a + b, 0));
    let g = 0;
    const seen = new Set();
    for (let t = states.length - 1; t >= 0; t--) {
      g = gamma * g + rewards[t];
      const s = states[t], a = actions[t];
      const key = s * A + a;
      if (seen.has(key)) continue;
      seen.add(key);
      sums[s][a] += g;
      cnts[s][a]++;
      Q[s][a] = sums[s][a] / cnts[s][a];
    }
  }
  const policy = Array.from({ length: S }, (_, s) => greedyAction(Q, s));
  return { Q, policy, episodeReturns };
}

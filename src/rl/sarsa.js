/**
 * 旗舰AI引擎 · SARSA / SARSA(λ)（同策略时序差分控制）
 * 站在：Rummery & Niranjan 1994《On-Line Q-Learning Using Connectionist Systems》
 *       （原名 Modified Q-learning）· Sutton 1996（更名为 SARSA）
 *       · Sutton & Barto《Reinforcement Learning: An Introduction》§6.4 / §12.7
 *       · Singh & Sutton 1996（替换迹 replacing traces）。
 *
 * 与 Q-learning（免模型 **离策略**）的分水岭在 TD 目标：
 *   Q-learning : Q(s,a) ← Q(s,a) + α[r + γ·max_{a'} Q(s',a') − Q(s,a)]   （贪婪目标）
 *   SARSA      : Q(s,a) ← Q(s,a) + α[r + γ·Q(s',a')      − Q(s,a)]   （a' ~ π(·|s')）
 * SARSA 是 **同策略（on-policy）**：目标里的 a' 由行为策略 ε-贪心真实采样，
 * 因此收敛到 Q^{π_ε} 而非 Q*；ε → 0（GLIE）时 Q^{π_ε} → Q*。
 *
 * SARSA(λ)：用资格迹把单步回报「回溯」给历史状态-动作对
 *   迹置位 e(s,a) ← 1（替换迹：e(s,·) ← 0 后 e(s,a) ← 1；累积迹：e(s,a) += 1）
 *   全表更新 Q ← Q + α·δ·e，δ = r + γQ(s',a') − Q(s,a)
 *   迹衰减 e ← γλ·e
 * λ = 0 时迹在一步内衰减为 0 ⇒ SARSA(λ) **逐位** 退化为 SARSA（不变量 ②）。
 *
 * P/R 约定与 valueIteration.js / qLearning.js 完全一致：
 *   P[s][a] = [[ns, p], …]，R[s][a] = 与 P[s][a] 逐项对齐的奖励。
 *
 * 不变量：
 *   ① 退化恒等式：ε=0（贪心）+ 确定性 MDP ⇒ SARSA 单步 ≡ Q-learning 单步（1e-15）；
 *   ② λ=0 时 SARSA(λ) 与 SARSA 逐位相等（0）；
 *   ③ GLIE 收敛：chainMDP 上收敛 Q ≡ qFromV(V*)（1e-4）；
 *   ④ 在策略 Bellman 残差：‖Q − B^π Q‖ → 0（π 为 ε-贪心策略）；
 *   ⑤ 资格迹恒等式：访问 (s,a) 后 e(s,a) = 1（替换迹）且按 γλ 衰减；
 *   ⑥ 同 seed 两次训练 Q 逐位一致。
 */

import { mulberry32 } from '../core/stats.js';
import { greedyAction } from './qLearning.js';

/** 零矩阵（S×A）：资格迹与 Polyak 累加器共用 */
function sarsaZeros(S, A) {
  const M = new Array(S);
  for (let s = 0; s < S; s++) M[s] = new Array(A).fill(0);
  return M;
}

/**
 * ε-贪心策略的概率向量（确定性 tie-break：贪心取最小下标动作）。
 * @param {number[][]} Q 状态-动作值表
 * @param {number} s 状态
 * @param {number} eps 探索率
 * @returns {number[]} π(a|s)：贪心动作 1−ε+ε/A，其余 ε/A
 */
export function sarsaEpsilonProbs(Q, s, eps) {
  const A = Q[s].length;
  const greedy = greedyAction(Q, s);
  const probs = new Array(A).fill(eps / A);
  probs[greedy] += 1 - eps;
  return probs;
}

/**
 * ε-贪心采样行为策略（同策略 TD 的核心：a' 必须真从 π 采样，而非取 max）。
 * @param {number[][]} Q 状态-动作值表
 * @param {number} s 状态
 * @param {number} eps 探索率
 * @param {function} rng 随机数发生器
 * @returns {{a:number, probs:number[]}} 采样动作与完整策略概率
 */
export function epsilonGreedyAction(Q, s, eps, rng) {
  const probs = sarsaEpsilonProbs(Q, s, eps);
  const u = rng();
  let acc = 0;
  let chosen = probs.length - 1;
  for (let a = 0; a < probs.length; a++) {
    acc += probs[a];
    if (u <= acc) { chosen = a; break; }
  }
  return { a: chosen, probs };
}

/** 依 P[s][a] 采样下一状态与即时奖励（与 qLearning.js 同一采样序，保证 rng 流一致） */
function sarsaSampleTransition(P, R, s, a, rng) {
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
 * SARSA 单步更新（原地修改 Q）。
 *   Q(s,a) ← Q(s,a) + α[r + γQ(s',a') − Q(s,a)]
 * @param {number[][]} Q 状态-动作值表（原地修改）
 * @param {number} s 当前状态
 * @param {number} a 当前动作
 * @param {number} r 即时奖励
 * @param {number} s2 下一状态
 * @param {number} a2 下一动作（由行为策略采样，同策略之魂）
 * @param {{alpha:number, gamma:number}} hp 学习率与折扣
 * @returns {{Q:number[][], delta:number}} 同一 Q 引用与 TD 误差 δ
 */
export function sarsaStep(Q, s, a, r, s2, a2, { alpha, gamma }) {
  const delta = r + gamma * Q[s2][a2] - Q[s][a];
  Q[s][a] += alpha * delta;
  return { Q, delta };
}

/**
 * 资格迹置位：替换迹 e(s,·) ← 0 且 e(s,a) ← 1；累积迹 e(s,a) += 1。
 * 原地修改 e。
 * @param {number[][]} e 迹矩阵（S×A）
 * @param {number} s 状态
 * @param {number} a 动作
 * @param {{replacing?:boolean}} [opts] 默认替换迹
 * @returns {number[][]} 同一 e 引用
 */
export function sarsaTraceSet(e, s, a, { replacing = true } = {}) {
  if (replacing) {
    const row = e[s];
    for (let b = 0; b < row.length; b++) row[b] = 0;
  }
  e[s][a] += 1;
  return e;
}

/**
 * 资格迹衰减 e ← γλ·e（原地）。λ = 0 时精确清零——SARSA(λ) 退化为 SARSA 的代数原因。
 * @param {number[][]} e 迹矩阵
 * @param {number} gamma 折扣
 * @param {number} lambda 迹衰减参数
 * @returns {number[][]} 同一 e 引用
 */
export function sarsaTraceDecay(e, gamma, lambda) {
  const factor = gamma * lambda;
  for (let s = 0; s < e.length; s++) {
    const row = e[s];
    for (let a = 0; a < row.length; a++) {
      row[a] = factor === 0 ? 0 : row[a] * factor;
    }
  }
  return e;
}

/**
 * ε 调度：给定 epsStart/epsEnd 时在前 epsDecayFrac 比例的回合内线性衰减到 epsEnd，
 * 之后恒定在 epsEnd（留出「纯利用」阶段让 TD 迭代几何收敛——GLIE 的收敛条件）。
 */
function sarsaEpsAt(ep, episodes, eps, epsStart, epsEnd, epsDecayFrac = 1) {
  if (epsStart === null || epsStart === undefined) return eps;
  const e0 = epsStart;
  const e1 = (epsEnd === null || epsEnd === undefined) ? eps : epsEnd;
  const span = Math.max(1, Math.floor(episodes * epsDecayFrac));
  const frac = span > 1 ? Math.min(1, ep / (span - 1)) : 1;
  return e0 + (e1 - e0) * frac;
}

/** 尾部 Polyak（Polyak–Juditsky）平均累加器：抑制 TD 噪声的 √α 尾巴 */
function sarsaAccumulate(acc, Q, S, A) {
  for (let s = 0; s < S; s++) {
    const accRow = acc[s], qRow = Q[s];
    for (let a = 0; a < A; a++) accRow[a] += qRow[a];
  }
}

/** 平均表归一化（无样本时返回 null） */
function sarsaAverage(acc, S, A, count) {
  if (count <= 0) return null;
  const out = new Array(S);
  for (let s = 0; s < S; s++) {
    out[s] = new Array(A);
    for (let a = 0; a < A; a++) out[s][a] = acc[s][a] / count;
  }
  return out;
}

/** α 调度（Robbins–Monro 风格）：alphaDecay > 0 时 α_t = α₀/(1 + t·decay) */
function sarsaAlphaAt(t, alpha, alphaDecay) {
  return alphaDecay > 0 ? alpha / (1 + t * alphaDecay) : alpha;
}

/**
 * SARSA 主循环（单步、无迹）。λ > 0 时自动转交 sarsaLambda。
 * @param {number[][][]} P 转移核 P[s][a] = [[ns, p], …]
 * @param {number[][][]} R 奖励（与 P 逐项对齐）
 * @param {{gamma?:number, alpha?:number, eps?:number, epsStart?:number, epsEnd?:number,
 *          episodes?:number, stepsPerEpisode?:number, seed?:number, lambda?:number,
 *          alphaDecay?:number, replacing?:boolean, epsDecayFrac?:number,
 *          average?:boolean, avgFrom?:number}} opts
 *   epsDecayFrac：ε 衰减发生在其前 ×× 比例的回合内；average/avgFrom：尾部 Polyak 平均。
 * @returns {{Q:number[][], Qavg:number[][]|null, policy:number[],
 *            episodeReturns:number[], gamma:number, eps:number, lambda:number}}
 */
export function sarsa(P, R, {
  gamma = 0.9, alpha = 0.2, eps = 0.1, epsStart = null, epsEnd = null,
  episodes = 5000, stepsPerEpisode = 40, seed = 42, lambda = 0,
  alphaDecay = 0, replacing = true, epsDecayFrac = 1, average = false, avgFrom = 0.5,
} = {}) {
  if (lambda > 0) {
    return sarsaLambda(P, R, {
      gamma, alpha, eps, epsStart, epsEnd, episodes, stepsPerEpisode,
      seed, lambda, alphaDecay, replacing, epsDecayFrac, average, avgFrom,
    });
  }
  const S = P.length;
  const A = P[0].length;
  const rng = mulberry32(seed);
  const Q = Array.from({ length: S }, () => new Array(A).fill(0));
  const episodeReturns = [];
  const avgStart = Math.floor(episodes * avgFrom);
  const acc = average ? sarsaZeros(S, A) : null;
  let avgCount = 0;
  let t = 0; // 全局步计数（供 α 调度）

  for (let ep = 0; ep < episodes; ep++) {
    const epsNow = sarsaEpsAt(ep, episodes, eps, epsStart, epsEnd, epsDecayFrac);
    let s = Math.floor(rng() * S);
    let a = epsilonGreedyAction(Q, s, epsNow, rng).a;
    let total = 0;
    for (let step = 0; step < stepsPerEpisode; step++) {
      const { ns, r } = sarsaSampleTransition(P, R, s, a, rng);
      const a2 = epsilonGreedyAction(Q, ns, epsNow, rng).a;
      sarsaStep(Q, s, a, r, ns, a2, { alpha: sarsaAlphaAt(t, alpha, alphaDecay), gamma });
      total += r;
      s = ns;
      a = a2;
      t++;
    }
    if (average && ep >= avgStart) { sarsaAccumulate(acc, Q, S, A); avgCount++; }
    episodeReturns.push(total);
  }

  const policy = Array.from({ length: S }, (_, s) => greedyAction(Q, s));
  const Qavg = average ? sarsaAverage(acc, S, A, avgCount) : null;
  return { Q, Qavg, policy, episodeReturns, gamma, eps, lambda };
}

/**
 * SARSA(λ)：带资格迹的同策略 TD 控制。
 * 每步：置位 → Q ← Q + αδe（全表）→ e ← γλe。λ = 0 时逐位退化为 SARSA。
 * @param {number[][][]} P 转移核
 * @param {number[][][]} R 奖励
 * @param {{gamma?:number, alpha?:number, eps?:number, epsStart?:number, epsEnd?:number,
 *          episodes?:number, stepsPerEpisode?:number, seed?:number, lambda?:number,
 *          alphaDecay?:number, replacing?:boolean, epsDecayFrac?:number,
 *          average?:boolean, avgFrom?:number}} opts
 * @returns {{Q:number[][], Qavg:number[][]|null, policy:number[],
 *            episodeReturns:number[], gamma:number, eps:number, lambda:number}}
 */
export function sarsaLambda(P, R, {
  gamma = 0.9, alpha = 0.2, eps = 0.1, epsStart = null, epsEnd = null,
  episodes = 5000, stepsPerEpisode = 40, seed = 42, lambda = 0.9,
  alphaDecay = 0, replacing = true, epsDecayFrac = 1, average = false, avgFrom = 0.5,
} = {}) {
  const S = P.length;
  const A = P[0].length;
  const rng = mulberry32(seed);
  const Q = Array.from({ length: S }, () => new Array(A).fill(0));
  const episodeReturns = [];
  const avgStart = Math.floor(episodes * avgFrom);
  const acc = average ? sarsaZeros(S, A) : null;
  let avgCount = 0;
  let t = 0;

  for (let ep = 0; ep < episodes; ep++) {
    const epsNow = sarsaEpsAt(ep, episodes, eps, epsStart, epsEnd, epsDecayFrac);
    let e = sarsaZeros(S, A); // 每回合重置迹（回合式任务的标准做法）
    let s = Math.floor(rng() * S);
    let a = epsilonGreedyAction(Q, s, epsNow, rng).a;
    let total = 0;
    for (let step = 0; step < stepsPerEpisode; step++) {
      const { ns, r } = sarsaSampleTransition(P, R, s, a, rng);
      const a2 = epsilonGreedyAction(Q, ns, epsNow, rng).a;
      const alphaNow = sarsaAlphaAt(t, alpha, alphaDecay);
      const delta = r + gamma * Q[ns][a2] - Q[s][a];
      sarsaTraceSet(e, s, a, { replacing });
      for (let i = 0; i < S; i++) {
        for (let b = 0; b < A; b++) {
          if (e[i][b] !== 0) Q[i][b] += alphaNow * delta * e[i][b];
        }
      }
      sarsaTraceDecay(e, gamma, lambda);
      total += r;
      s = ns;
      a = a2;
      t++;
    }
    if (average && ep >= avgStart) { sarsaAccumulate(acc, Q, S, A); avgCount++; }
    episodeReturns.push(total);
  }

  const policy = Array.from({ length: S }, (_, s) => greedyAction(Q, s));
  const Qavg = average ? sarsaAverage(acc, S, A, avgCount) : null;
  return { Q, Qavg, policy, episodeReturns, gamma, eps, lambda };
}

/**
 * 在策略 Bellman 残差：max_{s,a} |Q(s,a) − B^π Q(s,a)|，
 *   B^πQ(s,a) = Σ_k p_k [ r_k + γ Σ_{a'} π(a'|s_k') Q(s_k', a') ]
 * 其中 π 为 Q 导出的 ε-贪心策略。SARSA 收敛 ⇒ 残差 → 0（同策略 TD 不动点）。
 * @param {number[][][]} P 转移核
 * @param {number[][][]} R 奖励
 * @param {number} gamma 折扣
 * @param {number[][]} Q 待检验的 Q 表
 * @param {number} eps 行为策略探索率
 * @returns {number} 最大绝对残差
 */
export function sarsaBellmanResidual(P, R, gamma, Q, eps) {
  const S = P.length;
  let worst = 0;
  for (let s = 0; s < S; s++) {
    const A = P[s].length;
    for (let a = 0; a < A; a++) {
      let bq = 0;
      const list = P[s][a];
      for (let k = 0; k < list.length; k++) {
        const ns = list[k][0];
        const probs = sarsaEpsilonProbs(Q, ns, eps);
        let vQ = 0;
        for (let b = 0; b < probs.length; b++) vQ += probs[b] * Q[ns][b];
        bq += list[k][1] * (R[s][a][k] + gamma * vQ);
      }
      worst = Math.max(worst, Math.abs(Q[s][a] - bq));
    }
  }
  return worst;
}

/**
 * 旗舰AI引擎 · 策略梯度 / REINFORCE（Williams 1992；Sutton et al. 2000）
 * 站在：Williams 1992《Simple Statistical Gradient-Following Algorithms for
 *       Connectionist Reinforcement Learning》（REINFORCE）
 *       · Sutton, McAllester, Singh & Mansour 2000 NIPS《Policy Gradient Methods
 *       for Reinforcement Learning with Function Approximation》（策略梯度定理）
 *       · Sutton & Barto《Reinforcement Learning》§13（蒙特卡洛策略梯度）
 *       · Weaver & Tao 2001《The Optimal Reward Baseline》（基线不改变梯度期望）。
 *
 * 参数化：**tabular softmax 策略**——每个状态 s 有独立参数向量 θ_s（长度 = 动作数）：
 *   π(a|s) = softmax(θ_s)_a
 *   ∇_θs log π(a|s) = e_a − π(·|s)          （softmax 的 score-function 形式）
 *   ∇_θs' log π(a|s) = 0  (s' ≠ s)           （tabular：状态间参数不共享）
 * REINFORCE 更新：θ += lr · γ^t · (G_t − b) · ∇log π(a_t|s_t)，G_t = Σ_{k≥t} γ^{k−t} r_k。
 *
 * 不变量（核心是精确代数恒等式，拒绝「看起来收敛」）：
 *   ① Score function 恒等式（精确）：Σ_a π(a|s)·∇log π(a|s) = 0（softmax 下逐分量 = 0，maxAbs < 1e-14）；
 *   ② 策略梯度定理（解析 ≡ 数值）：多臂老虎机 J(θ) = Σ_a π(a)r_a 的解析梯度
 *      Σ_a π(a)∇logπ(a)r_a ≡ J 对 θ 的中心差分（maxRelErr < 1e-6）；
 *   ③ 基线不改变梯度期望（精确）：由 ① 知 E[∇logπ]·b = 0 ⇒ 带/不带基线的**全路径枚举**期望梯度逐位一致（maxAbs < 1e-12）；
 *   ④ 采样梯度期望 ≡ 解析梯度：轨迹枚举（O(A^T) 全路径）≡ 动态规划（分层 Q_t × 折扣状态访问）
 *      两条独立推导的期望更新量一致（relErr < 1e-6）；
 *   ⑤ 概率单纯形：Σ_a π(a|s) = 1 且 π(a|s) > 0（softmax 保证，maxAbs < 1e-15）；
 *   ⑥ 收敛性（辅助判据）：chainMDP 上训练足够回合后贪心策略 ≡ valueIteration 最优策略；
 *   ⑦ 同 seed 一致：两次训练 θ 逐位一致（maxDiff = 0）。
 */

import { zeros } from '../core/linalg.js';
import { softmax } from '../core/numerical.js';
import { mulberry32 } from '../core/stats.js';

/**
 * tabular softmax 策略：π = [softmax(θ_0), softmax(θ_1), …]。
 * @param {number[][]} theta 参数矩阵（S×A）
 * @returns {number[][]} 每状态的动作概率
 */
export function softmaxPolicy(theta) {
  return theta.map((row) => softmax(row));
}

/**
 * ∇_{θ_s} log π(a|s) = e_a − π(·|s)（softmax 的 score-function 形式）。
 * @param {number[][]} theta 参数矩阵
 * @param {number} s 状态
 * @param {number} a 动作
 * @returns {number[]} 长度 A 的梯度向量（其余状态的分量恒为 0，由 tabular 结构保证）
 */
export function policyLogGrad(theta, s, a) {
  const probs = softmax(theta[s]);
  const g = new Array(probs.length);
  for (let b = 0; b < probs.length; b++) g[b] = (b === a ? 1 : 0) - probs[b];
  return g;
}

/** 依概率向量采样一个下标（与 qLearning/sarsa 同一采样序：u ≤ 累积概率） */
function pgSampleCategorical(probs, rng) {
  const u = rng();
  let acc = 0;
  for (let i = 0; i < probs.length; i++) {
    acc += probs[i];
    if (u <= acc) return i;
  }
  return probs.length - 1;
}

/** 依 P[s][a] 采样下一状态与奖励 */
function pgSampleTransition(P, R, s, a, rng) {
  const list = P[s][a];
  const u = rng();
  let acc = 0;
  let k = 0;
  for (; k < list.length; k++) {
    acc += list[k][1];
    if (u <= acc) break;
  }
  if (k >= list.length) k = list.length - 1;
  return { ns: list[k][0], r: R[s][a][k] };
}

/** 逐状态去均值（softmax 平移不变：不改变策略，只防 θ 数值漂移导致 exp 溢出） */
function pgRecenter(theta) {
  for (const row of theta) {
    let mu = 0;
    for (let i = 0; i < row.length; i++) mu += row[i];
    mu /= row.length;
    for (let i = 0; i < row.length; i++) row[i] -= mu;
  }
  return theta;
}

/**
 * 按当前策略采样一条轨迹（长度 maxSteps）。
 * @param {number[][][]} P 转移核 P[s][a] = [[ns,p],…]
 * @param {number[][][]} R 奖励（与 P 逐项对齐）
 * @param {number[][]} theta 参数矩阵
 * @param {{gamma?:number, rng?:function, maxSteps?:number, startState?:number}} opts
 *   gamma 仅用于回报折算（在 reinforce 内使用），此处保留以保证接口一致。
 * @returns {{states:number[], actions:number[], rewards:number[]}}
 */
export function reinforceEpisode(P, R, theta, { gamma = 0.9, rng, maxSteps = 20, startState = null } = {}) {
  const S = P.length;
  const policy = softmaxPolicy(theta);
  let s = startState === null ? Math.floor(rng() * S) : startState;
  const states = new Array(maxSteps);
  const actions = new Array(maxSteps);
  const rewards = new Array(maxSteps);
  for (let t = 0; t < maxSteps; t++) {
    const a = pgSampleCategorical(policy[s], rng);
    const { ns, r } = pgSampleTransition(P, R, s, a, rng);
    states[t] = s;
    actions[t] = a;
    rewards[t] = r;
    s = ns;
  }
  return { states, actions, rewards, gamma };
}

/**
 * REINFORCE 主循环（蒙特卡洛策略梯度，可选基线）。
 * @param {number[][][]} P 转移核
 * @param {number[][][]} R 奖励
 * @param {{gamma?:number, lr?:number, episodes?:number, maxSteps?:number, seed?:number,
 *          theta0?:number[][], baseline?:number|string|null, startState?:number|null}} opts
 *   baseline：常数基线（null 表示无）；'mean' 表示用历史回报的**运行均值**作基线。
 * @returns {{theta:number[][], policy:number[][], episodeReturns:number[]}}
 */
export function reinforce(P, R, {
  gamma = 0.9, lr = 0.05, episodes = 5000, maxSteps = 20, seed = 42,
  theta0 = null, baseline = null, startState = null,
} = {}) {
  const S = P.length;
  const A = P[0].length;
  const rng = mulberry32(seed);
  const theta = theta0 ? theta0.map((row) => row.slice()) : zeros(S, A);
  const useMeanBaseline = (baseline === 'mean' || baseline === true);
  let retSum = 0;
  let retCnt = 0;
  const episodeReturns = [];

  for (let ep = 0; ep < episodes; ep++) {
    const { states, actions, rewards } = reinforceEpisode(P, R, theta, { gamma, rng, maxSteps, startState });
    // 逆序折算回报：G_t = r_t + γ·G_{t+1}（末端 G_T = 0）
    const G = new Array(maxSteps);
    let g = 0;
    for (let t = maxSteps - 1; t >= 0; t--) {
      g = rewards[t] + gamma * g;
      G[t] = g;
    }
    episodeReturns.push(G[0]);
    let b = 0;
    if (useMeanBaseline) {
      b = retCnt > 0 ? retSum / retCnt : 0;   // 用「历史」均值，避免自引入偏差
      retSum += G[0];
      retCnt++;
    } else if (baseline !== null && baseline !== undefined) {
      b = Number(baseline);
    }
    // θ += lr·γ^t·(G_t − b)·∇logπ(a_t|s_t)
    for (let t = 0; t < maxSteps; t++) {
      const grad = policyLogGrad(theta, states[t], actions[t]);
      const w = lr * Math.pow(gamma, t) * (G[t] - b);
      const row = theta[states[t]];
      for (let c = 0; c < A; c++) row[c] += w * grad[c];
    }
    pgRecenter(theta); // 平移不变 ⇒ 策略不变，仅防数值漂移
  }

  return { theta, policy: softmaxPolicy(theta), episodeReturns };
}

/**
 * 多臂老虎机的期望回报 J(θ) = Σ_a π(a)·r_a（单状态情形）。
 * @param {number[]} thetaRow 单状态参数向量
 * @param {number[]} r 各动作的（确定性）奖励
 * @returns {number} J(θ)
 */
export function banditExpectedReturn(thetaRow, r) {
  const probs = softmax(thetaRow);
  let j = 0;
  for (let a = 0; a < probs.length; a++) j += probs[a] * r[a];
  return j;
}

/**
 * 多臂老虎机 J 的解析梯度：∂J/∂θ_b = π_b·(r_b − Σ_a π_a r_a)。
 * （由 ① 的 score-function 形式直接推出：Σ_a π_a ∇logπ_a r_a = π∘(r − J)）
 * @param {number[]} thetaRow 单状态参数向量
 * @param {number[]} r 各动作奖励
 * @returns {number[]} 解析梯度
 */
export function banditAnalyticGrad(thetaRow, r) {
  const probs = softmax(thetaRow);
  const j = banditExpectedReturn(thetaRow, r);
  const g = new Array(probs.length);
  for (let b = 0; b < probs.length; b++) g[b] = probs[b] * (r[b] - j);
  return g;
}

/**
 * 有限 horizon 的分层 Q^π（反向递推，Q_T ≡ 0）。供解析策略梯度使用。
 * @param {number[][][]} P 转移核
 * @param {number[][][]} R 奖励
 * @param {number[][]} policy 策略概率矩阵
 * @param {number} gamma 折扣
 * @param {number} horizon 回合长度 T
 * @returns {number[][][]} Q[t] 为 S×A，t = 0..T−1
 */
export function pgPolicyQ(P, R, policy, gamma, horizon) {
  const S = P.length;
  const A = P[0].length;
  const Q = new Array(horizon);
  let next = zeros(S, A); // Q_T ≡ 0
  for (let t = horizon - 1; t >= 0; t--) {
    const Qt = zeros(S, A);
    for (let s = 0; s < S; s++) {
      for (let a = 0; a < A; a++) {
        let q = 0;
        const list = P[s][a];
        for (let k = 0; k < list.length; k++) {
          const ns = list[k][0];
          let v = 0;
          for (let b = 0; b < A; b++) v += policy[ns][b] * next[ns][b];
          q += list[k][1] * (R[s][a][k] + gamma * v);
        }
        Qt[s][a] = q;
      }
    }
    Q[t] = Qt;
    next = Qt;
  }
  return Q;
}

/** 一步状态分布传播：P_{t+1}(s') = Σ_s P_t(s) Σ_a π(a|s) p(s'|s,a) */
function pgPropagate(P, policy, dist) {
  const S = P.length;
  const A = P[0].length;
  const nd = new Array(S).fill(0);
  for (let s = 0; s < S; s++) {
    if (dist[s] === 0) continue;
    for (let a = 0; a < A; a++) {
      const pa = policy[s][a] * dist[s];
      if (pa === 0) continue;
      const list = P[s][a];
      for (let k = 0; k < list.length; k++) nd[list[k][0]] += pa * list[k][1];
    }
  }
  return nd;
}

/**
 * 折扣状态访问分布 d(s) = Σ_{t=0}^{T−1} γ^t·P(s_t = s)（前向传播，精确无采样）。
 * @param {number[][][]} P 转移核
 * @param {number[][]} policy 策略概率矩阵
 * @param {number[]} startDist 初始状态分布（缺省均匀）
 * @param {number} gamma 折扣
 * @param {number} horizon 回合长度 T
 * @returns {number[]} d(s)
 */
export function pgVisitation(P, policy, startDist, gamma, horizon) {
  const S = P.length;
  const A = P[0].length;
  const dist = startDist.slice();
  const d = new Array(S).fill(0);
  for (let t = 0; t < horizon; t++) {
    const w = Math.pow(gamma, t);
    for (let s = 0; s < S; s++) d[s] += w * dist[s];
    const nd = pgPropagate(P, policy, dist);
    for (let s = 0; s < S; s++) dist[s] = nd[s];
  }
  return d;
}

/**
 * 解析策略梯度（动态规划路径，不变量 ④ 的一侧）：
 *   ∂J/∂θ_{s,b} = Σ_t γ^t·P(s_t=s)·π(b|s)·(Q_t(s,b) − V_t(s))
 * 与「轨迹枚举」的 REINFORCE 期望更新量是同一量的两条独立推导。
 * @param {number[][][]} P 转移核
 * @param {number[][][]} R 奖励
 * @param {number[][]} theta 参数矩阵
 * @param {{gamma?:number, horizon?:number, startDist?:number[]}} [opts]
 * @returns {number[][]} 梯度矩阵（S×A）
 */
export function policyGradientAnalytic(P, R, theta, { gamma = 0.9, horizon = 6, startDist = null } = {}) {
  const S = P.length;
  const A = P[0].length;
  const policy = softmaxPolicy(theta);
  const Q = pgPolicyQ(P, R, policy, gamma, horizon);
  const mu = startDist || new Array(S).fill(1 / S);
  const grad = zeros(S, A);
  const dist = mu.slice();
  for (let t = 0; t < horizon; t++) {
    const w = Math.pow(gamma, t);
    for (let s = 0; s < S; s++) {
      if (dist[s] === 0) continue;
      let v = 0;
      for (let a = 0; a < A; a++) v += policy[s][a] * Q[t][s][a];
      for (let b = 0; b < A; b++) {
        grad[s][b] += w * dist[s] * policy[s][b] * (Q[t][s][b] - v);
      }
    }
    const nd = pgPropagate(P, policy, dist);
    for (let s = 0; s < S; s++) dist[s] = nd[s];
  }
  return grad;
}

/**
 * REINFORCE 更新量的**精确期望**（全路径枚举，不变量 ③④ 的另一侧）：
 *   E[ Σ_t γ^t (G_t − b) ∇logπ(a_t|s_t) ]
 * 枚举所有动作序列（含随机的转移分支），按出现概率加权——无蒙特卡洛噪声。
 * @param {number[][][]} P 转移核
 * @param {number[][][]} R 奖励
 * @param {number[][]} theta 参数矩阵
 * @param {{gamma?:number, horizon?:number, baseline?:number, startDist?:number[]}} [opts]
 * @returns {number[][]} 期望更新量（S×A）
 */
export function reinforceExpectedUpdate(P, R, theta, {
  gamma = 0.9, horizon = 6, baseline = 0, startDist = null,
} = {}) {
  const S = P.length;
  const A = P[0].length;
  const policy = softmaxPolicy(theta);
  const T = horizon;
  const mu = startDist || new Array(S).fill(1 / S);
  const grad = zeros(S, A);
  const states = new Array(T);
  const actions = new Array(T);
  const rewards = new Array(T);

  /** 深度优先枚举：到达终点时回溯折算回报并累加梯度 */
  function dfs(s, t, prob) {
    if (t === T) {
      const G = new Array(T);
      let g = 0;
      for (let k = T - 1; k >= 0; k--) {
        g = rewards[k] + gamma * g;
        G[k] = g;
      }
      for (let k = 0; k < T; k++) {
        const gr = policyLogGrad(theta, states[k], actions[k]);
        const w = prob * Math.pow(gamma, k) * (G[k] - baseline);
        const row = grad[states[k]];
        for (let b = 0; b < A; b++) row[b] += w * gr[b];
      }
      return;
    }
    for (let a = 0; a < A; a++) {
      const pa = policy[s][a];
      if (pa <= 0) continue;
      const list = P[s][a];
      for (let k = 0; k < list.length; k++) {
        const ns = list[k][0];
        const pk = list[k][1];
        states[t] = s;
        actions[t] = a;
        rewards[t] = R[s][a][k];
        dfs(ns, t + 1, prob * pa * pk);
      }
    }
  }

  for (let s = 0; s < S; s++) {
    if (mu[s] > 0) dfs(s, 0, mu[s]);
  }
  return grad;
}

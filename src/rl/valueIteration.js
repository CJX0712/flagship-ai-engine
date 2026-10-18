/**
 * 旗舰AI引擎 · 强化学习：马尔可夫决策过程（MDP）最优求解
 * 站在 Bellman 最优性原理之上：V*(s) = max_a Σ p(s'|s,a)[r + γ·V*(s')]
 * 不变量：
 *   ① Bellman 残差收敛 → 0（不动点定理，压缩映射 γ<1）
 *   ② 贪心策略评估值 V^{π*} ≡ V*（策略评估与值迭代两个独立算法互证）
 *   ③ 小链式 MDP：值迭代 V* ≡ 2^n 全策略暴力枚举最优值 ≡ 解析闭式 γ^(n−1−s)
 */

/**
 * 值迭代（同步更新，无穷范数收敛判据）。
 * @param {number[][][]} P P[s][a] = [[s',p],…]
 * @param {number[][][]} R R[s][a] = [r(s,a,s'),…] 与 P[s][a] 对齐
 * @param {number} gamma 折扣因子
 * @returns {{V:number[], policy:number[], iterations:number, residual:number}}
 */
export function valueIteration(P, R, gamma, { tol = 1e-12, maxIter = 100000 } = {}) {
  const S = P.length, A = P[0].length;
  let V = new Array(S).fill(0);
  let policy = new Array(S).fill(0);
  let iterations = 0, residual = Infinity;
  for (let k = 0; k < maxIter; k++) {
    iterations++;
    let delta = 0;
    const Vn = new Array(S);
    for (let s = 0; s < S; s++) {
      let best = -Infinity, bestA = 0;
      for (let a = 0; a < A; a++) {
        let q = 0;
        for (let idx = 0; idx < P[s][a].length; idx++) {
          q += P[s][a][idx][1] * (R[s][a][idx] + gamma * V[P[s][a][idx][0]]);
        }
        if (q > best) { best = q; bestA = a; }
      }
      Vn[s] = best;
      policy[s] = bestA;
      delta = Math.max(delta, Math.abs(best - V[s]));
    }
    V = Vn;
    residual = delta;
    if (delta < tol) break;
  }
  return { V, policy, iterations, residual };
}

/**
 * 策略评估（给定策略，迭代求解 Bellman 期望方程 V^π = r^π + γP^πV^π）。
 * @returns {{V:number[], iterations:number, residual:number}}
 */
export function policyEvaluation(P, R, gamma, policy, { tol = 1e-12, maxIter = 100000 } = {}) {
  const S = P.length;
  let V = new Array(S).fill(0);
  let iterations = 0, residual = Infinity;
  for (let k = 0; k < maxIter; k++) {
    iterations++;
    let delta = 0;
    const Vn = new Array(S);
    for (let s = 0; s < S; s++) {
      const a = policy[s];
      let q = 0;
      for (let idx = 0; idx < P[s][a].length; idx++) {
        q += P[s][a][idx][1] * (R[s][a][idx] + gamma * V[P[s][a][idx][0]]);
      }
      Vn[s] = q;
      delta = Math.max(delta, Math.abs(q - V[s]));
    }
    V = Vn;
    residual = delta;
    if (delta < tol) break;
  }
  return { V, iterations, residual };
}

/**
 * 小链式 MDP：状态 0..n−1，动作 {LEFT=0, RIGHT=1}，确定性转移。
 * 首次进入终点 n−1 得奖励 +1，终点吸收（后续奖励 0）。
 * 闭式最优值：V*(s) = γ^(n−1−s)。
 */
export function chainMDP({ n = 6, gamma = 0.9 } = {}) {
  const P = [], R = [];
  for (let s = 0; s < n; s++) {
    const pa = [], ra = [];
    for (let a = 0; a < 2; a++) {
      let ns = a === 1 ? Math.min(s + 1, n - 1) : Math.max(s - 1, 0);
      if (s === n - 1) ns = n - 1;                    // 终点吸收
      const r = s !== n - 1 && ns === n - 1 ? 1 : 0;  // 首次进入终点 +1
      pa.push([[ns, 1]]);
      ra.push([r]);
    }
    P.push(pa);
    R.push(ra);
  }
  return { P, R, S: n, A: 2, gamma };
}

/**
 * 暴力枚举链式 MDP 全部 2^n 个确定性策略，返回最优值与最优策略。
 * 轨迹模拟评估（确定性 → 单路径；环检测：进入无奖励循环则剩余值为 0）。
 */
export function bruteForceChain({ n = 6, gamma = 0.9 } = {}) {
  const { P, R } = chainMDP({ n, gamma });
  let bestV = null, bestPolicy = null;
  for (let mask = 0; mask < (1 << n); mask++) {
    const policy = [];
    for (let s = 0; s < n; s++) policy.push((mask >> s) & 1);
    const V = new Array(n).fill(0);
    for (let s = 0; s < n; s++) {
      const seen = new Set([s]);
      let cur = s, val = 0, disc = 1;
      for (let step = 0; step < 4 * n + 4; step++) {
        const a = policy[cur];
        const ns = P[cur][a][0][0];
        val += disc * R[cur][a][0];
        disc *= gamma;
        cur = ns;
        if (cur === n - 1 && R[cur][policy[cur]][0] === 0 && seen.has(cur)) { break; }
        if (seen.has(cur)) break;
        seen.add(cur);
      }
      V[s] = val;
    }
    const total = V.reduce((x, y) => x + y, 0);
    if (!bestV || total > bestV.total) { bestV = { V, total }; bestPolicy = policy; }
  }
  return { V: bestV.V, policy: bestPolicy, total: bestV.total };
}

/**
 * 4×4 随机网格世界：动作 {上,下,左,右}；slip 概率平分给两个垂直方向。
 * (3,3) 终点、(1,1) 陷阱均为吸收态：每步持续奖励 ±1 ⇒ 闭式 V = r/(1−γ)。
 * 其余格子每步 stepReward。
 */
export function gridworld({
  size = 4, slip = 0.1, goalReward = 1, trapReward = -1, stepReward = -0.02, gamma = 0.9,
} = {}) {
  const S = size * size, A = 4;
  const idx = (r, c) => r * size + c;
  const goal = idx(size - 1, size - 1);
  const trap = idx(1, 1);
  // 动作方向：上(行−1) 下(行+1) 左(列−1) 右(列+1)；垂直映射用于打滑
  const dirs = [[-1, 0], [1, 0], [0, -1], [0, 1]];
  const perp = { 0: [2, 3], 1: [2, 3], 2: [0, 1], 3: [0, 1] };
  const P = [], R = [];
  for (let s = 0; s < S; s++) {
    const r = Math.floor(s / size), c = s % size;
    const absorbing = s === goal ? goalReward : (s === trap ? trapReward : null);
    const pa = [], ra = [];
    for (let a = 0; a < A; a++) {
      if (absorbing !== null) { // 吸收态：原地自环，持续奖励
        pa.push([[s, 1]]);
        ra.push([absorbing]);
        continue;
      }
      const options = [];
      const probs = [1 - 2 * slip, slip, slip];
      const acts = [a, ...perp[a]];
      for (let k = 0; k < 3; k++) {
        const nr = r + dirs[acts[k]][0], nc = c + dirs[acts[k]][1];
        const inside = nr >= 0 && nr < size && nc >= 0 && nc < size;
        const ns = inside ? idx(nr, nc) : s; // 撞墙停留
        options.push([ns, probs[k]]);
      }
      // 合并同目标概率（如停留+撞墙）
      const merged = new Map();
      for (const [ns, p] of options) merged.set(ns, (merged.get(ns) || 0) + p);
      const list = [...merged.entries()].map(([ns, p]) => [ns, p]);
      const rewards = list.map(() => stepReward);
      pa.push(list);
      ra.push(rewards);
    }
    P.push(pa);
    R.push(ra);
  }
  return { P, R, S, A, gamma, size, goal, trap, goalReward, trapReward };
}

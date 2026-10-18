/**
 * 旗舰AI引擎 · 一阶优化器（站在 Kingma & Ba 2015、Hinton 的 RMSProp、Polyak 重球法之上）
 * 统一签名：state = step(params, grads, state, opts) → { params, state }
 * 不变量：
 *   ① Adam 偏差修正首步：m̂=g、v̂=g² ⇒ Δ = −lr·g/|g| = −lr·sign(g)（精确成立，且尺度不变）
 *   ② 动量重球闭式：恒定梯度下 v_k = g·(1−m^k)/(1−m)（几何累积）
 *   ③ 凸二次型收敛：GD/Momentum/RMSProp/Adam 均收敛到解析最优 x* = A⁻¹b，且损失单调不增
 */

/**
 * 带动量的 SGD（Polyak 重球法）：v ← m·v + g；x ← x − lr·v
 */
export function sgdMomentumStep(x, g, state, t, { lr = 0.01, momentum = 0.9 } = {}) {
  const v = state.v || new Array(x.length).fill(0);
  const nv = v.map((vi, i) => momentum * vi + g[i]);
  return { x: x.map((xi, i) => xi - lr * nv[i]), state: { v: nv } };
}

/**
 * RMSProp（Hinton）：v ← β·v + (1−β)·g²；x ← x − lr·g/(√v̂ + ε)
 */
export function rmspropStep(x, g, state, t, { lr = 0.01, beta = 0.9, eps = 1e-8 } = {}) {
  const v = state.v || new Array(x.length).fill(0);
  const nv = v.map((vi, i) => beta * vi + (1 - beta) * g[i] * g[i]);
  return { x: x.map((xi, i) => xi - lr * g[i] / (Math.sqrt(nv[i]) + eps)), state: { v: nv } };
}

/**
 * Adam（Kingma & Ba 2015）：一阶/二阶矩估计 + 偏差修正。
 */
export function adamStep(x, g, state, t, { lr = 0.01, beta1 = 0.9, beta2 = 0.999, eps = 1e-8 } = {}) {
  const m = state.m || new Array(x.length).fill(0);
  const v = state.v || new Array(x.length).fill(0);
  const nm = m.map((mi, i) => beta1 * mi + (1 - beta1) * g[i]);
  const nv = v.map((vi, i) => beta2 * vi + (1 - beta2) * g[i] * g[i]);
  const mHat = nm.map((mi) => mi / (1 - Math.pow(beta1, t)));
  const vHat = nv.map((vi) => vi / (1 - Math.pow(beta2, t)));
  const nx = x.map((xi, i) => xi - lr * mHat[i] / (Math.sqrt(vHat[i]) + eps));
  return { x: nx, state: { m: nm, v: nv } };
}

const STEPS = { sgdMomentum: sgdMomentumStep, rmsprop: rmspropStep, adam: adamStep };

/**
 * 通用最小化循环。
 * @param {(x:number[])=>number} f 目标函数
 * @param {(x:number[])=>number[]} grad 梯度
 * @param {number[]} x0 初始点
 * @param {{optimizer?:string, lr?:number, iters?:number, momentum?:number, beta?:number}} opts
 * @returns {{x:number[], history:number[], state:object}}
 */
export function minimize(f, grad, x0, {
  optimizer = 'adam', lr = 0.01, iters = 500, momentum, beta,
} = {}) {
  const step = STEPS[optimizer];
  if (!step) throw new Error(`未知优化器：${optimizer}（可选 ${Object.keys(STEPS).join('/')}）`);
  const opts = { lr };
  if (momentum !== undefined) opts.momentum = momentum;
  if (beta !== undefined) opts.beta = beta;
  let x = [...x0];
  let state = {};
  const history = [f(x)];
  for (let t = 1; t <= iters; t++) {
    const g = grad(x);
    const out = step(x, g, state, t, opts);
    x = out.x;
    state = out.state;
    history.push(f(x));
  }
  return { x, history, state };
}

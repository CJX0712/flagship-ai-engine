/**
 * 旗舰AI引擎 · 学习率调度器
 * 站在：Vaswani et al. 2017（warmup + 逆平方根，"Noam" 调度）
 *       Loshchilov & Hutter 2016（SGDR 余弦退火）· 经典阶梯/指数衰减
 * 全部提供闭式公式，调度曲线本身即被不变量钉死（端点/峰值/连续性/单调性）。
 */

/**
 * Warmup + 逆平方根（Vaswani et al. 2017）：
 * lr(s) = d_model^{-0.5} · min(s^{-0.5}, s·W^{-1.5})，s 从 1 起（1-based 避免除零）。
 * 结构：s < W 时线性上升（s·W^{-1.5} 支），s ≥ W 时按 s^{-0.5} 衰减。
 * 不变量：两支在 s = W 处精确连续，且峰值恰在 s = W，lr_peak = d_model^{-0.5}·W^{-0.5}。
 */
export function warmupInverseSqrt(step, { warmupSteps = 4000, dModel = 512 } = {}) {
  const s = Math.max(1, step | 0);
  return Math.min(s ** -0.5, s * warmupSteps ** -1.5) * dModel ** -0.5;
}

/** 线性 warmup：lr(s) = base·min(1, s/W)（到 W 封顶） */
export function linearWarmup(step, { warmupSteps = 4000, base = 1 } = {}) {
  const s = Math.max(1, step | 0);
  return base * Math.min(1, s / warmupSteps);
}

/**
 * 余弦退火（SGDR，Loshchilov & Hutter 2016）：
 * lr(t) = η_min + ½(η_max − η_min)(1 + cos(πt/T))，t ∈ [0, T]。
 * 端点闭式：lr(0) = η_max（cos 0 = 1 精确）、lr(T) = η_min（cos π = −1 精确）、lr(T/2) = 中点。
 */
export function cosineAnnealing(step, { T = 1000, etaMax = 1, etaMin = 0 } = {}) {
  const t = Math.max(0, Math.min(T, step));
  return etaMin + 0.5 * (etaMax - etaMin) * (1 + Math.cos(Math.PI * t / T));
}

/** 阶梯衰减：lr(t) = base·γ^{⌊t/stepSize⌋}（段内恒定，整数倍处精确掉落） */
export function stepDecay(step, { stepSize = 30, gamma = 0.1, base = 1 } = {}) {
  const t = Math.max(0, step | 0);
  return base * gamma ** Math.floor(t / stepSize);
}

/** 指数衰减：lr(t) = base·γ^t（每步严格等比，lr(t+1)/lr(t) = γ） */
export function exponentialDecay(step, { gamma = 0.99, base = 1 } = {}) {
  const t = Math.max(0, step | 0);
  return base * gamma ** t;
}

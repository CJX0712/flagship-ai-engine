/**
 * 旗舰AI引擎 · 数值稳定原语
 * 与 scipy / PyTorch 同款思想：max-shift 抑制上溢、双支式抑制下溢。
 */

/**
 * logsumexp：log Σ e^{x_i} —— 数值稳定版。
 * 恒等式：logsumexp(x) = m + log Σ e^{x_i − m}，m = max(x)。
 * 验证点：logsumexp([1000, 1001]) = 1001 + log(1 + e^{-1})，普通加法直接溢出。
 */
export function logsumexp(arr) {
  let m = -Infinity;
  for (const x of arr) if (x > m) m = x;
  if (!isFinite(m)) return m; // 全 -Inf → -Inf；含 +Inf → +Inf
  let s = 0;
  for (const x of arr) s += Math.exp(x - m);
  return m + Math.log(s);
}

/** softmax：数值稳定（max-shift），输出和恒为 1 */
export function softmax(arr) {
  let m = -Infinity;
  for (const x of arr) if (x > m) m = x;
  const e = arr.map(x => Math.exp(x - m));
  let s = 0;
  for (const v of e) s += v;
  return e.map(v => v / s);
}

/** sigmoid：双支式写法，x 很负时不再 exp 溢出 */
export function sigmoid(x) {
  if (x >= 0) return 1 / (1 + Math.exp(-x));
  const z = Math.exp(x);
  return z / (1 + z);
}

/** ReLU */
export function relu(x) {
  return x > 0 ? x : 0;
}

/** 相对误差：|a − b| / max(1e-8, |a| + |b|)，梯度检验的标尺 */
export function relErr(a, b) {
  return Math.abs(a - b) / Math.max(1e-8, Math.abs(a) + Math.abs(b));
}

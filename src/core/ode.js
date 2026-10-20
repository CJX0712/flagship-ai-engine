/**
 * 旗舰AI引擎 · 常微分方程数值求解（ODE Integrators）
 * 站在：Runge 1895 / Kutta 1901（经典四级四阶 RK4）·
 *       Fehlberg 1969 与 Dormand & Prince 1980（嵌入式 RK5(4) 系数、自适应步长）·
 *       Hairer, Nørsett & Wanner《Solving ODEs I》（阶条件、局部/全局误差、稳定性理论）。
 *
 * 约定：状态统一为向量 y(t) ∈ R^d（标量用长度 1 的数组）；右端 f(t, y) 同型返回。
 * 步进器纯函数：返回新数组，不修改入参（供梯度/不变量安全复用）。
 *
 * 不变量（精确代数恒等式 + 定理背书的阶条件）：
 *   ① RK4 单步 ≡ 线性系统 4 阶 Taylor：‖rk4Step − Σ_{k=0..4} h^kA^k/k!‖ < 1e-12；
 *   ② 标量 y'=λy：RK4 单步 ≡ 1+hλ+h²λ²/2+h³λ³/6+h⁴λ⁴/24（< 1e-15）；
 *   ③ 全局收敛阶 4：步长折半 ⇒ 误差比 → 16（对 y'=y 闭式解 e^t，观测阶 ∈ [3.5, 4.5]）；
 *   ④ 谐振荡器 y''=−y：RK4 能量漂移远小于 Euler（四阶 vs 一阶稳定性对照）；
 *   ⑤ RK45 自适应：末点误差 ≤ tol 且所用步数 < 固定细步（效率 + 精度双约束）；
 *   ⑥ 定步长网格闭式：t_i = t0 + i·h，步数 = round((t1−t0)/h)+1，长度精确一致。
 */

/** 前向 Euler（一阶，仅作对照） */
export function eulerStep(f, t, y, h) {
  const k1 = f(t, y);
  return y.map((v, i) => v + h * k1[i]);
}

/** 显式中点法（二阶） */
export function midpointStep(f, t, y, h) {
  const k1 = f(t, y);
  const ym = y.map((v, i) => v + 0.5 * h * k1[i]);
  const k2 = f(t + 0.5 * h, ym);
  return y.map((v, i) => v + h * k2[i]);
}

/** 经典四级四阶 Runge–Kutta（局部误差 O(h^5)，全局 O(h^4)） */
export function rk4Step(f, t, y, h) {
  const k1 = f(t, y);
  const y2 = y.map((v, i) => v + 0.5 * h * k1[i]);
  const k2 = f(t + 0.5 * h, y2);
  const y3 = y.map((v, i) => v + 0.5 * h * k2[i]);
  const k3 = f(t + 0.5 * h, y3);
  const y4 = y.map((v, i) => v + h * k3[i]);
  const k4 = f(t + h, y4);
  return y.map((v, i) => v + (h / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]));
}

const STEPPERS = { euler: eulerStep, midpoint: midpointStep, rk4: rk4Step };

/**
 * 定步长求解 y' = f(t, y)，t: t0 → t1。
 * @returns {{t:number[], y:number[][], steps:number, h:number}}
 */
export function solveODE(f, t0, y0, t1, { h, method = 'rk4' } = {}) {
  const step = STEPPERS[method];
  if (!step) throw new Error('solveODE: 未知方法 ' + method);
  const steps = Math.round((t1 - t0) / h);
  const t = new Array(steps + 1), y = new Array(steps + 1);
  let cur = y0.slice();
  t[0] = t0; y[0] = cur.slice();
  for (let i = 0; i < steps; i++) {
    cur = step(f, t0 + i * h, cur, h);
    t[i + 1] = t0 + (i + 1) * h; y[i + 1] = cur;
  }
  return { t, y, steps, h };
}

/* ── Dormand–Prince 5(4) 嵌入式系数（DP1980 表，CFL 友好的 FSAL 首列） ── */
const DP_C = [0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1];
const DP_A = [
  [],
  [1 / 5],
  [3 / 40, 9 / 40],
  [44 / 45, -56 / 15, 32 / 9],
  [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
  [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
  [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
];
const DP_B5 = [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0];
const DP_B4 = [5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40];

/**
 * Dormand–Prince 5(4) 系数表（供不变量校验结构恒等式）。
 * 正确性恒等式：ΣB5=ΣB4=1（一致性）、Σ_j A[s][j] = C[s]（阶段一致性）。
 * @returns {{C:number[], A:number[][], B5:number[], B4:number[]}}
 */
export function dpTableau() {
  return { C: DP_C.slice(), A: DP_A.map((r) => r.slice()), B5: DP_B5.slice(), B4: DP_B4.slice() };
}

/**
 * 自适应 RK5(4)（Dormand–Prince）：5 阶推进、4 阶估计误差，按 tol 调节步长。
 *
 * 语义约定（与 solveODE 对齐）：`steps` = **接受步数**（= t.length − 1），
 * `attempts` = 尝试步数（含被误差判据拒绝的步，≥ steps）。
 * 两者分开，避免 `y[steps]` 越界与"步数"口径漂移。
 * @returns {{t:number[], y:number[][], steps:number, attempts:number}}
 */
export function solveODERK45(f, t0, y0, t1, { tol = 1e-8, h0 = 0.1, safety = 0.9 } = {}) {
  let t = t0;
  let y = y0.slice();
  let h = Math.min(h0, t1 - t0);
  const ts = [t], ys = [y.slice()];
  let attempts = 0;
  while (t < t1 - 1e-12) {
    if (t + h > t1) h = t1 - t;
    const k = [];
    for (let s = 0; s < 7; s++) {
      const yi = y.map((v, i) => {
        let acc = v;
        for (let j = 0; j < s; j++) acc += h * DP_A[s][j] * k[j][i];
        return acc;
      });
      k.push(f(t + DP_C[s] * h, yi));
    }
    const y5 = y.map((v, i) => {
      let a = v;
      for (let s = 0; s < 7; s++) a += h * DP_B5[s] * k[s][i];
      return a;
    });
    const y4 = y.map((v, i) => {
      let a = v;
      for (let s = 0; s < 7; s++) a += h * DP_B4[s] * k[s][i];
      return a;
    });
    let err = 0;
    for (let i = 0; i < y.length; i++) err = Math.max(err, Math.abs(y5[i] - y4[i]));
    if (err <= tol || h <= 1e-14) {
      t += h; y = y5; ts.push(t); ys.push(y.slice());
    }
    const fac = err === 0 ? 5 : safety * Math.pow(tol / Math.max(err, 1e-300), 0.2);
    h *= Math.max(0.2, Math.min(5, fac));
    if (++attempts > 1000000) throw new Error('solveODERK45: 超过步数上限');
  }
  return { t: ts, y: ys, steps: ts.length - 1, attempts };
}

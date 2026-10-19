/**
 * 旗舰AI引擎 · 激活函数族（现代深度学习标准件）
 * 站在：Hendrycks & Gimpel 2016（GELU）· Ramachandran, Zhao & Le 2017（Swish/SiLU）
 *       Misra 2019（Mish）· Howard et al. 2019（HardSwish）· Dugas et al. 2001（Softplus）
 *       Abramowitz & Stegun 7.1.26（erf 有理逼近，最大绝对误差 1.5e-7）
 * 每个激活都提供解析导数，供梯度检验交叉验证（中心差分 vs 解析）。
 */

import { sigmoid, relErr } from '../core/numerical.js';

/** erfc 主支（x ≥ 0）：A&S 7.1.26，t = 1/(1+0.5x) 的九项 Padé 型逼近 */
function erfcPos(x) {
  const t = 1 / (1 + 0.5 * x);
  const y = t * Math.exp(
    -x * x - 1.26551223
    + 1.00002368 * t + 0.37409196 * t * t + 0.09678418 * t ** 3
    - 0.18628806 * t ** 4 + 0.27886807 * t ** 5 - 1.13520398 * t ** 6
    + 1.48851587 * t ** 7 - 0.82215223 * t ** 8 + 0.17087277 * t ** 9,
  );
  return y;
}

/** 互补误差函数 erfc(x)：全定义域稳定（负支用 erfc(−x) = 2 − erfc(x) 反射） */
export function erfc(x) {
  return x >= 0 ? erfcPos(x) : 2 - erfcPos(-x);
}

/** 误差函数 erf(x)：奇函数，erf(−x) = −erf(x)（bit 级精确） */
export function erf(x) {
  return x >= 0 ? 1 - erfcPos(x) : erfcPos(-x) - 1;
}

/** 标准正态 CDF：Φ(x) = ½·erfc(−x/√2)，全定义域稳定 */
export function normalCDF(x) {
  return 0.5 * erfc(-x / Math.SQRT2);
}

/** 标准正态 PDF：φ(x) = e^{−x²/2}/√(2π) */
export function normalPDF(x) {
  return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
}

/** Softplus：ln(1+e^x)，数值稳定双支式；恒等式 softplus'(x) ≡ sigmoid(x) */
export function softplus(x) {
  return x > 0 ? x + Math.log1p(Math.exp(-x)) : Math.log1p(Math.exp(x));
}

/** Softplus 导数：σ(x)（解析恒等） */
export function softplusD(x) {
  return sigmoid(x);
}

/** GELU 精确版：x·Φ(x)（Hendrycks & Gimpel 2016） */
export function gelu(x) {
  return 0.5 * x * erfc(-x / Math.SQRT2);
}

/** GELU 导数：Φ(x) + x·φ(x)（乘积法则） */
export function geluD(x) {
  return normalCDF(x) + x * normalPDF(x);
}

/** GELU tanh 逼近版：½x(1+tanh(√(2/π)(x+0.044715x³))) */
export function geluTanh(x) {
  const c = Math.sqrt(2 / Math.PI);
  return 0.5 * x * (1 + Math.tanh(c * (x + 0.044715 * x * x * x)));
}

/** GELU tanh 版导数（链式法则） */
export function geluTanhD(x) {
  const c = Math.sqrt(2 / Math.PI);
  const u = c * (x + 0.044715 * x * x * x);
  const du = c * (1 + 3 * 0.044715 * x * x);
  return 0.5 * (1 + Math.tanh(u)) + 0.5 * x * (1 - Math.tanh(u) ** 2) * du;
}

/** SiLU / Swish：x·σ(x)（Ramachandran et al. 2017，两名为同一函数） */
export function silu(x) {
  return x * sigmoid(x);
}

/** SiLU 导数：σ(x)·(1 + x·(1 − σ(x))) */
export function siluD(x) {
  const s = sigmoid(x);
  return s * (1 + x * (1 - s));
}

/** Swish ≡ SiLU（β=1 特例，别名导出保持文献兼容） */
export function swish(x) {
  return silu(x);
}

/** Mish：x·tanh(softplus(x))（Misra 2019） */
export function mish(x) {
  return x * Math.tanh(softplus(x));
}

/** Mish 导数：tanh(sp) + x·sech²(sp)·σ(x) */
export function mishD(x) {
  const sp = softplus(x);
  const th = Math.tanh(sp);
  return th + x * (1 - th * th) * sigmoid(x);
}

/** HardSwish：x·relu6(x+3)/6（Howard et al. 2019，分段线性） */
export function hardswish(x) {
  return x * Math.min(Math.max(x + 3, 0), 6) / 6;
}

/** HardSwish 导数（分段：x<−3 → 0；−3<x<3 → (2x+3)/6；x>3 → 1；±3 为折点不可导） */
export function hardswishD(x) {
  if (x <= -3) return 0;
  if (x >= 3) return 1;
  return (2 * x + 3) / 6;
}

/** 中心差分相对误差检验：max |解析 − 数值| / (|a|+|b|)（双判据标尺 relErr） */
export function activationGradCheck(f, df, xs, { eps = 1e-5 } = {}) {
  let worst = 0;
  for (const x of xs) {
    const num = (f(x + eps) - f(x - eps)) / (2 * eps);
    worst = Math.max(worst, relErr(num, df(x)));
  }
  return worst;
}

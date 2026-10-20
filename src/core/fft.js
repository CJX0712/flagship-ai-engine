/**
 * 旗舰AI引擎 · 快速傅里叶变换（FFT, Cooley & Tukey 1965）
 * 站在：Cooley & Tukey 1965《An Algorithm for the Machine Calculation of
 *       Complex Fourier Series》（基-2 分治 FFT，O(n log n) 击败 O(n²) DFT）·
 *       Numerical Recipes in C §12.2–12.3（复数变换与卷积定理）·
 *       Oppenheim & Schafer《Discrete-Time Signal Processing》§8–9（DFT/圆周卷积）。
 *
 * 约定：实部/虚部分别用等长 number[] 表示；长度须为 2 的幂；返回新数组不修改入参。
 *   前向：X_k = Σ_t x_t·e^{−2πi·tk/n}（工程符号，与 NumPy/NR 一致）
 *   逆向：x_t = (1/n)·Σ_k X_k·e^{+2πi·tk/n}
 *
 * 不变量（全部为可独立交叉验证的硬恒等式）：
 *   ① FFT ≡ 朴素 DFT：分治结果与 O(n²) 参照逐分量一致（maxRelErr < 1e-10）；
 *   ② 逆变换闭合：ifft(fft(x)) ≡ x（maxAbs < 1e-12）；
 *   ③ Parseval 能量守恒：Σ_k|X_k|² = n·Σ_t|x_t|²（相对误差 < 1e-12）；
 *   ④ 线性卷积定理：FFT 卷积（零填充）≡ 朴素线性卷积（maxAbs < 1e-10）；
 *   ⑤ 线性性：fft(αx+βy) = α·fft(x)+β·fft(y)（maxAbs < 1e-12）；
 *   ⑥ 圆周卷积定理：ifft(fft(a)·fft(b)) ≡ 圆周卷积（maxAbs < 1e-10）。
 */

/** 不小于 n 的最小 2 的幂 */
export function nextPow2(n) {
  let p = 1;
  while (p < n) p *= 2;
  return p;
}

function isPow2(n) {
  return n >= 1 && (n & (n - 1)) === 0;
}

/** 基-2 Cooley–Tukey 蝶形（原地递归，re/im 同步分裂偶奇子序列） */
function fftRec(re, im) {
  const n = re.length;
  if (n <= 1) return;
  const half = n >> 1;
  const reE = new Array(half), imE = new Array(half), reO = new Array(half), imO = new Array(half);
  for (let i = 0; i < half; i++) {
    reE[i] = re[2 * i]; imE[i] = im[2 * i];
    reO[i] = re[2 * i + 1]; imO[i] = im[2 * i + 1];
  }
  fftRec(reE, imE);
  fftRec(reO, imO);
  for (let k = 0; k < half; k++) {
    const ang = -2 * Math.PI * k / n;
    const wr = Math.cos(ang), wi = Math.sin(ang);
    const ur = reE[k], ui = imE[k];
    const vr = reO[k] * wr - imO[k] * wi;
    const vi = reO[k] * wi + imO[k] * wr;
    re[k] = ur + vr; im[k] = ui + vi;
    re[k + half] = ur - vr; im[k + half] = ui - vi;
  }
}

/**
 * 基-2 FFT（前向）。长度须为 2 的幂。
 * @param {number[]} re 实部
 * @param {number[]} im 虚部
 * @returns {{re:number[], im:number[]}}
 */
export function fftRadix2(re, im) {
  const n = re.length;
  if (im.length !== n) throw new Error('fftRadix2: 实部/虚部长度不一致');
  if (!isPow2(n)) throw new Error('fftRadix2: 长度须为 2 的幂，得到 ' + n);
  const R = re.slice();
  const I = im.length === n ? im.slice() : new Array(n).fill(0);
  fftRec(R, I);
  return { re: R, im: I };
}

/** 实数信号前向 FFT（虚部置零） */
export function fft(x) {
  return fftRadix2(x.slice(), new Array(x.length).fill(0));
}

/**
 * 逆 FFT：共轭 → 前向 → 共轭 → 除 n。
 * @returns {{re:number[], im:number[]}}
 */
export function ifftRadix2(re, im) {
  const n = re.length;
  if (im.length !== n) throw new Error('ifftRadix2: 实部/虚部长度不一致');
  if (!isPow2(n)) throw new Error('ifftRadix2: 长度须为 2 的幂，得到 ' + n);
  const cr = re.slice(), ci = im.map((v) => -v);
  fftRec(cr, ci);
  return { re: cr.map((v) => v / n), im: ci.map((v) => -v / n) };
}

/** 朴素 DFT（O(n²) 参照实现，不变量①的黄金标准） */
export function dftNaive(re, im) {
  const n = re.length;
  const R = new Array(n), I = new Array(n);
  for (let k = 0; k < n; k++) {
    let sr = 0, si = 0;
    for (let t = 0; t < n; t++) {
      const ang = -2 * Math.PI * k * t / n;
      const c = Math.cos(ang), s = Math.sin(ang);
      sr += re[t] * c - im[t] * s;
      si += re[t] * s + im[t] * c;
    }
    R[k] = sr; I[k] = si;
  }
  return { re: R, im: I };
}

/** 复数逐点相乘（长度须一致） */
function spectrumMultiply(fa, fb) {
  const n = fa.re.length;
  const pr = new Array(n), pi = new Array(n);
  for (let k = 0; k < n; k++) {
    pr[k] = fa.re[k] * fb.re[k] - fa.im[k] * fb.im[k];
    pi[k] = fa.re[k] * fb.im[k] + fa.im[k] * fb.re[k];
  }
  return { re: pr, im: pi };
}

/** 圆周卷积（周期边界，等长序列；FFT 实现） */
export function circularConvolve(a, b) {
  const n = a.length;
  if (b.length !== n) throw new Error('circularConvolve: 两序列长度须一致');
  const prod = spectrumMultiply(fftRadix2(a.slice(), new Array(n).fill(0)), fftRadix2(b.slice(), new Array(n).fill(0)));
  return ifftRadix2(prod.re, prod.im).re;
}

/** 线性卷积（零填充至 2 的幂），结果长度 a.length + b.length − 1 */
export function convolveLinear(a, b) {
  const n = nextPow2(a.length + b.length - 1);
  const pa = new Array(n).fill(0), pb = new Array(n).fill(0);
  for (let i = 0; i < a.length; i++) pa[i] = a[i];
  for (let i = 0; i < b.length; i++) pb[i] = b[i];
  const prod = spectrumMultiply(fft(pa), fft(pb));
  return ifftRadix2(prod.re, prod.im).re.slice(0, a.length + b.length - 1);
}

/** 朴素线性卷积（O(n·m) 参照实现） */
export function convolveNaive(a, b) {
  const out = new Array(a.length + b.length - 1).fill(0);
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < b.length; j++) out[i + j] += a[i] * b[j];
  }
  return out;
}

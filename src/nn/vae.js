/**
 * 旗舰AI引擎 · 变分自编码器（VAE）
 * 站在：Kingma & Welling 2013《Auto-Encoding Variational Bayes》(ICLR)
 *       Rezende, Mohamed & Wierstra 2014（重参数化 + 随机反向传播）
 *       Kingma & Welling 2019《An Introduction to Variational Autoencoders》
 * 核心机制：
 *   ① 重参数化 z = μ + σ⊙ε（ε~N(0,I)）——把随机节点移出梯度路径，
 *      使 ELBO 对编码器参数是**确定性可微**的（这是 VAE 能训练的根本原因）。
 *   ② 对角高斯后验 vs 标准正态先验的 KL **解析闭式**：
 *      KL = ½Σ_d (μ_d² + σ_d² − log σ_d² − 1)（无需采样即可精确计算）。
 * 不变量：
 *   ① KL 解析闭式 ≡ 蒙特卡洛估计（大样本下相对误差 → 0）
 *   ② KL ≥ 0，且 μ=0、σ²=1（后验 ≡ 先验）时 KL = 0（精确）
 *   ③ 重参数化后 ELBO 为确定性函数 ⇒ 解析梯度 ≡ 中心差分（含编码器/解码器全部参数）
 *   ④ ELBO ≤ log p(x)（变分下界）；KL 项随后验逼近先验单调趋零
 *   ⑤ μ=0、logvar=0 时重参数化 z ≡ ε（确定性恒等）
 */

import { zeros, randomMatrix, matmul, matvec, transpose, norm, dot } from '../core/linalg.js';
import { relErr, sigmoid } from '../core/numerical.js';
import { mulberry32, gaussianSample } from '../core/stats.js';

/** 构造 VAE 参数（编码器 x→μ,logvar；解码器 z→x̂） */
export function vaeMake({ dimX = 6, dimH = 8, dimZ = 3, seed = 42 } = {}) {
  const rng = mulberry32(seed);
  const s = (m, n) => randomMatrix(m, n, rng, -0.5, 0.5);
  return {
    dimX, dimH, dimZ,
    W1: s(dimH, dimX), b1: new Array(dimH).fill(0),
    Wmu: s(dimZ, dimH), bmu: new Array(dimZ).fill(0),
    Wlv: s(dimZ, dimH), blv: new Array(dimZ).fill(0),
    W2: s(dimH, dimZ), b2: new Array(dimH).fill(0),
    W3: s(dimX, dimH), b3: new Array(dimX).fill(0),
  };
}

/** 编码器：x → {mu, logvar} */
export function vaeEncode(net, x) {
  const a1 = matvec(net.W1, x).map((v, i) => v + net.b1[i]);
  const h1 = a1.map(Math.tanh);
  const mu = matvec(net.Wmu, h1).map((v, i) => v + net.bmu[i]);
  const logvar = matvec(net.Wlv, h1).map((v, i) => v + net.blv[i]);
  return { h1, mu, logvar };
}

/** 重参数化采样：z = μ + exp(½·logvar) ⊙ ε */
export function vaeReparametrize(mu, logvar, eps) {
  return mu.map((m, i) => m + Math.exp(0.5 * logvar[i]) * eps[i]);
}

/** 解码器：z → x̂ */
export function vaeDecode(net, z) {
  const a2 = matvec(net.W2, z).map((v, i) => v + net.b2[i]);
  const h2 = a2.map(Math.tanh);
  const xhat = matvec(net.W3, h2).map((v, i) => v + net.b3[i]);
  return { h2, xhat };
}

/**
 * 对角高斯后验 N(μ,diag σ²) 与标准正态先验的 KL **解析闭式**：
 *   KL = ½ Σ_d (μ_d² + σ_d² − log σ_d² − 1)
 */
export function vaeKL(mu, logvar) {
  let s = 0;
  for (let i = 0; i < mu.length; i++) s += mu[i] ** 2 + Math.exp(logvar[i]) - logvar[i] - 1;
  return 0.5 * s;
}

/**
 * 同一 KL 的蒙特卡洛估计（解析闭式的交叉验证标尺）：
 *   KL ≈ E_q[log q(z|x) − log p(z)]，z = μ + σ⊙ε
 */
export function vaeKLMonteCarlo(mu, logvar, rng, samples = 200000) {
  let s = 0;
  for (let t = 0; t < samples; t++) {
    let lq = 0, lp = 0;
    for (let i = 0; i < mu.length; i++) {
      const sig = Math.exp(0.5 * logvar[i]);
      const z = mu[i] + sig * gaussianSample(rng, 0, 1);
      lq += -0.5 * Math.log(2 * Math.PI) - logvar[i] / 2 - 0.5 * ((z - mu[i]) / sig) ** 2;
      lp += -0.5 * Math.log(2 * Math.PI) - 0.5 * z * z;
    }
    s += lq - lp;
  }
  return s / samples;
}

/** 生成标准正态噪声向量（可复现，供重参数化） */
export function vaeSampleEps(dimZ, rng) {
  const e = new Array(dimZ);
  for (let i = 0; i < dimZ; i++) e[i] = gaussianSample(rng, 0, 1);
  return e;
}

/**
 * VAE 前向：给 x 与 ε，返回 {mu, logvar, z, xhat, kl, recon, elbo}
 * 目标（高斯似然，观测方差 1）：−ELBO = ½‖x − x̂‖² + KL
 */
export function vaeForward(net, x, eps) {
  const { h1, mu, logvar } = vaeEncode(net, x);
  const z = vaeReparametrize(mu, logvar, eps);
  const { h2, xhat } = vaeDecode(net, z);
  let recon = 0;
  for (let i = 0; i < x.length; i++) recon += 0.5 * (x[i] - xhat[i]) ** 2;
  const kl = vaeKL(mu, logvar);
  return { h1, mu, logvar, z, h2, xhat, kl, recon, elbo: -(recon + kl) };
}

/**
 * VAE 反向（重参数化路径上的精确解析梯度）。
 * 目标 loss = ½‖x−x̂‖² + KL（即 −ELBO，最小化）。
 */
export function vaeBackward(net, fwd, x, eps) {
  const { dimX, dimH, dimZ } = net;
  const { h1, mu, logvar, z, h2, xhat } = fwd;
  const g = {
    W1: zeros(dimH, dimX), b1: new Array(dimH).fill(0),
    Wmu: zeros(dimZ, dimH), bmu: new Array(dimZ).fill(0),
    Wlv: zeros(dimZ, dimH), blv: new Array(dimZ).fill(0),
    W2: zeros(dimH, dimZ), b2: new Array(dimH).fill(0),
    W3: zeros(dimX, dimH), b3: new Array(dimX).fill(0),
  };
  // 解码器：∂loss/∂x̂ = x̂ − x
  const dxhat = xhat.map((v, i) => v - x[i]);
  for (let i = 0; i < dimX; i++) {
    for (let j = 0; j < dimH; j++) g.W3[i][j] += dxhat[i] * h2[j];
    g.b3[i] += dxhat[i];
  }
  const dh2raw = new Array(dimH).fill(0);
  for (let j = 0; j < dimH; j++) {
    for (let i = 0; i < dimX; i++) dh2raw[j] += net.W3[i][j] * dxhat[i];
  }
  const dh2 = dh2raw.map((v, j) => v * (1 - h2[j] ** 2));
  for (let j = 0; j < dimH; j++) {
    for (let k = 0; k < dimZ; k++) g.W2[j][k] += dh2[j] * z[k];
    g.b2[j] += dh2[j];
  }
  const dz = new Array(dimZ).fill(0);
  for (let k = 0; k < dimZ; k++) {
    for (let j = 0; j < dimH; j++) dz[k] += net.W2[j][k] * dh2[j];
  }
  // 编码器：z 路径 + KL 路径
  const dmu = dz.map((v, i) => v + mu[i]);                       // ∂KL/∂μ = μ
  const dlogvar = new Array(dimZ);
  for (let k = 0; k < dimZ; k++) {
    const sig = Math.exp(0.5 * logvar[k]);
    dlogvar[k] = dz[k] * sig * 0.5 * eps[k] + 0.5 * (Math.exp(logvar[k]) - 1);
  }
  for (let k = 0; k < dimZ; k++) {
    for (let j = 0; j < dimH; j++) {
      g.Wmu[k][j] += dmu[k] * h1[j];
      g.Wlv[k][j] += dlogvar[k] * h1[j];
    }
    g.bmu[k] += dmu[k];
    g.blv[k] += dlogvar[k];
  }
  const dh1raw = new Array(dimH).fill(0);
  for (let j = 0; j < dimH; j++) {
    for (let k = 0; k < dimZ; k++) {
      dh1raw[j] += net.Wmu[k][j] * dmu[k] + net.Wlv[k][j] * dlogvar[k];
    }
  }
  const dh1 = dh1raw.map((v, j) => v * (1 - h1[j] ** 2));
  for (let j = 0; j < dimH; j++) {
    for (let i = 0; i < dimX; i++) g.W1[j][i] += dh1[j] * x[i];
    g.b1[j] += dh1[j];
  }
  return g;
}

/**
 * VAE 梯度检验：固定 ε 后 loss 是确定性函数 ⇒ 中心差分可精确比对。
 * @returns {{maxRel:number, maxAbs:number}}
 */
export function vaeGradientCheck(net, x, eps, { epsFD = 1e-5 } = {}) {
  const loss = () => {
    const f = vaeForward(net, x, eps);
    return f.recon + f.kl;
  };
  const fwd = vaeForward(net, x, eps);
  const g = vaeBackward(net, fwd, x, eps);
  let maxRel = 0, maxAbs = 0;
  const mats = ['W1', 'Wmu', 'Wlv', 'W2', 'W3'];
  const vecs = ['b1', 'bmu', 'blv', 'b2', 'b3'];
  for (const name of mats) {
    const P = net[name], G = g[name];
    for (let i = 0; i < P.length; i++) {
      for (let j = 0; j < P[i].length; j++) {
        const orig = P[i][j];
        P[i][j] = orig + epsFD;
        const lp = loss();
        P[i][j] = orig - epsFD;
        const lm = loss();
        P[i][j] = orig;
        const num = (lp - lm) / (2 * epsFD);
        maxRel = Math.max(maxRel, relErr(num, G[i][j]));
        maxAbs = Math.max(maxAbs, Math.abs(num - G[i][j]));
      }
    }
  }
  for (const name of vecs) {
    const p = net[name], gr = g[name];
    for (let i = 0; i < p.length; i++) {
      const orig = p[i];
      p[i] = orig + epsFD;
      const lp = loss();
      p[i] = orig - epsFD;
      const lm = loss();
      p[i] = orig;
      const num = (lp - lm) / (2 * epsFD);
      maxRel = Math.max(maxRel, relErr(num, gr[i]));
      maxAbs = Math.max(maxAbs, Math.abs(num - gr[i]));
    }
  }
  return { maxRel, maxAbs };
}

/** 词/向量余弦相似度（表示学习的通用标尺） */
export function cosineSim(a, b) {
  const na = norm(a), nb = norm(b);
  return na > 0 && nb > 0 ? dot(a, b) / (na * nb) : 0;
}

/** σ(−x) = 1 − σ(x)（负采样损失的对偶恒等） */
export function sigmoidNegIdentity(x) {
  return Math.abs(sigmoid(-x) - (1 - sigmoid(x)));
}

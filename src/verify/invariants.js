/**
 * 旗舰AI引擎 · 不变量自检套件
 * 每条算法至少 1 条可独立交叉验证的硬不变量；Node 与浏览器共用同一份逻辑。
 * 8 组套件全绿 = 引擎数学正确性的机器证明。
 */

import {
  zeros, transpose, matmul, matvec, dot, norm, frobeniusNorm, maxAbs,
  solve, inverse, cholesky, qr, jacobiEig, svd, powerIteration, randomMatrix,
} from '../core/linalg.js';
import { logsumexp, softmax, sigmoid, relErr } from '../core/numerical.js';
import {
  gaussianPDF, entropy, crossEntropy, klDivergence,
  mean as colMean, mulberry32, gaussianSample,
} from '../core/stats.js';
import {
  addIntercept, linearRegression, ridgeRegression,
  logisticRegression, logisticGradientCheck,
} from '../ml/regression.js';
import { knnPredict, naiveBayesFit, naiveBayesPredict } from '../ml/classification.js';
import { kmeans, gmm } from '../ml/clustering.js';
import { pca } from '../ml/dimreduction.js';
import { DecisionTree, giniImpurity } from '../ml/tree.js';
import { DenseNet, denseGradientCheck, trainDense } from '../nn/dense.js';
import {
  scaledDotProductAttention, attentionGradients, attentionGradientCheck,
  multiHeadAttention, randomAttentionTensors,
} from '../nn/attention.js';
import { layerNorm, positionalEncoding, transformerBlock } from '../nn/transformer.js';

/** 向量最大绝对值 */
function vinf(v) {
  let m = 0;
  for (const x of v) m = Math.max(m, Math.abs(x));
  return m;
}

/** 矩阵逐元素最大绝对差 */
function mdiff(A, B) {
  let m = 0;
  for (let i = 0; i < A.length; i++) {
    for (let j = 0; j < A[i].length; j++) m = Math.max(m, Math.abs(A[i][j] - B[i][j]));
  }
  return m;
}

/** 单位阵 */
function eye(n) {
  const I = zeros(n, n);
  for (let i = 0; i < n; i++) I[i][i] = 1;
  return I;
}

/** 高斯团簇数据 */
function blobs(rng, centers, nPer, sigma) {
  const X = [];
  const y = [];
  centers.forEach((c, ci) => {
    for (let i = 0; i < nPer; i++) {
      X.push(c.map(v => gaussianSample(rng, v, sigma)));
      y.push(ci);
    }
  });
  return { X, y };
}

/**
 * 运行全部 8 组不变量自检。
 * @returns {{results:Array, passed:number, total:number, allOk:boolean}}
 */
export function runAllInvariants() {
  const results = [];
  const suites = [
    ['核心线性代数 · Linalg Core', suiteLinalg],
    ['特征分解与SVD · Eigen & SVD', suiteEigen],
    ['数值稳定 · Numerical Stability', suiteNumerical],
    ['概率与信息 · Probability & Info', suiteStats],
    ['回归与梯度检验 · Regression', suiteRegression],
    ['聚类与生成 · Clustering & GMM', suiteClustering],
    ['降维与决策树 · PCA & Tree', suiteDimTree],
    ['深度学习 · Deep Learning', suiteDeep],
  ];
  for (const [name, fn] of suites) {
    const checks = [];
    const t0 = Date.now();
    const check = (label, ok, info = '') => checks.push({ label, ok: !!ok, info: String(info) });
    try {
      fn(check);
    } catch (e) {
      checks.push({ label: '执行异常: ' + (e && e.message), ok: false, info: String((e && e.stack) || e).split('\n').slice(0, 3).join(' | ') });
    }
    results.push({
      name,
      ok: checks.length > 0 && checks.every(c => c.ok),
      checks,
      ms: Date.now() - t0,
    });
  }
  const passed = results.filter(r => r.ok).length;
  return { results, passed, total: results.length, allOk: passed === results.length };
}

/* ───────────────────────── 1. 核心线性代数 ───────────────────────── */
function suiteLinalg(check) {
  const rng = mulberry32(7);
  const B = randomMatrix(4, 4, rng);
  const M = matmul(transpose(B), B).map((row, i) => row.map((v, j) => v + (i === j ? 2 : 0)));

  const b = [rng(), rng(), rng(), rng()];
  const x = solve(M, b);
  const resS = vinf(matvec(M, x).map((v, i) => v - b[i]));
  check('solve：‖Ax − b‖∞ < 1e-9（部分主元高斯消元）', resS < 1e-9, `残差=${resS.toExponential(2)}`);

  const Minv = inverse(M);
  const resI = mdiff(matmul(M, Minv), eye(4));
  check('inverse：‖M·M⁻¹ − I‖max < 1e-9', resI < 1e-9, `误差=${resI.toExponential(2)}`);

  const L = cholesky(M);
  const resC = mdiff(matmul(L, transpose(L)), M);
  check('cholesky：L·Lᵀ = A（SPD）', resC < 1e-9, `误差=${resC.toExponential(2)}`);

  const A52 = randomMatrix(5, 3, rng);
  const { Q, R } = qr(A52);
  const resQ = mdiff(matmul(transpose(Q), Q), eye(5));
  const resQR = frobeniusNorm(matmul(Q, R).map((row, i) => row.map((v, j) => v - A52[i][j])));
  let upperTri = true;
  for (let i = 1; i < R.length; i++) for (let j = 0; j < Math.min(i, R[0].length); j++) if (Math.abs(R[i][j]) > 1e-9) upperTri = false;
  check('qr：QᵀQ = I（Householder 正交性）', resQ < 1e-9, `误差=${resQ.toExponential(2)}`);
  check('qr：Q·R = A 且 R 上三角', resQR < 1e-9 * (1 + frobeniusNorm(A52)) && upperTri,
    `‖QR−A‖_F=${resQR.toExponential(2)}，上三角=${upperTri}`);
}

/* ───────────────────────── 2. 特征分解与SVD ───────────────────────── */
function suiteEigen(check) {
  const rng = mulberry32(11);
  const B = randomMatrix(5, 5, rng);
  const S = B.map((row, i) => row.map((v, j) => (v + B[j][i]) / 2 + (i === j ? 2 : 0)));

  const { values, vectors } = jacobiEig(S);
  const scale = 1 + Math.max(...values.map(Math.abs));
  let maxRes = 0;
  for (let k = 0; k < values.length; k++) {
    const Av = matvec(S, vectors[k]);
    maxRes = Math.max(maxRes, norm(Av.map((v, i) => v - values[k] * vectors[k][i])));
  }
  check('jacobiEig：‖S·v − λ·v‖ < 1e-9（全部特征对）', maxRes < 1e-9 * scale, `最大残差=${maxRes.toExponential(2)}`);

  let orthoErr = 0;
  for (let a = 0; a < vectors.length; a++) {
    orthoErr = Math.max(orthoErr, Math.abs(norm(vectors[a]) - 1));
    for (let b = a + 1; b < vectors.length; b++) {
      orthoErr = Math.max(orthoErr, Math.abs(dot(vectors[a], vectors[b])));
    }
  }
  check('jacobiEig：特征向量正交归一（对称阵）', orthoErr < 1e-9, `最大偏差=${orthoErr.toExponential(2)}`);

  const pi = powerIteration(S, { rng: mulberry32(3), iters: 2000 });
  const maxLam = Math.max(...values.map(Math.abs));
  check('powerIteration：主特征值与 Jacobi 一致', Math.abs(pi.value - maxLam) < 1e-8,
    `幂迭代=${pi.value.toFixed(10)}，Jacobi=${maxLam.toFixed(10)}`);
  check('powerIteration：Rayleigh 残差 ‖Av − λv‖ < 1e-8', pi.residual < 1e-8, `残差=${pi.residual.toExponential(2)}`);

  const A = randomMatrix(6, 4, rng);
  const { U, s, V } = svd(A);
  const errU = mdiff(matmul(transpose(U), U), eye(4));
  const errV = mdiff(matmul(transpose(V), V), eye(4));
  check('svd：UᵀU = I 且 VᵀV = I', errU < 1e-9 && errV < 1e-9, `errU=${errU.toExponential(2)}，errV=${errV.toExponential(2)}`);

  const Vcol = (k) => V.map(row => row[k]);
  const recon = zeros(6, 4);
  for (let k = 0; k < 4; k++) {
    for (let i = 0; i < 6; i++) for (let j = 0; j < 4; j++) recon[i][j] += s[k] * U[i][k] * Vcol(k)[j];
  }
  const resR = frobeniusNorm(recon.map((row, i) => row.map((v, j) => v - A[i][j])));
  check('svd：‖A − UΣVᵀ‖_F ≈ 0（完全重构）', resR < 1e-9 * (1 + frobeniusNorm(A)), `误差=${resR.toExponential(2)}`);

  const A2 = zeros(6, 4);
  for (let k = 0; k < 2; k++) {
    for (let i = 0; i < 6; i++) for (let j = 0; j < 4; j++) A2[i][j] += s[k] * U[i][k] * Vcol(k)[j];
  }
  const errTrunc = frobeniusNorm(A2.map((row, i) => row.map((v, j) => v - A[i][j])));
  const formula = Math.sqrt(s.slice(2).reduce((a, b) => a + b * b, 0));
  check('svd：秩2截断误差 = √Σᵢ>₂σᵢ²（Eckart–Young）', Math.abs(errTrunc - formula) < 1e-9,
    `实测=${errTrunc.toFixed(12)}，理论=${formula.toFixed(12)}`);

  let sorted = true;
  for (let k = 0; k + 1 < s.length; k++) if (s[k] < s[k + 1] - 1e-12) sorted = false;
  check('svd：奇异值降序', sorted, `s=[${s.map(v => v.toFixed(4)).join(', ')}]`);
}

/* ───────────────────────── 3. 数值稳定 ───────────────────────── */
function suiteNumerical(check) {
  const lse = logsumexp([1000, 1001]);
  const expect = 1001 + Math.log(1 + Math.exp(-1));
  check('logsumexp：[1000,1001] 不溢出且解析一致', Math.abs(lse - expect) < 1e-9,
    `实测=${lse.toFixed(12)}，解析=${expect.toFixed(12)}（朴素加法 = Infinity）`);

  const lse2 = logsumexp([-1000, -1001]);
  const expect2 = -1000 + Math.log(1 + Math.exp(-1));
  check('logsumexp：[−1000,−1001] 不下溢且解析一致', Math.abs(lse2 - expect2) < 1e-9,
    `实测=${lse2.toFixed(12)}`);

  const sm = softmax([1000, 1001, 999]);
  const smRef = softmax([0, 1, -1]);
  const smSum = sm.reduce((a, b) => a + b, 0);
  check('softmax：大数输入行和=1、与平移等价（max-shift）',
    Math.abs(smSum - 1) < 1e-12 && mdiff([sm], [smRef]) < 1e-12 && sm.every(isFinite),
    `和=${smSum.toFixed(15)}，最大偏差=${mdiff([sm], [smRef]).toExponential(2)}`);

  let symErr = 0;
  for (let x = 1; x <= 800; x += 7) symErr = Math.max(symErr, Math.abs(sigmoid(-x) - (1 - sigmoid(x))));
  check('sigmoid：σ(−x) = 1 − σ(x)（双支式无溢出）', symErr < 1e-15 && isFinite(sigmoid(-800)),
    `最大偏差=${symErr.toExponential(2)}，σ(−800)=${sigmoid(-800).toExponential(2)}`);

  const gradErr = relErr(0.123456789, 0.123456789);
  check('relErr：相同值 → 0（梯度检验标尺自洽）', gradErr === 0, `relErr=${gradErr}`);
}

/* ───────────────────────── 4. 概率与信息 ───────────────────────── */
function suiteStats(check) {
  const rng = mulberry32(21);
  const p = softmax([rng(), rng(), rng(), rng(), rng(), rng()].map(v => v * 4 - 2));
  const q = softmax([rng(), rng(), rng(), rng(), rng(), rng()].map(v => v * 4 - 2));

  const klp = klDivergence(p, p);
  check('KL：KL(p‖p) = 0', klp < 1e-12, `KL=${klp.toExponential(2)}`);

  const kl = klDivergence(p, q);
  check('KL：KL(p‖q) ≥ 0（Gibbs 不等式）', kl > -1e-12, `KL=${kl.toFixed(12)}`);

  const u = new Array(6).fill(1 / 6);
  check('entropy：H(均匀6) = log 6（上界取等）', Math.abs(entropy(u) - Math.log(6)) < 1e-12,
    `H=${entropy(u).toFixed(12)}，log6=${Math.log(6).toFixed(12)}`);
  check('entropy：H(p) ≤ log 6 对任意分布成立', entropy(p) <= Math.log(6) + 1e-12,
    `H(p)=${entropy(p).toFixed(6)}`);

  const cep = crossEntropy(p, p);
  check('crossEntropy：H(p,p) = H(p)', Math.abs(cep - entropy(p)) < 1e-12,
    `偏差=${Math.abs(cep - entropy(p)).toExponential(2)}`);

  const mu = 1.3, sigma = 0.7, N = 4000;
  const a = mu - 14 * sigma, b = mu + 14 * sigma, h = (b - a) / N;
  const f = (x) => gaussianPDF(x, mu, sigma);
  let sum = f(a) + f(b);
  for (let i = 1; i < N; i++) sum += f(a + i * h) * (i % 2 === 1 ? 4 : 2);
  const integral = (h / 3) * sum;
  check('gaussianPDF：Simpson 数值积分 ≈ 1（±14σ, N=4000）', Math.abs(integral - 1) < 1e-6,
    `积分=${integral.toFixed(10)}，|误差|=${Math.abs(integral - 1).toExponential(2)}`);
}

/* ───────────────────────── 5. 回归与梯度检验 ───────────────────────── */
function suiteRegression(check) {
  const rng = mulberry32(31);
  const X = randomMatrix(30, 3, rng).map(row => row.map(v => v * 2));
  const y = X.map(row => 3 + 1 * row[0] - 2 * row[1] + 0.5 * row[2]);

  const { w } = linearRegression(X, y);
  const wErr = Math.max(Math.abs(w[0] - 3), Math.abs(w[1] - 1), Math.abs(w[2] + 2), Math.abs(w[3] - 0.5));
  check('linearRegression：精确恢复真值 w*（无噪声）', wErr < 1e-8,
    `ŵ=[${w.map(v => v.toFixed(8)).join(', ')}]，误差=${wErr.toExponential(2)}`);

  const Xb = addIntercept(X);
  const r = y.map((yi, i) => yi - Xb[i].reduce((s, x, j) => s + x * w[j], 0));
  const gradRes = matvec(transpose(Xb), r);
  check('linearRegression：Xᵀ(y − Xŵ) = 0（残差⊥设计矩阵）',
    vinf(gradRes) < 1e-7 * (1 + norm(y)), `‖Xᵀr‖∞=${vinf(gradRes).toExponential(2)}`);

  const { w: wR } = ridgeRegression(X, y, 0.5);
  const n2 = (v) => Math.sqrt(v.reduce((a, b) => a + b * b, 0));
  check('ridgeRegression：系数收缩 ‖w_λ‖ ≤ ‖w_OLS‖', n2(wR.slice(1)) <= n2(w.slice(1)) + 1e-12,
    `λ=0.5：${n2(wR.slice(1)).toFixed(6)} ≤ OLS：${n2(w.slice(1)).toFixed(6)}`);
  const { w: wT } = ridgeRegression(X, y, 1e-10);
  const wDiff = Math.max(...wT.map((v, i) => Math.abs(v - w[i])));
  check('ridgeRegression：λ→0 退化为 OLS', wDiff < 1e-6, `最大差=${wDiff.toExponential(2)}`);

  const rng2 = mulberry32(5);
  const Xl = [], yl = [];
  for (let i = 0; i < 20; i++) { Xl.push([gaussianSample(rng2, -2, 0.8), gaussianSample(rng2, -1, 0.8)]); yl.push(0); }
  for (let i = 0; i < 20; i++) { Xl.push([gaussianSample(rng2, 2, 0.8), gaussianSample(rng2, 1, 0.8)]); yl.push(1); }

  const gerr = logisticGradientCheck(Xl, yl, { eps: 1e-5, seed: 9 });
  check('logisticRegression：梯度检验 maxRelErr < 1e-6（中心差分 vs 解析梯度）', gerr < 1e-6,
    `maxRelErr=${gerr.toExponential(2)}`);

  const lg = logisticRegression(Xl, yl, { lr: 0.5, iters: 400, seed: 2 });
  const acc = lg.predict(Xl).filter((p, i) => p === yl[i]).length / yl.length;
  check('logisticRegression：训练收敛（损失下降 + 精度≥0.95）',
    lg.losses[lg.losses.length - 1] < lg.losses[0] && acc >= 0.95,
    `损失 ${lg.losses[0].toFixed(4)} → ${lg.losses[lg.losses.length - 1].toFixed(4)}，acc=${acc}`);

  // kNN：k=1 训练集零误差；朴素贝叶斯后验归一
  const nb = naiveBayesFit(Xl, yl);
  const nbp = nb ? naiveBayesPredict(nb, Xl[0]) : null;
  const knnAcc = Xl.filter((x, i) => knnPredict(Xl, yl, x, 1) === yl[i]).length / Xl.length;
  check('kNN：k=1 训练集精度 = 1（过拟合边界不变量）', knnAcc === 1, `acc=${knnAcc}`);
  check('naiveBayes：后验概率归一 Σ=1 且训练精度≥0.95',
    nbp && Math.abs(nbp.probs.reduce((a, b) => a + b, 0) - 1) < 1e-9 &&
    Xl.filter((x, i) => naiveBayesPredict(nb, x).label === yl[i]).length / Xl.length >= 0.95,
    `Σp=${nbp ? nbp.probs.reduce((a, b) => a + b, 0).toFixed(12) : 'n/a'}`);
}

/* ───────────────────────── 6. 聚类与生成 ───────────────────────── */
function suiteClustering(check) {
  const rng = mulberry32(77);
  const { X, y } = blobs(rng, [[0, 0], [5, 5], [0, 7]], 15, 0.6);
  const km = kmeans(X, 3, { seed: 99, iters: 50 });

  let mono = true;
  const hist = km.sseHistory;
  for (let t = 1; t < hist.length; t++) {
    if (hist[t] > hist[t - 1] + 1e-8 * (1 + Math.abs(hist[t - 1]))) mono = false;
  }
  check('kmeans：SSE 逐迭代单调不增（Lloyd 收敛定理）', mono,
    `SSE ${hist[0].toFixed(3)} → ${hist[hist.length - 1].toFixed(3)}（${hist.length} 轮）`);

  let minCd = Infinity;
  for (let a = 0; a < 3; a++) for (let b = a + 1; b < 3; b++) {
    minCd = Math.min(minCd, norm(km.centers[a].map((v, i) => v - km.centers[b][i])));
  }
  check('kmeans：3 簇中心彼此分离（min 距 > 1）', minCd > 1, `min中心距=${minCd.toFixed(3)}`);

  const rng2 = mulberry32(13);
  const g = blobs(rng2, [[0, 0], [4, 3]], 20, 0.7);
  const gm = gmm(g.X, 2, { seed: 13, iters: 40 });

  let rowSumErr = 0;
  for (const row of gm.resp) rowSumErr = Math.max(rowSumErr, Math.abs(row.reduce((a, b) => a + b, 0) - 1));
  check('GMM：责任度每行 Σ=1（logsumexp 稳定归一）', rowSumErr < 1e-9, `最大偏差=${rowSumErr.toExponential(2)}`);

  const wSum = gm.weights.reduce((a, b) => a + b, 0);
  check('GMM：混合权重 Σ=1', Math.abs(wSum - 1) < 1e-9, `Σw=${wSum.toFixed(12)}`);

  let symErr = 0, minEig = Infinity;
  for (const C of gm.covs) {
    symErr = Math.max(symErr, mdiff(C, transpose(C)));
    minEig = Math.min(minEig, ...jacobiEig(C).values);
  }
  check('GMM：协方差对称半正定', symErr < 1e-9 && minEig > -1e-9,
    `对称偏差=${symErr.toExponential(2)}，min特征值=${minEig.toExponential(2)}`);

  const ll = gm.llHistory;
  let llMono = true;
  for (let t = 1; t < ll.length; t++) {
    if (ll[t] < ll[t - 1] - 1e-8 * (1 + Math.abs(ll[t - 1]))) llMono = false;
  }
  check('GMM：对数似然单调不减（EM 定理）', llMono,
    `LL ${ll[0].toFixed(3)} → ${ll[ll.length - 1].toFixed(3)}（${ll.length} 轮）`);
}

/* ───────────────────────── 7. 降维与决策树 ───────────────────────── */
function suiteDimTree(check) {
  const rng = mulberry32(41);
  const X = randomMatrix(40, 4, rng).map(row => row.map(v => v * 2 + 1));
  const r = pca(X, 4);

  const V = r.components;
  const orthoErr = mdiff(matmul(V, transpose(V)), eye(4));
  check('pca：主轴正交归一 V·Vᵀ = I', orthoErr < 1e-9, `偏差=${orthoErr.toExponential(2)}`);

  let sorted = true;
  for (let i = 0; i + 1 < r.explained.length; i++) {
    if (r.explained[i] < r.explained[i + 1] - 1e-10) sorted = false;
  }
  check('pca：解释方差降序（λ₁ ≥ λ₂ ≥ …）', sorted,
    `λ=[${r.explained.map(v => v.toFixed(3)).join(', ')}]`);

  const sumExpl = r.explained.reduce((a, b) => a + b, 0);
  check('pca：Σ解释方差 = 总方差（k=全维）', Math.abs(sumExpl - r.totalVar) < 1e-9 * (1 + Math.abs(r.totalVar)),
    `Σλ=${sumExpl.toFixed(8)}，总方差=${r.totalVar.toFixed(8)}`);

  const recErr = mdiff(r.reconstruct, X);
  check('pca：全秩重构恒等 X̂ = X', recErr < 1e-8, `最大偏差=${recErr.toExponential(2)}`);

  // XOR：非线性可分，深度 3 完美拟合
  const rng2 = mulberry32(51);
  const Xx = [], Yx = [];
  const patterns = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]];
  for (const pt of patterns) {
    for (let t = 0; t < 3; t++) {
      Xx.push(pt.map(v => v + (rng2() - 0.5) * 0.1));
      Yx.push(pt[2]);
    }
  }
  const treeX = new DecisionTree({ maxDepth: 3, minSamples: 2 }).fit(Xx, Yx);
  const xorAcc = treeX.score(Xx, Yx);
  let giniMono = true;
  for (const s of treeX.splits) if (s.childGini > s.parentGini + 1e-12) giniMono = false;
  check('decisionTree：XOR 训练精度 = 1（深度3）', xorAcc === 1, `acc=${xorAcc}`);
  check('decisionTree：任意分裂 Gini 单调不增（凸性定理）', giniMono,
    `分裂 ${treeX.splits.length} 次，root ${treeX.splits[0].parentGini.toFixed(4)}→${treeX.splits[0].childGini.toFixed(4)}`);

  const rng3 = mulberry32(61);
  const sp = blobs(rng3, [[-3, -3], [3, 3]], 12, 0.5);
  const treeS = new DecisionTree({ maxDepth: 2, minSamples: 2 }).fit(sp.X, sp.y);
  const root = treeS.splits[0];
  check('decisionTree：线性可分数据 root 分裂 Gini 严格下降',
    root && root.childGini < root.parentGini - 1e-9 && treeS.score(sp.X, sp.y) === 1,
    `${root.parentGini.toFixed(4)} → ${root.childGini.toFixed(4)}，acc=${treeS.score(sp.X, sp.y)}`);
}

/* ───────────────────────── 8. 深度学习 ───────────────────────── */
function suiteDeep(check) {
  const rng = mulberry32(55);
  const net = new DenseNet([3, 6, 3], { seed: 5 });
  const X = randomMatrix(12, 3, rng);
  const y = X.map((_, i) => i % 3);

  const gerr = denseGradientCheck(net, X, y);
  check('denseNet：梯度检验 maxRelErr < 1e-6（反向传播 = 中心差分）', gerr < 1e-6,
    `maxRelErr=${gerr.toExponential(2)}`);

  const losses = trainDense(net, X, y, { lr: 0.5, iters: 300 });
  check('denseNet：训练收敛（损失下降）', losses[losses.length - 1] < losses[0],
    `损失 ${losses[0].toFixed(4)} → ${losses[losses.length - 1].toFixed(4)}`);

  const { Q, K, V } = randomAttentionTensors(4, 5, 3, 66);
  const { A, O } = scaledDotProductAttention(Q, K, V);
  let rowErr = 0;
  for (const row of A) rowErr = Math.max(rowErr, Math.abs(row.reduce((a, b) => a + b, 0) - 1));
  check('attention：注意力矩阵行和 = 1（softmax 归一）', rowErr < 1e-12, `最大偏差=${rowErr.toExponential(2)}`);

  const aerr = attentionGradientCheck(Q, K, V);
  check('attention：梯度检验 maxRelErr < 1e-6（手推雅可比 = 中心差分）', aerr < 1e-6,
    `maxRelErr=${aerr.toExponential(2)}`);

  const mha = multiHeadAttention(Q, K, V, { heads: 1 });
  const mhaErr = mdiff(mha.O, O);
  check('multiHead：单头+单位投影 ≡ 缩放点积注意力', mhaErr < 1e-12, `最大偏差=${mhaErr.toExponential(2)}`);

  const Xs = randomMatrix(5, 4, mulberry32(88));
  const { out, attn } = transformerBlock(Xs, { heads: 2, seed: 8 });
  const finite = out.every(row => row.every(isFinite));
  check('transformer：pre-LN 块前向有效（形状 5×4、全有限值）',
    out.length === 5 && out[0].length === 4 && finite, `shape=${out.length}×${out[0].length}`);

  let tRowErr = 0;
  for (const row of attn) tRowErr = Math.max(tRowErr, Math.abs(row.reduce((a, b) => a + b, 0) - 1));
  check('transformer：块内注意力行和 = 1（heads=2 分头正确）', tRowErr < 1e-12, `最大偏差=${tRowErr.toExponential(2)}`);

  const ln = layerNorm([3, 4, 5, 8], { eps: 1e-8 });
  const lnMean = ln.reduce((a, b) => a + b, 0) / ln.length;
  const lnVar = ln.reduce((a, b) => a + (b - lnMean) ** 2, 0) / ln.length;
  check('layerNorm：输出均值≈0、方差≈1', Math.abs(lnMean) < 1e-9 && Math.abs(lnVar - 1) < 1e-6,
    `mean=${lnMean.toExponential(2)}，var=${lnVar.toFixed(12)}`);

  const PE = positionalEncoding(3, 4);
  check('positionalEncoding：PE(0,even)=sin0=0、PE(0,odd)=cos0=1、|PE|≤1',
    Math.abs(PE[0][0]) < 1e-12 && Math.abs(PE[0][1] - 1) < 1e-12 &&
    PE.every(row => row.every(v => Math.abs(v) <= 1 + 1e-12)),
    `PE[0]=[${PE[0].map(v => v.toFixed(4)).join(', ')}]`);
}

/**
 * 旗舰AI引擎 · 不变量自检套件
 * 每条算法至少 1 条可独立交叉验证的硬不变量；Node 与浏览器共用同一份逻辑。
 * 18 组套件全绿 = 引擎数学正确性的机器证明。
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
import { RNN, rnnGradientCheck, trainRNN } from '../nn/rnn.js';
import { LSTM, lstmGradientCheck, trainLSTM } from '../nn/lstm.js';
import { svmTrain, svmPredict, svmDecision, svmDualObjective, svmSupportVectors } from '../ml/svm.js';
import { adaBoost } from '../ml/boosting.js';
import {
  hmmForward, hmmBackward, hmmGamma, hmmXi, hmmBaumWelch, hmmBruteForceLogLik,
} from '../ml/hmm.js';
import {
  valueIteration, policyEvaluation, chainMDP, bruteForceChain, gridworld,
} from '../rl/valueIteration.js';
import { qLearning, greedyAction, qFromV, bellmanResidualQ } from '../rl/qLearning.js';
import { dbscan, adjustedRandIndex } from '../ml/dbscan.js';
import { tsne, knnOverlap, searchBetas } from '../ml/tsne.js';
import {
  sgdMomentumStep, rmspropStep, adamStep, minimize,
} from '../nn/optimizers.js';

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
 * 运行全部 14 组不变量自检。
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
    ['循环神经网络 · RNN & BPTT', suiteRNN],
    ['门控循环网络 · LSTM & BPTT', suiteLSTM],
    ['核支持向量机 · Kernel SVM', suiteSVM],
    ['AdaBoost 集成 · Boosting', suiteBoosting],
    ['隐马尔可夫模型 · HMM', suiteHMM],
    ['强化学习 · Reinforcement Learning', suiteRL],
    ['Q-learning · 免模型TD控制', suiteQLearning],
    ['密度聚类 · DBSCAN', suiteDBSCAN],
    ['非线性降维 · t-SNE', suiteTSNE],
    ['一阶优化器 · Optimizers', suiteOptim],
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

/* ───────────────────────── 9. 循环神经网络 RNN ───────────────────────── */
function suiteRNN(check) {
  const rng = mulberry32(101);
  const net = new RNN({ inputSize: 3, hiddenSize: 5, outputSize: 2, seed: 5 });
  const X = randomMatrix(10, 3, rng, -1, 1);
  const targets = X.map(row => [row[0] * 0.5 - 0.2, row[1] + 0.1]);

  const gerr = rnnGradientCheck(net, X, targets);
  check('RNN：BPTT 解析梯度 ≡ 中心差分（显著分量 maxRelErr < 1e-6）',
    gerr.maxRelErr < 1e-6 && gerr.maxAbsErr < 1e-7,
    `maxRelErr=${gerr.maxRelErr.toExponential(2)}，maxAbsErr=${gerr.maxAbsErr.toExponential(2)}（${gerr.significant} 个显著分量）`);

  const cache = net.forward(X);
  const bounded = cache.H.every(row => row.every(v => Math.abs(v) < 1));
  check('RNN：隐藏态 tanh 有界 h_t ∈ (−1,1)', bounded && cache.H.length === 10,
    `T=${cache.H.length}，全有限有界=${bounded}`);

  const { losses } = trainRNN(net, X, targets, { lr: 0.05, iters: 300 });
  check('RNN：序列回归训练收敛（MSE 下降）',
    losses[losses.length - 1] < losses[0],
    `损失 ${losses[0].toFixed(4)} → ${losses[losses.length - 1].toFixed(4)}`);

  // 长序列（T=40）BPTT 仍精确（时间梯度传播稳定性）
  const rng2 = mulberry32(102);
  const Xl = randomMatrix(40, 3, rng2, -1, 1);
  const tl = Xl.map(row => [row[2] * 0.3]);
  const net2 = new RNN({ inputSize: 3, hiddenSize: 4, outputSize: 1, seed: 9 });
  const gerr2 = rnnGradientCheck(net2, Xl, tl);
  check('RNN：T=40 长序列 BPTT 梯度检验仍 < 1e-6', gerr2.maxRelErr < 1e-6 && gerr2.maxAbsErr < 1e-7,
    `maxRelErr=${gerr2.maxRelErr.toExponential(2)}，maxAbsErr=${gerr2.maxAbsErr.toExponential(2)}`);
}

/* ───────────────────────── 10. LSTM 门控网络 ───────────────────────── */
function suiteLSTM(check) {
  const rng = mulberry32(103);
  const net = new LSTM({ inputSize: 3, hiddenSize: 5, outputSize: 2, seed: 6 });
  const X = randomMatrix(12, 3, rng, -1, 1);
  const targets = X.map(row => [row[0] * 0.5, row[1] - 0.3]);

  const gerr = lstmGradientCheck(net, X, targets);
  check('LSTM：BPTT 解析梯度 ≡ 中心差分（显著分量 maxRelErr < 1e-6）',
    gerr.maxRelErr < 1e-6 && gerr.maxAbsErr < 1e-7,
    `maxRelErr=${gerr.maxRelErr.toExponential(2)}，maxAbsErr=${gerr.maxAbsErr.toExponential(2)}（${gerr.significant} 个显著分量）`);

  const cache = net.forward(X);
  const gatesOk = cache.I.every(r => r.every(v => v > 0 && v < 1))
    && cache.F.every(r => r.every(v => v > 0 && v < 1))
    && cache.O.every(r => r.every(v => v > 0 && v < 1))
    && cache.G.every(r => r.every(v => Math.abs(v) < 1));
  check('LSTM：门控值域（i/f/o ∈ (0,1)，候选 g ∈ (−1,1)）', gatesOk, `全部 ${cache.I.length} 时间步`);

  const { losses } = trainLSTM(net, X, targets, { lr: 0.05, iters: 300 });
  check('LSTM：序列回归训练收敛（MSE 下降）',
    losses[losses.length - 1] < losses[0],
    `损失 ${losses[0].toFixed(4)} → ${losses[losses.length - 1].toFixed(4)}`);

  // 长程依赖：T=50 时细胞态仍保持非退化（梯度高速公路不消失）
  const rng2 = mulberry32(104);
  const Xl = randomMatrix(50, 2, rng2, -1, 1);
  const tl = Xl.map(row => [row[0] * 0.4]);
  const net2 = new LSTM({ inputSize: 2, hiddenSize: 4, outputSize: 1, seed: 11 });
  const gerr2 = lstmGradientCheck(net2, Xl, tl);
  check('LSTM：T=50 长序列 BPTT 梯度检验仍 < 1e-6', gerr2.maxRelErr < 1e-6 && gerr2.maxAbsErr < 1e-7,
    `maxRelErr=${gerr2.maxRelErr.toExponential(2)}，maxAbsErr=${gerr2.maxAbsErr.toExponential(2)}`);
}

/* ───────────────────────── 11. 核支持向量机 ───────────────────────── */
function suiteSVM(check) {
  const rng = mulberry32(107);
  const X = [], y = [];
  for (let i = 0; i < 15; i++) { X.push([gaussianSample(rng, -2, 0.6), gaussianSample(rng, -2, 0.6)]); y.push(-1); }
  for (let i = 0; i < 15; i++) { X.push([gaussianSample(rng, 2, 0.6), gaussianSample(rng, 2, 0.6)]); y.push(1); }

  const model = svmTrain(X, y, { C: 1, gamma: 0.5 });

  const sumAy = model.alpha.reduce((s, a, i) => s + a * y[i], 0);
  check('SVM：对偶等式约束 Σαᵢyᵢ = 0（SMO 成对更新恒保持）',
    Math.abs(sumAy) < 1e-6, `Σαy=${sumAy.toExponential(2)}`);

  let kktMax = 0;
  for (let i = 0; i < X.length; i++) {
    const m = y[i] * svmDecision(model, X[i]);
    if (model.alpha[i] < 1e-6) kktMax = Math.max(kktMax, Math.max(0, 1 - m));
    else if (model.alpha[i] > model.C - 1e-6) kktMax = Math.max(kktMax, Math.max(0, m - 1));
    else kktMax = Math.max(kktMax, Math.abs(m - 1));
  }
  check('SVM：KKT 互补松弛（α=0→yf≥1；0<α<C→yf=1；α=C→yf≤1）',
    kktMax < 0.05, `最大违反=${kktMax.toFixed(4)}`);

  const acc = X.filter((x, i) => svmPredict(model, x) === y[i]).length / X.length;
  check('SVM：线性可分数据训练精度 = 1（最大间隔几何意义）', acc === 1, `acc=${acc}`);

  const dual = svmDualObjective(model);
  check('SVM：对偶目标 W(α) > 0（非平凡最大间隔解）', dual > 0, `W(α)=${dual.toFixed(4)}`);

  const sv = svmSupportVectors(model);
  check('SVM：存在支持向量（0 < α < C，其余点 α=0）', sv.length > 0 && sv.length < X.length,
    `|SV|=${sv.length}/${X.length}`);
}

/* ───────────────────────── 12. AdaBoost 集成 ───────────────────────── */
function suiteBoosting(check) {
  const rng = mulberry32(109);
  const X = [], y = [];
  for (let i = 0; i < 20; i++) { X.push([gaussianSample(rng, -1.5, 0.5), gaussianSample(rng, -1.5, 0.5)]); y.push(-1); }
  for (let i = 0; i < 20; i++) { X.push([gaussianSample(rng, 1.5, 0.5), gaussianSample(rng, 1.5, 0.5)]); y.push(1); }

  const ab = adaBoost(X, y, { rounds: 25 });

  const maxEps = Math.max(...ab.errors);
  check('AdaBoost：每轮加权误差 ε_t < ½（弱学习器假设）', maxEps < 0.5,
    `max ε=${maxEps.toFixed(4)}（25 轮）`);

  let mono = true;
  for (let t = 1; t < ab.expLoss.length; t++) {
    if (ab.expLoss[t] > ab.expLoss[t - 1] + 1e-9) mono = false;
  }
  check('AdaBoost：指数损失 E_t 单调不增（Z_t = 2√(ε_t(1−ε_t)) ≤ 1）', mono,
    `E ${ab.expLoss[0].toFixed(3)} → ${ab.expLoss[ab.expLoss.length - 1].toFixed(4)}`);

  let bound = 1;
  for (const e of ab.errors) bound *= 2 * Math.sqrt(e * (1 - e));
  const trainErr = X.filter((x, i) => ab.predict(x) !== y[i]).length / X.length;
  check('AdaBoost：训练误差 ≤ Π_t 2√(ε_t(1−ε_t))（Freund–Schapire 定理）',
    trainErr <= bound + 1e-12, `实测=${trainErr}，理论上界=${bound.toFixed(4)}`);

  check('AdaBoost：线性可分数据训练误差 → 0', trainErr === 0, `训练误差=${trainErr}`);
}

/* ───────────────────────── 13. 隐马尔可夫模型 ───────────────────────── */
function suiteHMM(check) {
  const rng = mulberry32(113);
  const randStoch = (rows, cols) => Array.from({ length: rows }, () => {
    const v = Array.from({ length: cols }, () => rng() + 0.1);
    const s = v.reduce((a, b) => a + b, 0);
    return v.map(x => x / s);
  });
  const N = 3, M = 2;
  const A = randStoch(N, N), B = randStoch(N, M);
  const pv = Array.from({ length: N }, () => rng() + 0.1);
  const pi = pv.map(v => v / pv.reduce((a, b) => a + b, 0));
  const obs = Array.from({ length: 6 }, () => Math.floor(rng() * M));

  // ① 缩放前向 vs 3^6 = 729 条路径暴力枚举（两个独立实现）
  const fw = hmmForward(A, B, pi, obs);
  const bf = hmmBruteForceLogLik(A, B, pi, obs);
  check('HMM：缩放前向 logP(O) ≡ 729 路径暴力枚举（logsumexp）',
    Math.abs(fw.logLik - bf) < 1e-9,
    `前向=${fw.logLik.toFixed(10)}，暴力=${bf.toFixed(10)}`);

  // ② 平滑后验归一
  const beta = hmmBackward(A, B, obs, fw.c);
  const gamma = hmmGamma(fw.alphaHat, beta);
  let gErr = 0;
  for (const row of gamma) gErr = Math.max(gErr, Math.abs(row.reduce((a, b) => a + b, 0) - 1));
  check('HMM：平滑后验 γ_t 每行 Σ=1', gErr < 1e-9, `最大偏差=${gErr.toExponential(2)}`);

  // ③ ξ 归一 + 前后向一致性 Σⱼξ_t(i,j) = γ_t(i)
  const xi = hmmXi(A, B, fw.alphaHat, beta, obs, fw.c);
  let xiErr = 0, consErr = 0;
  for (let t = 0; t < xi.length; t++) {
    let s = 0;
    for (const row of xi[t]) for (const v of row) s += v;
    xiErr = Math.max(xiErr, Math.abs(s - 1));
    for (let i = 0; i < N; i++) {
      const rowSum = xi[t][i].reduce((a, b) => a + b, 0);
      consErr = Math.max(consErr, Math.abs(rowSum - gamma[t][i]));
    }
  }
  check('HMM：ξ_t 归一 Σ=1 且 Σⱼξ_t(i,j)=γ_t(i)（前后向一致）',
    xiErr < 1e-9 && consErr < 1e-9, `ξ偏差=${xiErr.toExponential(2)}，一致性=${consErr.toExponential(2)}`);

  // ④ Baum-Welch 对数似然单调不减（EM 定理）
  const obs2 = Array.from({ length: 24 }, () => Math.floor(rng() * M));
  const bw = hmmBaumWelch(A, B, pi, obs2, { iters: 25 });
  const ll = bw.logLikHistory;
  let llMono = true;
  for (let t = 1; t < ll.length; t++) {
    if (ll[t] < ll[t - 1] - 1e-8 * (1 + Math.abs(ll[t - 1]))) llMono = false;
  }
  check('HMM：Baum-Welch 对数似然单调不减（EM 定理）', llMono && ll.length >= 2,
    `LL ${ll[0].toFixed(3)} → ${ll[ll.length - 1].toFixed(3)}（${ll.length} 轮）`);

  // ⑤ 重估参数仍是合法随机矩阵
  const rowOk = (m) => m.every(r => Math.abs(r.reduce((a, b) => a + b, 0) - 1) < 1e-9 && r.every(v => v >= -1e-12));
  check('HMM：Baum-Welch 输出为合法分布（行随机、非负）',
    rowOk(bw.A) && rowOk(bw.B) && rowOk([bw.pi]), 'A/B/π 均归一');
}

/* ───────────────────────── 14. 强化学习 ───────────────────────── */
function suiteRL(check) {
  // 链式 MDP：值迭代 ≡ 2^6 全策略暴力枚举 ≡ 解析闭式
  const n = 6, gamma = 0.9;
  const { P, R } = chainMDP({ n, gamma });
  const vi = valueIteration(P, R, gamma);
  const bf = bruteForceChain({ n, gamma });
  let chainErr = 0;
  for (let s = 0; s < n; s++) chainErr = Math.max(chainErr, Math.abs(vi.V[s] - bf.V[s]));
  check('RL：链式 MDP 值迭代 ≡ 2⁶ 全策略暴力枚举', chainErr < 1e-9,
    `最大偏差=${chainErr.toExponential(2)}`);

  // 闭式：奖励在"进入终点的前一步"支付 ⇒ V*(s) = γ^(n−2−s)（s < n−1）
  let closedErr = 0;
  for (let s = 0; s < n - 1; s++) {
    closedErr = Math.max(closedErr, Math.abs(vi.V[s] - Math.pow(gamma, n - 2 - s)));
  }
  check('RL：链式 MDP V*(s) = γ^(n−2−s)（解析闭式）', closedErr < 1e-9,
    `最大偏差=${closedErr.toExponential(2)}，V*(0)=${vi.V[0].toFixed(6)}=γ⁴`);

  check('RL：Bellman 残差 → 0（压缩不动点定理）', vi.residual < 1e-10,
    `残差=${vi.residual.toExponential(2)}（${vi.iterations} 轮）`);

  // 随机网格世界：双算法互证 + 吸收态闭式
  const gw = gridworld({ size: 4 });
  const gv = valueIteration(gw.P, gw.R, gw.gamma);
  check('RL：随机网格世界 Bellman 残差 → 0（γ=0.9 压缩映射）', gv.residual < 1e-10,
    `残差=${gv.residual.toExponential(2)}（${gv.iterations} 轮）`);

  const pe = policyEvaluation(gw.P, gw.R, gw.gamma, gv.policy);
  let peErr = 0;
  for (let s = 0; s < gw.S; s++) peErr = Math.max(peErr, Math.abs(pe.V[s] - gv.V[s]));
  check('RL：贪心策略评估 V^{π*} ≡ V*（值迭代 vs 策略评估互证）', peErr < 1e-6,
    `最大偏差=${peErr.toExponential(2)}`);

  const goalClosed = gw.goalReward / (1 - gw.gamma);
  const trapClosed = gw.trapReward / (1 - gw.gamma);
  const absorbErr = Math.max(Math.abs(gv.V[gw.goal] - goalClosed), Math.abs(gv.V[gw.trap] - trapClosed));
  check('RL：吸收态闭式 V = r/(1−γ)（几何级数）', absorbErr < 1e-6,
    `V(终点)=${gv.V[gw.goal].toFixed(6)} vs ${goalClosed}；V(陷阱)=${gv.V[gw.trap].toFixed(6)} vs ${trapClosed}`);

  check('RL：V*(终点) > V*(陷阱)（最优策略趋利避害）',
    gv.V[gw.goal] > gv.V[gw.trap] && gv.V[gw.goal] > 0 && gv.V[gw.trap] < 0,
    `V(终点)=${gv.V[gw.goal].toFixed(3)} > V(陷阱)=${gv.V[gw.trap].toFixed(3)}`);
}

/** ⑮ Q-learning：免模型时序差分控制（Watkins 1989） */
function suiteQLearning(check) {
  // ① 链式 MDP：Q-learning ≡ 值迭代 Q* ≡ 解析闭式 γ^(n−2−s)
  const n = 6, gamma = 0.9;
  const cm = chainMDP({ n, gamma });
  const ql = qLearning(cm.P, cm.R, {
    episodes: 20000, alpha: 0.2, epsStart: 1, epsEnd: 0.05, gamma, maxSteps: 60, seed: 42,
  });
  const vi = valueIteration(cm.P, cm.R, gamma);
  const Qstar = qFromV(cm.P, cm.R, gamma, vi.V);

  let closedErr = 0;
  for (let s = 0; s < n - 1; s++) {
    closedErr = Math.max(closedErr, Math.abs(ql.Q[s][1] - Math.pow(gamma, n - 2 - s)));
  }
  check('Q-learning：链式 MDP Q*(s,→) ≡ γ^(n−2−s)（解析闭式）', closedErr < 5e-2,
    `最大偏差=${closedErr.toExponential(2)}，Q(0,→)=${ql.Q[0][1].toFixed(4)} vs γ⁴=${Math.pow(gamma, 4).toFixed(4)}`);

  let viErr = 0;
  for (let s = 0; s < n; s++) {
    for (let a = 0; a < 2; a++) viErr = Math.max(viErr, Math.abs(ql.Q[s][a] - Qstar[s][a]));
  }
  check('Q-learning：Q ≡ 值迭代反推 Q*（免模型 vs 基于模型互证）', viErr < 5e-2,
    `最大偏差=${viErr.toExponential(2)}`);

  const resid = bellmanResidualQ(cm.P, cm.R, gamma, ql.Q);
  check('Q-learning：Bellman 残差 → 0（收敛至不动点）', resid < 5e-2,
    `残差=${resid.toExponential(2)}`);

  // ② 网格世界：贪心策略 ≡ 值迭代最优策略
  const gw = gridworld();
  const ql2 = qLearning(gw.P, gw.R, {
    episodes: 30000, alpha: 0.2, epsStart: 1, epsEnd: 0.05, gamma: gw.gamma, maxSteps: 60, seed: 7,
  });
  const vi2 = valueIteration(gw.P, gw.R, gw.gamma);
  let optErr = 0, exact = 0;
  for (let s = 0; s < gw.S; s++) {
    optErr = Math.max(optErr, ql2.Q[s][vi2.policy[s]] - Math.max(...ql2.Q[s]));
    if (greedyAction(ql2.Q, s) === vi2.policy[s]) exact++;
  }
  check('Q-learning：网格世界 π*(s) 动作在 Q 中最优（贪心策略 ≡ 值迭代 π*）', optErr < 5e-2,
    `次优缺口=${optErr.toExponential(2)}，逐状态一致 ${exact}/${gw.S}`);
  check('Q-learning：网格世界策略逐状态一致率 ≥ 75%', exact / gw.S >= 0.75,
    `一致率=${(100 * exact / gw.S).toFixed(1)}%`);

  // ③ 学习改善：后期回合平均回报 > 前期
  const er = ql.episodeReturns;
  const k = Math.floor(er.length / 10);
  const early = er.slice(0, k).reduce((a, b) => a + b, 0) / k;
  const late = er.slice(-k).reduce((a, b) => a + b, 0) / k;
  check('Q-learning：后期回合回报 > 前期（探索 → 利用）', late > early,
    `早期=${early.toFixed(3)} → 后期=${late.toFixed(3)}`);
}

/** ⑯ DBSCAN：密度聚类（Ester et al. 1996） */
function suiteDBSCAN(check) {
  const rng = mulberry32(123);
  const X = [], truth = [];
  for (let i = 0; i < 40; i++) { X.push([gaussianSample(rng, 0, 0.3), gaussianSample(rng, 0, 0.3)]); truth.push(0); }
  for (let i = 0; i < 40; i++) { X.push([gaussianSample(rng, 8, 0.3), gaussianSample(rng, 8, 0.3)]); truth.push(1); }
  for (let i = 0; i < 5; i++) { X.push([gaussianSample(rng, 20, 1), gaussianSample(rng, 20, 1)]); truth.push(-1); }
  const n = X.length;
  const { labels, core, nClusters, noise } = dbscan(X, { eps: 1.0, minPts: 5 });

  const ari = adjustedRandIndex(labels, truth);
  check('DBSCAN：两团簇 ARI = 1（划分与真实标注完全一致）', ari > 0.999,
    `ARI=${ari.toFixed(6)}，簇数=${nClusters}，噪声=${noise}`);
  check('DBSCAN：恰好 2 簇 + 5 个离群点（稀疏区全部识别为噪声）',
    nClusters === 2 && noise === 5, `簇数=${nClusters}，噪声=${noise}`);

  // 核心见证：每个非噪声点在其标签簇内有 ε-邻域核心点
  let witnessOk = true;
  for (let i = 0; i < n; i++) {
    if (labels[i] === -1) continue;
    let ok = false;
    for (let j = 0; j < n; j++) {
      if (labels[j] !== labels[i] || !core[j]) continue;
      const d2 = (X[i][0] - X[j][0]) ** 2 + (X[i][1] - X[j][1]) ** 2;
      if (d2 <= 1.0 + 1e-12) { ok = true; break; }
    }
    if (!ok) { witnessOk = false; break; }
  }
  check('DBSCAN：核心点见证（每个非噪声点邻域内必有同簇核心点）', witnessOk);

  // 密度连通传递性：每簇核心点在 ε-邻域图上构成单一连通分量
  let connectedOk = true;
  for (let c = 0; c < nClusters; c++) {
    const cores = [];
    for (let i = 0; i < n; i++) if (labels[i] === c && core[i]) cores.push(i);
    if (cores.length === 0) { connectedOk = false; break; }
    const seen = new Set([cores[0]]);
    const queue = [cores[0]];
    while (queue.length) {
      const p = queue.shift();
      for (const q of cores) {
        if (seen.has(q)) continue;
        const d2 = (X[p][0] - X[q][0]) ** 2 + (X[p][1] - X[q][1]) ** 2;
        if (d2 <= 1.0 + 1e-12) { seen.add(q); queue.push(q); }
      }
    }
    if (seen.size !== cores.length) connectedOk = false;
  }
  check('DBSCAN：密度连通传递性（每簇核心点单一连通分量）', connectedOk);

  // 离群稳健：追加孤立远点 → 噪声，且原有划分不变
  const X2 = [...X, [100, 100]];
  const r2 = dbscan(X2, { eps: 1.0, minPts: 5 });
  const stable = r2.labels.slice(0, n).every((v, i) => v === labels[i]);
  check('DBSCAN：追加孤立远点 → 识别为噪声且原划分逐点不变',
    r2.labels[n] === -1 && stable && r2.nClusters === 2, `新点标签=${r2.labels[n]}，划分稳定=${stable}`);
}

/** ⑰ t-SNE：非线性降维（van der Maaten & Hinton 2008） */
function suiteTSNE(check) {
  const rng = mulberry32(7);
  const X = [];
  for (const c of [[0, 0], [6, 0], [3, 5]]) {
    for (let i = 0; i < 30; i++) {
      X.push([gaussianSample(rng, c[0], 0.5), gaussianSample(rng, c[1], 0.5)]);
    }
  }
  const n = X.length;
  const perplexity = 15;
  const { Y, P, Q, costs, entropies } = tsne(X, {
    dims: 2, perplexity, iters: 400, lr: 100, seed: 42,
  });

  // ① P/Q 对称联合分布、概率质量守恒
  let symErr = 0, massP = 0, massQ = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      symErr = Math.max(symErr, Math.abs(P[i][j] - P[j][i]), Math.abs(Q[i][j] - Q[j][i]));
      massP += P[i][j];
      massQ += Q[i][j];
    }
  }
  check('t-SNE：P/Q 对称联合分布（ΣP = ΣQ = 1）',
    symErr < 1e-12 && Math.abs(massP - 1) < 1e-9 && Math.abs(massQ - 1) < 1e-9,
    `对称误差=${symErr.toExponential(2)}，ΣP=${massP.toFixed(9)}，ΣQ=${massQ.toFixed(9)}`);

  // ② 困惑度达成
  let perpErr = 0;
  for (const h of entropies) perpErr = Math.max(perpErr, Math.abs(Math.pow(2, h) - perplexity) / perplexity);
  check('t-SNE：二分搜索困惑度达成（2^H ≈ perplexity）', perpErr < 0.05,
    `最大相对偏差=${(100 * perpErr).toFixed(2)}%（目标=${perplexity}）`);

  // ③ KL 代价单调不增
  let mono = true;
  for (let k = 1; k < costs.length; k++) if (costs[k] > costs[k - 1] + 1e-3) mono = false;
  check('t-SNE：KL 代价随迭代单调不增（学习有效）',
    mono && costs[costs.length - 1] < costs[0],
    `初=${costs[0].toFixed(4)} → 末=${costs[costs.length - 1].toFixed(4)}（${costs.length} 个采样点）`);

  // ④ 结构保持：可分团簇 k-NN 邻域重叠
  const ov = knnOverlap(X, Y, 5);
  check('t-SNE：可分团簇 k-NN 邻域重叠率 ≥ 0.85（局部结构保真）', ov >= 0.85,
    `重叠率=${ov.toFixed(4)}`);
}

/** ⑱ 一阶优化器：SGD-Momentum / RMSProp / Adam */
function suiteOptim(check) {
  // ① Adam 偏差修正首步：Δ = −lr·sign(g)（eps→0 极限下精确成立且尺度不变）
  const g = [0.5, -2, 0.001];
  const lr = 0.01;
  const expected = g.map((v) => -lr * Math.sign(v));
  const r1 = adamStep([0, 0, 0], g, {}, 1, { lr, eps: 0 });
  const r2 = adamStep([0, 0, 0], g.map((v) => v * 1000), {}, 1, { lr, eps: 0 });
  let e1 = 0, e2 = 0;
  for (let i = 0; i < 3; i++) {
    e1 = Math.max(e1, Math.abs(r1.x[i] - expected[i]));
    e2 = Math.max(e2, Math.abs(r2.x[i] - expected[i]));
  }
  check('Adam：偏差修正首步 Δ = −lr·sign(g)（m̂/√v̂ = g/|g|，ulp 级精确）', e1 < 1e-12,
    `首步=[${r1.x.map((v) => v.toFixed(6)).join(', ')}]，期望=[${expected.map((v) => v.toFixed(6)).join(', ')}]`);
  check('Adam：首步尺度不变（g 与 1000g 产生相同步长）', e2 < 1e-12,
    `偏差=${e2.toExponential(2)}`);

  // ② 动量重球闭式：v_k = g·(1−m^k)/(1−m)
  const m = 0.9;
  let st = {};
  let x = [0];
  const vs = [];
  for (let t = 1; t <= 6; t++) {
    const out = sgdMomentumStep(x, [1.5], st, t, { lr: 0.1, momentum: m });
    st = out.state;
    x = out.x;
    vs.push(st.v[0]);
  }
  let vErr = 0;
  for (let k = 1; k <= 6; k++) {
    vErr = Math.max(vErr, Math.abs(vs[k - 1] - 1.5 * (1 - Math.pow(m, k)) / (1 - m)));
  }
  check('SGD-Momentum：速度闭式 v_k = g(1−m^k)/(1−m)（重球几何累积）', vErr < 1e-12,
    `v_6=${vs[5].toFixed(9)} vs 闭式=${(1.5 * (1 - Math.pow(m, 6)) / (1 - m)).toFixed(9)}`);

  // ③ 凸二次型：三优化器均收敛到解析最优 x* = A⁻¹b，损失单调不增
  const A = [[4, 1], [1, 3]], b = [1, 2];
  const xStar = solve(A, b);
  const f = (x) => 0.5 * (A[0][0] * x[0] ** 2 + 2 * A[0][1] * x[0] * x[1] + A[1][1] * x[1] ** 2)
    - (b[0] * x[0] + b[1] * x[1]);
  const grad = (x) => [
    A[0][0] * x[0] + A[0][1] * x[1] - b[0],
    A[1][0] * x[0] + A[1][1] * x[1] - b[1],
  ];
  const cases = [
    ['SGD-Momentum', 'sgdMomentum', { lr: 0.01, momentum: 0.9 }],
    ['RMSProp', 'rmsprop', { lr: 0.1, beta: 0.9 }],
    ['Adam', 'adam', { lr: 0.2 }],
  ];
  for (const [name, opt, opts] of cases) {
    const { x: xf, history } = minimize(f, grad, [3, -2], { optimizer: opt, iters: 2000, ...opts });
    const err = Math.max(Math.abs(xf[0] - xStar[0]), Math.abs(xf[1] - xStar[1]));
    // 包络单调：每 20 步采样非增（一阶方法早期过冲属经典行为，逐步单调非普遍保证）
    let mono = true;
    for (let k = 20; k < history.length; k += 20) {
      if (history[k] > history[k - 20] + 1e-6) mono = false;
    }
    check(`${name}：凸二次型收敛到解析最优 x* = A⁻¹b`, err < 1e-6,
      `x=[${xf.map((v) => v.toFixed(8)).join(', ')}]，x*=[${xStar.map((v) => v.toFixed(8)).join(', ')}]`);
    check(`${name}：损失包络单调不增（每 20 步采样）`,
      mono && history[history.length - 1] < history[0] * 1e-6,
      `f: ${history[0].toExponential(3)} → ${history[history.length - 1].toExponential(3)}`);
  }
}

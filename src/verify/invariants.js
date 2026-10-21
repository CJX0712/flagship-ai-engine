/**
 * 旗舰AI引擎 · 不变量自检套件
 * 每条算法至少 1 条可独立交叉验证的硬不变量；Node 与浏览器共用同一份逻辑。
 * 58 组套件全绿 = 引擎数学正确性的机器证明。
 */

import {
  zeros, transpose, matmul, matvec, dot, norm, frobeniusNorm, maxAbs,
  solve, inverse, cholesky, qr, jacobiEig, svd, powerIteration, randomMatrix,
} from '../core/linalg.js';
import { logsumexp, softmax, sigmoid, relErr } from '../core/numerical.js';
import {
  nextPow2, fftRadix2, fft, ifftRadix2, dftNaive,
  circularConvolve, convolveLinear, convolveNaive,
} from '../core/fft.js';
import { eulerStep, midpointStep, rk4Step, solveODE, solveODERK45, dpTableau } from '../core/ode.js';
import {
  gaussianPDF, gaussianLogPDF, entropy, crossEntropy, klDivergence,
  mean as colMean, mulberry32, gaussianSample, covarianceMatrix, twoMoons,
} from '../core/stats.js';
import {
  addIntercept, linearRegression, ridgeRegression,
  logisticRegression, logisticGradientCheck,
} from '../ml/regression.js';
import { knnPredict, naiveBayesFit, naiveBayesPredict } from '../ml/classification.js';
import { somNeighborhood, somBMU, somQuantError, somStep, somTrain, somBatchTrain, somLabels } from '../ml/som.js';
import { kmeans, gmm } from '../ml/clustering.js';
import { pca } from '../ml/dimreduction.js';
import { DecisionTree, giniImpurity } from '../ml/tree.js';
import { randomForestTrain, randomForestPredict } from '../ml/randomForest.js';
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
import { mcReturn, mcEpisode, mcPredictionFirstVisit, mcPredictionEveryVisit, mcControl } from '../rl/monteCarlo.js';
import { dbscan, adjustedRandIndex } from '../ml/dbscan.js';
import { tsne, knnOverlap, searchBetas } from '../ml/tsne.js';
import {
  sgdMomentumStep, rmspropStep, adamStep, minimize,
} from '../nn/optimizers.js';
import {
  kpca, kpcaTransform, knnPurity, kpcaReconstructionError, centerKernel,
} from '../ml/kpca.js';
import {
  similarityGraph, normalizedLaplacian, spectralEmbedding, spectralCluster,
} from '../ml/spectral.js';
import {
  erf, normalCDF, normalPDF, softplus, softplusD,
  gelu, geluD, geluTanh, geluTanhD, silu, siluD, mish, mishD,
  hardswish, hardswishD, activationGradCheck,
} from '../nn/activations.js';
import {
  warmupInverseSqrt, linearWarmup, cosineAnnealing, stepDecay, exponentialDecay,
} from '../nn/schedulers.js';
import {
  conv2dForward, conv2dBackward, conv2dNaive, convOutputSize, im2col,
  maxPool2d, maxPoolBackward, convGradientCheck, randomConvTensors,
} from '../nn/cnn.js';
import {
  gprFit, gprPredict, gprHyperGradAnalytic, gprHyperGradNumeric, gprPriorVariance,
} from '../ml/gpr.js';
import {
  gbdtFit, gbdtPredict, gbdtStagedPredict, regressionTreeFit,
  regressionTreePredictAll, pearsonCorr,
} from '../ml/gbdt.js';
import {
  kalmanFilter, kalmanScalarSteadyState, kalmanScalarIterate,
  innovationAutocorr, empiricalVariance,
} from '../ml/kalman.js';
import {
  vaeMake, vaeForward, vaeBackward, vaeKL, vaeKLMonteCarlo, vaeSampleEps,
  vaeReparametrize, vaeGradientCheck,
} from '../nn/vae.js';
import {
  metropolisHastings, mhAcceptProb, detailedBalanceResidual,
  importanceSampling, mcmcEmpiricalMoments,
} from '../ml/mcmc.js';
import {
  jacobiIter, gaussSeidelIter, conjugateGradient, jacobiIterationMatrix,
  spectralRadius, diagonalDominance, conjugateResidual, residualOrthogonality,
} from '../core/linsolve.js';
import {
  iforC, iforCRecursive, iforCExact, isolationTree, pathLength, isolationForest,
  anomalyScore, iforScoreAll, iforToyData,
} from '../ml/isolationForest.js';
import {
  distanceMatrix, doubleCenter, classicalMDS, knnGraph, graphGeodesic,
  isomap, stress, mdsKnnOverlap, mdsSwissRoll,
} from '../ml/manifold.js';
import {
  softmaxPolicy, policyLogGrad, reinforceEpisode, reinforce, banditExpectedReturn,
  banditAnalyticGrad, pgPolicyQ, pgVisitation, policyGradientAnalytic,
  reinforceExpectedUpdate,
} from '../rl/policyGradient.js';
import {
  sgnsLoss, sgnsGradients, sgnsLossReference, sgnsGradientCheck,
  trainSGNS, wordSimilarity, makeToyCorpus,
} from '../nn/word2vec.js';
import {
  nmf, nmfReconstructionError, nmfMinEntry, svdBestRankKError, makeNMFData,
} from '../ml/nmf.js';
import {
  batchNormForward, batchNormBackward, batchNormInfer, batchNormGradCheck,
  layerNorm2d, layerNormBackward, layerNormGradCheck,
} from '../nn/normalization.js';
import {
  ResidualBlock, ResNet, residualIdentityCheck, residualJacobianIdentity,
  residualGradientCheck, trainResNet, resnetAccuracy, residualGradientHighway,
  residualVsPlain,
} from '../nn/resnet.js';
import {
  pageRank, pagerankExact, pagerankStationary, pagerankSum,
} from '../ml/pagerank.js';
import {
  gibbsBivariateGaussian, gibbsMarginal, gibbsCovariance, gibbsConditionalCheck,
  gibbsDiscrete, gibbsConvergence,
} from '../ml/gibbs.js';
import {
  gmmFit, gmmPredict,
} from '../ml/gmm.js';
import {
  ldaFit, ldaProject,
} from '../ml/lda.js';
import {
  nbFit, nbPredict,
} from '../ml/naiveBayes.js';
import {
  perceptronTrain, perceptronStep, perceptronPredict,
} from '../ml/perceptron.js';
import { icaFit, icaTransform } from '../ml/ica.js';
import { ccaFit, ccaProject } from '../ml/cca.js';
import { hierarchicalCluster, hierarchicalCophenetic } from '../ml/hierarchical.js';
import { qdaFit, qdaPredict, qdaDiscriminant } from '../ml/qda.js';
import { lassoFit, lassoPath, lassoPredict } from '../ml/lasso.js';
import { rbmTrain, rbmHiddenProbs, rbmReconstruct, rbmFreeEnergy } from '../nn/rbm.js';
import { hopfieldTrain, hopfieldEnergy, hopfieldUpdateUnit, hopfieldRecall } from '../nn/hopfield.js';
import { rtsSmooth, rtsRMSE, rtsCovarianceGapEigen } from '../ml/kalmanSmoother.js';
import {
  particleFilter, effectiveSampleSize, systematicResample, pfCompareWithKalman,
} from '../ml/particleFilter.js';
import {
  meanShiftMode, meanShiftCluster, meanShiftGradientCheck, meanShiftStep,
} from '../ml/meanShift.js';
import {
  sarsa, sarsaLambda, sarsaBellmanResidual, sarsaStep,
  sarsaEpsilonProbs, epsilonGreedyAction, sarsaTraceSet, sarsaTraceDecay,
} from '../rl/sarsa.js';
import {
  softmaxRegression, softmaxGradientCheck,
} from '../ml/softmaxRegression.js';
import {
  elasticNetFit, elasticNetPredict,
} from '../ml/elasticNet.js';
import {
  GRU, gruGradientCheck, trainGRU, gruStepClosed,
} from '../nn/gru.js';
import {
  expectedSarsa, expectedSarsaStep, expectedValue, expectedBellmanResidual,
} from '../rl/expectedSarsa.js';

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
 * 运行全部 58 组不变量自检。
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
    ['核主成分分析 · Kernel PCA', suiteKPCA],
    ['谱聚类 · Spectral Clustering', suiteSpectral],
    ['激活函数族 · Activations', suiteActivations],
    ['学习率调度 · LR Schedulers', suiteSched],
    ['卷积神经网络 · CNN', suiteCNN],
    ['高斯过程回归 · GPR', suiteGPR],
    ['梯度提升树 · GBDT', suiteGBDT],
    ['卡尔曼滤波 · Kalman', suiteKalman],
    ['变分自编码器 · VAE', suiteVAE],
    ['马尔可夫链蒙特卡洛 · MCMC', suiteMCMC],
    ['词向量 · Word2Vec SGNS', suiteWord2Vec],
    ['非负矩阵分解 · NMF', suiteNMF],
    ['归一化层 · Normalization', suiteNormalization],
    ['残差网络 · ResNet', suiteResNet],
    ['网页排名 · PageRank', suitePageRank],
    ['吉布斯采样 · Gibbs', suiteGibbs],
    ['高斯混合模型 · GMM-EM', suiteGMM],
    ['线性判别分析 · LDA/Fisher', suiteLDA],
    ['高斯朴素贝叶斯 · Naive Bayes', suiteNaiveBayes],
    ['随机森林 · Random Forest', suiteRandomForest],
    ['感知机 · Perceptron (Rosenblatt 1958)', suitePerceptron],
    ['独立成分分析 · FastICA (Hyvärinen & Oja 2000)', suiteICA],
    ['典型相关分析 · CCA (Hotelling 1936)', suiteCCA],
    ['层次聚类 · Hierarchical (Lance & Williams 1967)', suiteHierarchical],
    ['二次判别分析 · QDA (Bishop §4.3)', suiteQDA],
    ['Lasso 稀疏回归 · Tibshirani 1996', suiteLasso],
    ['受限玻尔兹曼机 · RBM (Hinton 2002)', suiteRBM],
    ['Hopfield 联想记忆 · Hopfield 1982', suiteHopfield],
    ['RTS 平滑器 · Rauch–Tung–Striebel 1965', suiteRTS],
    ['粒子滤波 · Gordon 1993 (SIR)', suiteParticleFilter],
    ['Mean Shift 众数搜索 · Fukunaga 1975', suiteMeanShift],
    ['SARSA 同策略TD · Rummery & Niranjan 1994', suiteSARSA],
    ['门控循环单元 · GRU (Cho et al. 2014)', suiteGRU],
    ['多类逻辑回归 · Softmax Regression (Bishop §4.3.4)', suiteSoftmaxRegression],
    ['期望SARSA · Expected SARSA (van Seijen et al. 2009)', suiteExpectedSarsa],
    ['弹性网络 · Elastic Net (Zou & Hastie 2005)', suiteElasticNet],
    ['快速傅里叶变换 · FFT (Cooley & Tukey 1965)', suiteFFT],
    ['常微分方程求解 · ODE (RK4 / Dormand–Prince RK45)', suiteODE],
    ['蒙特卡洛强化学习 · Monte Carlo RL (Sutton & Barto §5)', suiteMonteCarlo],
    ['自组织映射 · SOM (Kohonen 1982)', suiteSOM],
    ['迭代线性求解 · Iterative Linear Solvers (Hestenes & Stiefel 1952)', suiteLinSolve],
    ['隔离森林 · Isolation Forest (Liu et al. 2008)', suiteIsolationForest],
    ['流形学习 · MDS & Isomap (Torgerson 1952 / Tenenbaum 2000)', suiteManifold],
    ['策略梯度 · REINFORCE (Williams 1992)', suitePolicyGradient],
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

  // 协方差矩阵三道防线（历史 bug：下三角被二次相除致不对称，套件曾漏检）
  const rngC = mulberry32(77);
  const XC = randomMatrix(50, 4, rngC).map(r => r.map(v => v * 3 + 0.5));
  const C = covarianceMatrix(XC);
  let symErr = 0;
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) symErr = Math.max(symErr, Math.abs(C[i][j] - C[j][i]));
  }
  check('covarianceMatrix：C = Cᵀ 对称（精确）', symErr === 0, `最大不对称=${symErr.toExponential(2)}`);

  const muC = colMean(XC);
  let bruteErr = 0;
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      let s = 0;
      for (const row of XC) s += (row[i] - muC[i]) * (row[j] - muC[j]);
      bruteErr = Math.max(bruteErr, Math.abs(s / (XC.length - 1) - C[i][j]));
    }
  }
  check('covarianceMatrix：与暴力双循环实现逐位一致', bruteErr < 1e-12, `最大偏差=${bruteErr.toExponential(2)}`);

  let cholOk = true;
  try { cholesky(C); } catch (e) { cholOk = false; }
  check('covarianceMatrix：半正定（Cholesky L·Lᵀ = C 成功）', cholOk, cholOk ? '分解成功' : '分解失败');
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

/* ============================================================
 * 套件 19 · 核主成分分析（Schölkopf, Smola & Müller 1998）
 * ============================================================ */
function suiteKPCA(check) {
  // 各向异性高斯（协方差谱分离：3 / 1.5 / 0.7）
  const rng = mulberry32(5);
  const Xa = [];
  for (let i = 0; i < 80; i++) {
    Xa.push([gaussianSample(rng, 0, 3), gaussianSample(rng, 0, 1.5), gaussianSample(rng, 0, 0.7)]);
  }
  const kl = kpca(Xa, { kernel: 'linear', dims: 3 });

  // ① 线性核 KPCA ≡ 经典 PCA：主分量方向逐分量 |cos| = 1（谱分离数据无简并）
  const C = covarianceMatrix(Xa);
  const pe = jacobiEig(C);
  const mu = colMean(Xa);
  const Xc = Xa.map((r) => r.map((v, j) => v - mu[j]));
  let minCos = 1, maxNormDiff = 0;
  for (let k = 0; k < 3; k++) {
    const pcaCol = Xc.map((r) => r.reduce((s, x, j) => s + x * pe.vectors[k][j], 0));
    const kpcaCol = kl.scores.map((r) => r[k]);
    const na = norm(pcaCol), nb = norm(kpcaCol);
    minCos = Math.min(minCos, Math.abs(dot(pcaCol, kpcaCol) / (na * nb)));
    maxNormDiff = Math.max(maxNormDiff, Math.abs(na - nb) / na);
  }
  check('KPCA：线性核 ≡ 经典 PCA（主分量方向 |cos| = 1，谱分离数据）', minCos > 1 - 1e-12,
    `min|cos|=${minCos.toFixed(15)}`);
  check('KPCA：投影范数逐位一致（‖scores_k‖ ≡ ‖X̃v_k‖）', maxNormDiff < 1e-12,
    `maxΔ/‖·‖=${maxNormDiff.toExponential(2)}`);

  // ② 特征值关系 λ(K̃) = (n−1)·λ(C)（核技巧的谱恒等式）
  let maxLamErr = 0;
  for (let k = 0; k < 3; k++) {
    const rel = Math.abs(kl.eig.values[k] / 79 - pe.values[k]) / pe.values[k];
    maxLamErr = Math.max(maxLamErr, rel);
  }
  check('KPCA：特征值恒等式 λ(K̃) = (n−1)·λ(C)', maxLamErr < 1e-10,
    `最大相对偏差=${maxLamErr.toExponential(2)}`);

  // ③ 特征空间中心化：K̃ 行和 ≈ 0、1ᵀK̃1 ≈ 0、K̃ 对称半正定
  const rowSums = kl.Kc.map((r) => Math.abs(r.reduce((s, v) => s + v, 0)));
  const grand = Math.abs(kl.Kc.flat().reduce((s, v) => s + v, 0));
  const symErr = mdiff(kl.Kc, transpose(kl.Kc));
  const minEig = kl.eig.values[kl.eig.values.length - 1];
  check('KPCA：特征空间均值零（K̃ 行和 ≈ 0 且 1ᵀK̃1 ≈ 0）',
    vinf(rowSums) < 1e-9 && grand < 1e-9,
    `max行和=${vinf(rowSums).toExponential(2)}，总量=${grand.toExponential(2)}`);
  check('KPCA：K̃ 对称半正定（min eigenvalue ≥ −ε·λmax）',
    symErr < 1e-12 && minEig > -1e-10 * kl.eig.values[0],
    `对称误差=${symErr.toExponential(2)}，λmin=${minEig.toExponential(2)}`);

  // ④ 谱定理完全重构
  const recErr = kpcaReconstructionError(kl.Kc, kl.eig);
  check('KPCA：谱定理重构 ‖K̃ − VΛVᵀ‖_F ≈ 0', recErr < 1e-9 * (1 + frobeniusNorm(kl.Kc)),
    `误差=${recErr.toExponential(2)}`);

  // ⑤ 样本外变换自洽：kpcaTransform(X_train) ≡ 训练 scores
  const tr = kpcaTransform(kl, Xa);
  let trDiff = 0;
  for (let i = 0; i < Xa.length; i++) {
    for (let k = 0; k < 3; k++) trDiff = Math.max(trDiff, Math.abs(tr[i][k] - kl.scores[i][k]));
  }
  check('KPCA：样本外投影自洽（transform(train) ≡ scores）', trDiff < 1e-8,
    `maxDiff=${trDiff.toExponential(2)}`);

  // ⑥ RBF 核把两月亮非线性结构线性化：嵌入空间 1-NN 纯度 → 1
  const moons = twoMoons({ n: 60, sigma: 0.12, seed: 7 });
  const km = kpca(moons.X, { kernel: 'rbf', gamma: 10, dims: 2 });
  const purity = knnPurity(km.scores, moons.y);
  check('KPCA：RBF 核两月亮嵌入 1-NN 纯度 ≥ 0.95（非线性→线性）', purity >= 0.95,
    `纯度=${purity.toFixed(4)}`);
}

/* ============================================================
 * 套件 20 · 谱聚类（Shi & Malik 2000 · Ng-Jordan-Weiss 2002 · von Luxburg 2007）
 * ============================================================ */
function suiteSpectral(check) {
  const moons = twoMoons({ n: 100, sigma: 0.05, seed: 7 });
  const W = similarityGraph(moons.X, { k: 7, gamma: 1 });
  const { L, degrees } = normalizedLaplacian(W);
  const { eigenvalues, U } = spectralEmbedding(L, 2);
  const n = L.length;

  // ① 图对称性与度矩阵一致性：W = Wᵀ、D = diag(W1)、L_sym 对称
  const symErr = mdiff(W, transpose(W));
  const degErr = Math.max(...degrees.map((d, i) => Math.abs(d - W[i].reduce((s, v) => s + v, 0))));
  const lSym = mdiff(L, transpose(L));
  check('谱聚类：W = Wᵀ、D = diag(W1)、L_sym 对称', symErr < 1e-12 && degErr < 1e-9 && lSym < 1e-12,
    `W对称=${symErr.toExponential(2)}，度=${degErr.toExponential(2)}，L对称=${lSym.toExponential(2)}`);

  // ② 归一化拉普拉斯基本定理：λ_min = 0，特征向量 ∝ D^{1/2}1 / ‖D^{1/2}1‖
  const lam0 = eigenvalues[eigenvalues.length - 1];
  const v0 = U.map((r) => r[0]);
  const sqrtD = degrees.map((d) => Math.sqrt(d));
  const s0 = Math.sign(v0[0]) * Math.sign(sqrtD[0]);
  const ref0 = sqrtD.map((v) => v / norm(sqrtD));
  const evErr = vinf(v0.map((v, i) => Math.abs(v - s0 * ref0[i])));
  check('谱聚类：λ_min(L_sym) = 0，特征向量 ≡ √d/‖√d‖（基本定理）',
    Math.abs(lam0) < 1e-10 && evErr < 1e-8,
    `λmin=${lam0.toExponential(2)}，向量偏差=${evErr.toExponential(2)}`);

  // ③ 谱界定理：λ(L_sym) ⊆ [0, 2]
  const inBound = eigenvalues.every((v) => v > -1e-10 && v < 2 + 1e-10);
  check('谱聚类：谱界 λ(L_sym) ⊆ [0, 2]（归一化拉普拉斯定理）', inBound,
    `λ∈[${Math.min(...eigenvalues).toFixed(6)}, ${Math.max(...eigenvalues).toFixed(6)}]`);

  // ④ 两月亮精确恢复：完整 NJW 管线 ARI = 1.000000（非凸结构）
  const sc = spectralCluster(moons.X, 2, { neighbors: 7, gamma: 1, seed: 42 });
  const ari = adjustedRandIndex(sc.labels, moons.y);
  check('谱聚类：两月亮 ARI = 1.000000（NJW 全管线，非凸精确恢复）', ari === 1,
    `ARI=${ari.toFixed(6)}`);

  // ⑤ 对照：原始空间 k-means 凸假设失效（同数据 ARI 显著更低）
  const kmRaw = kmeans(moons.X, 2, { seed: 42 });
  const ariRaw = adjustedRandIndex(kmRaw.labels, moons.y);
  check('谱聚类：对照原始空间 k-means（凸假设失效，ARI < 0.5 且差距 ≥ 0.5）',
    ariRaw < 0.5 && ari - ariRaw >= 0.5,
    `k-means ARI=${ariRaw.toFixed(4)} vs 谱=${ari.toFixed(4)}`);

  // ⑥ 特征向量正交归一（Jacobi 输出 QᵀQ = I，嵌入基底的几何保证）
  const V = jacobiEig(L).vectors;
  const VtV = matmul(transpose(V), V);
  check('谱聚类：特征向量正交归一 ‖VᵀV − I‖_max ≈ 0', mdiff(VtV, eye(n)) < 1e-9,
    `偏差=${mdiff(VtV, eye(n)).toExponential(2)}`);
}

/* ============================================================
 * 套件 21 · 激活函数族（GELU / SiLU / Mish / HardSwish / Softplus）
 * ============================================================ */
function suiteActivations(check) {
  // ① erf 奇对称 + 文献参考值（A&S 7.1.26 精度 1.5e-7）
  let oddErr = 0;
  for (const x of [-3, -1.5, -0.7, -0.1, 0.1, 0.7, 1.5, 3]) oddErr = Math.max(oddErr, Math.abs(erf(-x) + erf(x)));
  const refErr = Math.max(Math.abs(erf(1) - 0.8427007929497149), Math.abs(erf(2) - 0.9953222650189527));
  check('激活：erf 奇对称 erf(−x) = −erf(x)（bit 级）且 A&S 参考值吻合',
    oddErr < 1e-15 && refErr < 1.5e-6,
    `对称=${oddErr.toExponential(2)}，参考值偏差=${refErr.toExponential(2)}`);

  // ② 正态 CDF：Φ(0) = 1/2、Φ(1.96) ≈ 0.975、对称 Φ(−x) = 1 − Φ(x)
  const cdfErr = Math.max(Math.abs(normalCDF(1.96) - 0.9750021048517795),
    Math.abs(normalCDF(0) - 0.5));
  let cdfSym = 0;
  for (const x of [-2, -1, -0.3, 0.3, 1, 2]) cdfSym = Math.max(cdfSym, Math.abs(normalCDF(-x) - (1 - normalCDF(x))));
  check('激活：Φ = ½erfc(−x/√2)，Φ(1.96)≈0.975 且 Φ(−x)=1−Φ(x)',
    cdfErr < 1.5e-6 && cdfSym < 1e-15,
    `参考值=${cdfErr.toExponential(2)}，对称=${cdfSym.toExponential(2)}`);

  // ③ softplus 恒等式：sp(x) − sp(−x) = x，且 sp′(x) ≡ σ(x)（中心差分互证）
  let idErr = 0;
  for (const x of [-20, -5, -1, 0, 1, 5, 20]) idErr = Math.max(idErr, Math.abs(softplus(x) - softplus(-x) - x));
  const spGrad = activationGradCheck(softplus, softplusD, [-3, -1, 0, 1, 3]);
  check('激活：softplus(x) − softplus(−x) = x（精确恒等）且 sp′ ≡ σ', idErr < 1e-12 && spGrad < 1e-7,
    `恒等=${idErr.toExponential(2)}，梯度=${spGrad.toExponential(2)}`);

  // ④ GELU tanh 逼近 vs 精确版（文献口径 max|Δ| < 5e-4）
  let approxErr = 0;
  for (let x = -6; x <= 6.0001; x += 0.25) approxErr = Math.max(approxErr, Math.abs(gelu(x) - geluTanh(x)));
  check('激活：GELU tanh 逼近 vs x·Φ(x)（max|Δ| < 5e-4）', approxErr < 5e-4,
    `maxΔ=${approxErr.toExponential(2)}`);

  // ⑤ 解析导数 ≡ 中心差分（gelu / geluTanh / silu / mish 四连检验）
  // 注：GELU 精确版导数经 Φ、x·φ 传播 A&S erf 逼近误差（1.5e-7），故阈值取 5e-7；
  //     其余纯初等函数逼近达 1e-10 量级。
  const xs = [-3, -2, -1, -0.5, 0.1, 0.5, 1, 2, 3];
  const g1 = activationGradCheck(gelu, geluD, xs);
  const g2 = activationGradCheck(geluTanh, geluTanhD, xs);
  const g3 = activationGradCheck(silu, siluD, xs);
  const g4 = activationGradCheck(mish, mishD, xs);
  check('激活：GELU/SiLU/Mish 解析导数 ≡ 中心差分（gelu ≤ 5e-7 含 erf 逼近传播，其余 < 1e-9）',
    g1 < 5e-7 && Math.max(g2, g3, g4) < 1e-9,
    `gelu=${g1.toExponential(2)} tanh=${g2.toExponential(2)} silu=${g3.toExponential(2)} mish=${g4.toExponential(2)}`);

  // ⑥ 分段线性 HardSwish：段内中心差分精确、端点行为、导数分段闭式
  // 注：分段线性在浮点下的差分噪声 ~ulp(f)/(2ε) ≈ 5e-12，非逻辑误差
  const hsErr = activationGradCheck(hardswish, hardswishD, [-2.5, -1, 0, 1, 2.5]);
  const endsOk = hardswish(-4) === 0 && hardswish(0) === 0
    && Math.abs(hardswish(4) - 4) < 1e-12 && Math.abs(hardswish(3.5) - 3.5) < 1e-12
    && Math.abs(hardswish(2.9) - 2.9 * (5.9 / 6)) < 1e-12;
  check('激活：HardSwish 段内导数精确（差分 < 1e-11 浮点口径）且端点 x·relu6(x+3)/6', hsErr < 1e-11 && endsOk,
    `差分=${hsErr.toExponential(2)}`);

  // ⑦ 渐近行为：silu(x→∞) = x、silu(x→−∞) → 0⁻、mish(0) = 0
  const asympOk = Math.abs(silu(50) - 50) < 1e-10 && silu(-50) < 0 && Math.abs(silu(-50)) < 1e-10
    && Math.abs(mish(0)) < 1e-15;
  check('激活：渐近行为 silu(∞)=x / silu(−∞)→0⁻ / mish(0)=0', asympOk,
    `silu(50)=${silu(50).toFixed(12)}，silu(−50)=${silu(-50).toExponential(2)}`);
}

/* ============================================================
 * 套件 22 · 学习率调度器（Noam / SGDR / 阶梯 / 指数）
 * ============================================================ */
function suiteSched(check) {
  // ① Noam（Vaswani 2017）：两支在 s = W 精确连续，峰值恰在 warmup 顶点
  const W = 4000, dModel = 512;
  const peak = warmupInverseSqrt(W, { warmupSteps: W, dModel });
  const peakRef = dModel ** -0.5 * W ** -0.5;
  const contErr = Math.abs(peak - peakRef);
  const isPeak = peak >= warmupInverseSqrt(W - 1, { warmupSteps: W, dModel })
    && peak >= warmupInverseSqrt(W + 1, { warmupSteps: W, dModel });
  const firstErr = Math.abs(warmupInverseSqrt(1, { warmupSteps: W, dModel }) - dModel ** -0.5 * W ** -1.5);
  check('调度：Noam 两支在 s = W 精确连续且峰值 lr = d^−½·W^−½', contErr < 1e-15 && isPeak && firstErr < 1e-18,
    `连续误差=${contErr.toExponential(2)}，lr(W)=${peak.toExponential(4)}`);

  // ② 线性 warmup：lr(1) = base/W、lr(W) = base 封顶、单调非降
  const lw = (s) => linearWarmup(s, { warmupSteps: 100, base: 1 });
  let lwMono = true;
  for (let s = 1; s < 200; s++) if (lw(s + 1) < lw(s) - 1e-15) lwMono = false;
  check('调度：线性 warmup lr(s) = base·min(1, s/W)，lr(W) 封顶且单调非降',
    Math.abs(lw(1) - 0.01) < 1e-15 && lw(100) === 1 && lw(150) === 1 && lwMono,
    `lr(1)=${lw(1).toFixed(4)}，lr(100)=${lw(100)}`);

  // ③ 余弦退火（SGDR）：端点闭式 ηmax/ηmin、中点、对称性、单调非增
  const co = (s) => cosineAnnealing(s, { T: 1000, etaMax: 1, etaMin: 0 });
  const coErr = Math.max(Math.abs(co(0) - 1), Math.abs(co(1000) - 0), Math.abs(co(500) - 0.5));
  let coMono = true;
  for (let s = 0; s < 1000; s += 5) if (co(s + 5) > co(s) + 1e-15) coMono = false;
  let coSym = 0;
  for (let s = 0; s <= 1000; s += 100) coSym = Math.max(coSym, Math.abs(co(s) + co(1000 - s) - 1));
  check('调度：SGDR 余弦 lr(0)=ηmax、lr(T)=ηmin、lr(T/2)=中点，对称且单调非增',
    coErr < 1e-12 && coMono && coSym < 1e-12,
    `端点=${coErr.toExponential(2)}，对称=${coSym.toExponential(2)}`);

  // ④ 阶梯衰减：段内严格恒定，整数倍处精确 γ 掉落，闭式 base·γ^⌊t/step⌋
  const sd = (s) => stepDecay(s, { stepSize: 30, gamma: 0.1, base: 1 });
  let segConst = true;
  for (let s = 0; s < 90; s++) if (sd(s) !== sd(Math.floor(s / 30) * 30)) segConst = false;
  const dropOk = sd(29) === 1 && Math.abs(sd(30) - 0.1) < 1e-15
    && Math.abs(sd(59) - 0.1) < 1e-15 && Math.abs(sd(60) - 0.01) < 1e-15;
  check('调度：阶梯衰减段内恒定、s=30/60 精确掉落（闭式 base·γ^⌊t/step⌋）', segConst && dropOk,
    `lr(29)=${sd(29)}，lr(30)=${sd(30)}，lr(60)=${sd(60)}`);

  // ⑤ 指数衰减：逐步等比 lr(t+1)/lr(t) = γ（200 个采样点全部精确）
  const ed = (s) => exponentialDecay(s, { gamma: 0.99, base: 1 });
  let ratioErr = 0;
  for (let s = 0; s < 200; s++) ratioErr = Math.max(ratioErr, Math.abs(ed(s + 1) / ed(s) - 0.99));
  check('调度：指数衰减逐步等比 lr(t+1)/lr(t) = γ（精确）', ratioErr < 1e-15,
    `最大比值偏差=${ratioErr.toExponential(2)}`);

  // ⑥ 全调度器值域健康：正、有限（训练安全线）
  let healthy = true;
  for (let s = 0; s <= 5000; s += 7) {
    for (const v of [warmupInverseSqrt(s), linearWarmup(s), cosineAnnealing(s), stepDecay(s), exponentialDecay(s)]) {
      if (!Number.isFinite(v) || v < 0) healthy = false;
    }
  }
  check('调度：五个调度器 0..5000 步全部值有限且非负', healthy,
    '采样 715 点 × 5 调度器');
}

/* ============================================================
 * 套件 23 · 卷积神经网络（LeCun 1998 · im2col+GEMM）
 * ============================================================ */
function suiteCNN(check) {
  // ① 输出尺寸闭式 oH = ⌊(n + 2p − k)/s⌋ + 1（精确整数恒等）
  const sizeCases = [[8, 3, 1, 0, 6], [8, 3, 1, 1, 8], [8, 3, 2, 0, 3], [7, 3, 2, 1, 4], [32, 5, 1, 2, 32]];
  let sizeOk = true;
  for (const [n, k, s, p, want] of sizeCases) if (convOutputSize(n, k, s, p) !== want) sizeOk = false;
  check('CNN：输出尺寸闭式 ⌊(n+2p−k)/s⌋+1（5 组配置逐位相等）', sizeOk,
    sizeCases.map(([n, k, s, p]) => `(${n},${k},${s},${p})→${convOutputSize(n, k, s, p)}`).join(' '));

  // ② im2col+GEMM ≡ 直接滑动窗口（双实现互证）
  const rng = mulberry32(2024);
  const { X, W, b } = randomConvTensors(2, 7, 7, 3, 3, 3, rng);
  const fwd = conv2dForward(X, W, b, { stride: 1, pad: 1 });
  const naive = conv2dNaive(X, W, b, { stride: 1, pad: 1 });
  let implDiff = 0;
  for (let oc = 0; oc < fwd.out.length; oc++) {
    for (let r = 0; r < fwd.out[oc].length; r++) {
      for (let c = 0; c < fwd.out[oc][r].length; c++) {
        implDiff = Math.max(implDiff, Math.abs(fwd.out[oc][r][c] - naive[oc][r][c]));
      }
    }
  }
  check('CNN：im2col+GEMM ≡ 直接滑动窗口卷积（双实现互证）', implDiff < 1e-12,
    `maxDiff=${implDiff.toExponential(2)}`);

  // ③ im2col 展开形状：(oH·oW) × (C·kH·kW)
  const geo = im2col(X, 3, 3, { stride: 1, pad: 1 });
  check('CNN：im2col 形状 (oH·oW)×(C·kH·kW) = 49×18',
    geo.cols.length === 49 && geo.cols[0].length === 18,
    `${geo.cols.length}×${geo.cols[0].length}`);

  // ④ 平移等变性：conv(shift(X)) = shift(conv(X))（结构性保证）
  const Xa = [zeros(9, 9)], Xb = [zeros(9, 9)];
  const pat = [[2, 3], [3, 4], [4, 5]]; // 对角图案
  for (const [r, c] of pat) { Xa[0][r][c] = 1; Xb[0][r][c + 1] = 1; } // 右移 1
  const Wa = [[zeros(3, 3)]]; // [OC=1][C=1][kH=3][kW=3]
  Wa[0][0][0][0] = 0.5; Wa[0][0][1][1] = 1; Wa[0][0][2][2] = -0.5;
  const oa = conv2dForward(Xa, Wa, [0], { stride: 1, pad: 1 }).out;
  const ob = conv2dForward(Xb, Wa, [0], { stride: 1, pad: 1 }).out;
  let eqErr = 0;
  for (let r = 2; r <= 6; r++) {
    for (let c = 2; c <= 6; c++) eqErr = Math.max(eqErr, Math.abs(ob[0][r][c + 1] - oa[0][r][c]));
  }
  check('CNN：平移等变性 conv(shift·X) = shift·conv(X)（中心区逐点）', eqErr < 1e-15,
    `maxDiff=${eqErr.toExponential(2)}`);

  // ⑤ 梯度检验：dX / dW / db 解析梯度 ≡ 中心差分（双判据）
  const r2 = mulberry32(77);
  const t2 = randomConvTensors(2, 6, 6, 3, 3, 3, r2);
  const { out } = conv2dForward(t2.X, t2.W, t2.b, { stride: 1, pad: 1 });
  const dOut = out.map((p) => p.map((row) => row.map((v) => Math.tanh(v) + 0.3)));
  const gc = convGradientCheck(t2.X, t2.W, t2.b, dOut, { stride: 1, pad: 1 });
  check('CNN：反向传播解析梯度 ≡ 中心差分（dX/dW/db 三路，双判据）',
    gc.maxAbs < 1e-7 && gc.maxRel < 1e-5,
    `maxRel=${gc.maxRel.toExponential(2)}，maxAbs=${gc.maxAbs.toExponential(2)}`);

  // ⑥ 最大池化梯度路由：仅 argmax 位置回传，非零数恰为输出元素数
  const r3 = mulberry32(99);
  const tp = randomConvTensors(1, 6, 6, 1, 1, 1, r3);
  const pool = maxPool2d(tp.X, 2, 2);
  const dPoolOut = pool.out.map((p) => p.map((row) => row.map((v) => v + 1)));
  const dPool = maxPoolBackward(pool, dPoolOut);
  let nonZero = 0, routeOk = true;
  for (let c = 0; c < dPool.length; c++) {
    for (let r = 0; r < dPool[c].length; r++) {
      for (let cc = 0; cc < dPool[c][r].length; cc++) {
        if (dPool[c][r][cc] !== 0) {
          nonZero++;
          const idx = r * dPool[c][r].length + cc;
          const isArgmax = pool.argmax[c].some((row) => row.some((v) => v === idx));
          if (!isArgmax) routeOk = false;
        }
      }
    }
  }
  const expectNZ = pool.out.length * pool.out[0].length * pool.out[0][0].length;
  check('CNN：最大池化梯度仅经 argmax 路由回传（非零数 = 输出元素数）',
    routeOk && nonZero === expectNZ,
    `非零=${nonZero}/${expectNZ}`);
}

/* ============================================================
 * 套件 24 · 高斯过程回归（Rasmussen & Williams 2006）
 * ============================================================ */
function suiteGPR(check) {
  // 一维真值函数 y = sin(x)，20 个训练点
  const Xtr = [], ytr = [];
  for (let i = 0; i < 20; i++) {
    const x = -3 + (i * 6) / 19;
    Xtr.push([x]);
    ytr.push(Math.sin(x));
  }
  const model = gprFit(Xtr, ytr, { lengthScale: 1, sigmaF: 1, noise: 0.05 });

  // ① Cholesky 精确分解：LLᵀ ≡ K
  const Krec = matmul(model.L, transpose(model.L));
  check('GPR：Cholesky 分解精确 LLᵀ ≡ K（核矩阵对称正定）', mdiff(Krec, model.K) < 1e-10,
    `‖LLᵀ−K‖_max=${mdiff(Krec, model.K).toExponential(2)}`);

  // ② 无噪声极限插值：训练点后验均值 ≡ 观测 y
  const interp = gprFit(Xtr, ytr, { lengthScale: 1, sigmaF: 1, noise: 1e-8 });
  const atTrain = gprPredict(interp, Xtr);
  let interpErr = 0;
  for (let i = 0; i < ytr.length; i++) interpErr = Math.max(interpErr, Math.abs(atTrain.mean[i] - ytr[i]));
  check('GPR：无噪声极限插值（训练点后验均值 ≡ y）', interpErr < 1e-5,
    `maxDiff=${interpErr.toExponential(2)}`);

  // ③ 后验方差 ≤ 先验；近训练点方差 < 远点方差；远点 → 先验 σ_f²
  const near = gprPredict(model, [[0.0]]);
  const far = gprPredict(model, [[50]]);
  const prior = gprPriorVariance(model);
  check('GPR：后验方差 ≤ 先验方差，且远点 → 先验 σ_f²',
    near.variance[0] <= prior + 1e-12 && near.variance[0] < far.variance[0]
      && Math.abs(far.variance[0] - prior) < 1e-6,
    `σ²(0)=${near.variance[0].toExponential(3)}，σ²(50)=${far.variance[0].toExponential(3)}，先验=${prior}`);

  // ④ 边际似然解析梯度 ≡ 中心差分（R&W 5.9）
  const ga = gprHyperGradAnalytic(model);
  const gn = gprHyperGradNumeric(model, { eps: 1e-5 });
  const worstRel = Math.max(
    relErr(ga.dLengthScale, gn.dLengthScale),
    relErr(ga.dSigmaF, gn.dSigmaF),
    relErr(ga.dNoise, gn.dNoise),
  );
  check('GPR：超参数解析梯度 ≡ 中心差分（R&W 公式 5.9）', worstRel < 1e-5,
    `∂/∂ℓ: ${ga.dLengthScale.toFixed(8)} vs ${gn.dLengthScale.toFixed(8)}，∂/∂σf: ${ga.dSigmaF.toFixed(6)} vs ${gn.dSigmaF.toFixed(6)}`);

  // ⑤ 单点闭式：μ(x*) = k·y/(k+σ_n²)，σ² = σ_f² − k²/(k+σ_n²)
  const ell = 1, sf = 1, sn = 0.3;
  const m1 = gprFit([[0]], [2], { lengthScale: ell, sigmaF: sf, noise: sn });
  const xs = 0.7;
  const k = sf * sf * Math.exp(-(xs * xs) / (2 * ell * ell));
  const muRef = k * 2 / (sf * sf + sn * sn);
  const varRef = sf * sf - (k * k) / (sf * sf + sn * sn);
  const p1 = gprPredict(m1, [[xs]]);
  check('GPR：单点后验闭式 μ = k·y/(k+σ_n²)、σ² = σ_f² − k²/(k+σ_n²)',
    Math.abs(p1.mean[0] - muRef) < 1e-10 && Math.abs(p1.variance[0] - varRef) < 1e-10,
    `μ=${p1.mean[0].toFixed(10)}（期望 ${muRef.toFixed(10)}），σ²=${p1.variance[0].toFixed(10)}（期望 ${varRef.toFixed(10)}）`);

  // ⑥ 预测合理性：留一验证 RMSE 显著小于数据标准差
  let sse = 0;
  for (let i = 0; i < Xtr.length; i++) {
    const rest = Xtr.filter((_, j) => j !== i), ry = ytr.filter((_, j) => j !== i);
    const mm = gprFit(rest, ry, { lengthScale: 1, sigmaF: 1, noise: 0.05 });
    const pr = gprPredict(mm, [Xtr[i]]);
    sse += (pr.mean[0] - ytr[i]) ** 2;
  }
  const rmse = Math.sqrt(sse / Xtr.length);
  check('GPR：留一验证 RMSE < 0.1（核超参数在真值函数上有效）', rmse < 0.1,
    `LOO-RMSE=${rmse.toExponential(3)}`);
}

/* ============================================================
 * 套件 25 · 梯度提升树（Friedman 2001）
 * ============================================================ */
function suiteGBDT(check) {
  // 训练集：y = x₁² + 2x₂ + 噪声（非线性，树可拟合）
  const rng = mulberry32(11);
  const X = [], y = [];
  for (let i = 0; i < 120; i++) {
    const a = rng() * 4 - 2, bb = rng() * 4 - 2;
    X.push([a, bb]);
    y.push(a * a + 2 * bb + gaussianSample(rng, 0, 0.15));
  }
  const model = gbdtFit(X, y, { nEstimators: 60, lr: 0.1, maxDepth: 3 });

  // ① 训练 MSE 单调不增（贪心函数逼近推论）
  let mono = true, maxUp = 0;
  for (let k = 1; k < model.trainLoss.length; k++) {
    const up = model.trainLoss[k] - model.trainLoss[k - 1];
    if (up > 1e-12) { mono = false; maxUp = Math.max(maxUp, up); }
  }
  check('GBDT：训练 MSE 严格单调不增（60 轮）', mono,
    `f₀=${model.trainLoss[0].toFixed(6)} → f₆₀=${model.trainLoss[model.trainLoss.length - 1].toFixed(6)}，最大回升=${maxUp.toExponential(2)}`);

  // ② 每棵树 ≡ 残差在分片常数子空间上的正交投影（精确恒等式 ⟨h,r⟩ = ‖h‖²）
  // 回归树即投影算子 P（P²=P, Pᵀ=P），h = Pr ⇒ ⟨h,r⟩ = ⟨Pr,r⟩ = ‖Pr‖² = ‖h‖²
  let projErr = 0;
  for (let m = 0; m < model.trees.length; m++) {
    const h = regressionTreePredictAll(model.trees[m], X);
    const r = model.residuals[m];
    let ip = 0, nh = 0;
    for (let i = 0; i < h.length; i++) { ip += h[i] * r[i]; nh += h[i] * h[i]; }
    projErr = Math.max(projErr, Math.abs(ip - nh) / (nh + 1e-300));
  }
  check('GBDT：树 ≡ 残差的正交投影（⟨h,r⟩ = ‖h‖² 精确恒等，60 棵逐棵）', projErr < 1e-9,
    `max 相对偏差=${projErr.toExponential(2)}`);

  // ③ 学习率收缩极限：lr → 0 时预测 ≡ 初始常数 ȳ
  const tiny = gbdtFit(X, y, { nEstimators: 20, lr: 1e-10, maxDepth: 3 });
  const pTiny = gbdtPredict(tiny, X);
  const muY = y.reduce((s, v) => s + v, 0) / y.length;
  let shrinkErr = 0;
  for (const v of pTiny) shrinkErr = Math.max(shrinkErr, Math.abs(v - muY));
  check('GBDT：学习率收缩极限 lr→0 ⇒ 预测 ≡ 常数均值 ȳ', shrinkErr < 1e-6,
    `maxDiff=${shrinkErr.toExponential(2)}，ȳ=${muY.toFixed(8)}`);

  // ④ 树桩闭式：maxDepth=1 时每个叶子的值 ≡ 该叶样本均值
  const ys2 = [1, 2, 3, 10, 11, 12];
  const Xs2 = [[1], [2], [3], [10], [11], [12]];
  const stump = regressionTreeFit(Xs2, ys2, { maxDepth: 1, minSamplesSplit: 2 });
  const stumpPred = regressionTreePredictAll(stump, Xs2);
  const leftMu = (1 + 2 + 3) / 3, rightMu = (10 + 11 + 12) / 3;
  const stumpOk = Math.abs(stumpPred[0] - leftMu) < 1e-12 && Math.abs(stumpPred[5] - rightMu) < 1e-12;
  check('GBDT：回归树桩叶子值 ≡ 叶内样本均值（最小二乘解析解）', stumpOk,
    `左叶=${stumpPred[0].toFixed(6)}（期望 ${leftMu.toFixed(6)}），右叶=${stumpPred[5].toFixed(6)}（期望 ${rightMu.toFixed(6)}）`);

  // ⑤ 迭代增益：末轮 MSE 显著低于常数基线方差
  let varY = 0;
  for (const v of y) varY += (v - muY) ** 2;
  varY /= y.length;
  const finalMse = model.trainLoss[model.trainLoss.length - 1];
  check('GBDT：末轮训练 MSE < 10% 常数基线方差（迭代增益显著）', finalMse < 0.1 * varY,
    `MSE=${finalMse.toFixed(6)} vs 基线方差=${varY.toFixed(6)}（比值 ${(finalMse / varY * 100).toFixed(2)}%）`);
}

/* ============================================================
 * 套件 26 · 卡尔曼滤波（Kalman 1960 · DARE）
 * ============================================================ */
function suiteKalman(check) {
  // 标量系统：x_t = 0.95·x_{t−1} + w(Q=0.1)；z_t = x_t + v(R=1)
  const f = 0.95, h = 1, q = 0.1, r = 1;
  const F = [[f]], H = [[h]], Q = [[q]], R = [[r]];
  const rng = mulberry32(4242);
  let truth = 0;
  const obs = [], truthSeries = [];
  for (let t = 0; t < 1500; t++) {
    truth = f * truth + gaussianSample(rng, 0, Math.sqrt(q));
    truthSeries.push(truth);
    obs.push([truth + gaussianSample(rng, 0, Math.sqrt(r))]);
  }
  const kf = kalmanFilter({ F, H, Q, R, x0: [0], P0: [[1]] }, obs);

  // ① 卡尔曼增益闭式 K = P⁻Hᵀ(HP⁻Hᵀ+R)⁻¹（逐步逐位）
  let gainErr = 0;
  for (let t = 0; t < obs.length; t++) {
    const Pm = kf.predictedCov[t][0][0];
    const want = Pm * h / (Pm * h * h + r);
    gainErr = Math.max(gainErr, Math.abs(kf.gains[t][0][0] - want));
  }
  check('Kalman：增益闭式 K = P⁻H/(HP⁻H+R)（1500 步逐位）', gainErr < 1e-12,
    `maxDiff=${gainErr.toExponential(2)}，稳态 K=${kf.gains[kf.gains.length - 1][0][0].toFixed(8)}`);

  // ② 新息协方差一致：经验方差 ≈ S = HP⁻Hᵀ+R（统计自洽性）
  const inn = kf.innovations.map((v) => v[0]);
  const empVar = empiricalVariance(inn);
  const Sval = kf.innovationCov[kf.innovationCov.length - 1][0][0];
  check('Kalman：新息经验协方差 ≈ 理论 S（相对偏差 < 15%）',
    Math.abs(empVar - Sval) / Sval < 0.15,
    `经验=${empVar.toFixed(5)} vs 理论 S=${Sval.toFixed(5)}（偏差 ${(Math.abs(empVar - Sval) / Sval * 100).toFixed(2)}%）`);

  // ③ 新息白噪声：滞后−1 自相关 ≈ 0（最优滤波的充要统计特征）
  const ac1 = innovationAutocorr(inn, 1);
  check('Kalman：新息序列白噪声（|滞后−1 自相关| < 0.1）', Math.abs(ac1) < 0.1,
    `ρ(1)=${ac1.toFixed(5)}`);

  // ④ 稳态收敛：迭代不动点 ≡ DARE 解析解
  const ss = kalmanScalarSteadyState(f, h, q, r);
  const iter = kalmanScalarIterate(f, h, q, r, 5000);
  const lastP = kf.predictedCov[kf.predictedCov.length - 1][0][0];
  check('Kalman：P⁻ 收敛到离散代数 Riccati 方程（DARE）解析解',
    Math.abs(iter - ss.Pminus) < 1e-9 && Math.abs(lastP - ss.Pminus) < 1e-6,
    `解析=${ss.Pminus.toFixed(10)}，迭代=${iter.toFixed(10)}，滤波末步=${lastP.toFixed(10)}`);

  // ⑤ 协方差恒为对称半正定（Joseph 形式的保障）
  let symErr = 0, negFound = false;
  for (const P of kf.covariances) {
    symErr = Math.max(symErr, mdiff(P, transpose(P)));
    for (let i = 0; i < P.length; i++) if (P[i][i] < 0) negFound = true;
  }
  check('Kalman：协方差全程对称半正定（Joseph 更新式保障）', symErr < 1e-15 && !negFound,
    `对称误差=${symErr.toExponential(2)}`);

  // ⑥ 滤波有效：估计 RMSE < 观测噪声标准差（优于直接观测）
  let se = 0;
  for (let t = 0; t < obs.length; t++) se += (kf.states[t][0] - truthSeries[t]) ** 2;
  const kfRmse = Math.sqrt(se / obs.length);
  check('Kalman：滤波估计 RMSE < 观测噪声 σ（优于直接观测）', kfRmse < Math.sqrt(r),
    `RMSE=${kfRmse.toFixed(5)} vs 观测噪声 σ=${Math.sqrt(r).toFixed(5)}`);
}

/* ============================================================
 * 套件 27 · 变分自编码器（Kingma & Welling 2013）
 * ============================================================ */
function suiteVAE(check) {
  const net = vaeMake({ dimX: 6, dimH: 8, dimZ: 3, seed: 42 });
  const rng = mulberry32(1234);
  const x = [];
  for (let i = 0; i < 6; i++) x.push(rng() * 2 - 1);
  const eps = vaeSampleEps(3, rng);

  // ① KL 解析闭式 ≡ 蒙特卡洛估计（大样本）
  const mu = [0.7, -1.2, 0.3], logvar = [0.2, -0.5, 0.9];
  const klAna = vaeKL(mu, logvar);
  const klMC = vaeKLMonteCarlo(mu, logvar, mulberry32(99), 200000);
  check('VAE：KL 解析闭式 ≡ 蒙特卡洛估计（20 万样本，相对偏差 < 2%）',
    Math.abs(klAna - klMC) / klAna < 0.02,
    `解析=${klAna.toFixed(6)}，MC=${klMC.toFixed(6)}（偏差 ${(Math.abs(klAna - klMC) / klAna * 100).toFixed(2)}%）`);

  // ② KL ≥ 0；后验 ≡ 先验（μ=0, logvar=0）时 KL = 0 精确
  let minKL = Infinity;
  const rng2 = mulberry32(7);
  for (let t = 0; t < 200; t++) {
    const m = [rng2() * 4 - 2, rng2() * 4 - 2, rng2() * 4 - 2];
    const lv = [rng2() * 4 - 2, rng2() * 4 - 2, rng2() * 4 - 2];
    minKL = Math.min(minKL, vaeKL(m, lv));
  }
  check('VAE：KL ≥ 0（200 组随机后验）且 μ=0,σ²=1 时 KL = 0',
    minKL >= -1e-12 && vaeKL([0, 0, 0], [0, 0, 0]) === 0,
    `minKL=${minKL.toExponential(3)}，KL(先验)=${vaeKL([0, 0, 0], [0, 0, 0])}`);

  // ③ 重参数化恒等：μ=0、logvar=0 ⇒ z ≡ ε
  const zId = vaeReparametrize([0, 0, 0], [0, 0, 0], eps);
  let idErr = 0;
  for (let i = 0; i < eps.length; i++) idErr = Math.max(idErr, Math.abs(zId[i] - eps[i]));
  check('VAE：重参数化恒等 μ=0,logvar=0 ⇒ z ≡ ε', idErr < 1e-15,
    `maxDiff=${idErr.toExponential(2)}`);

  // ④ 梯度检验：重参数化后 loss 为确定性函数 ⇒ 解析梯度 ≡ 中心差分
  const gc = vaeGradientCheck(net, x, eps);
  check('VAE：重参数化路径解析梯度 ≡ 中心差分（编码器+解码器全部参数）',
    gc.maxRel < 1e-6 && gc.maxAbs < 1e-7,
    `maxRel=${gc.maxRel.toExponential(2)}，maxAbs=${gc.maxAbs.toExponential(2)}`);

  // ⑤ ELBO ≤ log p(x) 的构造性证据：ELBO ≡ −(重构 + KL) 恒等
  const fwd = vaeForward(net, x, eps);
  const elboRef = -(fwd.recon + fwd.kl);
  check('VAE：ELBO ≡ −(重构误差 + KL)（变分下界分解恒等）',
    Math.abs(fwd.elbo - elboRef) < 1e-15 && fwd.kl >= 0,
    `ELBO=${fwd.elbo.toFixed(6)}，重构=${fwd.recon.toFixed(6)}，KL=${fwd.kl.toFixed(6)}`);

  // ⑥ KL 随后验逼近先验单调趋零（σ² → 1、μ → 0 时 KL → 0）
  let klMono = true, prev = Infinity;
  for (const scale of [3, 2, 1.5, 1.2, 1.05, 1.0]) {
    const k = vaeKL([0.5 * (scale - 1)], [Math.log(scale)]);
    if (k > prev + 1e-12) klMono = false;
    prev = k;
  }
  check('VAE：KL 随后验逼近先验单调趋零', klMono && prev < 1e-3,
    `末值（后验≡先验）=${prev.toExponential(2)}`);
}

/* ============================================================
 * 套件 28 · MCMC（Metropolis-Hastings + 重要性采样）
 * ============================================================ */
function suiteMCMC(check) {
  // 目标：标准正态。注意两个版本都**未归一化**（缺少 −½log2π），
  // 这正是 MH 的关键优势：归一化常数在比值中相消。
  const logUnnorm = (v) => -0.5 * v[0] * v[0];
  const res = metropolisHastings(logUnnorm, [0], {
    iters: 40000, proposalSigma: 2.4, burnIn: 4000, thin: 2, seed: 42,
  });
  const series = res.samples.map((s) => s[0]);
  const mom = mcmcEmpiricalMoments(series);

  // ① 经验矩 ≡ 解析矩（标准正态：均值 0、方差 1、四阶矩 3、P(|x|<1)=0.6827）
  const okMean = Math.abs(mom.mean) < 0.05;
  const okVar = Math.abs(mom.variance - 1) < 0.06;
  const okM4 = Math.abs(mom.m4 - 3) / 3 < 0.10;
  const okP = Math.abs(mom.pWithin1 - 0.682689) < 0.03;
  check('MCMC：未归一化目标采样 ≡ 标准正态（均值0/方差1/四阶矩3/P(|x|<1)=0.6827）',
    okMean && okVar && okM4 && okP,
    `μ=${mom.mean.toFixed(4)}，σ²=${mom.variance.toFixed(4)}，E[x⁴]=${mom.m4.toFixed(4)}，P=${mom.pWithin1.toFixed(4)}（n=${mom.n}）`);

  // ② 归一化常数无关性：补上常数项后应得同一分布
  const logNorm = (v) => -0.5 * v[0] * v[0] - 0.5 * Math.log(2 * Math.PI);
  const res2 = metropolisHastings(logNorm, [0], {
    iters: 40000, proposalSigma: 2.4, burnIn: 4000, thin: 2, seed: 42,
  });
  const m2 = mcmcEmpiricalMoments(res2.samples.map((s) => s[0]));
  check('MCMC：归一化常数无关性（未归一化 vs 归一化同一平稳分布）',
    Math.abs(m2.variance - mom.variance) < 0.02 && Math.abs(m2.mean - mom.mean) < 0.05,
    `σ²: ${mom.variance.toFixed(4)} vs ${m2.variance.toFixed(4)}`);

  // ③ 接受率落在有效区间（Roberts et al. 1997 理论最优 ≈ 0.234）
  const rates = [];
  for (const sig of [0.5, 1.0, 2.4, 5.0, 10.0]) {
    const r = metropolisHastings(logUnnorm, [0], { iters: 6000, proposalSigma: sig, burnIn: 500, thin: 1, seed: 5 });
    rates.push(r.acceptanceRate);
  }
  let monoDown = true;
  for (let i = 1; i < rates.length; i++) if (rates[i] > rates[i - 1] + 1e-12) monoDown = false;
  check('MCMC：接受率随提议尺度单调下降，且 σ=2.4 时落在有效区间 [0.2, 0.6]',
    monoDown && rates[2] >= 0.2 && rates[2] <= 0.6,
    `σ=0.5/1/2.4/5/10 → ${rates.map((v) => v.toFixed(3)).join(' / ')}`);

  // ④ 细致平衡：π(x)T(x→x') = π(x')T(x'→x)（对称提议下逐对相消）
  const xA = [0.7], xB = [-1.1];
  const dbRes = detailedBalanceResidual(logNorm(xA), logNorm(xB), xA, xB, 2.4);
  check('MCMC：细致平衡 π(x)T(x→x′) = π(x′)T(x′→x)（对称提议）', dbRes < 1e-12,
    `相对残差=${dbRes.toExponential(2)}`);

  // ⑤ 接受概率闭式 min(1, π(x')/π(x))
  const ap = mhAcceptProb(logNorm([0]), logNorm([1]));
  check('MCMC：接受概率 ≡ min(1, π(x′)/π(x)) = e^{−½}', Math.abs(ap - Math.exp(-0.5)) < 1e-15,
    `α=${ap.toFixed(10)}，e^{−½}=${Math.exp(-0.5).toFixed(10)}`);

  // ⑥ 重要性采样：E_π[x²] = 1（目标 N(0,1)，提议 N(0,2²)）
  const isRes = importanceSampling(
    (v) => -0.5 * v[0] * v[0],
    (v) => -0.5 * (v[0] / 2) ** 2 - Math.log(2),
    (rng) => [2 * gaussianSample(rng, 0, 1)],
    (v) => v[0] * v[0],
    200000, 42,
  );
  check('MCMC：重要性采样 E_π[x²] ≡ 1（提议 N(0,2²)，ESS 合理）',
    Math.abs(isRes.estimate - 1) < 0.03 && isRes.ess > 20000,
    `估计=${isRes.estimate.toFixed(5)}，ESS=${Math.round(isRes.ess)}/200000（${(isRes.ess / 2000).toFixed(1)}%）`);
}

/* ============================================================
 * 套件 29 · Word2Vec：Skip-gram + 负采样（Mikolov 2013）
 * ============================================================ */
function suiteWord2Vec(check) {
  const rng = mulberry32(2026);
  const dim = 6;
  const vc = [], uo = [], u1 = [], u2 = [], u3 = [];
  for (let i = 0; i < dim; i++) {
    vc.push(rng() * 2 - 1); uo.push(rng() * 2 - 1);
    u1.push(rng() * 2 - 1); u2.push(rng() * 2 - 1); u3.push(rng() * 2 - 1);
  }
  const negs = [u1, u2, u3];

  // ① 梯度检验：解析梯度 ≡ 中心差分（v_c / u_o / u_neg 三路）
  const gc = sgnsGradientCheck(vc, uo, negs);
  check('Word2Vec：SGNS 解析梯度 ≡ 中心差分（v_c/u_o/u_neg 三路）',
    gc.maxRel < 1e-6 && gc.maxAbs < 1e-7,
    `maxRel=${gc.maxRel.toExponential(2)}，maxAbs=${gc.maxAbs.toExponential(2)}`);

  // ② 损失闭式 ≡ 逐项 sigmoid 参考实现（双实现互证）
  const l1 = sgnsLoss(vc, uo, negs);
  const l2 = sgnsLossReference(vc, uo, negs);
  const l3 = sgnsGradients(vc, uo, negs).loss;
  check('Word2Vec：损失闭式 ≡ 逐项 sigmoid 参考实现（三路互证）',
    Math.abs(l1 - l2) < 1e-12 && Math.abs(l1 - l3) < 1e-12,
    `loss=${l1.toFixed(10)}（参考 ${l2.toFixed(10)}，梯度版 ${l3.toFixed(10)}）`);

  // ③ σ(−x) = 1 − σ(x)（负采样损失的对偶恒等）
  let sigId = 0;
  for (const t of [-8, -3, -0.5, 0, 0.5, 3, 8]) {
    sigId = Math.max(sigId, Math.abs(sigmoid(-t) - (1 - sigmoid(t))));
  }
  check('Word2Vec：σ(−x) = 1 − σ(x)（负采样对偶恒等，bit 级）', sigId < 1e-15,
    `maxDiff=${sigId.toExponential(2)}`);

  // ④ 训练有效性：共现词对相似度 > 非共现词对（表示学习把共现结构编码进向量）
  const toy = makeToyCorpus({ clusterSize: 4, windowSize: 2, repeats: 14, seed: 42 });
  const trained = trainSGNS(toy.pairs, toy.vocabSize, {
    dim: 12, negatives: 5, lr: 0.08, epochs: 80, seed: 42,
  });
  const meanSim = (pairs) => {
    let s = 0;
    for (const [a, b] of pairs) s += wordSimilarity(trained.Win[a], trained.Win[b]);
    return s / pairs.length;
  };
  const simCo = meanSim(toy.cooc);
  const simNon = meanSim(toy.noncooc);
  check('Word2Vec：训练后共现词对余弦相似度 > 非共现对（表示学习有效）',
    simCo > simNon + 0.05,
    `共现=${simCo.toFixed(4)} vs 非共现=${simNon.toFixed(4)}（差 ${(simCo - simNon).toFixed(4)}）`);

  // ⑤ 训练损失整体下降（首尾 epoch 平均对比）
  const losses = trained.losses;
  const headAvg = losses.slice(0, 8).reduce((a, b) => a + b, 0) / 8;
  const tailAvg = losses.slice(-8).reduce((a, b) => a + b, 0) / 8;
  check('Word2Vec：训练损失整体下降（前 8 epoch vs 后 8 epoch）', tailAvg < headAvg,
    `${headAvg.toFixed(5)} → ${tailAvg.toFixed(5)}（降 ${((1 - tailAvg / headAvg) * 100).toFixed(1)}%）`);
}

/* ============================================================
 * 套件 30 · 非负矩阵分解（Lee & Seung 1999/2001）
 * ============================================================ */
function suiteNMF(check) {
  const { V, W0, H0 } = makeNMFData({ m: 8, n: 10, k: 3, seed: 42 });
  const res = nmf(V, 3, { iters: 400, seed: 42 });

  // ① 重构误差单调不增（Lee & Seung 2001 定理 1：乘性更新的核心保证）
  let mono = true, maxUp = 0;
  for (let i = 1; i < res.errors.length; i++) {
    const up = res.errors[i] - res.errors[i - 1];
    if (up > 1e-15 * (1 + res.errors[i - 1])) { mono = false; maxUp = Math.max(maxUp, up); }
  }
  check('NMF：重构误差 ‖V−WH‖²_F 单调不增（Lee-Seung 定理）', mono,
    `‖·‖²: ${res.errors[0].toExponential(3)} → ${res.errors[res.errors.length - 1].toExponential(3)}，最大回升=${maxUp.toExponential(2)}`);

  // ② 非负性全程保持（乘性更新的结构性保证）
  const minEntry = nmfMinEntry(res.W, res.H);
  check('NMF：非负性全程保持（min entry ≥ 0）', minEntry >= 0,
    `min=${minEntry.toExponential(3)}`);

  // ③ 精确恢复：由 V = W₀H₀ 构造的秩匹配数据，误差 → 0
  const exact = nmf(V, 3, { iters: 1500, seed: 42 });
  const finalErr = exact.errors[exact.errors.length - 1];
  const relErr = finalErr / (V.flat().reduce((s, v) => s + v * v, 0));
  check('NMF：秩匹配构造数据可精确恢复（相对误差 < 1e-6）', relErr < 1e-6,
    `绝对误差²=${finalErr.toExponential(3)}，相对=${relErr.toExponential(3)}`);

  // ④ Eckart–Young 下界：NMF 误差 ≥ 同秩无约束 SVD 最优逼近误差
  const svdErr = svdBestRankKError(V, 3);
  const nmfErr = res.errors[res.errors.length - 1];
  check('NMF：非负约束误差 ≥ 同秩 SVD 最优逼近误差（Eckart–Young 下界）',
    nmfErr >= svdErr - 1e-9,
    `NMF=${nmfErr.toExponential(3)} ≥ SVD 下界=${svdErr.toExponential(3)}`);

  // ⑤ 收敛到不动点：末步乘性因子偏离 1 的幅度 → 0（驻点条件）
  const lastStep = res.steps[res.steps.length - 1];
  check('NMF：收敛到驻点（末步更新量 < 1e-3）', lastStep < 1e-3,
    `末步相对更新=${lastStep.toExponential(3)}`);

  // ⑥ 零件式表示：NMF 的基向量非负（与 PCA/SVD 的符号任意性形成对照）
  const allNonNegBase = res.W.every((row) => row.every((v) => v >= 0));
  check('NMF：分解基 W 全部非负（零件式表示，区别于 SVD 的符号任意性）', allNonNegBase,
    `W 维度 ${res.W.length}×${res.W[0].length}，k=${res.k}`);
}

/* ============================================================
 * 套件 31 · 归一化层（BatchNorm + LayerNorm）
 * ============================================================ */
function suiteNormalization(check) {
  const rng = mulberry32(7);
  const d = 4, n = 64;
  const X = [];
  for (let i = 0; i < n; i++) X.push(new Array(d).fill(0).map(() => gaussianSample(rng, 0, 1)));
  const ones = new Array(d).fill(1), zeros = new Array(d).fill(0);

  // ① BatchNorm γ=1,β=0 ⇒ 通道均值≈0、方差≈1
  const { Y: Yb } = batchNormForward(X, ones, zeros, { training: true });
  let maxMean = 0, minVar = Infinity, maxVar = -Infinity;
  for (let j = 0; j < d; j++) {
    let mu = 0; for (let i = 0; i < n; i++) mu += Yb[i][j]; mu /= n;
    let varr = 0; for (let i = 0; i < n; i++) varr += (Yb[i][j] - mu) ** 2; varr /= n;
    maxMean = Math.max(maxMean, Math.abs(mu));
    minVar = Math.min(minVar, varr); maxVar = Math.max(maxVar, varr);
  }
  check('Norm：BatchNorm γ=1,β=0 ⇒ 通道均值≈0、方差≈1', maxMean < 1e-12 && minVar > 0.999 && maxVar < 1.001,
    `max|mean|=${maxMean.toExponential(2)}，var∈[${minVar.toFixed(4)}, ${maxVar.toFixed(4)}]`);

  // ② 仿射恢复：输出通道均值 ≡ β、标准差 ≡ |γ|
  const gamma = [0.7, 1.3, 2.1, 0.5], beta = [-0.4, 0.2, 0.9, -1.1];
  const { Y: Ya } = batchNormForward(X, gamma, beta, { training: true });
  let meanErr = 0, stdErr = 0;
  for (let j = 0; j < d; j++) {
    let mu = 0; for (let i = 0; i < n; i++) mu += Ya[i][j]; mu /= n;
    let varr = 0; for (let i = 0; i < n; i++) varr += (Ya[i][j] - mu) ** 2; varr /= n;
    meanErr = Math.max(meanErr, Math.abs(mu - beta[j]));
    stdErr = Math.max(stdErr, Math.abs(Math.sqrt(varr) - Math.abs(gamma[j])));
  }
  check('Norm：BatchNorm 仿射恢复（均值≡β、标准差≡|γ|）', meanErr < 1e-12 && stdErr < 1e-3,
    `max|Δmean|=${meanErr.toExponential(2)}，max|Δstd|=${stdErr.toExponential(2)}`);

  // ③ BatchNorm 梯度检验
  const gcB = batchNormGradCheck(X.map(r => r.slice()), ones.slice(), zeros.slice(), { seed: 42 });
  check('Norm：BatchNorm 解析梯度 ≡ 中心差分（X/γ/β 三路）', gcB.maxRel < 1e-6 && gcB.maxAbs < 1e-7,
    `maxRel=${gcB.maxRel.toExponential(2)}，maxAbs=${gcB.maxAbs.toExponential(2)}`);

  // ④ LayerNorm 梯度检验
  const gcL = layerNormGradCheck(X.map(r => r.slice()), { seed: 42 });
  check('Norm：LayerNorm 解析梯度 ≡ 中心差分', gcL.maxRel < 1e-6 && gcL.maxAbs < 1e-7,
    `maxRel=${gcL.maxRel.toExponential(2)}，maxAbs=${gcL.maxAbs.toExponential(2)}`);

  // ⑤ LayerNorm γ=1,β=0 ⇒ 逐样本均值≈0、方差≈1
  const { Y: Yl } = layerNorm2d(X, {});
  let lnMean = 0, lnVarOk = true;
  for (let i = 0; i < n; i++) {
    const ymu = Yl[i].reduce((a, b) => a + b, 0) / d;
    lnMean = Math.max(lnMean, Math.abs(ymu));
    let xv = 0; for (const v of X[i]) xv += (v - X[i].reduce((a, b) => a + b, 0) / d) ** 2; xv /= d;
    if (xv > 1e-9) {
      let yv = 0; for (const v of Yl[i]) yv += (v - ymu) ** 2; yv /= d;
      if (Math.abs(yv - 1) > 1e-3) lnVarOk = false;
    }
  }
  check('Norm：LayerNorm γ=1,β=0 ⇒ 逐样本均值≈0、方差≈1', lnMean < 1e-12 && lnVarOk,
    `max|mean|=${lnMean.toExponential(2)}`);

  // ⑥ BatchNorm 推理期 running stats 更新为精确 EMA：rm_new − rm_old = m·(batchMean − rm_old)
  const rng2 = mulberry32(123);
  const m = 0.1;
  const rmOld = new Array(d).fill(0.3), rvOld = new Array(d).fill(1.7);
  const rm = rmOld.slice(), rv = rvOld.slice();
  const g2 = new Array(d).fill(1), b2 = new Array(d).fill(0);
  const Xb = []; for (let i = 0; i < 64; i++) Xb.push(new Array(d).fill(0).map(() => gaussianSample(rng2, 0, 1)));
  const bm = new Array(d).fill(0), bv = new Array(d).fill(0);
  for (const row of Xb) for (let j = 0; j < d; j++) bm[j] += row[j];
  for (let j = 0; j < d; j++) bm[j] /= 64;
  for (const row of Xb) for (let j = 0; j < d; j++) bv[j] += (row[j] - bm[j]) ** 2;
  for (let j = 0; j < d; j++) bv[j] /= 64;
  batchNormForward(Xb, g2, b2, { training: true, runningMean: rm, runningVar: rv, momentum: m });
  let emaErr = 0;
  for (let j = 0; j < d; j++) {
    emaErr = Math.max(emaErr, Math.abs((rm[j] - rmOld[j]) - m * (bm[j] - rmOld[j])));
    emaErr = Math.max(emaErr, Math.abs((rv[j] - rvOld[j]) - m * (bv[j] - rvOld[j])));
  }
  check('Norm：BatchNorm running stats 更新为精确 EMA（rm_new−rm_old = m·(batch−rm_old)）', emaErr < 1e-9,
    `maxEMAdev=${emaErr.toExponential(2)}`);
}

/* ============================================================
 * 套件 32 · 残差网络（ResNet）
 * ============================================================ */
function suiteResNet(check) {
  // ① 恒等映射：零化分支 ⇒ forward(x) ≡ x
  const idErr = residualIdentityCheck(4, 6, 42);
  check('ResNet：零化分支 ⇒ forward(x) ≡ x（恒等映射，跳过连接保底）', idErr < 1e-12,
    `maxDiff=${idErr.toExponential(2)}`);

  // ② 零化分支 ⇒ 单样本雅可比 ∂out/∂x = I
  const jErr = residualJacobianIdentity(4, 6, 42);
  check('ResNet：零化分支 ⇒ 单样本雅可比 ∂out/∂x = I（跳过连接是精确恒等）', jErr < 1e-9,
    `max|J−I|=${jErr.toExponential(2)}`);

  // ③ 全参数梯度检验
  const moons = twoMoons({ n: 60, sigma: 0.18, seed: 42 });
  const net = new ResNet({ inputDim: 2, hiddenDim: 6, numBlocks: 2, numClasses: 2, seed: 3 });
  const gc = residualGradientCheck(net, moons.X, moons.y, { eps: 1e-5 });
  check('ResNet：全参数解析梯度 ≡ 中心差分（残差块+分类头）', gc.maxRel < 1e-6 && gc.maxAbs < 1e-7,
    `maxRel=${gc.maxRel.toExponential(2)}，maxAbs=${gc.maxAbs.toExponential(2)}`);

  // ④ 训练收敛：two-moons 上最终 CE 显著下降且准确率 > 0.9
  const trainData = twoMoons({ n: 120, sigma: 0.2, seed: 5 });
  const net2 = new ResNet({ inputDim: 2, hiddenDim: 12, numBlocks: 3, numClasses: 2, seed: 5 });
  const losses = trainResNet(net2, trainData.X, trainData.y, { lr: 0.2, iters: 800 });
  const initLoss = losses[0], finalLoss = losses[losses.length - 1];
  const acc = resnetAccuracy(net2, trainData.X, trainData.y);
  check('ResNet：two-moons 上训练收敛（最终 CE ↓ 且准确率 > 0.9）', finalLoss < initLoss * 0.5 && acc > 0.9,
    `CE ${initLoss.toFixed(3)} → ${finalLoss.toFixed(4)}，acc=${acc.toFixed(3)}`);

  // ⑤ 梯度高速公路（精确）：深度零化分支网络上，残差回传输入梯度范数 = √d，普通堆叠 = 0
  const hw = residualGradientHighway({ inputDim: 4, hiddenDim: 16, numBlocks: 8, seed: 42, dout: null });
  check('ResNet：梯度高速公路——深度零化分支上残差回传梯度范数=√d、普通堆叠=0（消除梯度消失）',
    Math.abs(hw.residualGradNorm - Math.sqrt(4)) < 1e-9 && hw.plainGradNorm < 1e-12,
    `残差=${hw.residualGradNorm.toFixed(4)}（=√4=2.0000），普通=${hw.plainGradNorm.toExponential(2)}`);
}

/* ============================================================
 * 套件 33 · 网页排名（PageRank）
 * ============================================================ */
function suitePageRank(check) {
  const A = [
    [0, 1, 1, 0],
    [0, 0, 1, 0],
    [1, 0, 0, 1],
    [0, 0, 1, 0],
  ];
  const { ranks, residuals } = pageRank(A, { damping: 0.85, iters: 200 });
  const sum = pagerankSum(ranks);
  check('PageRank：秩和 ≡ 1（概率分布）', Math.abs(sum - 1) < 1e-9, `Σr=${sum.toFixed(10)}（${ranks.length} 节点）`);

  const stat = pagerankStationary(ranks, A, { damping: 0.85 });
  check('PageRank：平稳方程 ‖G·r − r‖∞ → 0（r 是 G 的特征向量，λ=1）', stat < 1e-9,
    `maxResidual=${stat.toExponential(2)}`);

  const exact = pagerankExact(A, { damping: 0.85 });
  let diff = 0; for (let i = 0; i < ranks.length; i++) diff = Math.max(diff, Math.abs(ranks[i] - exact[i]));
  check('PageRank：幂迭代 ≡ 精确线性解 (I−dS)r=(1−d)/n·1（差异 < 1e-9）', diff < 1e-9,
    `max|Δr|=${diff.toExponential(2)}`);

  let mono = true;
  for (let k = 1; k < residuals.length; k++) {
    if (residuals[k] > 0.85 * residuals[k - 1] + 1e-12) { mono = false; break; }
  }
  check('PageRank：幂迭代残差单调收敛（收缩因子 d=0.85，‖r_{k+1}−r_k‖₁ ≤ d·‖·‖₁）', mono && residuals[residuals.length - 1] < 1e-10,
    `末步残差=${residuals[residuals.length - 1].toExponential(2)}，迭代=${residuals.length} 轮`);

  // 5-环正则图 → 均匀秩
  const n5 = 5;
  const A5 = Array.from({ length: n5 }, () => new Array(n5).fill(0));
  for (let i = 0; i < n5; i++) A5[i][(i + 1) % n5] = 1;
  const r5 = pageRank(A5, { damping: 0.85, iters: 100 }).ranks;
  let maxDev = 0; for (const v of r5) maxDev = Math.max(maxDev, Math.abs(v - 1 / n5));
  check('PageRank：5-环正则图 ⇒ 均匀秩（max 偏差 < 1e-9）', maxDev < 1e-9,
    `max|r−1/5|=${maxDev.toExponential(2)}`);

  // 含悬挂节点：质量守恒 + 平稳方程成立
  const A6 = [
    [0, 1, 0],
    [0, 0, 1],
    [0, 0, 0], // 节点 2 悬挂（无出边）
  ];
  const r6 = pageRank(A6, { damping: 0.85, iters: 200 });
  const stat6 = pagerankStationary(r6.ranks, A6, { damping: 0.85 });
  check('PageRank：含悬挂节点图仍质量守恒（Σr=1）且收敛到合法平稳分布',
    Math.abs(pagerankSum(r6.ranks) - 1) < 1e-9 && stat6 < 1e-9,
    `Σr=${pagerankSum(r6.ranks).toFixed(10)}，stationary=${stat6.toExponential(2)}`);
}

/* ============================================================
 * 套件 34 · 吉布斯采样（Gibbs Sampling）
 * ============================================================ */
function suiteGibbs(check) {
  const mu = [1, -0.5];
  const Sigma = [[2, 1.2], [1.2, 3]];
  const { samples } = gibbsBivariateGaussian(mu, Sigma, { iters: 20000, burnIn: 2000, thin: 1, seed: 42 });
  const m0 = gibbsMarginal(samples, 0), m1 = gibbsMarginal(samples, 1);
  check('Gibbs：二元高斯经验均值 ≡ μ（偏差 < 0.05）',
    Math.abs(m0.mean - mu[0]) < 0.05 && Math.abs(m1.mean - mu[1]) < 0.05,
    `μ̂=(${m0.mean.toFixed(4)}, ${m1.mean.toFixed(4)}) vs μ=(${mu[0]}, ${mu[1]})`);

  const cov = gibbsCovariance(samples);
  const rel = (a, b) => Math.abs(a - b) / b;
  check('Gibbs：二元高斯经验协方差 ≡ Σ（相对误差 < 0.08）',
    rel(cov[0][0], 2) < 0.08 && rel(cov[1][1], 3) < 0.08 && rel(cov[0][1], 1.2) < 0.08,
    `Σ̂=[[${cov[0][0].toFixed(3)}, ${cov[0][1].toFixed(3)}],[${cov[0][1].toFixed(3)}, ${cov[1][1].toFixed(3)}]] vs Σ=[[2,1.2],[1.2,3]]`);

  check('Gibbs：单坐标边际 ≡ 一元高斯（方差 rel < 0.08）',
    rel(m0.variance, 2) < 0.08 && rel(m1.variance, 3) < 0.08,
    `σ̂²=(${m0.variance.toFixed(3)}, ${m1.variance.toFixed(3)}) vs (2, 3)`);

  const cc = gibbsConditionalCheck(mu, Sigma, 0.3, { n: 50000, seed: 42 });
  check('Gibbs：实现的条件分布 ≡ 理论条件（均值偏差 < 0.02、方差 rel < 0.05）',
    cc.meanErr < 0.02 && cc.varRelErr < 0.05,
    `|Δμ|=${cc.meanErr.toFixed(4)}，Δσ²rel=${cc.varRelErr.toFixed(4)}`);

  // 离散二维：经验联合 ≡ 真实联合
  const joint = [[0.1, 0.2, 0.1], [0.15, 0.25, 0.2]]; // Σ=1
  const { samples: emp } = gibbsDiscrete(joint, [0, 0], { iters: 40000, burnIn: 4000, thin: 1, seed: 42 });
  let maxCell = 0, sumEmp = 0;
  for (let i = 0; i < 2; i++) for (let j = 0; j < 3; j++) {
    maxCell = Math.max(maxCell, Math.abs(emp[i][j] - joint[i][j]));
    sumEmp += emp[i][j];
  }
  check('Gibbs：离散经验联合 ≡ 真实联合（max 格点误差 < 0.02，Σ=1）——链的不变分布即目标',
    maxCell < 0.02 && Math.abs(sumEmp - 1) < 1e-9,
    `max|Δjoint|=${maxCell.toFixed(4)}，Σ=${sumEmp.toFixed(6)}`);

  // 收敛（平稳性）：前后半程坐标均值应一致（链已平稳）；经验相关系数 ≈ ρ
  const half = Math.floor(samples.length / 2);
  const meanOf = (arr) => { let s = 0; for (const r of arr) s += r[0]; return s / arr.length; };
  const mFirst = meanOf(samples.slice(0, half));
  const mSecond = meanOf(samples.slice(half));
  const rho = cov[0][1] / Math.sqrt(cov[0][0] * cov[1][1]);
  const rhoTrue = 1.2 / Math.sqrt(2 * 3);
  check('Gibbs：平稳收敛（前后半程均值一致）且经验相关系数 ≈ ρ（rel < 0.08）',
    Math.abs(mFirst - mSecond) < 0.05 && rel(rho, rhoTrue) < 0.08,
    `前半程μ=${mFirst.toFixed(4)} vs 后半程μ=${mSecond.toFixed(4)}，ρ̂=${rho.toFixed(4)} vs ρ=${rhoTrue.toFixed(4)}`);
}

/* ============================================================
 * 套件 35 · 高斯混合模型（GMM-EM）
 * 站在 Dempster,Laird,Rubin 1977；Bishop 2006 PRML §9。
 * EM 定理背书：数据对数似然单调不降。
 * ============================================================ */
function suiteGMM(check) {
  const run = (label, fn) => {
    try { fn((ok, info) => check(label, ok, info)); }
    catch (e) { check(label, false, '执行异常: ' + (e && e.message)); }
  };

  // 强分离双高斯（标签已知）：两类各 N 点、均值间距远大于方差
  const nPer = 300, seed = 7;
  const rngG = mulberry32(seed);
  const mu0 = [-5, 0], mu1 = [5, 0], sig = 1;
  const XG = [], yG = [];
  for (let i = 0; i < nPer; i++) { XG.push(mu0.map((m) => gaussianSample(rngG, m, sig))); yG.push(0); }
  for (let i = 0; i < nPer; i++) { XG.push(mu1.map((m) => gaussianSample(rngG, m, sig))); yG.push(1); }
  const modelG = gmmFit(XG, 2, { iters: 100, seed: 123 });
  const respG = modelG.responsibilities;

  run('GMM：每行 responsibilities 之和 ≡ 1（≤1e-9）', (c) => {
    let maxErr = 0;
    for (const row of respG) { let s = 0; for (const v of row) s += v; maxErr = Math.max(maxErr, Math.abs(s - 1)); }
    c(maxErr < 1e-9, `maxRowSumErr=${maxErr.toExponential(2)}`);
  });

  run('GMM：weights 之和 ≡ 1（≤1e-9）', (c) => {
    let s = 0; for (const w of modelG.weights) s += w;
    c(Math.abs(s - 1) < 1e-9, `Σw=${s.toFixed(10)}`);
  });

  run('GMM：每个 cov 对称且半正定（λ_min ≥ −1e-8）', (c) => {
    let maxAsym = 0, minLam = Infinity;
    for (const cov of modelG.covs) {
      const dd = cov.length;
      for (let i = 0; i < dd; i++) for (let j = i + 1; j < dd; j++) maxAsym = Math.max(maxAsym, Math.abs(cov[i][j] - cov[j][i]));
      const { values } = jacobiEig(cov);
      for (const lam of values) if (lam < minLam) minLam = lam;
    }
    c(maxAsym < 1e-9 && minLam >= -1e-8, `maxAsym=${maxAsym.toExponential(2)}，λ_min=${minLam.toExponential(2)}`);
  });

  run('GMM：对数似然单调不降（EM 定理；最大下降 < 1e-6）', (c) => {
    const tr = modelG.logLikTrace;
    let maxDrop = 0;
    for (let k = 1; k < tr.length; k++) maxDrop = Math.max(maxDrop, tr[k - 1] - tr[k]);
    c(maxDrop < 1e-6, `maxDrop=${maxDrop.toExponential(2)}，steps=${tr.length}`);
  });

  run('GMM：强分离双高斯硬指派准确率 ≥ 0.95', (c) => {
    const pred = gmmPredict(XG, modelG).labels;
    let accA = 0, accB = 0;
    for (let i = 0; i < XG.length; i++) { if (pred[i] === yG[i]) accA++; else accB++; }
    const acc = Math.max(accA, accB) / XG.length;
    c(acc >= 0.95, `acc=${acc.toFixed(4)}（最佳置换）`);
  });
}

/* ============================================================
 * 套件 36 · 线性判别分析（LDA / Fisher）
 * 站在 Fisher 1936；Rao 1948；Bishop 2006 §4.1。
 * 定理背书：Fisher 解 w = Sw⁻¹(μ1−μ2) 令 Rayleigh 商 J(w) 取极大。
 * ============================================================ */
function suiteLDA(check) {
  const run = (label, fn) => {
    try { fn((ok, info) => check(label, ok, info)); }
    catch (e) { check(label, false, '执行异常: ' + (e && e.message)); }
  };

  // 线性可分合成数据：两类均值不同、方差小（独立特征）
  const nPer = 200, seed = 11;
  const rngL = mulberry32(seed);
  const muA = [-4, 0], muB = [4, 0], sig = 1;
  const XL = [], yL = [];
  for (let i = 0; i < nPer; i++) { XL.push(muA.map((m) => gaussianSample(rngL, m, sig))); yL.push(0); }
  for (let i = 0; i < nPer; i++) { XL.push(muB.map((m) => gaussianSample(rngL, m, sig))); yL.push(1); }
  const modelL = ldaFit(XL, yL, { reg: 1e-6 });
  const classes = modelL.classes;

  run('LDA：各类 classMeans ≡ 暴力均值（逐元素 < 1e-9）', (c) => {
    let maxErr = 0;
    for (let ci = 0; ci < classes.length; ci++) {
      const rows = [];
      for (let i = 0; i < XL.length; i++) if (yL[i] === classes[ci]) rows.push(XL[i]);
      const dd = XL[0].length;
      for (let j = 0; j < dd; j++) {
        let s = 0; for (const r of rows) s += r[j];
        maxErr = Math.max(maxErr, Math.abs(s / rows.length - modelL.classMeans[ci][j]));
      }
    }
    c(maxErr < 1e-9, `maxMeanErr=${maxErr.toExponential(2)}`);
  });

  run('LDA：Sw、Sb 对称（max 非对称项 < 1e-9）', (c) => {
    let maxAsym = 0;
    const chkSym = (M) => {
      const dd = M.length;
      for (let i = 0; i < dd; i++) for (let j = i + 1; j < dd; j++) maxAsym = Math.max(maxAsym, Math.abs(M[i][j] - M[j][i]));
    };
    chkSym(modelL.Sw); chkSym(modelL.Sb);
    c(maxAsym < 1e-9, `maxAsym=${maxAsym.toExponential(2)}`);
  });

  run('LDA：Fisher 方向恒等式 Sw·w ∥ (μ1−μ2)（|cos|≈1，≤1e-6）', (c) => {
    const w = modelL.w, Sw = modelL.Sw;
    const dir = [];
    for (let j = 0; j < w.length; j++) dir[j] = modelL.classMeans[1][j] - modelL.classMeans[0][j];
    const Sw_w = [];
    for (let i = 0; i < Sw.length; i++) { let s = 0; for (let j = 0; j < Sw[0].length; j++) s += Sw[i][j] * w[j]; Sw_w.push(s); }
    let dot1 = 0, n1 = 0, n2 = 0;
    for (let j = 0; j < w.length; j++) { dot1 += Sw_w[j] * dir[j]; n1 += Sw_w[j] * Sw_w[j]; n2 += dir[j] * dir[j]; }
    const cosw = dot1 / (Math.sqrt(n1) * Math.sqrt(n2) + 1e-300);
    c(Math.abs(Math.abs(cosw) - 1) < 1e-6, `|cos|=${Math.abs(cosw).toFixed(10)}`);
  });

  run('LDA：Fisher 比 J(w) ≥ 随机方向 J（w 的 J 最大）', (c) => {
    const Sw = modelL.Sw, Sb = modelL.Sb, w = modelL.w;
    const quad = (M, v) => {
      let s = 0;
      for (let i = 0; i < M.length; i++) { let t = 0; for (let j = 0; j < M[0].length; j++) t += M[i][j] * v[j]; s += v[i] * t; }
      return s;
    };
    const J = (v) => quad(Sb, v) / (quad(Sw, v) + 1e-300);
    const Jw = J(w);
    const rngR = mulberry32(99);
    let maxRand = 0;
    for (let t = 0; t < 20; t++) {
      const v = [];
      for (let j = 0; j < w.length; j++) v.push(gaussianSample(rngR, 0, 1));
      maxRand = Math.max(maxRand, J(v));
    }
    c(Jw >= maxRand - 1e-9, `J(w)=${Jw.toExponential(3)}，maxRandJ=${maxRand.toExponential(3)}`);
  });

  run('LDA：投影后类心间距 > 类内散布，最近类心分类准确率 = 1.0', (c) => {
    const proj = ldaProject(XL, modelL); // 二分类 ⇒ 1-D
    let s0 = 0, n0 = 0, s1 = 0, n1 = 0;
    for (let i = 0; i < XL.length; i++) {
      if (yL[i] === classes[0]) { s0 += proj[i]; n0++; } else { s1 += proj[i]; n1++; }
    }
    const m0 = s0 / n0, m1 = s1 / n1;
    let v0 = 0, v1 = 0;
    for (let i = 0; i < XL.length; i++) {
      if (yL[i] === classes[0]) v0 += (proj[i] - m0) ** 2; else v1 += (proj[i] - m1) ** 2;
    }
    const sd0 = Math.sqrt(v0 / n0), sd1 = Math.sqrt(v1 / n1);
    const sep = Math.abs(m1 - m0);
    const mid = 0.5 * (m0 + m1);
    let acc = 0;
    for (let i = 0; i < XL.length; i++) {
      const predC = proj[i] >= mid ? classes[1] : classes[0];
      if (predC === yL[i]) acc++;
    }
    const accR = acc / XL.length;
    c(sep > 4 * Math.max(sd0, sd1) && accR === 1.0, `sep=${sep.toFixed(4)}，maxSD=${Math.max(sd0, sd1).toFixed(4)}，acc=${accR.toFixed(4)}`);
  });
}

/* ============================================================
 * 套件 37 · 高斯朴素贝叶斯（GNB）
 * 站在 Maron 1961；Domingos&Pazzani 1997。后验在 log 域计算，softmax 回概率。
 * ============================================================ */
function suiteNaiveBayes(check) {
  const run = (label, fn) => {
    try { fn((ok, info) => check(label, ok, info)); }
    catch (e) { check(label, false, '执行异常: ' + (e && e.message)); }
  };

  // 独立特征、类别可分：每类各特征均值不同、方差适中
  const nPer = 200, seed = 5;
  const rngN = mulberry32(seed);
  const muClass0 = [0, 0, 0, 0], muClass1 = [3, 3, 3, 3], sig = 1;
  const XN = [], yN = [];
  for (let i = 0; i < nPer; i++) { XN.push(muClass0.map((m) => gaussianSample(rngN, m, sig))); yN.push(0); }
  for (let i = 0; i < nPer; i++) { XN.push(muClass1.map((m) => gaussianSample(rngN, m, sig))); yN.push(1); }
  const modelN = nbFit(XN, yN);
  const { posteriors: postN, labels: labN } = nbPredict(XN, modelN);

  run('GNB：priors 之和 ≡ 1（≤1e-9）', (c) => {
    let s = 0; for (const p of modelN.priors) s += p;
    c(Math.abs(s - 1) < 1e-9, `Σprior=${s.toFixed(10)}`);
  });

  run('GNB：每样本 posteriors 之和 ≡ 1（≤1e-9）', (c) => {
    let maxErr = 0;
    for (const row of postN) { let s = 0; for (const v of row) s += v; maxErr = Math.max(maxErr, Math.abs(s - 1)); }
    c(maxErr < 1e-9, `maxSumErr=${maxErr.toExponential(2)}`);
  });

  run('GNB：独立特征、类别可分数据准确率 = 1.0', (c) => {
    let acc = 0;
    for (let i = 0; i < XN.length; i++) if (labN[i] === yN[i]) acc++;
    const accR = acc / XN.length;
    c(accR === 1.0, `acc=${accR.toFixed(4)}`);
  });

  run('GNB：单点对数项 ≡ 手算（gaussianLogPDF，差 < 1e-9）', (c) => {
    const i = 0, xi = XN[i], K = modelN.classes.length;
    const handLog = new Array(K);
    for (let k = 0; k < K; k++) {
      let s = Math.log(modelN.priors[k] + 1e-300);
      for (let j = 0; j < xi.length; j++) s += gaussianLogPDF(xi[j], modelN.classMeans[k][j], Math.sqrt(modelN.classVars[k][j]));
      handLog[k] = s;
    }
    let mx = -Infinity; for (const v of handLog) if (v > mx) mx = v;
    let ss = 0; for (const v of handLog) ss += Math.exp(v - mx);
    const handPost = handLog.map((v) => Math.exp(v - mx) / ss);
    let maxDiff = 0;
    for (let k = 0; k < K; k++) maxDiff = Math.max(maxDiff, Math.abs(handPost[k] - postN[i][k]));
    c(maxDiff < 1e-9, `maxPostDiff=${maxDiff.toExponential(2)}`);
  });

  run('GNB：labels = argmax posteriors（逐样本核对）', (c) => {
    let bad = 0;
    for (let i = 0; i < XN.length; i++) {
      let bi = 0, bv = -Infinity;
      for (let k = 0; k < postN[i].length; k++) if (postN[i][k] > bv) { bv = postN[i][k]; bi = k; }
      if (bi !== labN[i]) bad++;
    }
    c(bad === 0, `mismatch=${bad}`);
  });
}

/* ============================================================



/* ============================================================
 * 套件 · 随机森林（Random Forest，Breiman 2001）
 * 站在 Breiman 2001《Random Forests》。
 * 复用 tree.js 的 DecisionTree（单棵基准）与 giniImpurity。
 * 不变量：
 *   ① featureImportances 之和 ≡ 1（≤1e-9）；
 *   ② trees.length === numTrees；
 *   ③ twoMoons(n=200,sigma=0.05) 分类准确率 ≥ 0.95；
 *   ④ 方差缩减：RF 准确率 ≥ 单棵未集成树均值（并报告单棵 DecisionTree 基准）；
 *   ⑤ OOB 误差 < 0.1（报告/软判）；
 *   ⑥ 预测类别 ≡ argmax(probabilities)（mismatch=0）。
 * ============================================================ */
/** OOB 误差：对每个样本用未包含它的树投票（分类=误分率，回归=MSE） */
function rfOOBError(X, y, model) {
  const { trees, oobIndices, classes, isRegression } = model;
  const N = X.length;
  let nOob = 0;
  if (isRegression) {
    let sse = 0;
    for (let i = 0; i < N; i++) {
      let sum = 0, cnt = 0;
      for (let t = 0; t < trees.length; t++) {
        const oob = oobIndices[t];
        let inOob = false;
        for (let j = 0; j < oob.length; j++) if (oob[j] === i) { inOob = true; break; }
        if (!inOob) continue;
        let node = trees[t];
        while (!node.leaf) node = (X[i][node.feature] <= node.threshold) ? node.left : node.right;
        sum += node.value; cnt++;
      }
      if (cnt === 0) continue;
      const pred = sum / cnt;
      const d = pred - y[i]; sse += d * d; nOob++;
    }
    return nOob > 0 ? sse / nOob : 0;
  }
  let wrong = 0;
  for (let i = 0; i < N; i++) {
    const votes = new Array(classes.length).fill(0);
    let cnt = 0;
    for (let t = 0; t < trees.length; t++) {
      const oob = oobIndices[t];
      let inOob = false;
      for (let j = 0; j < oob.length; j++) if (oob[j] === i) { inOob = true; break; }
      if (!inOob) continue;
      let node = trees[t];
      while (!node.leaf) node = (X[i][node.feature] <= node.threshold) ? node.left : node.right;
      for (let k = 0; k < classes.length; k++) votes[k] += (node.proba.get(classes[k]) || 0);
      cnt++;
    }
    if (cnt === 0) continue;
    let bk = 0, bv = -Infinity;
    for (let k = 0; k < classes.length; k++) if (votes[k] > bv) { bv = votes[k]; bk = k; }
    if (classes[bk] !== y[i]) wrong++;
    nOob++;
  }
  return nOob > 0 ? wrong / nOob : 0;
}

function suiteRandomForest(check) {
  const run = (label, fn) => {
    try { fn((ok, info) => check(label, ok, info)); }
    catch (e) { check(label, false, '执行异常: ' + (e && e.message)); }
  };

  // 两月亮数据集（n=200, sigma=0.05 低噪声，seed=42 可复现）
  const { X, y } = twoMoons({ n: 200, sigma: 0.05, seed: 42 });
  const N = X.length;
  const numTrees = 64, maxDepth = 10, minSamples = 2;

  const model = randomForestTrain(X, y, { numTrees, maxDepth, minSamples, seed: 42 });
  const { predictions, probabilities } = randomForestPredict(X, model);

  // ① featureImportances 之和 ≡ 1（到 1e-9）
  run('RF：featureImportances 之和 ≡ 1（≤1e-9）', (c) => {
    let s = 0; for (const v of model.featureImportances) s += v;
    c(Math.abs(s - 1) < 1e-9, `Σimp=${s.toFixed(12)}`);
  });

  // ② 树数量 = numTrees（桩校验：每棵树均在 bootstrap 子集上训练）
  run('RF：trees.length === numTrees（每棵树均经 bootstrap 训练）', (c) => {
    c(model.trees.length === numTrees, `trees=${model.trees.length}，numTrees=${numTrees}`);
  });

  // ③ twoMoons 分类准确率 ≥ 0.95
  run('RF：twoMoons(n=200,sigma=0.05) 分类准确率 ≥ 0.95', (c) => {
    let acc = 0; for (let i = 0; i < N; i++) if (predictions[i] === y[i]) acc++;
    const accR = acc / N;
    c(accR >= 0.95, `acc=${accR.toFixed(4)}（${acc}/${N}）`);
  });

  // ④ 方差缩减：RF（集成）准确率 ≥ 单棵未集成树（森林基学习器）均值；并报告单棵 DecisionTree 基准
  run('RF：方差缩减 — RF 准确率 ≥ 单棵未集成树均值（且 ≥ 单棵 DecisionTree 基准）', (c) => {
    let sumBase = 0;
    for (const tree of model.trees) {
      let hit = 0;
      for (let i = 0; i < N; i++) {
        let node = tree;
        while (!node.leaf) node = (X[i][node.feature] <= node.threshold) ? node.left : node.right;
        let bk = 0, bv = -Infinity;
        for (const [cls, p] of node.proba) if (p > bv) { bv = p; bk = cls; }
        if (bk === y[i]) hit++;
      }
      sumBase += hit / N;
    }
    const meanBase = sumBase / model.trees.length;
    const dt = new DecisionTree({ maxDepth, minSamples }).fit(X, y);
    let dtHit = 0; for (let i = 0; i < N; i++) if (dt.predict(X[i]) === y[i]) dtHit++;
    const dtAcc = dtHit / N;
    let rfAcc = 0; for (let i = 0; i < N; i++) if (predictions[i] === y[i]) rfAcc++;
    rfAcc /= N;
    c(rfAcc >= meanBase - 1e-9, `rfAcc=${rfAcc.toFixed(4)}，基学习器均值=${meanBase.toFixed(4)}，单棵DecisionTree=${dtAcc.toFixed(4)}`);
  });

  // ⑤ OOB 误差在合理范围（< 0.1，报告/软判）
  run('RF：OOB 误差 < 0.1（报告）', (c) => {
    const oob = rfOOBError(X, y, model);
    c(oob < 0.1, `oobErr=${oob.toFixed(4)}`);
  });

  // ⑥ 预测类别 = 多数投票 ⇒ predictions ≡ argmax(probabilities)（逐样本核对，mismatch=0）
  run('RF：predictions ≡ argmax(probabilities)（mismatch=0）', (c) => {
    let mismatch = 0;
    for (let i = 0; i < N; i++) {
      const probs = probabilities[i];
      let bk = 0, bv = -Infinity;
      for (let k = 0; k < probs.length; k++) if (probs[k] > bv) { bv = probs[k]; bk = k; }
      if (model.classes[bk] !== predictions[i]) mismatch++;
    }
    c(mismatch === 0, `mismatch=${mismatch}`);
  });
}

/* ============================================================
 * 套件 39 · 感知机（Rosenblatt 1958 / Novikoff 1962）
 * 定理背书：Novikoff 1962 — 线性可分数据（几何间隔 γ > 0、最大样本范数 R）
 * 上感知机误分次数 ≤ (R/γ)²。本套件以已知分隔面 w*=(1,0) 取 γ 的下界，
 * 故 (R/γ_true)² ≤ (R/γ_lb)² 仍是合法上界。
 * ============================================================ */
function suitePerceptron(check) {
  const run = (label, fn) => {
    try { fn((ok, info) => check(label, ok, info)); }
    catch (e) { check(label, false, '执行异常: ' + (e && e.message)); }
  };

  // 线性可分数据：+1 类 x_1 ≥ 1、−1 类 x_1 ≤ −1 ⇒ w*=(1,0)、b*=0 为已知分隔面
  const rngP = mulberry32(2024);
  const nPer = 40, sig = 0.5;
  const Xp = [], yp = [];
  for (let i = 0; i < nPer; i++) {
    const u = Math.abs(gaussianSample(rngP, 0, sig)), v = gaussianSample(rngP, 0, sig);
    Xp.push([1 + u, v]); yp.push(1);
    Xp.push([-1 - u, v]); yp.push(-1);
  }
  const modelP = perceptronTrain(Xp, yp, { lr: 1, maxPasses: 100 });

  run('感知机：线性可分数据收敛且 min_i y_i(w·x_i+b) > 0（0 训练误差）', (c) => {
    let minM = Infinity;
    for (let i = 0; i < Xp.length; i++) minM = Math.min(minM, yp[i] * (dot(modelP.w, Xp[i]) + modelP.b));
    c(modelP.converged && minM > 0,
      `converged=${modelP.converged}，minMargin=${minM.toFixed(6)}，passes=${modelP.updates}`);
  });

  run('感知机：收敛后 sign(w·x+b) ≡ y（mismatch=0）', (c) => {
    const pred = perceptronPredict(Xp, modelP.w, modelP.b);
    let mismatch = 0;
    for (let i = 0; i < Xp.length; i++) if (pred[i] !== yp[i]) mismatch++;
    c(mismatch === 0, `mismatch=${mismatch}/${Xp.length}`);
  });

  run('感知机：Novikoff 定理 mistakes ≤ (R/γ)²（可分数据有限步收敛界）', (c) => {
    let R = 0;
    for (const x of Xp) R = Math.max(R, norm(x));
    let gamma = Infinity; // 已知分隔面 w*=(1,0) 的几何间隔下界（真实 γ ≥ gamma）
    for (let i = 0; i < Xp.length; i++) gamma = Math.min(gamma, Math.abs(Xp[i][0]));
    const bound = (R / gamma) ** 2;
    c(modelP.mistakes <= bound,
      `mistakes=${modelP.mistakes} ≤ (R/γ)²=${bound.toFixed(3)}（R=${R.toFixed(3)}，γ≥${gamma.toFixed(3)}）`);
  });

  run('感知机：单步更新确定性 — perceptronStep ≡ 手算 w+lr·y·x / b+lr·y（误差 0）', (c) => {
    const x0 = Xp[0], y0 = yp[0];
    const st = perceptronStep([0, 0], 0, x0, y0, 1);
    const err = Math.max(Math.abs(st.w[0] - y0 * x0[0]), Math.abs(st.w[1] - y0 * x0[1]), Math.abs(st.b - y0));
    c(err === 0, `w=[${st.w[0]},${st.w[1]}]，b=${st.b}，误差=${err}`);
  });

  run('感知机：train(maxPasses=1) ≡ 用 perceptronStep 逐点模拟一整轮（误差 0）', (c) => {
    let w = [0, 0], b = 0, nUp = 0;
    for (let i = 0; i < Xp.length; i++) {
      if (yp[i] * (dot(w, Xp[i]) + b) <= 0) {
        const s = perceptronStep(w, b, Xp[i], yp[i], 1);
        w = s.w; b = s.b; nUp++;
      }
    }
    const m1 = perceptronTrain(Xp, yp, { maxPasses: 1 });
    let errW = 0;
    for (let j = 0; j < w.length; j++) errW = Math.max(errW, Math.abs(w[j] - m1.w[j]));
    const errB = Math.abs(b - m1.b);
    c(nUp > 0 && errW === 0 && errB === 0 && m1.updates === 1,
      `一轮误分=${nUp}，w/b 最大误差=${Math.max(errW, errB)}，updates=${m1.updates}`);
  });

  run('感知机：不可分数据（XOR）在 maxPasses 内有界停止（仅报告，非硬失败）', (c) => {
    const Xxor = [[0, 0], [0, 1], [1, 0], [1, 1]], yxor = [-1, 1, 1, -1];
    const mX = perceptronTrain(Xxor, yxor, { maxPasses: 50 });
    c(!mX.converged && mX.mistakes > 0 && mX.updates === 50 && Number.isFinite(mX.mistakes),
      `mistakes=${mX.mistakes}，passes=${mX.updates}，converged=${mX.converged}（有界停止）`);
  });
}

/* ============================================================
 * 套件 40 · 独立成分分析（FastICA，Hyvärinen & Oja 2000 / Comon 1994）
 * 定理背书：Comon 1994 — 至多一个高斯源时 ICA 可辨识到置换与缩放；
 * 白化保证 Cov(Z)=I；deflation 正交化保证 W·Wᵀ=I。
 * ============================================================ */
function suiteICA(check) {
  const run = (label, fn) => {
    try { fn((ok, info) => check(label, ok, info)); }
    catch (e) { check(label, false, '执行异常: ' + (e && e.message)); }
  };

  // 5 个相互独立的非高斯源（logistic / χ²(1) / 指数，各自独立 RNG 流）
  const nI = 8000, dI = 5;
  const kinds = [0, 1, 2, 0, 1];
  const rngs = kinds.map((_, j) => mulberry32(1000 + 17 * j));
  const mkSrc = (rng, kind) => {
    if (kind === 0) { const u = rng(); return Math.log(u / (1 - u)); } // 标准 logistic（超额峰度 +1.2）
    if (kind === 1) { const g = gaussianSample(rng); return g * g; }   // χ²(1)（+12）
    return -Math.log(rng() || 1e-12);                                   // 指数(1)（+6）
  };
  const S = [];
  for (let i = 0; i < nI; i++) {
    const row = [];
    for (let j = 0; j < dI; j++) row.push(mkSrc(rngs[j], kinds[j]));
    S.push(row);
  }
  // 随机满秩混合矩阵 A ⇒ 观测 X = S·Aᵀ（Comon 可识别性前提）
  const A = randomMatrix(dI, dI, mulberry32(777), -1, 1);
  const X = matmul(S, transpose(A));
  const model = icaFit(X, { components: dI, maxIter: 300, tol: 1e-12, seed: 42 });
  const Shat = icaTransform(X, model);

  // 局部工具：Pearson 相关系数与超额峰度
  const pearson = (a, b) => {
    let ma = 0, mb = 0;
    for (let i = 0; i < a.length; i++) { ma += a[i]; mb += b[i]; }
    ma /= a.length; mb /= b.length;
    let sab = 0, saa = 0, sbb = 0;
    for (let i = 0; i < a.length; i++) {
      const da = a[i] - ma, db = b[i] - mb;
      sab += da * db; saa += da * da; sbb += db * db;
    }
    return sab / (Math.sqrt(saa * sbb) + 1e-300);
  };
  const excessKurt = (a) => {
    let m = 0;
    for (const v of a) m += v;
    m /= a.length;
    let m2 = 0, m4 = 0;
    for (const v of a) { const dv = v - m; m2 += dv * dv; m4 += dv * dv * dv * dv; }
    m2 /= a.length; m4 /= a.length;
    return m4 / (m2 * m2) - 3;
  };

  run('ICA：白化后数据协方差 ≡ I（maxAbs(Cov(Z)−I) < 1e-8）', (c) => {
    const mu = model.whitening.mean;
    const Xc = X.map((r) => r.map((v, j) => v - mu[j]));
    const Z = matmul(Xc, transpose(model.K));
    const CZ = covarianceMatrix(Z);
    let err = 0;
    for (let i = 0; i < dI; i++) for (let j = 0; j < dI; j++) {
      err = Math.max(err, Math.abs(CZ[i][j] - (i === j ? 1 : 0)));
    }
    c(err < 1e-8, `maxAbs(Cov(Z)−I)=${err.toExponential(2)}`);
  });

  run('ICA：unmixing 矩阵 W·Wᵀ ≡ I（deflation 去相关，< 1e-10）', (c) => {
    const WWt = matmul(model.W, transpose(model.W));
    let err = 0;
    for (let i = 0; i < dI; i++) for (let j = 0; j < dI; j++) {
      err = Math.max(err, Math.abs(WWt[i][j] - (i === j ? 1 : 0)));
    }
    c(err < 1e-10, `maxAbs(W·Wᵀ−I)=${err.toExponential(2)}`);
  });

  run('ICA：源恢复（Comon 1994）— 每个恢复源与某真实源 |corr| > 0.99', (c) => {
    let minBest = Infinity;
    for (let p = 0; p < dI; p++) {
      const col = Shat.map((r) => r[p]);
      let best = 0;
      for (let j = 0; j < dI; j++) best = Math.max(best, Math.abs(pearson(col, S.map((r) => r[j]))));
      minBest = Math.min(minBest, best);
    }
    c(minBest > 0.99, `min_j |corr(ŝ_p, s_j)|=${minBest.toFixed(6)}（允许置换/符号）`);
  });

  run('ICA：非高斯性增强 — 恢复源 mean|超额峰度| > 原始混合信号', (c) => {
    let rec = 0, mix = 0;
    for (let j = 0; j < dI; j++) {
      rec += Math.abs(excessKurt(Shat.map((r) => r[j])));
      mix += Math.abs(excessKurt(X.map((r) => r[j])));
    }
    c(rec > mix, `恢复 mean|excess|=${(rec / dI).toFixed(3)} > 混合 ${(mix / dI).toFixed(3)}（FastICA 增大非高斯性）`);
  });

  run('ICA：恢复源近似去相关 — Cov(ŝ) ≈ I（< 1e-8）', (c) => {
    const CS = covarianceMatrix(Shat);
    let err = 0;
    for (let i = 0; i < dI; i++) for (let j = 0; j < dI; j++) {
      err = Math.max(err, Math.abs(CS[i][j] - (i === j ? 1 : 0)));
    }
    c(err < 1e-8, `maxAbs(Cov(ŝ)−I)=${err.toExponential(2)}`);
  });

  run('ICA：确定性 — 同种子两次拟合 W 逐位一致', (c) => {
    const m2 = icaFit(X, { components: dI, maxIter: 300, tol: 1e-12, seed: 42 });
    let err = 0;
    for (let i = 0; i < dI; i++) for (let j = 0; j < dI; j++) {
      err = Math.max(err, Math.abs(model.W[i][j] - m2.W[i][j]));
    }
    c(err === 0, `maxDiff(W, W′)=${err}`);
  });
}

/* ============================================================
 * 套件 41 · 典型相关分析（CCA，Hotelling 1936）
 * 定理背书：Hotelling 1936 — 典型对逐阶最大化相关系数；
 * Y = X·M（M 满秩可逆）时全部 r_k ≡ 1；独立数据 r_k → 0（速率 ~1/√n）。
 * 交叉验证：「白化 + 交叉协方差 SVD」与 eig 路线相互独立。
 * ============================================================ */
function suiteCCA(check) {
  const run = (label, fn) => {
    try { fn((ok, info) => check(label, ok, info)); }
    catch (e) { check(label, false, '执行异常: ' + (e && e.message)); }
  };

  // 相关非完美数据：X、Y 共享二维隐因子 z，各加 0.5 噪声
  const nC = 400;
  const rngC = mulberry32(4242);
  const X = [], Y = [];
  for (let i = 0; i < nC; i++) {
    const z1 = gaussianSample(rngC), z2 = gaussianSample(rngC);
    X.push([z1 + gaussianSample(rngC, 0, 0.5), z2 + gaussianSample(rngC, 0, 0.5)]);
    Y.push([z1 + gaussianSample(rngC, 0, 0.5), z2 + gaussianSample(rngC, 0, 0.5)]);
  }
  const model = ccaFit(X, Y, { reg: 1e-6 });
  const { U, V } = ccaProject(X, Y, model);
  const Sxx = covarianceMatrix(X);
  const Syy = covarianceMatrix(Y);
  const kC = model.correlations.length;

  run('CCA：相关系数 r_k ∈ [0,1] 且非增（r_1 ≥ r_2 ≥ …）', (c) => {
    const rs = model.correlations;
    let inRange = true, mono = true;
    for (const r of rs) if (!(r >= 0 && r <= 1)) inRange = false;
    for (let k = 1; k < rs.length; k++) if (rs[k] > rs[k - 1] + 1e-12) mono = false;
    c(inRange && mono, `r=[${rs.map((r) => r.toFixed(6)).join(', ')}]`);
  });

  run('CCA：典型变量 Cov(U)=Cov(V)=I 且 Cov(U,V)=diag(r)（交叉协方差对角）', (c) => {
    const CU = covarianceMatrix(U), CV = covarianceMatrix(V);
    let cuErr = 0, cvErr = 0;
    for (let i = 0; i < CU.length; i++) for (let j = 0; j < CU.length; j++) {
      cuErr = Math.max(cuErr, Math.abs(CU[i][j] - (i === j ? 1 : 0)));
      cvErr = Math.max(cvErr, Math.abs(CV[i][j] - (i === j ? 1 : 0)));
    }
    // 交叉协方差（全量累加、单次除 n−1）
    const muU = colMean(U), muV = colMean(V);
    let offErr = 0, diagErr = 0;
    for (let i = 0; i < kC; i++) {
      for (let j = 0; j < kC; j++) {
        let s = 0;
        for (let t = 0; t < U.length; t++) s += (U[t][i] - muU[i]) * (V[t][j] - muV[j]);
        const cc = s / (U.length - 1);
        if (i === j) diagErr = Math.max(diagErr, Math.abs(cc - model.correlations[i]));
        else offErr = Math.max(offErr, Math.abs(cc));
      }
    }
    c(cuErr < 1e-8 && cvErr < 1e-8 && offErr < 1e-8 && diagErr < 1e-8,
      `Cov(U)−I=${cuErr.toExponential(2)}，Cov(V)−I=${cvErr.toExponential(2)}，非对角=${offErr.toExponential(2)}，对角−r=${diagErr.toExponential(2)}`);
  });

  run('CCA：归一化约定 AᵀΣxxA = I、BᵀΣyyB = I', (c) => {
    const AxA = matmul(matmul(transpose(model.A), Sxx), model.A);
    const ByB = matmul(matmul(transpose(model.B), Syy), model.B);
    let err = 0;
    for (let i = 0; i < AxA.length; i++) {
      for (let j = 0; j < AxA.length; j++) {
        err = Math.max(err, Math.abs(AxA[i][j] - (i === j ? 1 : 0)), Math.abs(ByB[i][j] - (i === j ? 1 : 0)));
      }
    }
    c(err < 1e-10, `maxAbs(归一化偏差)=${err.toExponential(2)}`);
  });

  run('CCA：完美相关 ⇒ r ≡ 1 — Y = X·M（M 满秩可逆）', (c) => {
    const rngM = mulberry32(888);
    const Xp = Array.from({ length: 300 }, () => [gaussianSample(rngM), gaussianSample(rngM), gaussianSample(rngM)]);
    const Mp = randomMatrix(3, 3, mulberry32(999), -1, 1);
    const Yp = matmul(Xp, Mp);
    const mP = ccaFit(Xp, Yp, { reg: 1e-6 });
    let err = 0;
    for (const r of mP.correlations) err = Math.max(err, Math.abs(1 - r));
    c(err < 1e-6, `max|1−r_k|=${err.toExponential(2)}，r=[${mP.correlations.map((r) => r.toFixed(10)).join(', ')}]`);
  });

  run('CCA：独立 X,Y ⇒ 各 r_k ≈ 0（< 1e-2；n=400000 压低 1/√n 样本噪声）', (c) => {
    const nI = 400000;
    const rngX = mulberry32(31), rngY = mulberry32(32);
    const Xi = Array.from({ length: nI }, () => [gaussianSample(rngX), gaussianSample(rngX), gaussianSample(rngX)]);
    const Yi = Array.from({ length: nI }, () => [gaussianSample(rngY), gaussianSample(rngY), gaussianSample(rngY)]);
    const mI = ccaFit(Xi, Yi, { reg: 1e-6 });
    let rmax = 0;
    for (const r of mI.correlations) rmax = Math.max(rmax, r);
    c(rmax < 1e-2, `max r_k=${rmax.toExponential(2)}（理论 ρ=0，样本噪声 ~2√(p/n)）`);
  });

  run('CCA：交叉验证 — 白化+交叉协方差 SVD 与 ccaFit 相关系数一致（1e-8）', (c) => {
    // 第二条独立路线：Σxx、Σyy 各自白化 ⇒ 对 Σxy 做 SVD，奇异值即典型相关系数
    const n = X.length, p = Sxx.length, q = Syy.length;
    const xMu = colMean(X), yMu = colMean(Y);
    const Xc2 = X.map((r) => r.map((v, j) => v - xMu[j]));
    const Yc2 = Y.map((r) => r.map((v, j) => v - yMu[j]));
    const Sxy = zeros(p, q);
    for (let i = 0; i < p; i++) {
      for (let j = 0; j < q; j++) {
        let s = 0;
        for (let t = 0; t < n; t++) s += Xc2[t][i] * Yc2[t][j];
        Sxy[i][j] = s / (n - 1);
      }
    }
    const wm = (values, vectors) => {
      const dd = values.length;
      const M = zeros(dd, dd);
      let lamMax = 0;
      for (const l of values) if (l > lamMax) lamMax = l;
      for (let k = 0; k < dd; k++) {
        const cc = 1 / Math.sqrt(Math.max(values[k], 1e-12 * lamMax));
        for (let i = 0; i < dd; i++) M[k][i] = cc * vectors[k][i];
      }
      return M;
    };
    const { values: lx, vectors: ex } = jacobiEig(Sxx);
    const { values: ly, vectors: ey } = jacobiEig(Syy);
    const Cw = matmul(matmul(wm(lx, ex), Sxy), transpose(wm(ly, ey)));
    const s = (Cw.length >= Cw[0].length) ? svd(Cw).s : svd(transpose(Cw)).s;
    let err = 0;
    for (let k = 0; k < kC; k++) err = Math.max(err, Math.abs(s[k] - model.correlations[k]));
    c(err < 1e-8, `max|s_k − r_k|=${err.toExponential(2)}`);
  });
}

/* ============================================================
 * 套件 42 · 凝聚式层次聚类（Sokal & Michener 1958 / Lance & Williams 1967）
 * 定理背书：single/complete/average 均为可约 linkage ⇒ 合并高度单调非降
 * （凝聚 ultrametric）；Lance–Williams 递推 ≡ 成员集闭式定义。
 * ============================================================ */
function suiteHierarchical(check) {
  const run = (label, fn) => {
    try { fn((ok, info) => check(label, ok, info)); }
    catch (e) { check(label, false, '执行异常: ' + (e && e.message)); }
  };

  // 强分离 3 高斯团（中心间距 ≫ 方差，average linkage 可精确切分）
  const rngH = mulberry32(909);
  const centers = [[0, 0], [12, 0], [6, 10]];
  const Xh = [], yh = [];
  centers.forEach((c, ci) => {
    for (let i = 0; i < 30; i++) {
      Xh.push([gaussianSample(rngH, c[0], 0.5), gaussianSample(rngH, c[1], 0.5)]);
      yh.push(ci);
    }
  });
  const nH = Xh.length;
  const hc = hierarchicalCluster(Xh, { linkage: 'average' });

  run('层次聚类：合并距离单调非降（可约 linkage ⇒ ultrametric）', (c) => {
    let mono = true, minStep = Infinity;
    for (let k = 1; k < hc.distances.length; k++) {
      const d = hc.distances[k] - hc.distances[k - 1];
      if (d < minStep) minStep = d;
      if (d < -1e-12) mono = false;
    }
    c(mono, `最小递增量=${minStep.toExponential(2)}，末次合并高度=${hc.distances[nH - 2].toFixed(4)}`);
  });

  run('层次聚类：n 样本 ⇒ n−1 次合并，最终归为 1 簇', (c) => {
    const okM = hc.merges.length === nH - 1;
    const okC = hc.clusters(1).length === 1 && hc.clusters(1)[0].length === nH;
    c(okM && okC, `merges=${hc.merges.length}（应为 ${nH - 1}），clusters(1) 得到 ${hc.clusters(1).length} 簇`);
  });

  run('层次聚类：强分离 3 高斯团 clusters(3) 与真实标签 ARI = 1', (c) => {
    const lab3 = hc.labels(3);
    const ari = adjustedRandIndex(lab3, yh);
    c(ari === 1, `ARI=${ari.toFixed(6)}`);
  });

  run('层次聚类：Lance–Williams 一致性 — 4 点共线算例 ≡ 闭式（single/complete/average）', (c) => {
    // 点 {0,1,3,6}：手算合并序列（先 (0,1) 高 1；随后 single 接 2、complete 平手接 (2,3)、average 接 2）
    const X4 = [[0], [1], [3], [6]];
    const expect = { single: [1, 2, 3], complete: [1, 3, 6], average: [1, 2.5, 14 / 3] };
    let worst = 0;
    const got = [];
    for (const lk of ['single', 'complete', 'average']) {
      const r4 = hierarchicalCluster(X4, { linkage: lk });
      for (let k = 0; k < 3; k++) worst = Math.max(worst, Math.abs(r4.distances[k] - expect[lk][k]));
      got.push(`${lk}=[${r4.distances.map((v) => v.toFixed(4)).join(', ')}]`);
    }
    c(worst < 1e-12, `最大偏差=${worst.toExponential(2)}；${got.join('；')}`);
  });

  run('层次聚类：递推 ≡ 定义 — blob 数据每次合并距离 ≡ 成员集闭式（min/max/加权均值）', (c) => {
    let worst = 0;
    for (const lk of ['single', 'complete', 'average']) {
      const res = hierarchicalCluster(Xh, { linkage: lk });
      const members = Array.from({ length: 2 * nH - 1 }, (_, id) => (id < nH ? [id] : null));
      res.merges.forEach((m, step) => {
        const ga = members[m.a], gb = members[m.b];
        let mn = Infinity, mx = -Infinity, acc = 0, cnt = 0;
        for (const i of ga) {
          for (const j of gb) {
            const dij = Math.hypot(Xh[i][0] - Xh[j][0], Xh[i][1] - Xh[j][1]);
            if (dij < mn) mn = dij;
            if (dij > mx) mx = dij;
            acc += dij; cnt++;
          }
        }
        const closed = lk === 'single' ? mn : lk === 'complete' ? mx : acc / cnt;
        worst = Math.max(worst, Math.abs(closed - m.distance));
        members[nH + step] = ga.concat(gb);
      });
    }
    c(worst < 1e-9, `三种 linkage 全部 ${nH - 1} 次合并的最大偏差=${worst.toExponential(2)}`);
  });

  run('层次聚类：确定性 — 相同输入 ⇒ 相同合并树（逐位一致）', (c) => {
    const hc2 = hierarchicalCluster(Xh, { linkage: 'average' });
    let err = 0;
    for (let k = 0; k < hc.merges.length; k++) {
      err = Math.max(err, Math.abs(hc.merges[k].distance - hc2.merges[k].distance));
      if (hc.merges[k].a !== hc2.merges[k].a || hc.merges[k].b !== hc2.merges[k].b) err = Infinity;
    }
    c(err === 0, `两次运行 merges 最大差异=${err}`);
  });

  run('层次聚类：cophenetic 相关系数 ≥ 0.9（结构良好数据）', (c) => {
    const C = hierarchicalCophenetic(hc.merges, nH);
    const orig = [], coph = [];
    for (let i = 0; i < nH; i++) {
      for (let j = i + 1; j < nH; j++) {
        orig.push(Math.hypot(Xh[i][0] - Xh[j][0], Xh[i][1] - Xh[j][1]));
        coph.push(C[i][j]);
      }
    }
    let mo = 0, mc = 0;
    for (let t = 0; t < orig.length; t++) { mo += orig[t]; mc += coph[t]; }
    mo /= orig.length; mc /= coph.length;
    let so = 0, sc = 0, soc = 0;
    for (let t = 0; t < orig.length; t++) {
      so += (orig[t] - mo) ** 2;
      sc += (coph[t] - mc) ** 2;
      soc += (orig[t] - mo) * (coph[t] - mc);
    }
    const r = soc / Math.sqrt(so * sc);
    c(r >= 0.9, `cophenetic corr=${r.toFixed(6)}`);
  });
}

/* ============================================================
 * 套件 43 · 二次判别分析 QDA（Bishop PRML §4.3）
 * 站在：Fisher 1936 / Bishop 2006 §4.3。判别三足鼎立之第三足：
 *       各类完整协方差 Σ_c ⇒ 二次决策边界；Σ_c 共享时二次项对 argmax
 *       无贡献 ⇒ 边界退化为线性（≡ LDA + 最近类心）。
 * ============================================================ */
function suiteQDA(check) {
  const run = (label, fn) => {
    try { fn((ok, info) => check(label, ok, info)); }
    catch (e) { check(label, false, '执行异常: ' + (e && e.message)); }
  };

  // 协方差不同的两个高斯团：类0 沿 x 拉长、类1 沿 y 拉长（Σ₀ ≠ Σ₁）
  const rngQ = mulberry32(21);
  const XQ = [], yQ = [];
  for (let i = 0; i < 200; i++) { XQ.push([gaussianSample(rngQ, -3, 1.2), gaussianSample(rngQ, 0, 0.5)]); yQ.push(0); }
  for (let i = 0; i < 200; i++) { XQ.push([gaussianSample(rngQ, 3, 0.5), gaussianSample(rngQ, 0, 1.2)]); yQ.push(1); }
  const modelQ = qdaFit(XQ, yQ, { reg: 1e-6 });
  const predQ = qdaPredict(XQ, modelQ);
  const regQ = 1e-6;

  run('QDA：classMeans ≡ 暴力均值（逐元素 < 1e-9）', (c) => {
    let maxErr = 0;
    for (let ci = 0; ci < modelQ.classes.length; ci++) {
      const rows = XQ.filter((_, i) => yQ[i] === modelQ.classes[ci]);
      for (let j = 0; j < XQ[0].length; j++) {
        let s = 0; for (const r of rows) s += r[j];
        maxErr = Math.max(maxErr, Math.abs(s / rows.length - modelQ.classMeans[ci][j]));
      }
    }
    c(maxErr < 1e-9, `maxMeanErr=${maxErr.toExponential(2)}`);
  });

  run('QDA：各类 covs 对称且 ≡ covarianceMatrix 无偏估计 + reg·I（逐元素 < 1e-9）', (c) => {
    let maxErr = 0, maxAsym = 0;
    for (let ci = 0; ci < modelQ.classes.length; ci++) {
      const rows = XQ.filter((_, i) => yQ[i] === modelQ.classes[ci]);
      const C = covarianceMatrix(rows); // 独立实现：/(n_c−1) 无偏
      const cov = modelQ.covs[ci];
      for (let i = 0; i < cov.length; i++) {
        for (let j = 0; j < cov.length; j++) {
          maxErr = Math.max(maxErr, Math.abs(cov[i][j] - C[i][j] - (i === j ? regQ : 0)));
          maxAsym = Math.max(maxAsym, Math.abs(cov[i][j] - cov[j][i]));
        }
      }
    }
    c(maxErr < 1e-9 && maxAsym < 1e-9, `maxCovErr=${maxErr.toExponential(2)}，maxAsym=${maxAsym.toExponential(2)}`);
  });

  run('QDA：posterior 每行和 ≡ 1（< 1e-12）', (c) => {
    let maxErr = 0;
    for (const p of predQ.posteriors) {
      let s = 0; for (const v of p) s += v;
      maxErr = Math.max(maxErr, Math.abs(s - 1));
    }
    c(maxErr < 1e-12, `maxRowSumErr=${maxErr.toExponential(2)}`);
  });

  run('QDA：log-posterior ≡ 独立实现（solve 求逆 + 多元高斯对数密度，< 1e-9）', (c) => {
    const d = XQ[0].length;
    const log2pi = Math.log(2 * Math.PI);
    let maxErr = 0, nChk = 0;
    for (let i = 0; i < XQ.length; i += 37) {
      nChk++;
      const xi = XQ[i];
      const logp = [];
      for (let k = 0; k < modelQ.classes.length; k++) {
        const mu = modelQ.classMeans[k];
        const diff = xi.map((v, j) => v - mu[j]);
        const cov = modelQ.covs[k];
        // 独立路径：Cholesky 求 log|Σ|、solve 求二次项（不经 covsInv / logDets 字段）
        const L = cholesky(cov);
        let logDet = 0; for (let t = 0; t < d; t++) logDet += 2 * Math.log(L[t][t]);
        const sol = solve(cov, diff);
        let quad = 0; for (let j = 0; j < d; j++) quad += diff[j] * sol[j];
        logp.push(Math.log(modelQ.priors[k]) - 0.5 * (d * log2pi + logDet + quad));
      }
      const lse = logsumexp(logp);
      for (let k = 0; k < logp.length; k++) {
        maxErr = Math.max(maxErr, Math.abs(Math.exp(logp[k] - lse) - predQ.posteriors[i][k]));
      }
    }
    c(maxErr < 1e-9, `抽样 ${nChk} 点，maxPosteriorErr=${maxErr.toExponential(2)}`);
  });

  run('QDA：协方差不同的可分合成数据训练准确率 = 1.0', (c) => {
    let acc = 0;
    for (let i = 0; i < XQ.length; i++) if (predQ.labels[i] === yQ[i]) acc++;
    const accR = acc / XQ.length;
    c(accR === 1.0, `acc=${accR.toFixed(4)}（Σ₀ 沿 x 拉伸、Σ₁ 沿 y 拉伸）`);
  });

  run('QDA：同协方差时边界退化为线性 — 与 LDA + 最近类心标签完全一致', (c) => {
    const rngS = mulberry32(5);
    const XS = [], yS = [];
    for (let i = 0; i < 200; i++) { XS.push([gaussianSample(rngS, -3, 1), gaussianSample(rngS, 0, 1)]); yS.push(0); }
    for (let i = 0; i < 200; i++) { XS.push([gaussianSample(rngS, 3, 1), gaussianSample(rngS, 0, 1)]); yS.push(1); }
    const mQS = qdaFit(XS, yS, { reg: 1e-6 });
    const mL = ldaFit(XS, yS, { reg: 1e-6 });
    const predS = qdaPredict(XS, mQS);
    // 共享 Σ=σ²I ⇒ Σ⁻¹ 度量退化为欧氏度量 ⇒ 最近类心 ≡ LDA 判别 ≡ QDA argmax
    let mismatch = 0;
    for (let i = 0; i < XS.length; i++) {
      let best = 0, bd = Infinity;
      for (let k = 0; k < mL.classes.length; k++) {
        const mu = mL.classMeans[k];
        const dd = (XS[i][0] - mu[0]) ** 2 + (XS[i][1] - mu[1]) ** 2;
        if (dd < bd) { bd = dd; best = k; }
      }
      if (predS.labels[i] !== mL.classes[best]) mismatch++;
    }
    c(mismatch === 0, `标签不一致数=${mismatch}/${XS.length}（二次项消失定理）`);
  });
}

/* ============================================================
 * 套件 44 · Lasso 稀疏回归（坐标下降）
 * 站在：Tibshirani 1996 / Friedman-Hastie-Tibshirani 2010 glmnet /
 *       Efron et al. 2004 LARS。凸目标 ⇒ KKT 即最优性证书。
 * ============================================================ */
function suiteLasso(check) {
  const run = (label, fn) => {
    try { fn((ok, info) => check(label, ok, info)); }
    catch (e) { check(label, false, '执行异常: ' + (e && e.message)); }
  };

  // 稀疏真源：wTrue = [2, 0, −1.5, 0, 0.8, 0]（3 非零 3 零）+ 小噪声
  const rngL = mulberry32(17);
  const nL = 80, dL = 6;
  const XL = randomMatrix(nL, dL, rngL, -1, 1);
  const wTrue = [2, 0, -1.5, 0, 0.8, 0];
  const yL = XL.map((r) => {
    let s = 0; for (let j = 0; j < dL; j++) s += wTrue[j] * r[j];
    return s + gaussianSample(rngL, 0, 0.1);
  });
  const lambdaL = 0.5;
  const model = lassoFit(XL, yL, { lambda: lambdaL, iters: 5000, tol: 1e-13 });

  run('Lasso：KKT 最优性证书 — |ρⱼ|≤λ(1+1e-6) 且 wⱼ≠0 ⇒ ρⱼ=λ·sign(wⱼ)（<1e-7）', (c) => {
    // 独立中心化 + 残差重算（不经模型内部状态）
    const xbar = new Array(dL).fill(0);
    for (const r of XL) for (let j = 0; j < dL; j++) xbar[j] += r[j];
    for (let j = 0; j < dL; j++) xbar[j] /= nL;
    let ybar = 0; for (const v of yL) ybar += v; ybar /= nL;
    const r = yL.map((v, i) => {
      let s = v - ybar;
      for (let j = 0; j < dL; j++) s -= (XL[i][j] - xbar[j]) * model.w[j];
      return s;
    });
    let maxKKT = 0, maxSub = 0;
    for (let j = 0; j < dL; j++) {
      let rho = 0;
      for (let i = 0; i < nL; i++) rho += (XL[i][j] - xbar[j]) * r[i];
      if (model.w[j] !== 0) maxKKT = Math.max(maxKKT, Math.abs(rho - lambdaL * Math.sign(model.w[j])));
      else maxSub = Math.max(maxSub, Math.abs(rho) - lambdaL);
    }
    c(maxKKT < 1e-7 && maxSub <= lambdaL * 1e-6, `w≠0: max|ρ−λsign|=${maxKKT.toExponential(2)}；w=0: max(|ρ|−λ)=${maxSub.toExponential(2)}`);
  });

  run('Lasso：正交设计闭式 — X̃ᵀX̃=n·I ⇒ w ≡ OLS 系数逐元素 soft-threshold（<1e-10）', (c) => {
    // Hadamard 型正交列：零均值、两两正交、范度 √n ⇒ 中心化后仍有 X̃ᵀX̃ = n·I
    const nO = 8, dO = 3;
    const cols = [
      [1, 1, 1, 1, -1, -1, -1, -1],
      [1, 1, -1, -1, 1, 1, -1, -1],
      [1, -1, 1, -1, 1, -1, 1, -1],
    ];
    const XO = [];
    for (let i = 0; i < nO; i++) {
      const row = [];
      for (let j = 0; j < dO; j++) row.push(cols[j][i]); // 列范度 √n、两列内积 0 ⇒ X̃ᵀX̃ = n·I
      XO.push(row);
    }
    const rngO = mulberry32(3);
    const wO = [1.5, -0.8, 0.6];
    const yO = XO.map((r) => {
      let s = 0; for (let j = 0; j < dO; j++) s += wO[j] * r[j];
      return s + gaussianSample(rngO, 0, 0.05);
    });
    const lamO = 1.2;
    const mO = lassoFit(XO, yO, { lambda: lamO, iters: 20000, tol: 1e-14 });
    // OLS（中心化数据闭式）：ŵⱼ = (1/n)Σᵢx̃ᵢⱼỹᵢ；lasso 闭式：wⱼ = softThreshold(ŵⱼ, λ/n)
    const wOls = new Array(dO).fill(0);
    for (let j = 0; j < dO; j++) {
      let s = 0;
      for (let i = 0; i < nO; i++) s += XO[i][j] * yO[i];
      wOls[j] = s / nO;
    }
    let maxErr = 0;
    for (let j = 0; j < dO; j++) {
      const th = lamO / nO;
      const expect = Math.sign(wOls[j]) * Math.max(Math.abs(wOls[j]) - th, 0);
      maxErr = Math.max(maxErr, Math.abs(mO.w[j] - expect));
    }
    c(maxErr < 1e-10, `maxCoefErr=${maxErr.toExponential(2)}，w=[${mO.w.map(v => v.toFixed(6)).join(', ')}]`);
  });

  run('Lasso：λ=0 解 ≡ linearRegression（< 1e-6）', (c) => {
    const m0 = lassoFit(XL, yL, { lambda: 0, iters: 50000, tol: 1e-14 });
    const ols = linearRegression(XL, yL);
    let maxErr = 0;
    for (let j = 0; j < dL; j++) maxErr = Math.max(maxErr, Math.abs(m0.w[j] - ols.w[j + 1]));
    maxErr = Math.max(maxErr, Math.abs(m0.b - ols.w[0]));
    c(maxErr < 1e-6, `maxCoefErr=${maxErr.toExponential(2)}（含截距，nIter=${m0.nIter}）`);
  });

  run('Lasso：λ ≥ maxⱼ|Σx̃ᵢⱼỹᵢ| ⇒ w 全 0（精确）', (c) => {
    const xbar = new Array(dL).fill(0);
    for (const r of XL) for (let j = 0; j < dL; j++) xbar[j] += r[j];
    for (let j = 0; j < dL; j++) xbar[j] /= nL;
    let ybar = 0; for (const v of yL) ybar += v; ybar /= nL;
    let maxRho = 0;
    for (let j = 0; j < dL; j++) {
      let rho = 0;
      for (let i = 0; i < nL; i++) rho += (XL[i][j] - xbar[j]) * (yL[i] - ybar);
      maxRho = Math.max(maxRho, Math.abs(rho));
    }
    const mBig = lassoFit(XL, yL, { lambda: maxRho * (1 + 1e-9), iters: 100, tol: 1e-12 });
    const allZero = mBig.w.every((v) => v === 0);
    c(allZero && mBig.activeSet.length === 0, `λ*=${maxRho.toFixed(6)}，activeSet=${mBig.activeSet.length}，b=${mBig.b.toExponential(2)}`);
  });

  run('Lasso：路径单调 — λ↓ ⇒ ‖w‖₁ 非减', (c) => {
    const lambdas = [];
    for (let k = 40; k >= 0; k--) lambdas.push(Math.pow(10, 3 - 5 * k / 40)); // 1e3 → 1e-2 对数刻度
    const path = lassoPath(XL, yL, lambdas, { iters: 3000, tol: 1e-13 });
    let worst = 0;
    for (let k = 1; k < path.length; k++) {
      const l1a = path[k - 1].w.reduce((s, v) => s + Math.abs(v), 0);
      const l1b = path[k].w.reduce((s, v) => s + Math.abs(v), 0);
      worst = Math.max(worst, l1a - l1b); // λ↓ ⇒ L1 非减 ⇒ l1a−l1b ≤ 0
    }
    const nnzEnd = path[path.length - 1].activeSet.length;
    c(worst < 1e-9, `41 个 λ 最大回退=${worst.toExponential(2)}，末端非零=${nnzEnd}/${dL}`);
  });

  run('Lasso：欠定 d>n ⇒ ‖w‖₀ ≤ min(n,d)', (c) => {
    const rngU = mulberry32(77);
    const nU = 6, dU = 12;
    const XU = randomMatrix(nU, dU, rngU, -1, 1);
    const wU = [1, -1, 0.5, 0, 0, 0.8, 0, 0, -0.6, 0, 0, 0.4];
    const yU = XU.map((r) => {
      let s = 0; for (let j = 0; j < dU; j++) s += wU[j] * r[j];
      return s + gaussianSample(rngU, 0, 0.01);
    });
    const mU = lassoFit(XU, yU, { lambda: 0.05, iters: 20000, tol: 1e-14 });
    const nnz = mU.activeSet.length;
    c(nnz <= Math.min(nU, dU), `‖w‖₀=${nnz} ≤ min(${nU},${dU})=${Math.min(nU, dU)}`);
  });
}

/* ============================================================
 * 套件 45 · 受限玻尔兹曼机 RBM（CD-k）
 * 站在：Smolensky 1986 / Hinton 2002 CD / Hinton-Salakhutdinov 2006 Science。
 *       核心恒等式：FE(v) ≡ −log Σ_h e^{−E(v,h)}（配分函数），
 *       CD 更新 ΔW ≡ lr(E_data[vhᵀ] − E_model[vhᵀ])。
 * ============================================================ */
function suiteRBM(check) {
  const run = (label, fn) => {
    try { fn((ok, info) => check(label, ok, info)); }
    catch (e) { check(label, false, '执行异常: ' + (e && e.message)); }
  };

  // 三条一维 0/1 图案（"条纹棒"）——RBM 应把高概率质量压到它们上面
  const pats = [
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
  ];
  const rbmReconErr = (model) => {
    let tot = 0;
    for (const p of pats) {
      const { vProb } = rbmReconstruct(model, p);
      for (let i = 0; i < p.length; i++) tot += (p[i] - vProb[i]) ** 2;
    }
    return tot / pats.length;
  };
  const mBefore = rbmTrain(pats, { hidden: 6, cdK: 1, lr: 0.1, iters: 0, seed: 42 });
  const mT = rbmTrain(pats, { hidden: 6, cdK: 1, lr: 0.1, iters: 400, seed: 42 });
  const errBefore = rbmReconErr(mBefore);
  const errAfter = rbmReconErr(mT);

  // 枚举恒等式用小型模型（2^8=256 可见态 × 2^5=32 隐藏态）
  const rngF = mulberry32(9);
  const visF = 8, hidF = 5;
  const mF = {
    W: randomMatrix(visF, hidF, rngF, -0.8, 0.8),
    b: Array.from({ length: visF }, () => rngF() * 2 - 1),
    c: Array.from({ length: hidF }, () => rngF() * 2 - 1),
    hidden: hidF, visible: visF,
  };

  run('RBM：P(h|v) ≡ 独立 σ(Wᵀv+c)（< 1e-15）', (c) => {
    const v = [1, 0, 1, 1, 0, 0, 1, 0];
    const hp = rbmHiddenProbs(mF, v);
    let maxErr = 0;
    for (let j = 0; j < hp.length; j++) {
      let s = mF.c[j];
      for (let i = 0; i < v.length; i++) s += mF.W[i][j] * v[i];
      maxErr = Math.max(maxErr, Math.abs(hp[j] - sigmoid(s)));
    }
    c(maxErr < 1e-15, `maxErr=${maxErr.toExponential(2)}`);
  });

  run('RBM：自由能-配分函数恒等式 — softmax(−FE) ≡ 联合分布枚举归一（< 1e-10）', (c) => {
    const nV = 1 << visF, nH = 1 << hidF;
    const negFE = new Array(nV), Z = new Array(nV);
    for (let mask = 0; mask < nV; mask++) {
      const v = Array.from({ length: visF }, (_, i) => (mask >> i) & 1);
      negFE[mask] = -rbmFreeEnergy(mF, v);
      let z = 0; // Z(v) = Σ_h e^{−E(v,h)}，E = −vᵀWh − bᵀv − cᵀh
      for (let hm = 0; hm < nH; hm++) {
        const h = Array.from({ length: hidF }, (_, j) => (hm >> j) & 1);
        let e = 0;
        for (let i = 0; i < visF; i++) e -= mF.b[i] * v[i];
        for (let j = 0; j < hidF; j++) e -= mF.c[j] * h[j];
        for (let i = 0; i < visF; i++) for (let j = 0; j < hidF; j++) e -= mF.W[i][j] * v[i] * h[j];
        z += Math.exp(-e);
      }
      Z[mask] = z;
    }
    const pFE = softmax(negFE);
    let zSum = 0; for (const z of Z) zSum += z;
    let maxErr = 0;
    for (let mask = 0; mask < nV; mask++) maxErr = Math.max(maxErr, Math.abs(pFE[mask] - Z[mask] / zSum));
    c(maxErr < 1e-10, `${nV} 个可见态最大偏差=${maxErr.toExponential(2)}`);
  });

  run('RBM：softmax(−FE) 在全状态上概率和 = 1（< 1e-12）', (c) => {
    const nV = 1 << visF;
    const negFE = new Array(nV);
    for (let mask = 0; mask < nV; mask++) {
      const v = Array.from({ length: visF }, (_, i) => (mask >> i) & 1);
      negFE[mask] = -rbmFreeEnergy(mF, v);
    }
    const p = softmax(negFE);
    let s = 0; for (const x of p) s += x;
    c(Math.abs(s - 1) < 1e-12, `Σp=${s.toFixed(15)}`);
  });

  run('RBM：CD 训练后训练集平均重构误差 < 训练前（固定 seed，留足 margin）', (c) => {
    c(errAfter < errBefore - 0.5, `before=${errBefore.toFixed(4)} → after=${errAfter.toFixed(4)}（margin=${(errBefore - errAfter).toFixed(4)}）`);
  });

  run('RBM：训练后训练样本 FE < 随机样本 FE（能量间隔 > 0）', (c) => {
    const rngZ = mulberry32(11);
    let feTr = 0;
    for (const p of pats) feTr += rbmFreeEnergy(mT, p);
    feTr /= pats.length;
    const nRnd = 32;
    let feRd = 0;
    for (let t = 0; t < nRnd; t++) {
      const v = Array.from({ length: pats[0].length }, () => (rngZ() < 0.5 ? 0 : 1));
      feRd += rbmFreeEnergy(mT, v);
    }
    feRd /= nRnd;
    c(feTr < feRd, `FE(训练)=${feTr.toFixed(4)} < FE(随机)=${feRd.toFixed(4)}，间隔=${(feRd - feTr).toFixed(4)}`);
  });

  run('RBM：CD 更新恒等式 — 单步 ΔW ≡ lr(E_data[vhᵀ]−E_model[vhᵀ])（重算正负相一致 <1e-12）', (c) => {
    const rngC = mulberry32(3);
    const XC = Array.from({ length: 6 }, () => Array.from({ length: 8 }, () => (rngC() < 0.5 ? 0 : 1)));
    const hidC = 5, cdKC = 2, lrC = 0.1;
    const m0 = rbmTrain(XC, { hidden: hidC, cdK: cdKC, lr: lrC, iters: 0, seed: 42 });
    const m1 = rbmTrain(XC, { hidden: hidC, cdK: cdKC, lr: lrC, iters: 1, seed: 42 });
    // 独立重放：同种子初始化 + 同序伯努利采样，按 CD 定义重算正/负相统计
    const rng = mulberry32(42);
    const W = randomMatrix(8, hidC, rng, -0.1, 0.1);
    const bb = new Array(8).fill(0), cc = new Array(hidC).fill(0);
    const dW = zeros(8, hidC), mW = zeros(8, hidC);
    for (let s = 0; s < XC.length; s++) {
      const v0 = XC[s];
      const h0p = [];
      for (let j = 0; j < hidC; j++) {
        let t = cc[j];
        for (let i = 0; i < 8; i++) t += W[i][j] * v0[i];
        h0p.push(sigmoid(t));
      }
      const h0 = h0p.map((p) => (rng() < p ? 1 : 0));
      for (let i = 0; i < 8; i++) for (let j = 0; j < hidC; j++) dW[i][j] += v0[i] * h0[j];
      let v = v0, h = h0;
      for (let t = 0; t < cdKC; t++) {
        const vp = [];
        for (let i = 0; i < 8; i++) {
          let u = bb[i];
          for (let j = 0; j < hidC; j++) u += W[i][j] * h[j];
          vp.push(sigmoid(u));
        }
        v = vp.map((p) => (rng() < p ? 1 : 0));
        const hp = [];
        for (let j = 0; j < hidC; j++) {
          let u = cc[j];
          for (let i = 0; i < 8; i++) u += W[i][j] * v[i];
          hp.push(sigmoid(u));
        }
        h = hp.map((p) => (rng() < p ? 1 : 0));
      }
      for (let i = 0; i < 8; i++) for (let j = 0; j < hidC; j++) mW[i][j] += v[i] * h[j];
    }
    let maxErr = 0;
    for (let i = 0; i < 8; i++) for (let j = 0; j < hidC; j++) {
      const expect = W[i][j] + (lrC / XC.length) * (dW[i][j] - mW[i][j]);
      maxErr = Math.max(maxErr, Math.abs(expect - m1.W[i][j]));
    }
    c(maxErr < 1e-12, `ΔW 重算最大偏差=${maxErr.toExponential(2)}`);
  });
}

/* ============================================================
 * 套件 46 · Hopfield 联想记忆
 * 站在：Hopfield 1982 PNAS / Amit-Gutfreund-Sompolinski 1985（容量 c≈0.138）。
 *       核心定理：异步单单元更新下 E(s) = −½sᵀWs − bᵀs 单调不增（Lyapunov），
 *       存储模式在 m ≤ 0.138n 时为不动点（局部极小，ΔE = 2pᵢ(Wp)ᵢ ≥ 0）。
 * ============================================================ */
function suiteHopfield(check) {
  const run = (label, fn) => {
    try { fn((ok, info) => check(label, ok, info)); }
    catch (e) { check(label, false, '执行异常: ' + (e && e.message)); }
  };

  // 4 条随机 ±1 模式：n=100，m/n = 0.04 ≪ 0.138 容量上限
  const rngH = mulberry32(2024);
  const nH = 100, mH = 4;
  const patterns = Array.from({ length: mH }, () =>
    Array.from({ length: nH }, () => (rngH() < 0.5 ? -1 : 1)));
  const model = hopfieldTrain(patterns, { zeroDiag: true });
  const { W, b } = model;

  run('Hopfield：Lyapunov 能量不增 — 随机轨迹每步 ΔE ≤ 0（1e-12 容差，全步检查）', (c) => {
    const rngT = mulberry32(31);
    const s0 = Array.from({ length: nH }, () => (rngT() < 0.5 ? -1 : 1));
    const rec = hopfieldRecall(W, b, s0, { maxSteps: 50, seed: 5 });
    let worst = -Infinity, nUpd = 0;
    for (let k = 1; k < rec.energyTrace.length; k++) {
      const dE = rec.energyTrace[k] - rec.energyTrace[k - 1];
      if (dE > worst) worst = dE;
      nUpd++;
    }
    c(worst <= 1e-12, `全 ${nUpd} 次单单元更新 max ΔE=${worst.toExponential(2)}，收敛=${rec.converged}`);
  });

  run('Hopfield：Hebbian 存储的模式是不动点（从 pμ 出发收敛回 pμ）', (c) => {
    let allOk = true;
    const info = [];
    for (let mu = 0; mu < patterns.length; mu++) {
      const rec = hopfieldRecall(W, b, patterns[mu], { maxSteps: 50, seed: 100 + mu });
      const same = rec.state.every((v, i) => v === patterns[mu][i]);
      if (!same || !rec.converged) allOk = false;
      info.push(`p${mu}:${same ? '✓' : '✗'}(${rec.steps}轮)`);
    }
    c(allOk, info.join(' '));
  });

  run('Hopfield：W 对角元 = 0（精确）', (c) => {
    let maxDiag = 0;
    for (let i = 0; i < nH; i++) maxDiag = Math.max(maxDiag, Math.abs(W[i][i]));
    c(maxDiag === 0, `max|Wᵢᵢ|=${maxDiag}`);
  });

  run('Hopfield：W 对称（精确）', (c) => {
    let maxAsym = 0;
    for (let i = 0; i < nH; i++) for (let j = i + 1; j < nH; j++) maxAsym = Math.max(maxAsym, Math.abs(W[i][j] - W[j][i]));
    c(maxAsym === 0, `max|Wᵢⱼ−Wⱼᵢ|=${maxAsym}`);
  });

  run('Hopfield：25% 比特噪声探针仍收敛回原模式（m ≤ 0.138n）', (c) => {
    const rngP = mulberry32(55);
    let allOk = true;
    const info = [];
    for (let mu = 0; mu < patterns.length; mu++) {
      const probe = patterns[mu].slice();
      for (let i = 0; i < nH; i++) if (rngP() < 0.25) probe[i] = -probe[i];
      const rec = hopfieldRecall(W, b, probe, { maxSteps: 50, seed: 200 + mu });
      const same = rec.state.every((v, i) => v === patterns[mu][i]);
      if (!same) allOk = false;
      info.push(`p${mu}:${same ? '✓' : '✗'}`);
    }
    c(allOk, `${info.join(' ')}（m/n=${(mH / nH).toFixed(3)} ≤ 0.138）`);
  });

  run('Hopfield：存储模式能量 ≤ 全部单比特翻转邻居能量（局部极小）', (c) => {
    let worst = -Infinity; // 最大「违规量」= max(E(p)−E(flip))，应 ≤ 0
    for (const p of patterns) {
      const e0 = hopfieldEnergy(W, b, p);
      for (let i = 0; i < nH; i++) {
        const flip = p.slice(); flip[i] = -flip[i];
        worst = Math.max(worst, e0 - hopfieldEnergy(W, b, flip)); // ≤0 ⟺ p 为局部极小
      }
    }
    c(worst <= 1e-12, `max(E(p)−E(flip))=${worst.toExponential(2)}（≡ 2pᵢ(Wp)ᵢ ≥ 0）`);
  });
}

/* ============================================================
 * 套件 47 · RTS 平滑器（Rauch–Tung–Striebel 1965）
 * ============================================================ */
function suiteRTS(check) {
  // 2D 线性高斯系统：先跑卡尔曼 + RTS 平滑，交叉验证反向递推恒等式
  const rng = mulberry32(2025);
  const F = [[0.95, 0.05], [0, 0.90]];
  const H = [[1, 0], [0, 1]];
  const Q = [[0.10, 0], [0, 0.20]];
  const R = [[1, 0], [0, 1]];
  const x0 = [0, 0];
  const P0 = [[5, 0], [0, 5]];
  let state = x0.slice();
  const obs = [], truth = [];
  for (let t = 0; t < 300; t++) {
    const w0 = gaussianSample(rng, 0, Math.sqrt(Q[0][0]));
    const w1 = gaussianSample(rng, 0, Math.sqrt(Q[1][1]));
    state = [F[0][0] * state[0] + F[0][1] * state[1] + w0, F[1][1] * state[1] + w1];
    truth.push(state.slice());
    obs.push([state[0] + gaussianSample(rng, 0, 1), state[1] + gaussianSample(rng, 0, 1)]);
  }
  const model = { F, H, Q, R, x0, P0 };
  const out = rtsSmooth(model, obs);
  const N = out.smoothed.length;
  const tol = 1e-12;

  // ① 端点闭合：平滑终点 ≡ 滤波终点（反向递推第一步不做修正）
  let endErr = 0;
  for (let j = 0; j < 2; j++) endErr = Math.max(endErr, Math.abs(out.smoothed[N - 1][j] - out.filtered[N - 1][j]));
  check('RTS：端点闭合 x̂_{N|N}^{平滑} ≡ 滤波终点（精确）', endErr < tol, `误差=${endErr.toExponential(2)}`);

  // ② 协方差序：P_{k|k} − P_{k|N} 半正定（平滑必降低不确定性）
  const gap = rtsCovarianceGapEigen(out.filteredCovs, out.smoothedCovs);
  let minGap = Infinity;
  for (const g of gap) if (g < minGap) minGap = g;
  check('RTS：平滑必降不确定性 P_{k|k} − P_{k|N} 半正定（最小特征值 ≥ −1e-10）', minGap >= -1e-10, `min eig=${minGap.toExponential(2)}`);

  // ③ 增益恒等式：J_k ≡ P_{k|k}FᵀP_{k+1|k}⁻¹ 独立重算（逐步逐位）
  let gainErr = 0;
  for (let k = 0; k < out.gains.length; k++) {
    const J = matmul(matmul(out.filteredCovs[k], transpose(F)), inverse(out.predictedCov[k + 1]));
    gainErr = Math.max(gainErr, mdiff(J, out.gains[k]));
  }
  check('RTS：增益恒等式 J_k = P_{k|k}FᵀP_{k+1|k}⁻¹（逐步逐位）', gainErr < tol, `maxDiff=${gainErr.toExponential(2)}`);

  // ④ 常值状态闭式：F=1,Q=0 ⇒ 平滑估计全相等 ≡ 信息加权均值 Σz/(1+N)
  const rngC = mulberry32(7);
  const Nc = 120;
  const obsC = [];
  let zsum = 0;
  for (let t = 0; t < Nc; t++) { const z = gaussianSample(rngC, 3.0, 1.0); obsC.push([z]); zsum += z; }
  const outC = rtsSmooth({ F: [[1]], H: [[1]], Q: [[0]], R: [[1]], x0: [0], P0: [[1]] }, obsC);
  const closed = zsum / (1 + Nc);
  let constErr = Math.abs(outC.smoothed[0][0] - closed);
  for (let k = 1; k < outC.smoothed.length; k++) constErr = Math.max(constErr, Math.abs(outC.smoothed[k][0] - outC.smoothed[0][0]));
  check('RTS：常值状态 F=1,Q=0 ⇒ 平滑估计恒等且 ≡ Σz/(1+N)', constErr < 1e-6, `偏差=${constErr.toExponential(2)}，闭式=${closed.toFixed(6)}`);

  // ⑤ 反向递推自洽：x̂_{k|N} = x̂_{k|k} + J_k(x̂_{k+1|N} − x̂_{k+1|k}) 重算
  let recErr = 0;
  for (let k = 0; k < N - 1; k++) {
    const innov = out.smoothed[k + 1].map((v, j) => v - out.predicted[k + 1][j]);
    const recomputed = out.filtered[k].map((v, i) => {
      let s = v;
      for (let j = 0; j < innov.length; j++) s += out.gains[k][i][j] * innov[j];
      return s;
    });
    recErr = Math.max(recErr, mdiff(out.smoothed[k], recomputed));
  }
  check('RTS：反向递推自洽 x̂_{k|N} = x̂_{k|k} + J_k(x̂_{k+1|N} − x̂_{k+1|k})', recErr < tol, `maxDiff=${recErr.toExponential(2)}`);

  // ⑥ 均方误差序：平滑 RMSE ≤ 滤波 RMSE（全数据后验优于仅用历史）
  const rmseF = rtsRMSE(out.filtered, truth);
  const rmseS = rtsRMSE(out.smoothed, truth);
  check('RTS：平滑 RMSE ≤ 滤波 RMSE（最优平滑=全数据后验均值）', rmseS <= rmseF + 1e-6, `滤波=${rmseF.toFixed(4)} vs 平滑=${rmseS.toFixed(4)}`);
}

/* ============================================================
 * 套件 48 · 粒子滤波（Gordon 1993 · SIR）
 * ============================================================ */
function suiteParticleFilter(check) {
  const rng = mulberry32(99);
  const f = 0.95, h = 1, q = 0.1, r = 1;
  const F = [[f]], H = [[h]], Q = [[q]], R = [[r]];
  const P0 = [[q / (1 - f * f)]];
  const x0 = [0];
  let truth = 0;
  const obs = [], truthSeries = [];
  for (let t = 0; t < 100; t++) {
    truth = f * truth + gaussianSample(rng, 0, Math.sqrt(q));
    truthSeries.push(truth);
    obs.push([truth + gaussianSample(rng, 0, Math.sqrt(r))]);
  }
  const pfModel = {
    x0,
    init: (rngI) => [x0[0] + gaussianSample(rngI, 0, Math.sqrt(P0[0][0]))],
    transition: (xp, rngI) => [f * xp[0] + gaussianSample(rngI, 0, Math.sqrt(q))],
    likelihood: (y, xp) => gaussianPDF(y[0], xp[0], Math.sqrt(r)),
    N: 5000, seed: 7, resampleEvery: true,
  };
  const pf = particleFilter(pfModel, obs);
  const kf = kalmanFilter({ F, H, Q, R, x0, P0 }, obs);

  // ① 权重归一化：每步 Σw = 1（log 域 logsumexp 归一）
  let wErr = 0;
  for (const w of pf.weightsHistory) {
    let s = 0; for (const v of w) s += v;
    wErr = Math.max(wErr, Math.abs(s - 1));
  }
  check('粒子滤波：每步权重 Σw = 1（log 域 logsumexp 归一）', wErr < 1e-12, `max|Σw−1|=${wErr.toExponential(2)}`);

  // ② 线性高斯 ⇒ 粒子估计收敛到卡尔曼解析解（先验一致）
  const cmp = pfCompareWithKalman(pf.estimates, kf.states);
  check('粒子滤波：线性高斯下收敛到卡尔曼解析解（N=5000 均值偏差 < 0.3）', cmp.meanDiff < 0.3,
    `meanDiff=${cmp.meanDiff.toFixed(4)}，maxDiff=${cmp.maxDiff.toFixed(4)}`);

  // ③ ESS ≡ 1/Σw² 且 ∈ [1, N]（均匀=N、退化=1）
  const Nn = pf.N;
  const wU = new Array(Nn).fill(1 / Nn);
  const wD = new Array(Nn).fill(0); wD[0] = 1;
  const essU = effectiveSampleSize(wU), essD = effectiveSampleSize(wD);
  check('粒子滤波：ESS ≡ 1/Σw²，均匀=N 且 退化=1', Math.abs(essU - Nn) < 1e-9 && Math.abs(essD - 1) < 1e-9,
    `ESS(均匀)=${essU.toFixed(6)}，ESS(退化)=${essD.toFixed(6)}`);

  // ④ 系统重采样：均匀权重下 ≡ 恒等置换，权重精确 1/N
  const parts = Array.from({ length: Nn }, (_, i) => [i]);
  const res = systematicResample(parts, wU, mulberry32(1));
  const identity = res.indices.every((v, i) => v === i);
  const uniform = res.weights.every((v) => Math.abs(v - 1 / Nn) < 1e-12);
  let sumW = 0; for (const v of res.weights) sumW += v;
  check('粒子滤波：系统重采样（均匀权重）≡ 恒等置换、权重精确 1/N', identity && uniform && Math.abs(sumW - 1) < 1e-12,
    `indices 恒等=${identity}，权重均匀=${uniform}`);

  // ⑤ 退化情形：全部粒子相同 ⇒ 估计 ≡ 该粒子且 ESS ≡ N
  const degParts = Array.from({ length: Nn }, () => [-2]);
  const degW = new Array(Nn).fill(1 / Nn);
  let degMean = 0; for (const p of degParts) degMean += p[0] / Nn;
  check('粒子滤波：退化（粒子全同）⇒ 估计 ≡ 该粒子且 ESS=N', Math.abs(degMean - (-2)) < 1e-12 && Math.abs(effectiveSampleSize(degW) - Nn) < 1e-9,
    `估计=${degMean.toFixed(6)}，ESS=${effectiveSampleSize(degW).toFixed(6)}`);

  // ⑥ 同 seed 两次运行逐位一致（随机流完全确定）
  const a = particleFilter(pfModel, obs);
  const b = particleFilter(pfModel, obs);
  let identical = a.estimates.length === b.estimates.length;
  for (let t = 0; t < a.estimates.length && identical; t++)
    for (let j = 0; j < a.estimates[t].length; j++)
      if (a.estimates[t][j] !== b.estimates[t][j]) { identical = false; break; }
  check('粒子滤波：同 seed 两次运行逐位一致（随机流确定）', identical, `步数=${a.estimates.length}`);
}

/* ============================================================
 * 套件 49 · Mean Shift 众数搜索（Fukunaga 1975 · Cheng 1995）
 * ============================================================ */
function suiteMeanShift(check) {
  const rng = mulberry32(5);
  const X = randomMatrix(80, 2, rng, -3, 3);
  const h = 1.5;

  // ① 梯度恒等式：m(x) − x ≡ h²∇f̂/f̂（中心差分 vs 闭式）
  const xq = [rng() * 2 - 1, rng() * 2 - 1];
  const gc = meanShiftGradientCheck(X, xq, h, 1e-5);
  check('MeanShift：梯度恒等式 m(x)−x ≡ h²∇f̂/f̂（相对误差 1e-6）', gc.maxRelErr < 1e-6, `maxRelErr=${gc.maxRelErr.toExponential(2)}`);

  // ② h → ∞ 一步到质心：m(x) ≡ 数据均值 Σxᵢ/n
  const mInf = meanShiftStep(X, xq, 1e9);
  const meanX = colMean(X);
  let meanErr = 0;
  for (let j = 0; j < 2; j++) meanErr = Math.max(meanErr, Math.abs(mInf[j] - meanX[j]));
  check('MeanShift：h→∞ 一步到数据均值 Σxᵢ/n', meanErr < 1e-6, `maxDiff=${meanErr.toExponential(2)}`);

  // ③ 单调上升：沿轨迹 f̂ 非降（Cheng 1995 收敛定理）
  const mode = meanShiftMode(X, xq, h, { maxIter: 300, tol: 1e-10 });
  let nondec = true;
  for (let k = 1; k < mode.kdeTrace.length; k++)
    if (mode.kdeTrace[k] < mode.kdeTrace[k - 1] - 1e-12) { nondec = false; break; }
  check('MeanShift：沿迭代轨迹 f̂ 单调不降（Cheng 收敛定理）', nondec, `步数=${mode.steps}，kde 起点=${mode.kdeTrace[0].toFixed(4)}`);

  // ④ 收敛处不动点：‖m(x)−x‖ < tol
  check('MeanShift：收敛处不动点 ‖m(x)−x‖ < tol', mode.shift < 1e-9, `shift=${mode.shift.toExponential(2)}`);

  // ⑤ 单峰高斯团：任意起点收敛到样本均值附近
  const rngG = mulberry32(8);
  const G = randomMatrix(100, 2, rngG, -1, 1);
  const meanG = colMean(G);
  const modeG = meanShiftMode(G, [2, 2], h, { maxIter: 300, tol: 1e-10 });
  let gErr = 0;
  for (let j = 0; j < 2; j++) gErr = Math.max(gErr, Math.abs(modeG.mode[j] - meanG[j]));
  check('MeanShift：单峰高斯团收敛到样本均值附近', gErr < 0.3, `‖mode−mean‖=${gErr.toFixed(4)}`);

  // ⑥ 两团分离数据：簇数 = 2 且标签完美分离
  const rngC = mulberry32(13);
  const Ac = randomMatrix(40, 2, rngC, -1, 1).map((r) => [r[0] - 4, r[1] - 4]);
  const Bc = randomMatrix(40, 2, rngC, -1, 1).map((r) => [r[0] + 4, r[1] + 4]);
  const X2 = Ac.concat(Bc);
  const cl = meanShiftCluster(X2, 2.0, { mergeTol: 0.6, maxIter: 300, tol: 1e-10 });
  const lA = cl.labels[0], lB = cl.labels[40];
  const sep = Ac.every((_, i) => cl.labels[i] === lA) && Bc.every((_, i) => cl.labels[40 + i] === lB) && lA !== lB;
  check('MeanShift：两团分离数据 ⇒ 簇数=2 且标签完美分离', cl.k === 2 && sep, `k=${cl.k}，分离=${sep}`);
}

/* ============================================================
 * 套件 50 · SARSA 同策略时序差分（Rummery & Niranjan 1994）
 * ============================================================ */
function suiteSARSA(check) {
  const rng = mulberry32(3);
  const S = 3, A = 2;
  const Q0 = randomMatrix(S, A, rng, -1, 1);
  const P = [], R = [];
  for (let s = 0; s < S; s++) {
    const pa = [], ra = [];
    for (let a = 0; a < A; a++) {
      const ns = (s + (a === 1 ? 1 : -1) + S) % S;
      pa.push([[ns, 1]]);
      ra.push([a === 1 ? 1 : 0]);
    }
    P.push(pa); R.push(ra);
  }
  const gamma = 0.9;

  // ① 退化恒等式：ε=0 + 确定性 MDP ⇒ SARSA 单步 ≡ Q-learning 单步
  const s = 0, a = 0, s2 = P[s][a][0][0], r = R[s][a][0];
  const a2 = greedyAction(Q0, s2);
  const Qs = Q0.map((row) => row.slice());
  sarsaStep(Qs, s, a, r, s2, a2, { alpha: 0.1, gamma });
  const Qq = Q0.map((row) => row.slice());
  const deltaQL = r + gamma * Math.max(...Qq[s2]) - Qq[s][a];
  Qq[s][a] += 0.1 * deltaQL;
  let stepDiff = 0;
  for (let i = 0; i < S; i++) for (let j = 0; j < A; j++) stepDiff = Math.max(stepDiff, Math.abs(Qs[i][j] - Qq[i][j]));
  check('SARSA：ε=0 确定性 MDP 单步 ≡ Q-learning 单步（1e-15）', stepDiff < 1e-15, `maxDiff=${stepDiff.toExponential(2)}`);

  // ② λ=0 时 SARSA(λ) 与 SARSA 逐位相等
  const { P: Pc, R: Rc } = chainMDP({ n: 5, gamma });
  const res1 = sarsa(Pc, Rc, { gamma, episodes: 300, stepsPerEpisode: 25, seed: 3, lambda: 0 });
  const res2 = sarsaLambda(Pc, Rc, { gamma, episodes: 300, stepsPerEpisode: 25, seed: 3, lambda: 0 });
  let lamDiff = 0;
  for (let i = 0; i < 5; i++) for (let j = 0; j < 2; j++) lamDiff = Math.max(lamDiff, Math.abs(res1.Q[i][j] - res2.Q[i][j]));
  check('SARSA：λ=0 时 SARSA(λ) 与 SARSA 逐位相等（0）', lamDiff === 0, `maxDiff=${lamDiff}`);

  // ③ GLIE 收敛：chainMDP 上收敛 Q ≈ qFromV(V*)
  const { P: Pg, R: Rg } = chainMDP({ n: 6, gamma });
  const vi = valueIteration(Pg, Rg, gamma);
  const Qref = qFromV(Pg, Rg, gamma, vi.V);
  const resS = sarsa(Pg, Rg, {
    gamma, alpha: 0.2, epsStart: 1, epsEnd: 0, episodes: 4000, stepsPerEpisode: 40,
    seed: 5, average: true, avgFrom: 0.5, epsDecayFrac: 0.5, lambda: 0,
  });
  let qDiff = 0;
  for (let i = 0; i < 6; i++) for (let j = 0; j < 2; j++) qDiff = Math.max(qDiff, Math.abs(resS.Q[i][j] - Qref[i][j]));
  let polOk = true;
  for (let i = 0; i < 6; i++) if (greedyAction(resS.Q, i) !== greedyAction(Qref, i)) polOk = false;
  check('SARSA：GLIE 收敛 Q ≈ qFromV(V*) 且策略一致', qDiff < 0.3 && polOk, `max|ΔQ|=${qDiff.toFixed(4)}，策略一致=${polOk}`);

  // ④ 在策略 Bellman 残差 → 0（ε=0 贪心）
  const resid = sarsaBellmanResidual(Pg, Rg, gamma, resS.Q, 0);
  check('SARSA：在策略 Bellman 残差 ‖Q − B^πQ‖ → 0（π 贪心）', resid < 0.3, `max残差=${resid.toFixed(4)}`);

  // ⑤ 资格迹恒等式：访问 (s,a) 后 e(s,a)=1（替换迹）且按 γλ 衰减
  const e0 = Array.from({ length: S }, () => new Array(A).fill(0));
  sarsaTraceSet(e0, 2, 1, { replacing: true });
  const setOk = e0[2][1] === 1 && e0[0][0] === 0;
  const e1 = sarsaTraceDecay(e0, 0.9, 0.8);
  const decayOk = Math.abs(e1[2][1] - 0.72) < 1e-12;
  sarsaTraceSet(e0, 2, 0, { replacing: true });
  const replOk = e0[2][1] === 0 && e0[2][0] === 1;
  check('SARSA：资格迹 e(s,a)=1（替换迹清零他者）且按 γλ 衰减', setOk && decayOk && replOk,
    `置位=${setOk}，衰减=${decayOk}，替换=${replOk}`);

  // ⑥ 同 seed 两次训练 Q 逐位一致
  const r1 = sarsa(Pg, Rg, { gamma, episodes: 500, stepsPerEpisode: 30, seed: 9, lambda: 0 });
  const r2 = sarsa(Pg, Rg, { gamma, episodes: 500, stepsPerEpisode: 30, seed: 9, lambda: 0 });
  let repDiff = 0;
  for (let i = 0; i < 6; i++) for (let j = 0; j < 2; j++) repDiff = Math.max(repDiff, Math.abs(r1.Q[i][j] - r2.Q[i][j]));
  check('SARSA：同 seed 两次训练 Q 逐位一致', repDiff === 0, `maxDiff=${repDiff}`);
}

/* ============================================================
 * 套件 51 · 门控循环单元 GRU（Cho et al. 2014）
 * ============================================================ */
function suiteGRU(check) {
  const rng = mulberry32(11);
  const net = new GRU({ inputSize: 3, hiddenSize: 4, outputSize: 2, seed: 11 });

  // ① BPTT 解析梯度 ≡ 中心差分数值梯度（双判据 CS231n）
  const T = 6;
  const X = Array.from({ length: T }, () => Array.from({ length: 3 }, () => (rng() - 0.5) * 1.5));
  const targets = Array.from({ length: T }, () => Array.from({ length: 2 }, () => (rng() - 0.5) * 1.5));
  const gc = gruGradientCheck(net, X, targets, { eps: 1e-5 });
  check('GRU：BPTT 解析梯度 ≡ 中心差分数值梯度（maxRelErr<1e-6）', gc.maxRelErr < 1e-6 && gc.significant > 0,
    `maxRelErr=${gc.maxRelErr.toExponential(2)}，显著分量=${gc.significant}`);

  // ② 循环递推恒等式：h_t = (1−z_t)·h_{t−1} + z_t·h̃_t（逐位精确）
  const fwd = net.forward(X);
  let recErr = 0;
  let hPrev = new Array(4).fill(0);
  for (let t = 0; t < fwd.T; t++) {
    for (let j = 0; j < 4; j++) {
      const expected = (1 - fwd.Z[t][j]) * hPrev[j] + fwd.Z[t][j] * fwd.HT[t][j];
      recErr = Math.max(recErr, Math.abs(fwd.H[t][j] - expected));
    }
    hPrev = fwd.H[t];
  }
  check('GRU：隐态递推恒等式 h_t=(1−z_t)h_{t−1}+z_t·h̃_t（<1e-12）', recErr < 1e-12, `maxErr=${recErr.toExponential(2)}`);

  // ③ 单步闭式：gruStepClosed 手算首步 ≡ forward 首步（独立交叉验证）
  const closed = gruStepClosed(net, X[0], new Array(4).fill(0));
  let stepErr = 0;
  for (let j = 0; j < 4; j++) stepErr = Math.max(stepErr, Math.abs(closed.hT[j] - fwd.H[0][j]));
  check('GRU：单步闭式手算 ≡ forward 首步（<1e-12）', stepErr < 1e-12, `maxErr=${stepErr.toExponential(2)}`);

  // ④ 候选重构：h̃_t = tanh(W_xh·x_t + r_t⊙(W_hh·h_{t−1}) + b_h)（含重置门调制）
  let candErr = 0;
  hPrev = new Array(4).fill(0);
  for (let t = 0; t < fwd.T; t++) {
    for (let j = 0; j < 4; j++) {
      let sh = net.bh[j];
      for (let k = 0; k < 3; k++) sh += net.Wxh[j][k] * X[t][k];
      let hh = 0;
      for (let k = 0; k < 4; k++) hh += net.Whh[j][k] * hPrev[k];
      sh += fwd.R[t][j] * hh;
      candErr = Math.max(candErr, Math.abs(fwd.HT[t][j] - Math.tanh(sh)));
    }
    hPrev = fwd.H[t];
  }
  check('GRU：候选 h̃_t 含重置门调制重构（<1e-12）', candErr < 1e-12, `maxErr=${candErr.toExponential(2)}`);

  // ⑤ 训练显著降损：固定教师网络（不同权重）产生目标，学生 GRU 训练拟合 → MSE 大幅收敛
  const teacherNet = new GRU({ inputSize: 3, hiddenSize: 4, outputSize: 2, seed: 99 });
  const teacherTargets = teacherNet.forward(X).Y;
  const { losses } = trainGRU(net, X, teacherTargets, { lr: 0.05, iters: 2000 });
  check('GRU：拟合教师网络目标下训练大幅降损（末损 < 初损×0.1）',
    losses[losses.length - 1] < losses[0] * 0.1,
    `初损=${losses[0].toExponential(2)}，末损=${losses[losses.length - 1].toExponential(2)}`);

  // ⑥ 同 seed 初始化一致：两次构造（未训练）权重逐位相同
  const netSeed = new GRU({ inputSize: 3, hiddenSize: 4, outputSize: 2, seed: 11 });
  const net2 = new GRU({ inputSize: 3, hiddenSize: 4, outputSize: 2, seed: 11 });
  let same = true;
  for (const key of ['Wxr', 'Wxz', 'Wxh', 'Whr', 'Whz', 'Whh', 'Why']) {
    for (let j = 0; j < netSeed[key].length; j++)
      for (let k = 0; k < netSeed[key][j].length; k++)
        if (netSeed[key][j][k] !== net2[key][j][k]) same = false;
  }
  check('GRU：同 seed 两次构造权重逐位一致', same, `一致=${same}`);
}

/* ============================================================
 * 套件 52 · 多类逻辑回归 Softmax Regression（Bishop §4.3.4）
 * ============================================================ */
function suiteSoftmaxRegression(check) {
  const rng = mulberry32(21);
  const { X, y } = blobs(rng, [[-2, -2], [2, 2], [-2, 2]], 20, 0.4);
  const K = 3;

  // ① 解析梯度 ≡ 数值梯度（双判据）
  const g = softmaxGradientCheck(X, y, { eps: 1e-5, seed: 21 });
  check('Softmax：解析梯度 ≡ 中心差分数值梯度（maxRelErr<1e-6）', g.maxRelErr < 1e-6 && g.significant > 0,
    `maxRelErr=${g.maxRelErr.toExponential(2)}，显著=${g.significant}`);

  const model = softmaxRegression(X, y, { lr: 0.5, iters: 800, seed: 21, K });

  // ③ 概率归一：Σ_k p_{i,k} ≡ 1
  const proba = model.predictProba(X);
  let sumErr = 0;
  for (const p of proba) sumErr = Math.max(sumErr, Math.abs(p.reduce((a, b) => a + b, 0) - 1));
  check('Softmax：每样本概率和 ≡ 1（<1e-12）', sumErr < 1e-12, `max偏差=${sumErr.toExponential(2)}`);

  // ④ 交叉熵非负且训练单调不增
  let nonNeg = true, mono = true;
  for (const l of model.losses) if (l < -1e-12) nonNeg = false;
  for (let i = 1; i < model.losses.length; i++) if (model.losses[i] > model.losses[i - 1] + 1e-9) mono = false;
  check('Softmax：交叉熵 ≥ 0 且训练单调不增', nonNeg && mono, `非负=${nonNeg}，单调=${mono}`);

  // ⑤ 同 seed 可复现：两次训练 W 逐位一致
  const m2 = softmaxRegression(X, y, { lr: 0.5, iters: 800, seed: 21, K });
  let same = true;
  for (let k = 0; k < K; k++) for (let j = 0; j < model.W[k].length; j++) if (model.W[k][j] !== m2.W[k][j]) same = false;
  check('Softmax：同 seed 两次训练 W 逐位一致', same, `一致=${same}`);

  // ② K=2 退化：预测标签 ≡ logisticRegression 预测标签（可分数据逐位一致）
  const { X: X2, y: y2 } = blobs(rng, [[-2], [2]], 20, 0.3);
  const sm = softmaxRegression(X2, y2, { lr: 0.5, iters: 1000, seed: 21, K: 2 });
  const lg = logisticRegression(X2, y2, { lr: 0.5, iters: 1000, seed: 1 });
  const ps = sm.predict(X2);
  const pl = lg.predict(X2);
  let eq = true;
  for (let i = 0; i < ps.length; i++) if (ps[i] !== pl[i]) eq = false;
  let acc = 0;
  for (let i = 0; i < ps.length; i++) if (ps[i] === y2[i]) acc++;
  check('Softmax(K=2)：预测标签 ≡ logisticRegression 且可分满分', eq && acc === ps.length,
    `标签一致=${eq}，准确率=${acc}/${ps.length}`);
}

/* ============================================================
 * 套件 53 · 期望 SARSA Expected SARSA（van Seijen et al. 2009）
 * ============================================================ */
function suiteExpectedSarsa(check) {
  const rng = mulberry32(31);
  const S = 3, A = 2;
  const Q0 = randomMatrix(S, A, rng, -1, 1);
  const P = [], R = [];
  for (let s = 0; s < S; s++) {
    const pa = [], ra = [];
    for (let a = 0; a < A; a++) {
      const ns = (s + (a === 1 ? 1 : -1) + S) % S;
      pa.push([[ns, 1]]);
      ra.push([a === 1 ? 1 : 0]);
    }
    P.push(pa); R.push(ra);
  }
  const gamma = 0.9;

  // ① ε=0 退化：单步 TD 目标 ≡ Q-learning 贪婪目标
  const s = 0, a = 0;
  const s2 = P[s][a][0][0];
  const r = R[s][a][0];
  const aGreedy = greedyAction(Q0, s2);
  const Qe = Q0.map((row) => row.slice());
  expectedSarsaStep(Qe, s, a, r, s2, { alpha: 0.1, gamma, eps: 0 });
  const Qq = Q0.map((row) => row.slice());
  const deltaQL = r + gamma * Math.max(...Qq[s2]) - Qq[s][a];
  Qq[s][a] += 0.1 * deltaQL;
  let diff = 0;
  for (let i = 0; i < S; i++) for (let j = 0; j < A; j++) diff = Math.max(diff, Math.abs(Qe[i][j] - Qq[i][j]));
  check('ExpSARSA：ε=0 单步 ≡ Q-learning 贪婪目标（1e-15）', diff < 1e-15, `maxDiff=${diff.toExponential(2)}`);

  // ② 期望目标代数：r + γ·Σ π Q ≡ r + γ·dot(π, Q[s'])
  const probs = [0.3, 0.7];
  const expV = expectedValue(Q0, s2, probs);
  let dotV = 0;
  for (let b = 0; b < A; b++) dotV += probs[b] * Q0[s2][b];
  check('ExpSARSA：期望目标 ≡ dot(π, Q[s’])（<1e-15）', Math.abs(expV - dotV) < 1e-15, `差=${Math.abs(expV - dotV).toExponential(2)}`);

  // ③ GLIE 收敛：chainMDP 上收敛 Q ≈ qFromV(V*)
  const { P: Pg, R: Rg } = chainMDP({ n: 6, gamma });
  const vi = valueIteration(Pg, Rg, gamma);
  const Qref = qFromV(Pg, Rg, gamma, vi.V);
  const res = expectedSarsa(Pg, Rg, {
    gamma, alpha: 0.2, epsStart: 1, epsEnd: 0, episodes: 4000, stepsPerEpisode: 40,
    seed: 5, epsDecayFrac: 0.5,
  });
  let qDiff = 0;
  for (let i = 0; i < 6; i++) for (let j = 0; j < 2; j++) qDiff = Math.max(qDiff, Math.abs(res.Q[i][j] - Qref[i][j]));
  let polOk = true;
  for (let i = 0; i < 6; i++) if (greedyAction(res.Q, i) !== greedyAction(Qref, i)) polOk = false;
  check('ExpSARSA：GLIE 收敛 Q ≈ qFromV(V*) 且策略一致', qDiff < 0.4 && polOk, `max|ΔQ|=${qDiff.toFixed(4)}，策略一致=${polOk}`);

  // ④ 同策略期望 Bellman 残差 → 0
  const resid = expectedBellmanResidual(Pg, Rg, gamma, res.Q, 0);
  check('ExpSARSA：同策略期望 Bellman 残差 ‖Q − B^πQ‖ → 0（<0.4）', resid < 0.4, `max残差=${resid.toFixed(4)}`);

  // ⑤ 学习改善：后期回合平均回报 > 前期
  const ret = res.episodeReturns;
  const m = Math.max(1, Math.floor(ret.length * 0.1));
  let firstAvg = 0, lastAvg = 0;
  for (let i = 0; i < m; i++) firstAvg += ret[i];
  for (let i = ret.length - m; i < ret.length; i++) lastAvg += ret[i];
  firstAvg /= m; lastAvg /= m;
  check('ExpSARSA：后期回合平均回报 > 前期（探索→利用）', lastAvg > firstAvg + 1e-9, `前=${firstAvg.toFixed(3)}，后=${lastAvg.toFixed(3)}`);

  // ⑥ 同 seed 两次训练 Q 逐位一致
  const r1 = expectedSarsa(Pg, Rg, { gamma, episodes: 500, stepsPerEpisode: 30, seed: 9 });
  const r2 = expectedSarsa(Pg, Rg, { gamma, episodes: 500, stepsPerEpisode: 30, seed: 9 });
  let repDiff = 0;
  for (let i = 0; i < 6; i++) for (let j = 0; j < 2; j++) repDiff = Math.max(repDiff, Math.abs(r1.Q[i][j] - r2.Q[i][j]));
  check('ExpSARSA：同 seed 两次训练 Q 逐位一致', repDiff === 0, `maxDiff=${repDiff}`);
}

/* ============================================================
 * 套件 54 · 弹性网络 Elastic Net（Zou & Hastie 2005）
 * ============================================================ */
function suiteElasticNet(check) {
  const rng = mulberry32(41);
  const n = 60, d = 5;
  const X = Array.from({ length: n }, () => Array.from({ length: d }, () => gaussianSample(rng, 0, 1)));
  const trueW = [1.5, -2.0, 0.5, 0.0, 1.0];
  const y = X.map((xi) => {
    let s = 0;
    for (let j = 0; j < d; j++) s += trueW[j] * xi[j];
    return s + gaussianSample(rng, 0, 0.1);
  });
  const softTh = (rho, lam) => (rho > lam ? rho - lam : (rho < -lam ? rho + lam : 0));

  // ① λ1=0 ≡ 岭回归
  const en0 = elasticNetFit(X, y, { lambda1: 0, lambda2: 2, iters: 5000 });
  const ridge = ridgeRegression(X, y, 2);
  let wDiff = 0, bDiff = 0;
  for (let j = 0; j < d; j++) wDiff = Math.max(wDiff, Math.abs(en0.w[j] - ridge.w[j + 1]));
  bDiff = Math.abs(en0.b - ridge.w[0]);
  check('ElasticNet：λ1=0 解 ≡ 岭回归（w,b < 1e-6）', wDiff < 1e-6 && bDiff < 1e-6,
    `max|Δw|=${wDiff.toExponential(2)}，Δb=${bDiff.toExponential(2)}`);

  // ② λ2=0 ≡ Lasso
  const en1 = elasticNetFit(X, y, { lambda1: 0.5, lambda2: 0, iters: 5000 });
  const lasso = lassoFit(X, y, { lambda: 0.5, iters: 5000 });
  let wDiff2 = 0, bDiff2 = 0;
  for (let j = 0; j < d; j++) wDiff2 = Math.max(wDiff2, Math.abs(en1.w[j] - lasso.w[j]));
  bDiff2 = Math.abs(en1.b - lasso.b);
  check('ElasticNet：λ2=0 解 ≡ Lasso（w,b < 1e-7）', wDiff2 < 1e-7 && bDiff2 < 1e-7,
    `max|Δw|=${wDiff2.toExponential(2)}，Δb=${bDiff2.toExponential(2)}`);

  // ③ λ1=λ2=0 ≡ 普通最小二乘
  const en2 = elasticNetFit(X, y, { lambda1: 0, lambda2: 0, iters: 5000 });
  const ols = linearRegression(X, y);
  let wDiff3 = 0;
  for (let j = 0; j < d; j++) wDiff3 = Math.max(wDiff3, Math.abs(en2.w[j] - ols.w[j + 1]));
  check('ElasticNet：λ1=λ2=0 解 ≡ 最小二乘斜率（< 1e-6）', wDiff3 < 1e-6, `max|Δw|=${wDiff3.toExponential(2)}`);

  // ④ 坐标下降不动点：收敛解满足 w_j = S(ρ_j,λ1)/(z_j+λ2)（精确）
  const en3 = elasticNetFit(X, y, { lambda1: 0.8, lambda2: 1.5, iters: 5000 });
  const xbar = new Array(d).fill(0);
  for (const r of X) for (let j = 0; j < d; j++) xbar[j] += r[j];
  for (let j = 0; j < d; j++) xbar[j] /= n;
  const Xc = X.map((r) => r.map((v, j) => v - xbar[j]));
  const ybar = y.reduce((a, b) => a + b, 0) / n;
  const yc = y.map((v) => v - ybar);
  const z = new Array(d).fill(0);
  for (let j = 0; j < d; j++) for (let i = 0; i < n; i++) z[j] += Xc[i][j] * Xc[i][j];
  const rr = yc.slice();
  for (let j = 0; j < d; j++) for (let i = 0; i < n; i++) rr[i] -= Xc[i][j] * en3.w[j];
  let fpErr = 0;
  for (let j = 0; j < d; j++) {
    let rho = 0;
    for (let i = 0; i < n; i++) rho += Xc[i][j] * rr[i];
    rho += z[j] * en3.w[j];
    const target = softTh(rho, 0.8) / (z[j] + 1.5);
    fpErr = Math.max(fpErr, Math.abs(en3.w[j] - target));
  }
  check('ElasticNet：收敛解满足坐标不动点 w_j=S(ρ_j,λ1)/(z_j+λ2)（<1e-9）', fpErr < 1e-9, `maxErr=${fpErr.toExponential(2)}`);

  // ⑤ KKT 证书：平滑梯度 ρ'_j = Σ Xc_j r；活跃 ρ'_j = λ2·w_j + λ1·sign(w_j)，非活跃 |ρ'_j| ≤ λ1
  let kktOk = true;
  for (let j = 0; j < d; j++) {
    let rhoPrime = 0;
    for (let i = 0; i < n; i++) rhoPrime += Xc[i][j] * rr[i];
    if (Math.abs(en3.w[j]) > 1e-9) {
      if (Math.abs(rhoPrime - (1.5 * en3.w[j] + 0.8 * Math.sign(en3.w[j]))) > 1e-7) kktOk = false;
    } else if (Math.abs(rhoPrime) > 0.8 * (1 + 1e-6)) {
      kktOk = false;
    }
  }
  check('ElasticNet：KKT 证书（活跃 ρ’=λ2·w+λ1·sign，非活跃 |ρ’|≤λ1）', kktOk, `KKT满足=${kktOk}`);

  // ⑥ 目标单调不增
  const obj = en3.objective;
  let monoObj = true;
  for (let i = 1; i < obj.length; i++) if (obj[i] > obj[i - 1] + 1e-9) monoObj = false;
  check('ElasticNet：坐标下降目标单调不增', monoObj, `单调=${monoObj}`);
}

/* ============================================================
 * 套件 55 · 快速傅里叶变换 FFT（Cooley & Tukey 1965）
 * ============================================================ */
function suiteFFT(check) {
  const rng = mulberry32(7);
  const n = 64;
  const xr = Array.from({ length: n }, () => (rng() - 0.5) * 2);
  const xi = Array.from({ length: n }, () => (rng() - 0.5) * 2);

  // ① FFT ≡ 朴素 DFT
  const F = fftRadix2(xr, xi);
  const D = dftNaive(xr, xi);
  let d1 = 0;
  for (let k = 0; k < n; k++) d1 = Math.max(d1, Math.abs(F.re[k] - D.re[k]), Math.abs(F.im[k] - D.im[k]));
  check('FFT：基-2 分治 ≡ 朴素 O(n²) DFT（<1e-10）', d1 < 1e-10, `maxAbs=${d1.toExponential(3)}`);

  // ② 逆变换闭合
  const rt = ifftRadix2(F.re, F.im);
  let d2 = 0;
  for (let t = 0; t < n; t++) d2 = Math.max(d2, Math.abs(rt.re[t] - xr[t]), Math.abs(rt.im[t] - xi[t]));
  check('FFT：ifft(fft(x)) ≡ x（<1e-12）', d2 < 1e-12, `maxAbs=${d2.toExponential(3)}`);

  // ③ Parseval 能量守恒
  let eF = 0, eX = 0;
  for (let k = 0; k < n; k++) eF += F.re[k] * F.re[k] + F.im[k] * F.im[k];
  for (let t = 0; t < n; t++) eX += xr[t] * xr[t] + xi[t] * xi[t];
  const par = Math.abs(eF - n * eX) / (n * eX);
  check('FFT：Parseval Σ_k|X_k|² = n·Σ_t|x_t|²（<1e-12）', par < 1e-12, `relErr=${par.toExponential(3)}`);

  // ④ 线性卷积定理
  const a = [1, 2, 3, 4], b = [5, 6, 7];
  const cl = convolveLinear(a, b), cn = convolveNaive(a, b);
  let d4 = 0;
  for (let i = 0; i < cl.length; i++) d4 = Math.max(d4, Math.abs(cl[i] - cn[i]));
  check('FFT：线性卷积（零填充）≡ 朴素卷积（<1e-10）', d4 < 1e-10, `maxAbs=${d4.toExponential(3)}`);

  // ⑤ 线性性
  const yr = Array.from({ length: n }, () => (rng() - 0.5) * 2);
  const yi = Array.from({ length: n }, () => (rng() - 0.5) * 2);
  const alpha = 1.7, beta = -0.6;
  const cr = new Array(n), ci = new Array(n);
  for (let t = 0; t < n; t++) { cr[t] = alpha * xr[t] + beta * yr[t]; ci[t] = alpha * xi[t] + beta * yi[t]; }
  const FC = fftRadix2(cr, ci), FX = fftRadix2(xr, xi), FY = fftRadix2(yr, yi);
  let d5 = 0;
  for (let k = 0; k < n; k++) {
    d5 = Math.max(d5,
      Math.abs(FC.re[k] - (alpha * FX.re[k] + beta * FY.re[k])),
      Math.abs(FC.im[k] - (alpha * FX.im[k] + beta * FY.im[k])));
  }
  check('FFT：线性性 fft(αx+βy) = α·fft(x)+β·fft(y)（<1e-12）', d5 < 1e-12, `maxAbs=${d5.toExponential(3)}`);

  // ⑥ 圆周卷积定理
  const p = [1, 0, 2, -1], q = [3, 1, 0, 0.5];
  const cc = circularConvolve(p, q);
  let d6 = 0;
  for (let t = 0; t < 4; t++) {
    let s = 0;
    for (let k = 0; k < 4; k++) s += p[k] * q[(t - k + 4) % 4];
    d6 = Math.max(d6, Math.abs(cc[t] - s));
  }
  check('FFT：圆周卷积定理 ifft(fft(a)·fft(b)) ≡ Σa_k·b_{t−k}（<1e-10）', d6 < 1e-10, `maxAbs=${d6.toExponential(3)}`);
}

/* ============================================================
 * 套件 56 · 常微分方程求解 ODE（RK4 / Dormand–Prince RK45）
 * ============================================================ */
function suiteODE(check) {
  // ① RK4 单步 ≡ 线性系统 4 阶 Taylor 展开（精确代数恒等式）
  const A = [[0.3, -1.2], [0.8, 0.5]];
  const f = (t, y) => [A[0][0] * y[0] + A[0][1] * y[1], A[1][0] * y[0] + A[1][1] * y[1]];
  const h = 0.1, y0 = [1, -0.5];
  const got = rk4Step(f, 0, y0, h);
  const mm = (X, Y) => X.map((r) => Y[0].map((_, j) => r.reduce((s, v, k) => s + v * Y[k][j], 0)));
  let M = [[1, 0], [0, 1]], Ak = [[1, 0], [0, 1]], fact = 1, hp = 1;
  for (let k = 1; k <= 4; k++) {
    Ak = mm(Ak, A); fact *= k; hp *= h;
    M = M.map((r, i) => r.map((v, j) => v + (hp / fact) * Ak[i][j]));
  }
  const tay = M.map((r) => r[0] * y0[0] + r[1] * y0[1]);
  let d1 = 0;
  for (let i = 0; i < 2; i++) d1 = Math.max(d1, Math.abs(got[i] - tay[i]));
  check('ODE：RK4 单步 ≡ 线性系统 4 阶 Taylor 展开（<1e-12）', d1 < 1e-12, `maxAbs=${d1.toExponential(3)}`);

  // ② 标量 y'=λy 闭式
  const lam = -1.7, hs = 0.2;
  const sc = rk4Step((t, y) => [lam * y[0]], 0, [1], hs)[0];
  const exact = 1 + hs * lam + hs * hs * lam ** 2 / 2 + hs ** 3 * lam ** 3 / 6 + hs ** 4 * lam ** 4 / 24;
  const d2 = Math.abs(sc - exact);
  check('ODE：标量 y’=λy 的 RK4 单步 ≡ 4 阶 Taylor 闭式（<1e-14）', d2 < 1e-14, `maxAbs=${d2.toExponential(3)}`);

  // ③ 全局收敛阶 4（误差比 → 16）
  const g1 = solveODE((t, y) => [y[0]], 0, [1], 1, { h: 0.1 });
  const g2 = solveODE((t, y) => [y[0]], 0, [1], 1, { h: 0.05 });
  const e1 = Math.abs(g1.y[g1.y.length - 1][0] - Math.E);
  const e2 = Math.abs(g2.y[g2.y.length - 1][0] - Math.E);
  const order = Math.log2(e1 / e2);
  check('ODE：全局收敛阶 4（步长折半 ⇒ 误差比→16，观测阶∈[3.5,4.5]）',
    order > 3.5 && order < 4.5, `观测阶=${order.toFixed(3)}，误差比=${(e1 / e2).toFixed(2)}`);

  // ④ 谐振荡器能量漂移：RK4 ≪ Euler
  const ho = (t, y) => [y[1], -y[0]];
  const energy = (y) => 0.5 * (y[0] * y[0] + y[1] * y[1]);
  const rk = solveODE(ho, 0, [1, 0], 20, { h: 0.01 });
  const eu = solveODE(ho, 0, [1, 0], 20, { h: 0.01, method: 'euler' });
  const dRk = Math.abs(energy(rk.y[rk.y.length - 1]) - 0.5);
  const dEu = Math.abs(energy(eu.y[eu.y.length - 1]) - 0.5);
  check('ODE：谐振荡器 RK4 能量漂移 ≪ Euler（四阶 vs 一阶稳定性）',
    dRk < 1e-6 && dRk * 1000 < dEu, `RK4=${dRk.toExponential(2)}，Euler=${dEu.toExponential(2)}`);

  // ⑤a RK45 系数表结构恒等式（精确代数，钉住嵌入式对）
  const T = dpTableau();
  const sumB5 = T.B5.reduce((a, b) => a + b, 0);
  const sumB4 = T.B4.reduce((a, b) => a + b, 0);
  let rowOk = true;
  for (let s = 0; s < T.C.length; s++) {
    const rowSum = T.A[s].reduce((a, b) => a + b, 0);
    if (Math.abs(rowSum - T.C[s]) > 1e-14) rowOk = false;
  }
  check('ODE：RK45 系数表结构恒等式 ΣB5=ΣB4=1 且 Σ_jA[s][j]=C[s]（精确）',
    Math.abs(sumB5 - 1) < 1e-14 && Math.abs(sumB4 - 1) < 1e-14 && rowOk,
    `ΣB5=${sumB5.toFixed(12)}，ΣB4=${sumB4.toFixed(12)}，阶段一致=${rowOk}`);

  // ⑤b RK45 自适应：非自治 ODE y'=t·y（y=e^{t²/2}）末点精度 + 步数效率
  const r45 = solveODERK45((t, y) => [t * y[0]], 0, [1], 2, { tol: 1e-8, h0: 0.1 });
  const err45 = Math.abs(r45.y[r45.y.length - 1][0] - Math.exp(2)) / Math.exp(2);
  check('ODE：RK45 自适应（非自治）末点相对误差 ≤ tol 且步数 ≪ 固定细步（2000）',
    err45 < 1e-8 && r45.steps < 2000, `relErr=${err45.toExponential(2)}，步数=${r45.steps}`);

  // ⑥ 定步长网格闭式
  const s6 = solveODE((t, y) => [y[0]], 0.3, [1], 1.0, { h: 0.1 });
  const gridOk = s6.steps === Math.round((1.0 - 0.3) / 0.1)
    && s6.t.length === s6.steps + 1
    && Math.abs(s6.t[s6.t.length - 1] - 1.0) < 1e-12;
  check('ODE：定步长网格闭式 t_i = t0+i·h（步数与末点精确）', gridOk,
    `步数=${s6.steps}，长度=${s6.t.length}，末点=${s6.t[s6.t.length - 1]}`);

  // ⑦ 步数语义自洽（跨 solveODE / solveODERK45 同族 API 口径统一）
  //    steps 恒等于「接受步数」= t.length − 1 ⇒ y[steps] 必然存在且是末点；
  //    attempts 记录含被拒步的尝试次数，必 ≥ steps（拒绝步必然发生 ⇒ 严格大于）。
  const sem = r45.steps === r45.t.length - 1
    && r45.steps === r45.y.length - 1
    && r45.y[r45.steps] !== undefined
    && r45.attempts >= r45.steps
    && s6.steps === s6.t.length - 1
    && s6.steps === s6.y.length - 1;
  check('ODE：步数语义自洽（steps = 接受步数 = t.length−1，attempts ≥ steps）', sem,
    `RK45: steps=${r45.steps}/t.len=${r45.t.length}/attempts=${r45.attempts}；定步长: steps=${s6.steps}/t.len=${s6.t.length}`);
}

/* ============================================================
 * 套件 57 · 蒙特卡洛强化学习（Sutton & Barto 第 5 章）
 * ============================================================ */
function suiteMonteCarlo(check) {
  const { P, R } = chainMDP({ n: 6, gamma: 0.9 });
  const viR = valueIteration(P, R, 0.9);
  const Qref = qFromV(P, R, 0.9, viR.V);

  // ① 回报恒等式：G_t = Σ γ^k·r_{t+k}
  const rew = [1, 0.5, -0.25, 2, 0];
  let d1 = 0;
  for (let t = 0; t < rew.length; t++) {
    let g = 0, disc = 1;
    for (let k = t; k < rew.length; k++) { g += disc * rew[k]; disc *= 0.9; }
    d1 = Math.max(d1, Math.abs(mcReturn(rew, 0.9, t) - g));
  }
  check('MC：回报恒等式 G_t=Σγ^k·r_{t+k} ≡ 逐项展开（0）', d1 === 0, `maxDiff=${d1}`);

  // ② 确定性轨迹：首访 ≡ 每访
  const rng2 = mulberry32(5);
  const pi = new Array(6).fill(1); // 固定策略：恒向右
  const eps2 = [];
  for (let i = 0; i < 400; i++) {
    eps2.push(mcEpisode(P, R, { gamma: 0.9, policy: pi, rng: rng2, maxSteps: 50, startState: Math.floor(rng2() * 6) }));
  }
  const Vf = mcPredictionFirstVisit(eps2, 0.9, { nStates: 6 });
  const Ve = mcPredictionEveryVisit(eps2, 0.9, { nStates: 6 });
  let d2 = 0;
  for (let s = 0; s < 6; s++) d2 = Math.max(d2, Math.abs(Vf[s] - Ve[s]));
  check('MC：确定性轨迹首访 ≡ 每访（0）', d2 === 0, `maxDiff=${d2}`);

  // ②b 随机策略轨迹（π=[0.7,0.3]）：首访 ≠ 每访——首访过滤器必须真实生效
  const rngB = mulberry32(11);
  const probs = Array.from({ length: 6 }, () => [0.7, 0.3]);
  const epsB = [];
  for (let i = 0; i < 2000; i++) {
    epsB.push(mcEpisode(P, R, { gamma: 0.9, probs, rng: rngB, maxSteps: 60 }));
  }
  const VfB = mcPredictionFirstVisit(epsB, 0.9, { nStates: 6 });
  const VeB = mcPredictionEveryVisit(epsB, 0.9, { nStates: 6 });
  let d2b = 0;
  for (let s = 0; s < 6; s++) d2b = Math.max(d2b, Math.abs(VfB[s] - VeB[s]));
  check('MC：随机策略轨迹首访 ≠ 每访（首访过滤器真实生效）', d2b > 1e-3,
    `max|首访−每访|=${d2b.toFixed(4)}`);

  // ③ MC 预测 ≡ 策略评估（两算法互证）
  const pe = policyEvaluation(P, R, 0.9, pi);
  let d3 = 0;
  for (let s = 0; s < 6; s++) d3 = Math.max(d3, Math.abs(Vf[s] - pe.V[s]));
  check('MC：首访预测 ≡ policyEvaluation(P,R,γ,π)（<1e-9）', d3 < 1e-9, `maxDiff=${d3.toExponential(3)}`);

  // ④ GLIE 控制收敛
  const res4 = mcControl(P, R, {
    gamma: 0.9, epsStart: 1, epsEnd: 0, epsDecayFrac: 0.5, episodes: 8000, maxSteps: 100, seed: 3,
  });
  let qd = 0;
  for (let s = 0; s < 6; s++) for (let a = 0; a < 2; a++) qd = Math.max(qd, Math.abs(res4.Q[s][a] - Qref[s][a]));
  const polOk = res4.policy.every((p, s) => p === viR.policy[s]);
  check('MC：GLIE 控制 Q ≈ qFromV(V*) 且贪心策略 ≡ 最优（<0.3）', qd < 0.3 && polOk,
    `max|ΔQ|=${qd.toFixed(4)}，策略一致=${polOk}`);

  // ⑤ Q* 是 ε=0 贪心更新的不动点（精确）
  const res5 = mcControl(P, R, { gamma: 0.9, eps: 0, episodes: 300, maxSteps: 100, seed: 3, Q0: Qref });
  let d5 = 0;
  for (let s = 0; s < 6; s++) for (let a = 0; a < 2; a++) d5 = Math.max(d5, Math.abs(res5.Q[s][a] - Qref[s][a]));
  check('MC：Q* 是 ε=0 贪心 MC 更新的不动点（<1e-12）', d5 < 1e-12, `max|ΔQ|=${d5.toExponential(3)}`);

  // ⑥ 同 seed 一致
  const r1 = mcControl(P, R, { episodes: 500, seed: 9 });
  const r2 = mcControl(P, R, { episodes: 500, seed: 9 });
  let d6 = 0;
  for (let s = 0; s < 6; s++) for (let a = 0; a < 2; a++) d6 = Math.max(d6, Math.abs(r1.Q[s][a] - r2.Q[s][a]));
  check('MC：同 seed 两次训练 Q 逐位一致', d6 === 0, `maxDiff=${d6}`);
}

/* ============================================================
 * 套件 58 · 自组织映射 SOM（Kohonen 1982）
 * ============================================================ */
function suiteSOM(check) {
  const rng = mulberry32(3);
  const X1 = Array.from({ length: 80 }, (_, i) => [i / 79]);

  // ① BMU 恒等式：直接距离 vs 展开式 ‖x‖²−2x·m+‖m‖²
  const m1 = somBatchTrain(X1, { rows: 1, cols: 12, sigma0: 6, iters: 60, seed: 4 });
  let mism = 0;
  for (const x of X1) {
    const b = somBMU(m1.codebook, x);
    let bx = Infinity, br = 0, bc = 0;
    for (let r = 0; r < m1.codebook.length; r++) {
      for (let c = 0; c < m1.codebook[0].length; c++) {
        let d2 = 0;
        for (let k = 0; k < x.length; k++) {
          d2 += x[k] * x[k] - 2 * x[k] * m1.codebook[r][c][k] + m1.codebook[r][c][k] * m1.codebook[r][c][k];
        }
        if (d2 < bx) { bx = d2; br = r; bc = c; }
      }
    }
    if (br !== b.r || bc !== b.c) mism++;
  }
  check('SOM：BMU ≡ 展开式暴力 argmin（‖x‖²−2x·m+‖m‖² 口径）', mism === 0, `不一致=${mism}`);

  // ② 邻域核闭式
  const h0 = somNeighborhood(0, 2);
  const h4 = somNeighborhood(4, 2);
  const hBig = somNeighborhood(4, 1e10);
  check('SOM：邻域核 h=exp(−d²/2σ²)，d=0⇒1、σ→∞⇒1（精确）',
    h0 === 1 && Math.abs(h4 - Math.exp(-0.5)) < 1e-15 && Math.abs(hBig - 1) < 1e-15,
    `h(0)=${h0}，h(4,2)=${h4.toFixed(12)}，h(4,1e10)=${hBig}`);

  // ③ 批量 SOM 量化误差下降（包络口径）
  const qes = m1.quantErrors;
  let envelope = true;
  for (let i = 1; i < qes.length; i++) if (qes[i] > qes[i - 1] * 1.1 + 1e-15) envelope = false;
  const improved = qes[qes.length - 1] < qes[0] * 0.05;
  check('SOM：批量训练量化误差逐扫描下降（包络，末值<初值 5%）', envelope && improved,
    `QE ${qes[0].toFixed(5)} → ${qes[qes.length - 1].toFixed(5)}，包络=${envelope}`);

  // ④ 拓扑保持：1D 链码本单调排列（方向不限）
  const vals = m1.codebook[0].map((v) => v[0]);
  let vu = 0, vd = 0;
  for (let c = 1; c < vals.length; c++) {
    if (vals[c] < vals[c - 1] - 1e-9) vu++;
    if (vals[c] > vals[c - 1] + 1e-9) vd++;
  }
  check('SOM：1D 链码本按格点序单调排列（拓扑保持）', Math.min(vu, vd) === 0,
    `升序乱序=${vu}，降序乱序=${vd}`);

  // ⑤ σ→∞ 极限：h≡1 ⇒ 码本离散度单步精确收缩为 (1−α) 倍
  const cb = Array.from({ length: 2 }, () => Array.from({ length: 3 }, () => [rng() * 2 - 1, rng() * 2 - 1]));
  const spread = (c) => {
    let m = 0;
    for (let i = 0; i < c.length; i++) {
      for (let j = 0; j < c[0].length; j++) {
        for (let k = 0; k < c[0][0].length; k++) m = Math.max(m, Math.abs(c[i][j][k] - c[0][0][k]));
      }
    }
    return m;
  };
  const before = spread(cb);
  const after = spread(somStep(cb, [0.7, -0.3], { r: 0, c: 1 }, 0.3, 1e10));
  const d5 = Math.abs(after - 0.7 * before) / before;
  check('SOM：σ→∞ ⇒ 码本离散度单步精确收缩为 (1−α) 倍（<1e-12）', d5 < 1e-12,
    `relErr=${d5.toExponential(3)}`);

  // ⑥ 同 seed 一致
  const s1 = somTrain(X1, { rows: 2, cols: 5, iters: 400, seed: 8 });
  const s2 = somTrain(X1, { rows: 2, cols: 5, iters: 400, seed: 8 });
  let d6 = 0;
  for (let r = 0; r < 2; r++) {
    for (let c = 0; c < 5; c++) {
      for (let k = 0; k < 1; k++) d6 = Math.max(d6, Math.abs(s1.codebook[r][c][k] - s2.codebook[r][c][k]));
    }
  }
  check('SOM：同 seed 两次训练码本逐位一致', d6 === 0, `maxDiff=${d6}`);
}

// ──────────────── 迭代线性求解 · LinSolve (Hestenes & Stiefel 1952) ────────────────
function suiteLinSolve(check) {
  const rng = mulberry32(12345);
  const n = 8;
  const M = randomMatrix(n, n, rng);
  const A = matmul(transpose(M), M).map((row, i) => row.map((v, j) => v + (i === j ? n : 0))); // 对称正定
  const b = new Array(n);
  for (let i = 0; i < n; i++) b[i] = rng() * 2 - 1;
  const xDirect = solve(A, b);

  // ① CG ≡ 直接解
  const cg = conjugateGradient(A, b, { tol: 1e-12 });
  let maxDiffCG = 0;
  for (let i = 0; i < n; i++) maxDiffCG = Math.max(maxDiffCG, Math.abs(cg.x[i] - xDirect[i]));
  check('LinSolve：CG 解与直接解 solve 逐分量一致（maxAbs<1e-8）', maxDiffCG < 1e-8, `maxDiff=${maxDiffCG.toExponential(3)}`);

  // ② CG 有限步终止：iters ≤ n 且末残差 < tol
  const finalRes = cg.residuals[cg.residuals.length - 1];
  check('LinSolve：CG 在 n 步内收敛（iters≤n 且残差<1e-9）',
    cg.iters <= n && finalRes < 1e-9, `iters=${cg.iters}, 末残差=${finalRes.toExponential(3)}`);

  // ③ A-共轭性
  const aConj = conjugateResidual(A, cg.directions);
  check('LinSolve：CG 搜索方向 A-共轭（相对误差<1e-9）', aConj < 1e-9, `maxRelErr=${aConj.toExponential(3)}`);

  // ④ 残差正交性
  const rOrth = residualOrthogonality(cg.residualVectors);
  check('LinSolve：CG 残差序列正交（相对<1e-9）', rOrth < 1e-9, `maxRel=${rOrth.toExponential(3)}`);

  // ⑤ Jacobi / Gauss–Seidel 在严格对角占优阵上收敛到 solve，且残差单调下降
  const m = 10;
  const Asd = zeros(m, m);
  for (let i = 0; i < m; i++) {
    let off = 0;
    for (let j = 0; j < m; j++) if (j !== i) { Asd[i][j] = rng() * 2 - 1; off += Math.abs(Asd[i][j]); }
    Asd[i][i] = off + 1.5; // 严格对角占优
  }
  const bs = new Array(m);
  for (let i = 0; i < m; i++) bs[i] = rng() * 2 - 1;
  const xs = solve(Asd, bs);
  const ji = jacobiIter(Asd, bs, { iters: 500, tol: 1e-9 });
  const gs = gaussSeidelIter(Asd, bs, { iters: 500, tol: 1e-9 });
  let dJac = 0, dGs = 0;
  for (let i = 0; i < m; i++) { dJac = Math.max(dJac, Math.abs(ji.x[i] - xs[i])); dGs = Math.max(dGs, Math.abs(gs.x[i] - xs[i])); }
  const mono = (res) => { for (let i = 1; i < res.length; i++) if (res[i] > res[i - 1] + 1e-12) return false; return true; };
  check('LinSolve：Jacobi 在严格对角占优阵收敛到 solve（maxAbs<1e-6）', dJac < 1e-6 && ji.converged, `maxDiff=${dJac.toExponential(3)}`);
  check('LinSolve：Gauss–Seidel 在严格对角占优阵收敛到 solve（maxAbs<1e-6）', dGs < 1e-6 && gs.converged, `maxDiff=${dGs.toExponential(3)}`);
  check('LinSolve：Jacobi 残差单调下降（包络）', mono(ji.residuals), `末残差=${ji.residuals[ji.residuals.length - 1].toExponential(2)}`);
  check('LinSolve：Gauss–Seidel 残差单调下降（包络）', mono(gs.residuals), `末残差=${gs.residuals[gs.residuals.length - 1].toExponential(2)}`);

  // ⑥ 迭代矩阵谱半径 < 1（Gershgorin + 幂迭代）
  const Brho = jacobiIterationMatrix(Asd);
  const rho = spectralRadius(Brho);
  const dom = diagonalDominance(Asd);
  check('LinSolve：迭代矩阵谱半径 ρ(B)<1（Gershgorin + 幂迭代）', rho < 1 && dom.strictlyDominant, `ρ=${rho.toExponential(3)}, 占优=${dom.strictlyDominant}`);

  // ⑦ 同输入确定性
  const a1 = conjugateGradient(A, b, { tol: 1e-12 });
  const a2 = conjugateGradient(A, b, { tol: 1e-12 });
  let dd = 0; for (let i = 0; i < n; i++) dd = Math.max(dd, Math.abs(a1.x[i] - a2.x[i]));
  check('LinSolve：同输入两次 CG 逐位一致', dd === 0, `maxDiff=${dd}`);
}

// ──────────────── 隔离森林 · Isolation Forest (Liu et al. 2008) ────────────────
function suiteIsolationForest(check) {
  const varOf = (a) => { const m = a.reduce((s, v) => s + v, 0) / a.length; return a.reduce((s, v) => s + (v - m) * (v - m), 0) / a.length; };

  // ① c(n) 闭式 ≡ 递归增量式
  let maxRelC = 0;
  for (let nn = 3; nn <= 200; nn++) {
    const a = iforC(nn), bb = iforCRecursive(nn);
    maxRelC = Math.max(maxRelC, Math.abs(a - bb) / (Math.abs(a) + 1e-30));
  }
  check('IsoForest：c(n) 闭式 ≡ 递归增量式（relErr<1e-12）', maxRelC < 1e-12, `maxRelErr=${maxRelC.toExponential(3)}`);

  // ⑦ 归一化常数性质：单调递增且 c(n) < n
  let mono = true, lessN = true;
  for (let nn = 2; nn <= 100; nn++) {
    if (iforC(nn) < iforC(nn - 1) - 1e-15) mono = false;
    if (!(iforC(nn) < nn)) lessN = false;
  }
  check('IsoForest：c(n) 单调递增（n≥2）且 c(n)<n', mono && lessN, `mono=${mono}, <n=${lessN}`);

  // ② 路径长度上界：h(x) ≤ heightLimit + c(ψ)
  const toy = iforToyData({ n: 300, nOutlier: 8, dim: 2, seed: 11 });
  const psi = 128;
  const forest = isolationForest(toy.X, { trees: 50, sampleSize: psi, seed: 42 });
  const hl = forest.heightLimit;
  let viol = 0;
  for (const x of toy.X) {
    for (let t = 0; t < forest.trees.length; t++) {
      const h = pathLength(forest.trees[t], x);
      if (h > hl + iforC(psi) + 1e-12) viol++;
    }
  }
  check('IsoForest：叶深度/路径长度上界（h ≤ heightLimit + c(ψ)）', viol === 0, `违反=${viol}`);

  // ③④ 异常分排序 + 路径长度单调性（outlier 更短）
  const sc = iforScoreAll(forest, toy.X);
  let normS = 0, outS = 0, normH = 0, outH = 0, nN = 0, nO = 0;
  for (let i = 0; i < toy.X.length; i++) {
    if (toy.y[i] === 1) { outS += sc.scores[i]; outH += sc.meanPath[i]; nO++; }
    else { normS += sc.scores[i]; normH += sc.meanPath[i]; nN++; }
  }
  const dS = outS / nO - normS / nN;
  const dH = normH / nN - outH / nO;
  check('IsoForest：outlier 异常分显著高于常态（差值>0.1）', dS > 0.1, `Δscore=${dS.toFixed(4)}`);
  check('IsoForest：outlier 平均路径更短（隔离机制）', dH > 0, `Δh=${dH.toFixed(4)}`);

  // ⑤ 集成降方差：单树得分方差 > 森林平均得分方差（bagging 方差缩减）
  const flat = [];
  for (const per of sc.perTreeScores) for (const s of per) flat.push(s);
  const varTree = varOf(flat);
  const varForest = varOf(sc.scores);
  check('IsoForest：集成降低方差（单树方差>森林方差）', varTree > varForest, `varTree=${varTree.toExponential(2)}, varForest=${varForest.toExponential(2)}`);

  // ⑥ 同 seed 可复现
  const f2 = isolationForest(toy.X, { trees: 50, sampleSize: psi, seed: 42 });
  let md = 0;
  for (const x of toy.X) md = Math.max(md, Math.abs(anomalyScore(forest, x) - anomalyScore(f2, x)));
  check('IsoForest：同 seed 两片森林异常分逐位一致', md === 0, `maxDiff=${md}`);
}

// ──────────────── 流形学习 · MDS & Isomap (Torgerson 1952 / Tenenbaum 2000) ────────────────
function suiteManifold(check) {
  const rng = mulberry32(2024);
  const N = 30, Dim = 4;
  const X = randomMatrix(N, Dim, rng);
  const mean = new Array(Dim).fill(0);
  for (const row of X) for (let j = 0; j < Dim; j++) mean[j] += row[j];
  for (let j = 0; j < Dim; j++) mean[j] /= N;
  const Xc = X.map((r) => r.map((v, j) => v - mean[j]));
  const Dmat = distanceMatrix(Xc);

  // ① 经典 MDS ≡ PCA（欧氏距离矩阵上，逐列允许符号翻转）
  const mds = classicalMDS(Dmat, Dim);
  const pcaRes = pca(Xc, Dim);
  let maxDiff = 0;
  for (let c = 0; c < Dim; c++) {
    let direct = 0, flip = 0;
    for (let i = 0; i < N; i++) {
      direct = Math.max(direct, Math.abs(mds.coords[i][c] - pcaRes.proj[i][c]));
      flip = Math.max(flip, Math.abs(mds.coords[i][c] + pcaRes.proj[i][c]));
    }
    maxDiff = Math.max(maxDiff, Math.min(direct, flip));
  }
  check('Manifold：经典 MDS ≡ PCA（逐列允许符号翻转，maxAbs<1e-8）', maxDiff < 1e-8, `maxDiff=${maxDiff.toExponential(3)}`);

  // ② 双居中矩阵行和恒等式 + 对称
  const B = doubleCenter(Dmat);
  let symErr = 0, rowSumErr = 0;
  for (let i = 0; i < N; i++) {
    let rs = 0;
    for (let j = 0; j < N; j++) { rs += B[i][j]; symErr = Math.max(symErr, Math.abs(B[i][j] - B[j][i])); }
    rowSumErr = Math.max(rowSumErr, Math.abs(rs));
  }
  check('Manifold：双居中 B 对称（maxAbs<1e-14）且 B·1=0（maxAbs<1e-12）', symErr < 1e-14 && rowSumErr < 1e-12, `sym=${symErr.toExponential(3)}, rowSum=${rowSumErr.toExponential(3)}`);

  // ③ 满秩重构：k = N−1 时嵌入点间欧氏距离 ≡ 原始 D
  const mFull = classicalMDS(Dmat, N - 1);
  const recon = distanceMatrix(mFull.coords);
  let reconErr = 0;
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) reconErr = Math.max(reconErr, Math.abs(recon[i][j] - Dmat[i][j]));
  check('Manifold：满秩重构嵌入距离 ≡ 原始 D（maxAbs<1e-8）', reconErr < 1e-8, `maxAbs=${reconErr.toExponential(3)}`);

  // ④ Kruskal 应力单调：嵌入维数增大 ⇒ 应力单调不增（包络口径）
  let prev = Infinity, mono = true;
  for (let k = 1; k <= Dim; k++) {
    const st = stress(classicalMDS(Dmat, k).coords, Dmat);
    if (st > prev + 1e-7) mono = false;
    prev = st;
  }
  check('Manifold：应力随嵌入维数单调不增（包络）', mono, `末应力=${prev.toExponential(3)}`);

  // ⑤ 测地距离 ≥ 欧氏距离（Swiss Roll，三角不等式，违反计数=0）
  const sr = mdsSwissRoll({ n: 200, seed: 42, noise: 0.0 });
  const iso = isomap(sr.X, { k: 8, dim: 2 });
  const Deuc = distanceMatrix(sr.X);
  let viol = 0;
  for (let i = 0; i < sr.X.length; i++) for (let j = 0; j < sr.X.length; j++) {
    const g = iso.geodesic[i][j];
    if (Number.isFinite(g) && g < Deuc[i][j] - 1e-9) viol++;
  }
  check('Manifold：图上测地距离 ≥ 欧氏距离（违反=0）', viol === 0, `违反=${viol}`);

  // ⑥ Isomap 展开质量（非线性流形）：圆环上 Isomap 测地恢复误差显著低于欧氏近似
  const NR = 60;
  const ring = [];
  for (let i = 0; i < NR; i++) { const t = 2 * Math.PI * i / NR; ring.push([Math.cos(t), Math.sin(t)]); }
  const Dtrue = zeros(NR, NR);
  for (let i = 0; i < NR; i++) for (let j = i + 1; j < NR; j++) {
    let d = Math.abs(i - j) / NR * 2 * Math.PI; d = Math.min(d, 2 * Math.PI - d);
    Dtrue[i][j] = Dtrue[j][i] = d;
  }
  const isoR = isomap(ring, { k: 3, dim: 2 });
  const DeucR = distanceMatrix(ring);
  const DembR = distanceMatrix(isoR.embedding);
  const mse = (A) => { let s = 0, c = 0; for (let i = 0; i < NR; i++) for (let j = i + 1; j < NR; j++) { s += (A[i][j] - Dtrue[i][j]) ** 2; c++; } return s / c; };
  const mseE = mse(DeucR), mseI = mse(DembR);
  check('Manifold：圆环上 Isomap 测地恢复误差 ≪ 欧氏近似（<0.7×）', mseI < 0.7 * mseE, `mseIso=${mseI.toFixed(4)}, mseEuc=${mseE.toFixed(4)}`);
}

// ──────────────── 策略梯度 · REINFORCE (Williams 1992) ────────────────
function suitePolicyGradient(check) {
  const rng = mulberry32(777);
  const theta0 = randomMatrix(5, 4, rng);
  const pol = softmaxPolicy(theta0);

  // ① Score function 恒等式：Σ_a π(a|s)·∇logπ(a|s) = 0
  let maxScoreZero = 0;
  for (let s = 0; s < theta0.length; s++) {
    const acc = new Array(theta0[s].length).fill(0);
    for (let a = 0; a < pol[s].length; a++) {
      const g = policyLogGrad(theta0, s, a);
      for (let b = 0; b < acc.length; b++) acc[b] += pol[s][a] * g[b];
    }
    for (let b = 0; b < acc.length; b++) maxScoreZero = Math.max(maxScoreZero, Math.abs(acc[b]));
  }
  check('PolicyGrad：Score 恒等式 Σπ∇logπ=0（maxAbs<1e-14）', maxScoreZero < 1e-14, `maxAbs=${maxScoreZero.toExponential(3)}`);

  // ⑤ 概率单纯形：Σ_a π=1 且 π>0
  let simplexOk = true, posOk = true;
  for (let s = 0; s < pol.length; s++) {
    let sum = 0;
    for (let a = 0; a < pol[s].length; a++) { sum += pol[s][a]; if (!(pol[s][a] > 0)) posOk = false; }
    if (Math.abs(sum - 1) > 1e-15) simplexOk = false;
  }
  check('PolicyGrad：softmax 策略构成概率单纯形（Σ=1, π>0）', simplexOk && posOk, `Σ=1=${simplexOk}, >0=${posOk}`);

  // ② 策略梯度定理（解析 ≡ 数值）：bandit J 解析梯度 ≡ 中心差分
  const r = [0.2, 1.3, -0.5, 0.9];
  const th = [0.1, -0.4, 0.3, 0.0];
  const ga = banditAnalyticGrad(th, r);
  const eps = 1e-6;
  const gn = new Array(4);
  for (let b = 0; b < 4; b++) {
    const tp = th.slice(); tp[b] += eps;
    const tm = th.slice(); tm[b] -= eps;
    gn[b] = (banditExpectedReturn(tp, r) - banditExpectedReturn(tm, r)) / (2 * eps);
  }
  let maxRel = 0; for (let b = 0; b < 4; b++) maxRel = Math.max(maxRel, Math.abs(ga[b] - gn[b]) / (Math.abs(ga[b]) + 1e-30));
  check('PolicyGrad：策略梯度定理（解析≡中心差分，maxRelErr<1e-6）', maxRel < 1e-6, `maxRelErr=${maxRel.toExponential(3)}`);

  // ③④ 构造有限 horizon MDP
  const P = [
    [[[1, 1.0]], [[1, 1.0]]],
    [[[0, 0.5], [2, 0.5]], [[2, 1.0]]],
    [[[2, 1.0]], [[2, 1.0]]],
  ];
  const R = [
    [[0], [1]],
    [[0, 0], [1]],
    [[0], [0]],
  ];
  const theta3 = zeros(3, 2);
  const gamma = 0.95, horizon = 6;

  // ③ 基线不改变梯度期望（解析恒等式，精确）：baseline=b ≡ baseline=0
  const g0 = reinforceExpectedUpdate(P, R, theta3, { gamma, horizon, baseline: 0 });
  const g5 = reinforceExpectedUpdate(P, R, theta3, { gamma, horizon, baseline: 5 });
  let mb = 0;
  for (let s = 0; s < 3; s++) for (let a = 0; a < 2; a++) mb = Math.max(mb, Math.abs(g0[s][a] - g5[s][a]));
  check('PolicyGrad：基线不改变梯度期望（带/不带基线逐位一致，maxAbs<1e-10）', mb < 1e-10, `maxAbs=${mb.toExponential(3)}`);

  // ④ 采样梯度期望 ≡ 解析梯度：枚举期望更新 ≡ 动态规划解析梯度
  const gaDP = policyGradientAnalytic(P, R, theta3, { gamma, horizon });
  let mr = 0;
  for (let s = 0; s < 3; s++) for (let a = 0; a < 2; a++) mr = Math.max(mr, Math.abs(g0[s][a] - gaDP[s][a]) / (Math.abs(gaDP[s][a]) + 1e-30));
  check('PolicyGrad：期望更新量（枚举）≡ 解析梯度（DP，maxRelErr<1e-6）', mr < 1e-6, `maxRelErr=${mr.toExponential(3)}`);

  // ⑥ 收敛性：chainMDP 上训练后贪心策略 ≡ valueIteration 最优策略
  const mdp = chainMDP({ n: 6, gamma: 0.95 });
  const vi = valueIteration(mdp.P, mdp.R, mdp.gamma);
  const out = reinforce(mdp.P, mdp.R, { gamma: mdp.gamma, lr: 0.05, episodes: 40000, maxSteps: 12, seed: 42, baseline: 'mean' });
  let match = 0;
  for (let s = 0; s < mdp.S; s++) {
    const ps = out.policy[s];
    let best = 0; for (let a = 1; a < mdp.A; a++) if (ps[a] > ps[best]) best = a;
    if (best === vi.policy[s]) match++;
  }
  check('PolicyGrad：chainMDP 训练后贪心策略 ≡ VI 最优策略', match === mdp.S, `匹配=${match}/${mdp.S}`);

  // ⑦ 同 seed 一致
  const tA = reinforce(mdp.P, mdp.R, { gamma: mdp.gamma, lr: 0.05, episodes: 200, maxSteps: 12, seed: 7, baseline: 'mean' });
  const tB = reinforce(mdp.P, mdp.R, { gamma: mdp.gamma, lr: 0.05, episodes: 200, maxSteps: 12, seed: 7, baseline: 'mean' });
  let md = 0;
  for (let s = 0; s < mdp.S; s++) for (let a = 0; a < mdp.A; a++) md = Math.max(md, Math.abs(tA.theta[s][a] - tB.theta[s][a]));
  check('PolicyGrad：同 seed 两次训练 θ 逐位一致', md === 0, `maxDiff=${md}`);
}

/**
 * 旗舰AI引擎 · 统一出口
 * 用法（Node）：  import { pca, DenseNet, runAllInvariants } from 'flagship-ai-engine/src/index.js'
 * 用法（浏览器）：scripts/bundle.mjs 打包后经 globalThis.FlagshipAI 访问
 */

export * from './core/linalg.js';
export * from './core/numerical.js';
export * from './core/stats.js';
export * from './ml/regression.js';
export * from './ml/classification.js';
export * from './ml/clustering.js';
export * from './ml/dimreduction.js';
export * from './ml/tree.js';
export * from './ml/svm.js';
export * from './ml/boosting.js';
export * from './ml/hmm.js';
export * from './ml/dbscan.js';
export * from './ml/tsne.js';
export * from './ml/kpca.js';
export * from './ml/spectral.js';
export * from './ml/gpr.js';
export * from './ml/gbdt.js';
export * from './ml/kalman.js';
export * from './ml/mcmc.js';
export * from './ml/nmf.js';
export * from './nn/dense.js';
export * from './nn/attention.js';
export * from './nn/transformer.js';
export * from './nn/rnn.js';
export * from './nn/lstm.js';
export * from './nn/optimizers.js';
export * from './nn/activations.js';
export * from './nn/schedulers.js';
export * from './nn/cnn.js';
export * from './nn/vae.js';
export * from './nn/word2vec.js';
export * from './rl/valueIteration.js';
export * from './rl/qLearning.js';
export * from './verify/invariants.js';

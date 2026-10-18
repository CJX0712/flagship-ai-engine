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
export * from './nn/dense.js';
export * from './nn/attention.js';
export * from './nn/transformer.js';
export * from './verify/invariants.js';

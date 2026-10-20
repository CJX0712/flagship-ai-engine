#!/usr/bin/env node
/**
 * 旗舰AI引擎 · 打包器
 * 把 src 按依赖序拼接 → 剥离 import/export → IIFE 注入 globalThis.FlagshipAI
 * → node --check 语法校验 → 内联进 showcase/index.html（单文件、双击即用）。
 */
import { readFileSync, writeFileSync, unlinkSync } from 'node:fs';
import { execSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');

// 依赖序：被依赖者在前
const FILES = [
  'src/core/linalg.js',
  'src/core/numerical.js',
  'src/core/stats.js',
  'src/ml/regression.js',
  'src/ml/classification.js',
  'src/ml/clustering.js',
  'src/ml/dimreduction.js',
  'src/ml/tree.js',
  'src/ml/randomForest.js',
  'src/ml/perceptron.js',
  'src/ml/ica.js',
  'src/ml/cca.js',
  'src/ml/hierarchical.js',
  'src/ml/svm.js',
  'src/ml/boosting.js',
  'src/ml/hmm.js',
  'src/ml/dbscan.js',
  'src/ml/tsne.js',
  'src/ml/kpca.js',
  'src/ml/spectral.js',
  'src/ml/gpr.js',
  'src/ml/gbdt.js',
  'src/ml/kalman.js',
  'src/ml/kalmanSmoother.js',
  'src/ml/particleFilter.js',
  'src/ml/mcmc.js',
  'src/ml/nmf.js',
  'src/ml/pagerank.js',
  'src/ml/gibbs.js',
  'src/ml/gmm.js',
  'src/ml/lda.js',
  'src/ml/naiveBayes.js',
  'src/ml/qda.js',
  'src/ml/lasso.js',
  'src/ml/meanShift.js',
  'src/ml/softmaxRegression.js',
  'src/ml/elasticNet.js',
  'src/nn/dense.js',
  'src/nn/attention.js',
  'src/nn/transformer.js',
  'src/nn/rnn.js',
  'src/nn/lstm.js',
  'src/nn/gru.js',
  'src/nn/optimizers.js',
  'src/nn/activations.js',
  'src/nn/normalization.js',
  'src/nn/schedulers.js',
  'src/nn/cnn.js',
  'src/nn/vae.js',
  'src/nn/word2vec.js',
  'src/nn/resnet.js',
  'src/nn/rbm.js',
  'src/nn/hopfield.js',
  'src/rl/valueIteration.js',
  'src/rl/qLearning.js',
  'src/rl/sarsa.js',
  'src/rl/expectedSarsa.js',
  'src/verify/invariants.js',
];

const names = [];
let body = '';
for (const f of FILES) {
  let src = readFileSync(join(root, f), 'utf8');
  for (const m of src.matchAll(/^export\s+(?:function|class|const)\s+([A-Za-z0-9_$]+)/gm)) {
    names.push(m[1]);
  }
  // 本库 import 均为单行；剥离的同时重建「别名导入」（如 `import { mean as colMean }` → `const colMean = mean;`），
  // 否则被剥离后别名在单一 IIFE 作用域内成为未定义引用（导致展示页自检误报 FAIL）。
  const rebinds = [];
  src = src.replace(/^import\s[^;]*;.*$/gm, (line) => {
    const m = line.match(/import\s*\{([^}]*)\}/);
    if (m) {
      for (const part of m[1].split(',')) {
        const am = part.trim().match(/^([A-Za-z0-9_$]+)\s+as\s+([A-Za-z0-9_$]+)$/);
        if (am) rebinds.push(`const ${am[2]} = ${am[1]};`);
      }
    }
    return '';
  });
  src = src.replace(/^export\s+/gm, '');
  body += `\n/* ═══════════ ${f} ═══════════ */\n${src}`;
  if (rebinds.length) body += `\n${rebinds.join('\n')}`;
}

const bundle = `(function () {
'use strict';
${body}
/* ───── 挂载全局命名空间 ───── */
const __FlagshipAI__ = { ${names.join(', ')} };
if (typeof globalThis !== 'undefined') {
  globalThis.FlagshipAI = Object.assign({}, globalThis.FlagshipAI, __FlagshipAI__);
}
})();
`;

// 语法校验（Node 解析器兜底）
const tmp = join(root, 'scripts', '.engine-bundle.tmp.js');
writeFileSync(tmp, bundle, 'utf8');
execSync(`node --check "${tmp}"`, { stdio: 'inherit' });
unlinkSync(tmp);

const template = readFileSync(join(root, 'showcase', 'template.html'), 'utf8');
const html = template.replace('/*__FLAGSHIP_ENGINE__*/', () => bundle);
writeFileSync(join(root, 'showcase', 'index.html'), html, 'utf8');

console.log(`✅ showcase/index.html 已生成 — 引擎导出 ${names.length} 个，内联 ${(bundle.length / 1024).toFixed(1)} KB，node --check 语法校验通过`);

/**
 * 旗舰AI引擎 · AdaBoost 自适应提升（Freund & Schapire 1997）
 * 每轮训练加权决策树桩，按误差 ε_t 加权 α_t = ½·ln((1−ε_t)/ε_t)。
 * 不变量（均有定理背书）：
 *   ① 样本权重逐轮归一 Σwᵢ = 1
 *   ② 指数损失 E_t = Σᵢexp(−yᵢF_t(xᵢ)) 单调不增（Z_t = 2√(ε_t(1−ε_t)) ≤ 1）
 *   ③ 训练误差 ≤ Π_t 2√(ε_t(1−ε_t))（Freund–Schapire 泛化定理）
 *   ④ 可分数据上训练误差 → 0
 */

/** 加权决策树桩搜索：遍历 (特征, 阈值, 极性) 取加权误差最小 */
export function bestStump(X, y, w) {
  const n = X.length, d = X[0].length;
  let best = { error: Infinity, feature: 0, threshold: 0, polarity: 1 };
  for (let j = 0; j < d; j++) {
    const vals = X.map((x) => x[j]).sort((a, b) => a - b);
    for (let t = 0; t + 1 < vals.length; t++) {
      const theta = (vals[t] + vals[t + 1]) / 2;
      if (theta === vals[t]) continue; // 重复阈值跳过
      for (const p of [1, -1]) {
        let err = 0;
        for (let i = 0; i < n; i++) {
          const h = p * Math.sign(X[i][j] - theta) || 1;
          if (h !== y[i]) err += w[i];
        }
        if (err < best.error) best = { error: err, feature: j, threshold: theta, polarity: p };
      }
    }
  }
  return best;
}

/** 树桩预测 */
function stumpPredict(stump, x) {
  const h = stump.polarity * Math.sign(x[stump.feature] - stump.threshold) || 1;
  return h;
}

/**
 * AdaBoost（二分类，y ∈ {−1,+1}）。
 * @returns {{stumps:Array, alphas:number[], errors:number[], expLoss:number[], rounds:number,
 *            predict:function, decisionValue:function}}
 */
export function adaBoost(X, y, { rounds = 30 } = {}) {
  const n = X.length;
  let w = new Array(n).fill(1 / n);
  const stumps = [], alphas = [], errors = [], expLoss = [];
  for (let t = 0; t < rounds; t++) {
    const stump = bestStump(X, y, w);
    const eps = Math.min(Math.max(stump.error, 1e-10), 0.5 - 1e-10);
    const alpha = 0.5 * Math.log((1 - eps) / eps);
    let z = 0;
    for (let i = 0; i < n; i++) {
      w[i] *= Math.exp(-alpha * y[i] * stumpPredict(stump, X[i]));
      z += w[i];
    }
    for (let i = 0; i < n; i++) w[i] /= z;
    stumps.push(stump); alphas.push(alpha); errors.push(stump.error);
    // 当前集成指数损失 E_t = Σᵢexp(−yᵢF_t(xᵢ))
    let e = 0;
    for (let i = 0; i < n; i++) {
      let f = 0;
      for (let s = 0; s < stumps.length; s++) f += alphas[s] * stumpPredict(stumps[s], X[i]);
      e += Math.exp(-y[i] * f);
    }
    expLoss.push(e);
  }
  const decisionValue = (x) => stumps.reduce((s, st, k) => s + alphas[k] * stumpPredict(st, x), 0);
  return {
    stumps, alphas, errors, expLoss, rounds,
    decisionValue,
    predict: (x) => (decisionValue(x) >= 0 ? 1 : -1),
  };
}

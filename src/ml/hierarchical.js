/**
 * 旗舰AI引擎 · 凝聚式层次聚类（Sokal & Michener 1958 · Lance & Williams 1967）
 * 站在：Sokal & Michener 1958《A Statistical Method for Evaluating Systematic
 *       Relationships》（数值分类学，UPMGA 的思想源头）·
 *       Lance & Williams 1967《A General Theory of Classificatory Sorting
 *       Strategies》（簇间距递推的通式）之上。
 *
 * 算法：自底向上聚合——初始每点一簇，每轮合并「当前簇间距最小」的一对，
 * 共 n−1 次合并后归为一族。合并距离用 Lance–Williams 递推统一表达：
 *   d(A∪B, C) = α_A·d(A,C) + α_B·d(B,C) + β·d(A,B) + γ·|d(A,C) − d(B,C)|
 * 三种 linkage 的参数：
 *   single  （最近邻）  α_A=α_B=1/2, β=0, γ=−1/2
 *   complete（最远邻）  α_A=α_B=1/2, β=0, γ=+1/2
 *   average （类平均）  α_A=|A|/(|A|+|B|), α_B=|B|/(|A|+|B|), β=0, γ=0
 * 平手时取字典序最小的簇对 ⇒ 算法完全确定（同输入必得同一合并树）。
 *
 * 不变量：
 *   ① 合并距离单调非降（single/complete/average 均为可约 linkage ⇒ ultrametric）
 *   ② n 个样本 ⇒ 恰 n−1 次合并，最终归为 1 个簇
 *   ③ 强分离 3 高斯团：clusters(3) 与真实标签 ARI = 1
 *   ④ Lance–Williams 一致性：4 点共线算例的合并距离 ≡ 闭式（single/complete/average）
 *   ⑤ 递推 ≡ 定义：blob 数据上每次合并距离与成员集闭式（min/max/加权均值）一致
 *   ⑥ 确定性：相同输入两次运行为逐位一致的合并树
 *   ⑦ cophenetic：结构良好数据上 cophenetic 相关系数 ≥ 0.9
 */

import { zeros } from '../core/linalg.js';

/** 支持的 linkage 及其名称表（确定性错误提示用） */
const HC_SUPPORTED_LINKAGES = ['single', 'complete', 'average'];

/**
 * Lance–Williams 递推：两簇 A、B 合并后与第三簇 C 的距离。
 * 公式 d(A∪B,C) = α_A·d(A,C) + α_B·d(B,C) + β·d(A,B) + γ·|d(A,C) − d(B,C)|
 * 中 β·d(A,B) 项在 single/complete/average 三种 linkage 下恒为 0，
 * 故实现不收 d(A,B) 参数（等价于 β = 0 的代入）。
 * @param {number} dAC d(A, C)
 * @param {number} dBC d(B, C)
 * @param {number} sizeA 簇 A 的样本数
 * @param {number} sizeB 簇 B 的样本数
 * @param {string} linkage 'single' | 'complete' | 'average'
 * @returns {number} d(A∪B, C)
 */
function hcLanceWilliams(dAC, dBC, sizeA, sizeB, linkage) {
  if (linkage === 'single') return 0.5 * dAC + 0.5 * dBC - 0.5 * Math.abs(dAC - dBC);
  if (linkage === 'complete') return 0.5 * dAC + 0.5 * dBC + 0.5 * Math.abs(dAC - dBC);
  if (linkage === 'average') {
    const total = sizeA + sizeB;
    return (sizeA * dAC + sizeB * dBC) / total;
  }
  throw new Error(`hcLanceWilliams: 不支持的 linkage "${linkage}"`);
}

/**
 * 凝聚式层次聚类（O(n³) 朴素 Lance–Williams 递推，确定性平手打破）。
 * @param {number[][]} X 样本矩阵 n×d
 * @param {{linkage?:'single'|'complete'|'average'}} [opts]
 * @returns {{linkage:string, n:number, merges:{a:number,b:number,distance:number,size:number}[],
 *            distances:number[], labels:(k:number)=>number[], clusters:(k:number)=>number[][]}}
 *   merges    合并序列（a、b 为参与合并的簇 id：< n 为原始点，n+step 为第 step 次合并的新簇）
 *   distances 合并高度序列（长度 n−1，非降）
 *   labels(k)  割成 k 簇时的逐点标签（编号按簇内最小点排序，确定性）
 *   clusters(k) 割成 k 簇时的成员分组（每组为升序原始点索引）
 */
export function hierarchicalCluster(X, { linkage = 'average' } = {}) {
  if (!HC_SUPPORTED_LINKAGES.includes(linkage)) {
    throw new Error(`hierarchicalCluster: 不支持的 linkage "${linkage}"（可选 ${HC_SUPPORTED_LINKAGES.join('/')}）`);
  }
  const n = X.length;
  if (n === 0) throw new Error('hierarchicalCluster: 输入样本为空');
  const d = X[0].length;
  const size = 2 * n - 1;

  // 簇间距矩阵 D：行/列索引 = 簇 id（< n 单点，≥ n 为历次合并产物）
  const D = zeros(size, size);
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      let s = 0;
      for (let k = 0; k < d; k++) {
        const diff = X[i][k] - X[j][k];
        s += diff * diff;
      }
      D[i][j] = Math.sqrt(s);
      D[j][i] = D[i][j];
    }
  }

  const members = Array.from({ length: size }, (_, id) => (id < n ? [id] : null));
  const repOf = Array.from({ length: size }, (_, id) => (id < n ? id : -1));
  let active = Array.from({ length: n }, (_, i) => i); // 保持升序 ⇒ 平手取字典序最小对
  const merges = [];
  const distances = [];

  for (let step = 0; step < n - 1; step++) {
    // 最小间距簇对（严格小于 ⇒ 保留先扫到的字典序最小对）
    let bi = -1, bj = -1, best = Infinity;
    for (let ii = 0; ii < active.length; ii++) {
      for (let jj = ii + 1; jj < active.length; jj++) {
        const i = active[ii], j = active[jj];
        if (D[i][j] < best) { best = D[i][j]; bi = i; bj = j; }
      }
    }
    if (bi < 0) throw new Error('hierarchicalCluster: 找不到可合并的簇对');
    const sizeA = members[bi].length, sizeB = members[bj].length;
    const newId = n + step;
    merges.push({ a: bi, b: bj, distance: best, size: sizeA + sizeB });
    distances.push(best);
    // Lance–Williams 更新新簇与其余各簇的间距
    for (const k of active) {
      if (k === bi || k === bj) continue;
      const dNew = hcLanceWilliams(D[bi][k], D[bj][k], sizeA, sizeB, linkage);
      D[newId][k] = dNew;
      D[k][newId] = dNew;
    }
    members[newId] = members[bi].concat(members[bj]).sort((a, b) => a - b);
    repOf[newId] = members[newId][0];
    active = active.filter((id) => id !== bi && id !== bj);
    active.push(newId); // newId 恒大于现存 id，序列仍升序
  }

  /** 割成 k 簇：施加前 n−k 次合并（并查集），返回按最小成员排序的成员分组 */
  const cutAt = (k) => {
    if (!Number.isInteger(k) || k < 1 || k > n) {
      throw new Error(`hierarchicalCluster: k 必须为 [1, ${n}] 的整数，收到 ${k}`);
    }
    const uf = Array.from({ length: n }, (_, i) => i);
    const find = (x) => {
      while (uf[x] !== x) { uf[x] = uf[uf[x]]; x = uf[x]; }
      return x;
    };
    for (let m = 0; m < n - k; m++) {
      const ra = find(repOf[merges[m].a]);
      const rb = find(repOf[merges[m].b]);
      if (ra !== rb) uf[ra] = rb;
    }
    const groups = new Map();
    for (let i = 0; i < n; i++) {
      const r = find(i);
      if (!groups.has(r)) groups.set(r, []);
      groups.get(r).push(i);
    }
    return [...groups.values()].sort((g1, g2) => g1[0] - g2[0]);
  };

  return {
    linkage, n, merges, distances,
    labels: (k) => {
      const groups = cutAt(k);
      const labels = new Array(n).fill(0);
      groups.forEach((g, ci) => {
        for (const i of g) labels[i] = ci;
      });
      return labels;
    },
    clusters: (k) => cutAt(k),
  };
}

/**
 * Cophenetic 距离矩阵：d_c(i,j) = i 与 j 首次被合并进同一簇时的合并高度。
 * @param {{a:number,b:number,distance:number}[]} merges hierarchicalCluster 的合并序列
 * @param {number} n 原始样本数
 * @returns {number[][]} n×n 对称矩阵（对角为 0）
 */
export function hierarchicalCophenetic(merges, n) {
  const members = Array.from({ length: 2 * n - 1 }, (_, id) => (id < n ? [id] : null));
  const C = zeros(n, n);
  merges.forEach((m, step) => {
    const newId = n + step;
    const left = members[m.a];
    const right = members[m.b];
    members[newId] = left.concat(right).sort((a, b) => a - b);
    for (const i of left) {
      for (const j of right) {
        C[i][j] = m.distance;
        C[j][i] = m.distance;
      }
    }
  });
  return C;
}

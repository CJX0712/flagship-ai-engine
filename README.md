# 旗舰AI引擎 · Flagship AI Engine

> 零依赖 · 可验证 · 世界顶级 AI 算法集大成引擎
> 浏览器 + Node 双跑 · 作者：晨星 (CJX0712)

[![CI](https://github.com/CJX0712/flagship-ai-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/CJX0712/flagship-ai-engine/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Zero Dependency](https://img.shields.io/badge/dependencies-0-brightgreen.svg)](package.json)
[![Self-Test](https://img.shields.io/badge/self--test-8%2F8%20green-success.svg)](test/run.mjs)

---

## 旗舰宣言

- **零重写 · 集大成**：每一行代码都站在世界顶级成果之上。数学来自 Numerical Recipes / Golub & Van Loan；机器学习来自 ESL / Bishop；深度学习来自 Attention Is All You Need。凡有更强现成方案，立即替换自写。
- **可验证 · 不变量驱动**：每条算法至少带 1 条"可验证不变量"——解析解 vs 数值解、梯度检验、责任度和 1、KL≥0…… 浏览器一键自检 + 无头 Node `8/8` 全绿。
- **闭环交付**：完整开源、可运行、可复现、可验证。

## 快速开始

```bash
# 克隆
git clone https://github.com/CJX0712/flagship-ai-engine.git
cd flagship-ai-engine

# 运行不变量自检（无需安装任何依赖）
node test/run.mjs
# 期望输出：PASS 8/8 — 全部不变量自检通过
```

**旗舰展示页**：直接双击 `showcase/index.html`（引擎已内联，零服务器、零网络），
或 `node scripts/serve.mjs` 后访问 http://localhost:8080 。
修改 `src/` 后用 `node scripts/bundle.mjs` 重新内联打包（自动 `node --check` 语法校验）。

## 架构总览

（架构师高见远的设计文档见 `ARCHITECTURE.md`，数学证明与不变量见 `MATH.md`。）

```
数学基础层   core/  linalg · numerical · stats
经典机器学习层 ml/    regression · classification · clustering · dimreduction · generative · tree
深度学习层   nn/    denseNet · attention · transformerBlock
验证层       verify/ 每条算法的不变量自检
展示层       showcase/ 单文件 HTML 旗舰页
```

## 覆盖算法

| 层 | 模块 | 关键不变量 |
|---|---|---|
| 线性代数 | matmul / eig(Jacobi) / svd / cholesky / qr | 正交 QᵀQ=I，Jacobi 残差 ‖Av−λv‖≈ε |
| 数值 | logsumexp / softmax / powerIteration | 数值稳定，主特征与解析一致 |
| 概率 | gaussianPDF / klDivergence / entropy | KL≥0 且 =0⟺p=q，H≤log|X| |
| 经典ML | 线性回归 / 逻辑回归 / kNN / kmeans / PCA / GMM / 朴素贝叶斯 / 决策树 | 残差⊥X，SSE 单调不增，责任度和 1 |
| 深度学习 | denseNet(梯度检验) / attention / multiHead / transformerBlock | 梯度检验 maxRelErr≈1e-7 |

## 验证

```bash
node test/run.mjs
# 期望输出：PASS 8/8 — 全部不变量自检通过
```

## 许可证

[MIT](LICENSE) © 2026 晨星 (Morning Star / CJX0712)

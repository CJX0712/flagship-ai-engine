# 旗舰AI引擎 · Flagship AI Engine

> 零依赖 · 可验证 · 世界顶级 AI 算法集大成引擎
> 浏览器 + Node 双跑 · 作者：晨星 (CJX0712)

[![CI](https://github.com/CJX0712/flagship-ai-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/CJX0712/flagship-ai-engine/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Zero Dependency](https://img.shields.io/badge/dependencies-0-brightgreen.svg)](package.json)
[![Self-Test](https://img.shields.io/badge/self--test-14%2F14%20green-success.svg)](test/run.mjs)
[![Release](https://img.shields.io/badge/release-v1.1.0-blue.svg)](https://github.com/CJX0712/flagship-ai-engine/releases)

---

## 旗舰宣言

- **零重写 · 集大成**：每一行代码都站在世界顶级成果之上。数学来自 Numerical Recipes / Golub & Van Loan；机器学习来自 ESL / Bishop；深度学习来自 Attention Is All You Need；循环网络来自 Werbos 1990；SVM 来自 Cortes & Vapnik 1995；集成学习来自 Freund & Schapire 1997；HMM 来自 Baum & Petrie 1966；强化学习来自 Bellman 1957。凡有更强现成方案，立即替换自写。
- **可验证 · 不变量驱动**：每条算法至少带 1 条"可验证不变量"——解析解 vs 数值解、梯度检验、责任度和 1、KL≥0、EM 单调性、Bellman 残差→0…… 浏览器一键自检 + 无头 Node `14/14` 全绿。
- **闭环交付**：完整开源、可运行、可复现、可验证。

## 快速开始

```bash
# 克隆
git clone https://github.com/CJX0712/flagship-ai-engine.git
cd flagship-ai-engine

# 运行不变量自检（无需安装任何依赖）
node test/run.mjs
# 期望输出：PASS 14/14 — 全部不变量自检通过
```

**旗舰展示页**：直接双击 `showcase/index.html`（引擎已内联，零服务器、零网络），
或 `node scripts/serve.mjs` 后访问 http://localhost:8080 。
修改 `src/` 后用 `node scripts/bundle.mjs` 重新内联打包（自动 `node --check` 语法校验）。

## 架构总览

（架构师高见远的设计文档见 `ARCHITECTURE.md`，数学证明与不变量见 `MATH.md`。）

```
数学基础层   core/  linalg · numerical · stats
经典机器学习层 ml/    regression · classification · clustering · dimreduction · svm · boosting · hmm · tree
深度学习层   nn/    denseNet · attention · transformerBlock · rnn · lstm
强化学习层   rl/    valueIteration · policyEvaluation · gridworld
验证层       verify/ 每条算法的不变量自检（14 组套件）
展示层       showcase/ 单文件 HTML 旗舰页
```

## 覆盖算法

| 层 | 模块 | 关键不变量 |
|---|---|---|
| 线性代数 | matmul / eig(Jacobi) / svd / cholesky / qr(Householder) | 正交 QᵀQ=I，Jacobi 残差 ‖Av−λv‖≈ε |
| 数值 | logsumexp / softmax / powerIteration | 数值稳定，主特征与解析一致 |
| 概率 | gaussianPDF / klDivergence / entropy | KL≥0 且 =0⟺p=q，H≤log\|X\| |
| 经典ML | 线性/岭/逻辑回归 · kNN · kmeans++ · GMM-EM · PCA · 朴素贝叶斯 · CART 决策树 | 残差⊥X，SSE 单调不增，责任度和 1，Gini 凸性 |
| 核方法 | RBF 核 SVM（SMO 对偶求解） | Σαᵢyᵢ=0，KKT 互补松弛，可分数据 acc=1 |
| 集成学习 | AdaBoost（决策树桩加权集成） | 指数损失单调不增，训练误差 ≤ Π2√(ε(1−ε)) |
| 概率图模型 | HMM 前向/后向 + Baum-Welch | 前向 ≡ 729 路径暴力枚举，对数似然单调不减（EM 定理） |
| 深度学习 | denseNet · attention · multiHead · transformerBlock · RNN(BPTT) · LSTM(BPTT) | 梯度检验 maxRelErr<1e-6（解析 ≡ 中心差分） |
| 强化学习 | 值迭代 · 策略评估 · 4×4 随机网格世界 | Bellman 残差→0，V^{π*}≡V*，V*=r/(1−γ) |

## 验证

```bash
node test/run.mjs
# 期望输出：PASS 14/14 — 全部不变量自检通过
```

14 组套件覆盖 70+ 条硬不变量。CI 在 Node 16/18/20 三个版本上运行同一自检。

## 许可证

[MIT](LICENSE) © 2026 晨星 (Morning Star / CJX0712)

# 旗舰AI引擎 · Flagship AI Engine

> 零依赖 · 可验证 · 世界顶级 AI 算法集大成引擎
> 浏览器 + Node 双跑 · 作者：晨星 (CJX0712)

[![CI](https://github.com/CJX0712/flagship-ai-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/CJX0712/flagship-ai-engine/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Zero Dependency](https://img.shields.io/badge/dependencies-0-brightgreen.svg)](package.json)
[![Self-Test](https://img.shields.io/badge/self--test-54%2F54%20green-success.svg)](test/run.mjs)
[![Release](https://img.shields.io/badge/release-v1.11.0-blue.svg)](https://github.com/CJX0712/flagship-ai-engine/releases)

---

## 旗舰宣言

- **零重写 · 集大成**：每一行代码都站在世界顶级成果之上。数学来自 Numerical Recipes / Golub & Van Loan；机器学习来自 ESL / Bishop；深度学习来自 Attention Is All You Need；循环网络来自 Werbos 1990；SVM 来自 Cortes & Vapnik 1995；集成学习来自 Freund & Schapire 1997；HMM 来自 Baum & Petrie 1966；强化学习来自 Bellman 1957；免模型控制来自 Watkins 1989；密度聚类来自 Ester et al. 1996；流形嵌入来自 van der Maaten & Hinton 2008；自适应优化来自 Kingma & Ba 2015；核主成分来自 Schölkopf et al. 1998；谱聚类来自 Shi & Malik 2000 / Ng-Jordan-Weiss 2002；现代激活来自 Hendrycks & Gimpel 2016（GELU）；学习率调度来自 Vaswani et al. 2017（Noam）与 Loshchilov & Hutter 2016（SGDR）；归一化来自 Ioffe & Szegedy 2015（BatchNorm）/ Ba 2016（LayerNorm）；残差网络来自 He et al. 2015；网页排名来自 Page & Brin 1998；吉布斯采样来自 Geman & Geman 1984 / Casella & George 1992；二次判别来自 Fisher 1936 / Bishop §4.3；稀疏回归来自 Tibshirani 1996 / glmnet 2010；受限玻尔兹曼机来自 Smolensky 1986 / Hinton 2002；联想记忆来自 Hopfield 1982 / Amit-Gutfreund-Sompolinsky 1985；状态估计平滑来自 Rauch-Tung-Striebel 1965（RTS 平滑）/ Gordon 1993（粒子滤波 SIR）；同策略强化学习来自 Rummery & Niranjan 1994（SARSA）；核密度爬山来自 Fukunaga 1975 / Cheng 1995（Mean Shift 梯度上升 + 单调收敛）；门控循环单元来自 Cho et al. 2014（GRU）；多类逻辑回归来自 Bishop §4.3.4；期望 SARSA 来自 van Seijen et al. 2009；弹性网络来自 Zou & Hastie 2005 / Friedman 2010（glmnet）。凡有更强现成方案，立即替换自写。
- **可验证 · 不变量驱动**：每条算法至少带 1 条"可验证不变量"——解析解 vs 数值解、梯度检验、责任度和 1、KL≥0、EM 单调性、Bellman 残差→0、KKT 互补松弛、ARI=1、困惑度达成、Adam 首步闭式、线性核 KPCA ≡ 经典 PCA、谱聚类两月亮 ARI=1.000000、GELU/SiLU/Mish 导数 ≡ 中心差分、Noam 两支在 warmup 顶点精确连续、QDA 同协方差边界退化为线性、Lasso KKT 最优性证书、RBM 自由能 ≡ 配分函数枚举、Hopfield 能量每步不增、GRU BPTT 梯度双判据、Softmax K=2≡logistic、Expected SARSA ε=0≡Q-learning、Elastic Net 退化≡Ridge/Lasso/OLS…… 浏览器一键自检 + 无头 Node `54/54` 全绿。
- **闭环交付**：完整开源、可运行、可复现、可验证。

## 快速开始

```bash
# 克隆
git clone https://github.com/CJX0712/flagship-ai-engine.git
cd flagship-ai-engine

# 运行不变量自检（无需安装任何依赖）
node test/run.mjs
# 期望输出：PASS 54/54 — 全部不变量自检通过
```

**旗舰展示页**：直接双击 `showcase/index.html`（引擎已内联，零服务器、零网络），
或 `node scripts/serve.mjs` 后访问 http://localhost:8080 。
修改 `src/` 后用 `node scripts/bundle.mjs` 重新内联打包（自动 `node --check` 语法校验）。

## 架构总览

（架构师高见远的设计文档见 `ARCHITECTURE.md`，数学证明与不变量见 `MATH.md`。）

```
数学基础层   core/  linalg · numerical · stats(twoMoons)
经典机器学习层 ml/    regression · classification · clustering · dimreduction · dbscan · tsne · kpca · spectral · svm · boosting · hmm · tree · kalmanSmoother · particleFilter · meanShift
深度学习层   nn/    denseNet · attention · transformerBlock · rnn · lstm · optimizers · activations · schedulers
强化学习层   rl/    valueIteration · policyEvaluation · qLearning · sarsa · gridworld
验证层       verify/ 每条算法的不变量自检（54 组套件）
展示层       showcase/ 单文件 HTML 旗舰页
```

## 覆盖算法

| 层 | 模块 | 关键不变量 |
|---|---|---|
| 线性代数 | matmul / eig(Jacobi) / svd / cholesky / qr(Householder) | 正交 QᵀQ=I，Jacobi 残差 ‖Av−λv‖≈ε |
| 数值 | logsumexp / softmax / powerIteration | 数值稳定，主特征与解析一致 |
| 概率 | gaussianPDF / klDivergence / entropy / twoMoons | KL≥0 且 =0⟺p=q，H≤log\|X\| |
| 经典ML | 线性/岭/逻辑回归 · kNN · kmeans++ · GMM-EM · PCA · 朴素贝叶斯 · CART 决策树 | 残差⊥X，SSE 单调不增，责任度和 1，Gini 凸性 |
| 密度聚类 | DBSCAN（ε-邻域 + 密度可达） | ARI=1，核心点见证，密度连通传递性，离群稳健 |
| 非线性降维 | t-SNE（困惑度二分 + Student-t 嵌入） | P/Q 对称联合分布 Σ=1，困惑度达成，KL 单调不增，kNN 重叠 ≥0.85 |
| 核降维 | Kernel PCA（Schölkopf 1998，特征空间中心化 + 核技巧） | 线性核 ≡ 经典 PCA（\|cos\|=1），λ(K̃)=(n−1)λ(C)，谱定理重构，样本外自洽 |
| 谱聚类 | NJW（归一化拉普拉斯 + kNN RBF 图） | λ_min=0 且特征向量 ∝ √d/‖√d‖，谱界 [0,2]，两月亮 ARI=1.000000（k-means 对照 0.24） |
| 核方法 | RBF 核 SVM（SMO 对偶求解） | Σαᵢyᵢ=0，KKT 互补松弛，可分数据 acc=1 |
| 集成学习 | AdaBoost（决策树桩加权集成） | 指数损失单调不增，训练误差 ≤ Π2√(ε(1−ε)) |
| 概率图模型 | HMM 前向/后向 + Baum-Welch | 前向 ≡ 729 路径暴力枚举，对数似然单调不减（EM 定理） |
| 深度学习 | denseNet · attention · multiHead · transformerBlock · RNN(BPTT) · LSTM(BPTT) | 梯度检验 maxRelErr<1e-6（解析 ≡ 中心差分） |
| 激活函数 | GELU(x·Φ(x)) · SiLU/Swish · Mish · HardSwish · Softplus（erf 用 A&S 7.1.26） | erf 奇对称，softplus(x)−softplus(−x)=x，解析导数 ≡ 中心差分 |
| 学习率调度 | Noam(warmup+逆√) · SGDR 余弦 · 阶梯 · 指数 · 线性 warmup | Noam 两支在 s=W 精确连续，SGDR 端点闭式，指数逐步等比 |
| 卷积网络 | CNN（im2col+GEMM 卷积 / 反向传播 / 最大池化路由） | 输出尺寸闭式；im2col ≡ 滑动窗口；平移等变性；dX/dW/db 梯度 ≡ 中心差分 |
| 高斯过程 | GPR（RBF 核 + Cholesky + 边际似然 + 超参数梯度） | LLᵀ≡K；无噪声极限插值；后验方差≤先验且远点→σ_f²；解析梯度 ≡ 差分（R&W 5.9） |
| 梯度提升 | GBDT（回归树基学习器 + 负梯度 + 学习率收缩） | 训练 MSE 单调不增；树 ≡ 残差正交投影（⟨h,r⟩=‖h‖²）；lr→0 ⇒ 预测 ≡ ȳ |
| 状态空间 | 卡尔曼滤波（Joseph 协方差更新 + DARE 稳态） | K=P⁻H/(HP⁻H+R)；新息白噪声；P⁻ 收敛到 Riccati 解析解；协方差恒半正定 |
| 深度生成 | VAE（重参数化 + ELBO + 解析 KL） | KL 解析闭式 ≡ 蒙特卡洛；KL≥0 且后验≡先验时为 0；重参数化梯度 ≡ 中心差分 |
| 贝叶斯推断 | MCMC（Metropolis-Hastings + 重要性采样） | 未归一化密度即可采样；经验矩 ≡ 解析矩；细致平衡；接受率随尺度单调 |
| 表示学习 | Word2Vec SGNS（Skip-gram + 负采样） | 解析梯度 ≡ 中心差分；σ(−x)=1−σ(x)；训练后共现相似度 ≫ 非共现 |
| 非负分解 | NMF（Lee-Seung 乘性更新） | 重构误差单调不增（定理）；非负性保持；≥ 同秩 SVD 最优误差（Eckart–Young） |
| 一阶优化 | SGD-Momentum · RMSProp · Adam（统一 step 接口 + minimize） | Adam 首步 Δ=−lr·sign(g) 且尺度不变，动量速度闭式，二次型收敛到 x*=A⁻¹b |
| 强化学习 | 值迭代 · 策略评估 · Q-learning · 4×4 随机网格世界 | Bellman 残差→0，V^{π*}≡V*，V*=r/(1−γ)，Q≡qFromV(V*) |
| 归一化层 | BatchNorm（Ioffe & Szegedy 2015）· LayerNorm（Ba 2016） | γ=1,β=0 ⇒ 逐通道均值≈0、方差=1（<1e-9）；running stats 精确 EMA；解析梯度 ≡ 中心差分 |
| 残差网络 | ResNet（He et al. 2015，残差 MLP 块 + 分类头 + useSkip 对照） | 零化分支 ⇒ forward≡x、Jacobian=I；全参数解析梯度 ≡ 中心差分；两月亮训练收敛 acc>0.99；梯度高速公路（残差回传范数=√d、普通=0） |
| 网页排名 | PageRank（Page & Brin 1998，d=0.85 阻尼 + 悬挂节点质量重分配） | 秩和 ≡ 1；平稳方程 ‖G·r−r‖∞→0；幂迭代 ≡ 精确线性解 (I−dS)r=(1−d)/n·1；残差单调收敛；含悬挂节点仍守恒 |
| 吉布斯采样 | Gibbs（Geman & Geman 1984 / Casella & George 1992，二元高斯闭式条件） | 经验均值 ≡ μ（偏差<0.05）；协方差 ≡ Σ（rel<0.08）；单坐标边际 ≡ 一元高斯；条件分布 ≡ 理论；离散经验联合 ≡ 真实（Σ=1）；平稳收敛 |
| 高斯混合 | GMM-EM（Dempster 1977 / Bishop §9，软指派 + 协方差 PSD 正则） | 责任度行和=1；权重和=1；协方差对称半正定；对数似然单调不降；强分离双高斯硬指派 acc=1 |
| 线性判别 | LDA / Fisher（Fisher 1936 / Rao 1948，闭式 Sw⁻¹(μ₁−μ₂)） | 类心 ≡ 暴力均值；Sw/Sb 对称；Fisher 方向恒等式 w∥Sw⁻¹(μ₁−μ₂)；Fisher 比 ≥ 随机方向；线性可分 acc=1 |
| 朴素贝叶斯 | 高斯朴素贝叶斯（Maron 1961 / Domingos&Pazzani 1997，对数域 + softmax） | 先验和=1；后验行和=1；独立可分 acc=1；对数项 ≡ 手算；labels=argmax |
| 集成学习（扩展） | Random Forest（Breiman 2001，bootstrap + mtry 随机特征 + 多数投票） | 特征重要性 Σ=1；twoMoons acc≥0.95；OOB 误差<0.1；方差低于单树 |
| 基础神经网络 | 感知机 Perceptron（Rosenblatt 1958 / Novikoff 1962） | 可分数收敛 min yᵢ(w·xᵢ+b)>0、0 训练错误；mistakes ≤ (R/γ)²；单步更新确定性；XOR 有界停止 |
| 盲源分离 | ICA / FastICA（Hyvärinen & Oja 2000，白化 + 非高斯性最大化） | 白化协方差≡I；unmixing 行正交；源恢复 \|corr\|>0.99（Comon 可识别性）；非高斯性增强；源间去相关 |
| 多变量分析 | CCA 典型相关（Hotelling 1936） | r_k∈[0,1] 且非增；Cov(U,V)=diag(r)；完美相关 r₁≡1；独立数据 r≈0；白化+SVD 交叉验证一致 |
| 层次聚类 | Hierarchical（Sokal & Michener 1958 / Lance & Williams 1967） | 合并高度单调非降；n−1 次合并；强分离 3 团 ARI=1.000；Lance-Williams 递推≡闭式；cophenetic≥0.9 |
| 二次判别 | QDA（Fisher 1936 / Bishop §4.3，各类完整协方差 + Cholesky logDet） | 类心≡暴力均值；Σ_c 对称≡无偏估计；后验行和=1；log-后验≡独立实现；可分 acc=1；同协方差边界退化为线性（≡LDA） |
| 稀疏回归 | Lasso（Tibshirani 1996 / glmnet 2010，坐标下降 + 软阈值） | KKT 最优性证书；正交设计≡OLS 软阈值闭式；λ=0≡OLS；λ* 全零；路径 ‖w‖₁ 单调；欠定 ‖w‖₀≤min(n,d) |
| 能量模型 | RBM（Smolensky 1986 / Hinton 2002 CD-k，二部图 + 自由能） | 自由能≡配分函数枚举（256 态 2.6e-17）；P(h\|v)≡σ(Wᵀv+c)；CD 更新恒等式；训练后 FE 间隔>0；重构误差下降 |
| 联想记忆 | Hopfield（Hopfield 1982 / AGS 1985，Hebbian 存储 + 异步更新） | Lyapunov 能量每步不增（300 步 max ΔE=0）；存储模式是不动点；W 对角=0 对称；25% 噪声收敛；局部极小 |
| 状态估计 | 卡尔曼平滑 RTS（Rauch-Tung-Striebel 1965，反向递推增益） | 端点闭合 x₀^smooth=x₀^filt；P 全程半正定；增益恒等式 J=P_k\|k FᵀP_{k+1\|k}⁻¹；常值模型闭式一致；反向递推自洽；RMSE 序（平滑≤滤波） |
| 贝叶斯滤波 | 粒子滤波 SIR（Gordon 1993，log 权重 + 系统重采样 + ESS） | 权重归一 Σw=1；收敛邻卡尔曼（meanDiff<0.01）；ESS≤N 且重采样后=1；系统重采样守恒；退化抑制；同 seed 一致 |
| 核密度聚类 | Mean Shift（Fukunaga 1975 / Cheng 1995，KDE 梯度上升 + 模态） | 梯度恒等式 m(x)−x=h²∇f̂/f̂（maxRelErr<1e-8）；h→∞ 退化为质心；单调收敛；不动点；单峰唯一；两团分离 |
| 同策略强化学习 | SARSA / SARSA(λ)（Rummery & Niranjan 1994，在策略 TD + 替换迹） | ε=0≡Q-learning；λ=0≡单步 SARSA；GLIE 收敛 V→V*；在策略 Bellman 残差→0；资格迹替换+γλ 衰减；同 seed 一致 |
| 循环网络 | GRU（Cho et al. 2014，双门 r/z + 候选隐态） | BPTT 梯度 ≡ 中心差分（双判据 maxRelErr<1e-6）；门控闭合 z≡1⇒h=h̃、z≡0⇒h_{t−1}；单步闭式；重置门隔离；训练收敛；同 seed 一致 |
| 多类分类 | Softmax Regression（Bishop §4.3.4，softmax + 交叉熵） | 解析梯度 ≡ 中心差分；K=2≡logisticRegression；Σ_k p=1；交叉熵≥0 且单调不增；同 seed 一致 |
| 离策略期望 TD | Expected SARSA（van Seijen et al. 2009，对 π 求期望） | ε=0 单步≡Q-learning（1e-15）；期望≡dot(π,Q)；GLIE 收敛到 Q*；同策略期望 Bellman 残差→0；后期回报>前期；同 seed 一致 |
| 弹性正则 | Elastic Net（Zou & Hastie 2005，L1+L2 坐标下降） | λ1=0≡Ridge；λ2=0≡Lasso；皆0≡OLS；不动点 w_j=S(ρ_j,λ1)/(z_j+λ2)；KKT 证书；目标单调不增 |

## 展示页实时演示（39 个）

k-means 聚类 · 注意力热力图 · 核 SVM 决策边界 · AdaBoost 集成边界 · RL 网格世界（值迭代）
· LSTM 序列回归 · t-SNE 嵌入 · DBSCAN 含噪声识别 · 优化器等势面轨迹
· **谱聚类 vs k-means（两月亮，图边可视化 + ARI 对照）** · 现代激活函数族（GELU/SiLU/Mish 函数/导数视图） · **高斯过程回归（后验均值 + 95% 置信带）**
· 卡尔曼滤波（真值 vs 观测 vs 滤波估计） · **MCMC 采样分布（直方图 vs 理论密度）**
· **Word2Vec 词向量（SGNS 训练后 PCA 投影）**
· **归一化层（BatchNorm 逐通道 σ 前/后对照 + LayerNorm 单样本方差≈1）** · **残差网络 ResNet（两月亮决策边界 + 准确率）**
· **网页排名 PageRank（有向图节点半径 ∝ 重要性，含悬挂节点）** · **吉布斯采样 Gibbs（二元高斯样本云 + 理论 2σ 等值线）**
· **高斯混合模型 GMM（两高斯团软指派配色 + 聚类中心 + EM 对数似然单调）** · **线性判别 LDA（Fisher 投影方向 + 类心分离）**
· **朴素贝叶斯（二次决策面 + 后验和=1）** · **随机森林 Random Forest（twoMoons 决策边界 + 64 棵树 + 特征重要性）**
· **感知机 Perceptron（线性可分散点 + 决策边界 + mistakes）** · **ICA 盲源分离（混合信号 vs 恢复源波形 + |corr|）**
· **CCA 典型相关（典型变量 U₁–V₁ 散点 + r₁≥r₂）** · **层次聚类（凝聚树状图 + 切 3 簇 ARI）**
· **QDA 二次判别（异协方差团 + 二次决策边界）** · **Lasso 正则化路径（系数随 λ 收缩 + 非零计数）**
· **RBM 受限玻尔兹曼机（CD 训练前后重构热图对比）** · **Hopfield 联想记忆（噪声探针召回 + 能量下降轨迹）**
· **RTS 平滑（卡尔曼前向 + 反向 RTS 平滑，真值 vs 滤波 vs 平滑轨迹 + 协方差带）**
· **粒子滤波 SIR（重采样粒子云 vs 卡尔曼真值 + 有效样本数 ESS 曲线）**
· **Mean Shift 聚类（KDE 梯度上升模态收敛轨迹 + 自适应聚类中心）**
· **SARSA(λ) 同策略强化学习（网格世界策略收敛 + 回合回报曲线）**
· **GRU 序列回归（阶跃+正弦目标，BPTT 训练损失下降）**
· **Softmax 多类逻辑回归（三团分类决策面 + 准确率）**
· **Expected SARSA（chainMDP 回合回报 + 策略收敛）**
· **Elastic Net 稀疏回归（估计系数 vs 真实系数 + 非零计数）**
——全部由本页内联引擎实时计算，种子化随机可复现。

## 验证

```bash
node test/run.mjs
# 期望输出：PASS 54/54 — 全部不变量自检通过
```

54 组套件覆盖 324 条硬不变量。CI 在 Node 16/18/20 三个版本上运行同一自检（精确校验 `PASS 54/54`）。

## 许可证

[MIT](LICENSE) © 2026 晨星 (Morning Star / CJX0712)

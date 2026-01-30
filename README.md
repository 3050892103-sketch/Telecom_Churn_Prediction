# 📉 电信客户流失预警与决策系统
# Telecom Customer Churn Prediction System

<!-- Badges for visual appeal -->
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange?style=flat&logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Deployment-Streamlit-red?style=flat&logo=streamlit)
![Status](https://img.shields.io/badge/Status-Completed-success?style=flat)

> **An End-to-End Data Science Solution: From Mathematical Modeling to Web Deployment.**  
> **端到端数据科学解决方案：从数学建模到 Web 端部署。**

---

## 📑 目录 (Table of Contents)
- [🇨🇳 中文介绍 (Chinese)](#-中文介绍)
  - [1. 项目背景与商业价值](#1-项目背景与商业价值)
  - [2. 数学框架与优化](#2-数学框架与优化)
  - [3. 工程架构](#3-工程架构)
  - [4. 模型表现与洞察](#4-模型表现与洞察)
  - [5. 在线部署 (Web App)](#5-在线部署-web-app)
- [🇺🇸 English Introduction](#-english-introduction)
- [🛠️ 快速开始 (Quick Start)](#%EF%B8%8F-快速开始-quick-start)
- [👨‍💻 作者信息 (Author)](#-作者信息-author)

---

<a name="-中文介绍"></a>
## 🇨🇳 中文介绍

### 1. 项目背景与商业价值
在电信行业，获取新客的成本通常是挽留老客的 **5-10 倍**。本项目不满足于仅仅输出一个“预测结果”，而是构建了一套完整的**风控解决方案**。
我们利用用户行为数据（在网时长、月费、投诉次数），结合逻辑回归算法，旨在提前识别高危流失用户，并提供可解释的干预建议。

*   **核心策略**：打破传统的“高准确率”迷思，通过**阈值移动 (Threshold Tuning)** 技术，将判别阈值下调至 **0.2**，优先保证**高召回率 (Recall)**，确保高风险用户无一漏网。

### 2. 数学框架与优化
本项目深度结合了数学理论与工程实践：

*   **建模假设**：采用逻辑回归，假设 $P(y=1|x) = \sigma(\mathbf{w}^T \mathbf{x} + b)$，其中 $\sigma$ 为 Sigmoid 激活函数。
*   **凸优化**：最小化**对数损失函数 (Log Loss)**，利用 L-BFGS 算法在凸曲面上寻找全局最优解。
    $$ J(\theta) = - \frac{1}{m} \sum_{i=1}^{m} [y^{(i)}\log(h_\theta(x^{(i)})) + (1-y^{(i)})\log(1-h_\theta(x^{(i)}))] $$
*   **几何优化**：针对特征量纲差异（薪资 10k vs 投诉 5次）导致的梯度震荡问题，引入 **Z-Score 标准化**，将损失曲面从“狭长峡谷”优化为“正圆”，加速收敛。

### 3. 工程架构
本项目遵循工业级工程结构，包含数据生成、模型训练、模型持久化与 Web 部署。

```text
Telecom_Churn_Prediction/
├── data/               # 模拟生成的清洗后数据 (Data Source)
├── models/             # 持久化的 .pkl 模型与标准化器 (Serialized Objects)
├── src/                # 核心源代码 (Source Code)
│   ├── data_generation.py  # 数据生成 (ETL)
│   └── train_model.py      # 模型训练与评估 (Training)
├── images/             # 可视化图表资源 (Plots)
├── app.py              # Streamlit Web 应用入口 (Deployment)
└── requirements.txt    # 依赖清单
```
### 4. 模型表现与洞察
*   **ROC-AUC**: **0.85** (优异的泛化能力)
*   **业务洞察**: 权重分析显示，**投诉次数 (ServiceCalls)** 是流失的第一大推手（正系数最大），而**在网时长**则是最强的稳定剂。

![ROC Curve](./images/roc_curve.png)

### 5. 在线部署 (Web App)
本项目包含一个基于 **Streamlit** 的交互式 Web 应用。业务人员可以通过侧边栏调整客户特征，实时获取流失概率与挽留建议。

![Confusion Matrix](./images/confusion_matrix.png)
*(注：此处展示混淆矩阵，实际部署界面可参考 app.py)*

---

<a name="-english-introduction"></a>
## 🇺🇸 English Introduction

### 1. Business Context
In the telecom industry, customer acquisition costs are significantly higher than retention. This project delivers an **End-to-End Data Science Solution** to predict customer churn.
**Strategic Goal**: Instead of chasing pure accuracy, we utilize **Threshold Tuning** (setting threshold to **0.2**) to maximize **Recall**. This ensures that high-risk customers are identified for proactive retention campaigns.

### 2. Mathematical Framework
*   **Model**: Logistic Regression with Sigmoid activation.
*   **Optimization**: Minimizing **Log-Loss** (Cross-Entropy) via L-BFGS algorithm.
*   **Geometric Optimization**: Applied **StandardScaler** to normalize feature scales. This transforms the loss surface from an elongated valley to a spherical shape, ensuring efficient Gradient Descent convergence.

### 3. Engineering Pipeline
The project follows a modular engineering structure, separating Data ETL, Training, Serialization, and Web Deployment logic (see structure above).

### 4. Model Performance
*   **ROC-AUC Score**: **0.85**
*   **Key Insight**: Feature importance analysis reveals that **Service Calls** is the strongest driver for churn, while **Tenure** acts as the strongest retention factor.

### 5. Interactive Deployment
The project includes a **Streamlit Web App** for real-time inference. Stakeholders can adjust customer parameters via sliders and receive instant risk assessments and action recommendations.

---

<a name="%EF%B8%8F-快速开始-quick-start"></a>
## 🛠️ 快速开始 (Quick Start)

### 1. 环境安装 (Installation)
Clone the repository and install dependencies:
```bash
git clone https://github.com/[Your_Username]/Telecom_Churn_Prediction.git
cd Telecom_Churn_Prediction
pip install -r requirements.txt
```

### 2. 复现全流程 (Run Pipeline)

**Step 1: Generate Synthetic Data (生成数据)**
```bash
python src/data_generation.py
```

**Step 2: Train Model & Save Artifacts (训练并保存模型)**
```bash
python src/train_model.py
```
*This will generate `.pkl` files in the `models/` directory and plots in `images/`.*

**Step 3: Launch Web App (启动网页应用)**
```bash
streamlit run app.py
```

---

<a name="-作者信息-author"></a>
## 👨‍💻 作者信息 (Author)

| Item | Details |
| :--- | :--- |
| **Name** | **林明超（Linminchao）** |
| **Major** | 数学与应用数学(Mathematics and Applied Mathematics) |
| **Date** | 2026年1月30日（Jan 30, 2026） |
| **Focus** | Data Modeling, Machine Learning, Mathematical Optimization |

---
*Created with ❤️ by a Math Student pivoting to AI Engineering.*
```
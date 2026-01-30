import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import log_loss, roc_curve, auc

# ==========================================
# 0. 环境设置与数据加载
# ==========================================
# 设置专业绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif' # 防止字体报错
plt.rcParams['axes.unicode_minus'] = False

current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, '..', 'data', 'churn_data.csv')
image_dir = os.path.join(current_dir, '..', 'images')

if not os.path.exists(data_path):
    print("❌ 错误：找不到数据文件！")
    exit()

df = pd.read_csv(data_path)
X = df.drop('Churn', axis=1)
y = df['Churn']

# 数据标准化 (画Loss曲线必须要做！)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print("🎨 开始生成可视化图表...")

# ==========================================
# 图表 1: 损失函数收敛曲线 (Loss Convergence Curve)
# ==========================================
# 数学含义：展示梯度下降(Gradient Descent)如何一步步找到山谷底部的过程
# 我们使用 SGDClassifier (随机梯度下降) 并手动循环来记录 Loss

print("   1. 正在绘制损失函数收敛曲线...")
sgd_clf = SGDClassifier(loss='log_loss', learning_rate='constant', eta0=0.01, random_state=42, warm_start=True)

loss_history = []
epochs = 50

for epoch in range(epochs):
    # partial_fit 允许我们一步一步训练
    sgd_clf.partial_fit(X_train, y_train, classes=np.unique(y))
    # 计算当前的 Log Loss (对数损失)
    y_pred_proba = sgd_clf.predict_proba(X_train)
    loss = log_loss(y_train, y_pred_proba)
    loss_history.append(loss)

plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), loss_history, color='#e74c3c', linewidth=2.5, marker='o', markersize=5)
plt.title('Loss Function Convergence (Gradient Descent)', fontsize=14, fontweight='bold')
plt.xlabel('Iterations (Epochs)', fontsize=12)
plt.ylabel('Log Loss', fontsize=12)
plt.grid(True, alpha=0.3)
plt.annotate('Start Optimization', xy=(1, loss_history[0]), xytext=(1, loss_history[0]+0.01),
             arrowprops=dict(facecolor='black', shrink=0.05))
plt.annotate('Converged', xy=(epochs, loss_history[-1]), xytext=(epochs-4, loss_history[-1]+0.01),
             arrowprops=dict(facecolor='black', shrink=0.05))

save_path = os.path.join(image_dir, 'loss_curve.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"      -> 已保存: {save_path}")

# ==========================================
# 图表 2: ROC 曲线 (ROC Curve)
# ==========================================
# 数学含义：衡量模型区分正负样本的能力。AUC (曲线下积分面积) 越大越好。

print("   2. 正在绘制 ROC 曲线...")
# 重新训练一个标准的逻辑回归用于评估
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
y_score = lr_model.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # 纯随机猜测线
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Recall)')
plt.title('Receiver Operating Characteristic (ROC)', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)

save_path = os.path.join(image_dir, 'roc_curve.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"      -> 已保存: {save_path}")

# ==========================================
# 图表 3: 特征重要性排序 (Feature Importance)
# ==========================================
# 商业含义：告诉老板哪个因素最影响客户流失

print("   3. 正在绘制特征重要性条形图...")
# 获取权重
weights = lr_model.coef_[0]
features = X.columns

# 创建 DataFrame 并排序
importance_df = pd.DataFrame({'Feature': features, 'Weight': weights})
importance_df = importance_df.sort_values(by='Weight', key=abs, ascending=False) # 按绝对值大小排序

plt.figure(figsize=(10, 5))
# 用颜色区分正负：红色代表正相关(促进流失)，绿色代表负相关(抑制流失)
colors = ['#e74c3c' if x > 0 else '#2ecc71' for x in importance_df['Weight']]
sns.barplot(x='Weight', y='Feature', data=importance_df, palette=colors)

plt.title('Feature Importance (Logistic Regression Coefficients)', fontsize=14, fontweight='bold')
plt.xlabel('Weight Impact (Standardized)', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.axvline(0, color='black', linewidth=0.8) # 0 轴线

save_path = os.path.join(image_dir, 'feature_importance.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"      -> 已保存: {save_path}")

print("\n✅ 所有图表绘制完成！请去 images 文件夹查看。")
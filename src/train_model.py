import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# ==========================================
# 1. 路径设置 (连接数据与图片的管道)
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
# 数据源路径
data_path = os.path.join(current_dir, '..', 'data', 'churn_data.csv')
# 图片保存路径
image_path = os.path.join(current_dir, '..', 'images', 'confusion_matrix.png')

# 检查数据是否存在
if not os.path.exists(data_path):
    print("❌ 错误：找不到数据文件！请先运行 src/data_generation.py")
    exit()

# ==========================================
# 2. 加载与清洗
# ==========================================
print("📥 正在读取数据...")
df = pd.read_csv(data_path)

X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ==========================================
# 3. 训练与预测
# ==========================================
print("🚀 正在训练模型...")
model = LogisticRegression(solver='lbfgs')
model.fit(X_train_scaled, y_train)

# 采用高危预警策略 (阈值 0.2)
y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
y_pred_strict = (y_pred_prob > 0.2).astype(int)

# ==========================================
# 4. 评估与保存图片
# ==========================================
acc = accuracy_score(y_test, y_pred_strict)
print(f"✅ 模型准确率: {acc:.2%}")

# 绘制混淆矩阵
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred_strict)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Stay', 'Churn'], yticklabels=['Stay', 'Churn'])
plt.title('Confusion Matrix (Threshold=0.2)')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# 保存图片到 images 文件夹
plt.savefig(image_path)
print(f"🖼️ 混淆矩阵图已保存至: {os.path.abspath(image_path)}")

# 只有在非自动运行时才弹窗，防止卡死
# plt.show()
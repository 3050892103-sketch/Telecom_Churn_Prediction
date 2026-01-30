import pandas as pd
import numpy as np
import os

# 1. 动态获取路径 (无论你在哪运行，都能找到 data 文件夹)
# 当前脚本位置: .../Telecom_Churn_Prediction/src/data_generation.py
# 目标保存位置: .../Telecom_Churn_Prediction/data/churn_data.csv
current_dir = os.path.dirname(os.path.abspath(__file__))
# '..' 代表上一级目录
output_path = os.path.join(current_dir, '..', 'data', 'churn_data.csv')
# 2. 上帝视角造数据 (逻辑与之前一致)
print("🎲 正在模拟生成电信用户数据...")
np.random.seed(42)
n_samples = 1000

tenure = np.random.randint(1, 72, n_samples)
monthly_charge = np.random.randint(20, 120, n_samples)
service_calls = np.random.randint(0, 6, n_samples)

# 定义数学规律
z = -0.08 * tenure + 0.03 * monthly_charge + 0.8 * service_calls - 2
prob = 1 / (1 + np.exp(-z))
churn_labels = (prob > np.random.rand(n_samples)).astype(int)

# 3. 封装与保存
df = pd.DataFrame({
    'Tenure': tenure,
    'MonthlyCharge': monthly_charge,
    'ServiceCalls': service_calls,
    'Churn': churn_labels
})

# 保存到 data 文件夹
df.to_csv(output_path, index=False)
print(f"✅ 数据已生成并保存至: {os.path.abspath(output_path)}")
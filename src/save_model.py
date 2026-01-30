import pandas as pd
import numpy as np
import os 
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
current_dir=os.path.dirname(os.path.abspath(__file__))
data_path=os.path.join(current_dir,'..','data','churn_data.csv')
models_dir=os.path.join(current_dir,'..','models')
os.makedirs(models_dir,exist_ok=True)
print("🍳 正在重新训练模型...")
df=pd.read_csv(data_path)
X=df.drop('Churn',axis=1)
y=df['Churn']
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
model=LogisticRegression(solver='lbfgs')
model.fit(X_scaled,y)
print("🧊 正在保存模型文件...")
model_path=os.path.join(models_dir,'logistic_model.pkl')
joblib.dump(model,model_path)
scaler_path=os.path.join(models_dir,'scaler.pkl')
joblib.dump(scaler,scaler_path)
print(f"✅ 保存成功！\n模型位置: {model_path}\n标准化器位置: {scaler_path}")
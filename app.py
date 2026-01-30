import streamlit as st
import joblib
import numpy as np
import os
# ============================================
# 1. 餐厅开门：加载冷冻的模型 (Load Models)
# ============================================
# 这是一个“缓存”技巧。
# @st.cache_resource 意味着：模型只加载一次，不用每次刷新网页都重新读硬盘，速度更快。
@st.cache_resource
def load_assets():
    # 获取绝对路径，防止找不到文件
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'models', 'logistic_model.pkl')
    scaler_path = os.path.join(current_dir, 'models', 'scaler.pkl')
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

try:
    model, scaler = load_assets()
except FileNotFoundError:
    st.error("❌ 错误：找不到模型文件！请先运行 src/save_model.py")
    st.stop()

# ============================================
# 2. 装修店面：侧边栏与标题 (UI Layout)
# ============================================
# 网页标题
st.title("🔮 电信客户流失预警系统")
st.markdown("### AI 智能风控终端")
st.divider() # 画一条分割线

# 侧边栏 (Sidebar)：用来放输入框
st.sidebar.header("📝 请输入客户特征")

# 输入框 1：在网时长 (Tenure)
# 这里的 72 是最大值，1 是最小值，12 是默认值
tenure = st.sidebar.slider("在网时长 (月)", 1, 72, 12)

# 输入框 2：月租费 (Monthly Charge)
monthly_charge = st.sidebar.number_input("月租费 (元)", 20, 150, 70)

# 输入框 3：投诉次数 (Service Calls)
service_calls = st.sidebar.slider("客服投诉次数", 0, 10, 0)

# ============================================
# 3. 厨房做菜：模型推理 (Inference)
# ============================================
# 当用户修改左边的参数时，这部分代码会自动运行

# 步骤 A: 收集数据
# 变成模型认识的 2D 数组形式 [[x1, x2, x3]]
user_input = np.array([[tenure, monthly_charge, service_calls]])

# 步骤 B: 数据加工 (标准化)
# 必须用之前保存的 scaler，不能用新的！
user_input_scaled = scaler.transform(user_input)

# 步骤 C: AI 预测
# predict_proba 返回 [[不流失概率, 流失概率]]
probability = model.predict_proba(user_input_scaled)[0][1]

# ============================================
# 4. 上菜：展示结果 (Display Results)
# ============================================
st.subheader("📊 预测报告")

# 用两列布局展示关键指标
col1, col2 = st.columns(2)

with col1:
    st.metric("流失概率", f"{probability:.1%}")

with col2:
    if probability > 0.2: # 你的高危阈值
        st.error("⚠️ 高风险客户")
    else:
        st.success("✅ 安全客户")

# 画一个进度条，让概率更直观
st.progress(probability, text="流失可能性")

# 给业务人员的建议
st.write("---")
st.write("**💡 决策建议：**")
if probability > 0.2:
    st.write(f"该客户投诉了 {service_calls} 次，且月租高达 {monthly_charge} 元。")
    st.write("建议立即：**赠送 10GB 流量包 + 客服回访**。")
else:
    st.write("客户状态健康，无需特殊干预。")
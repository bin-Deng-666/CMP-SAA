import streamlit as st
import sys
import os
from PIL import Image
import torch
import time

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

st.set_page_config(page_title="对抗图像测试", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    #MainMenu, footer, header {visibility: hidden;}
    [data-testid="stSidebar"] {display: none !important;}
    .block-container {padding-top: 3rem; padding-bottom: 2rem; max-width: 900px; margin: 0 auto;}
    
    /* 美化标题 */
    h1 {text-align: center; color: #1e293b; font-weight: 700;}
    h3 {color: #334155; font-weight: 600; margin-top: 1.5rem;}
    
    /* 按钮美化 */
    .stButton > button {border-radius: 8px; font-weight: 500;}
    
    /* 输入框美化 */
    .stTextInput > div > div > input, .stSelectbox > div > div > select {border-radius: 6px;}
    
    /* 信息框美化 */
    .stInfo {border-radius: 8px; background-color: #f0f9ff; border-left-color: #0ea5e9;}
    .stSuccess {border-radius: 8px;}
    .stWarning {border-radius: 8px;}
    .stError {border-radius: 8px;}
</style>
""", unsafe_allow_html=True)

# 返回按钮 + 标题
header_col1, header_col2, header_col3 = st.columns([1, 3, 1])
with header_col1:
    if st.button("← 返回", key="back_btn"):
        st.switch_page("home.py")
with header_col2:
    st.title("📊 对抗图像测试")
    st.caption("对比原始图像和对抗图像的模型预测结果")

st.divider()

# 初始化会话状态
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'adversarial_image' not in st.session_state:
    st.session_state.adversarial_image = None
if 'original_answer' not in st.session_state:
    st.session_state.original_answer = None
if 'adversarial_answer' not in st.session_state:
    st.session_state.adversarial_answer = None


# ==================== 步骤1: 加载图像 ====================
st.markdown("### 🖼️ 步骤 1: 加载图像")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**🖼️ 原始图像**")
    uploaded_original = st.file_uploader("上传原始图像", type=['png', 'jpg', 'jpeg'], key="original_uploader")
    if uploaded_original:
        st.session_state.original_image = Image.open(uploaded_original)
        st.image(st.session_state.original_image, use_column_width=True)
    else:
        st.info("💡 请上传原始图像")

with col2:
    st.markdown("**🛡️ 对抗图像**")
    uploaded_adversarial = st.file_uploader("上传对抗图像", type=['png', 'jpg', 'jpeg'], key="adversarial_uploader")
    if uploaded_adversarial:
        st.session_state.adversarial_image = Image.open(uploaded_adversarial)
        st.image(st.session_state.adversarial_image, use_column_width=True)
    else:
        st.info("💡 请上传对抗图像")


# ==================== 步骤2: 输入问题 ====================
if st.session_state.original_image and st.session_state.adversarial_image:
    st.divider()
    st.markdown("### ❓ 步骤 2: 输入测试问题")
    
    question = st.text_area("请输入问题", placeholder="例如: What is in the image?", height=80)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        test_btn = st.button("🔍 开始测试", type="primary", use_container_width=True)
    
    if test_btn:
        if not question:
            st.warning("请输入测试问题")
        else:
            with st.spinner("正在获取模型回答..."):
                # TODO: 调用实际的模型推理
                # 这里模拟测试过程
                time.sleep(2)
                
                # 模拟结果（实际应调用模型推理）
                st.session_state.original_answer = f"[原始图像回答] 这是模型对原始图像的回答：{question}"
                st.session_state.adversarial_answer = f"[对抗图像回答] 这是模型对对抗图像的回答：{question}"
                
                st.success("✅ 测试完成！")
    
    # ==================== 步骤3: 显示结果 ====================
    if st.session_state.original_answer and st.session_state.adversarial_answer:
        st.divider()
        st.markdown("### 📊 步骤 3: 测试结果对比")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🖼️ 原始图像回答**")
            st.info(st.session_state.original_answer)
        
        with col2:
            st.markdown("**🛡️ 对抗图像回答**")
            st.info(st.session_state.adversarial_answer)
        
        # 攻击成功判断
        st.markdown("---")
        result_col1, result_col2, result_col3 = st.columns([1, 2, 1])
        with result_col2:
            if st.session_state.original_answer != st.session_state.adversarial_answer:
                st.success("🎯 攻击成功！模型输出存在差异")
            else:
                st.error("❌ 攻击失败！模型输出相同")
import streamlit as st
import sys
import os
import re
from PIL import Image
import torch
import time
import importlib

_frontend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _frontend_dir not in sys.path:
    sys.path.insert(0, _frontend_dir)
import bootstrap

bootstrap.setup_project_path()

from utils.attack_tool import load_model
from frontend.utils.page_state import (
    PAGE_EVALUATE,
    PAGE_HOME,
    clear_evaluate_state,
    on_page_enter,
)


def extract_vqa_answer(text: str) -> str:
    """从 BLIP2 解码结果中提取 Answer: 之后的回答内容。"""
    if not text:
        return text
    match = re.search(r"Answer:(.*?)(?:Question|$)", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    if "Answer:" in text:
        return text.split("Answer:", 1)[-1].strip()
    return text.strip()


st.set_page_config(page_title="对抗图像测试", layout="wide", initial_sidebar_state="collapsed")

on_page_enter(PAGE_EVALUATE, clear_evaluate_state)

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
        clear_evaluate_state()
        st.session_state["active_page"] = PAGE_HOME
        st.switch_page("home.py")
with header_col2:
    st.title("📊 对抗图像测试")
    st.caption("对比原始图像和对抗图像的模型预测结果")

st.divider()

# ==================== 步骤1: 加载图像 ====================
st.markdown("### 🖼️ 步骤 1: 加载图像")

# 输入图像ID
col1, col2 = st.columns([3, 1])
with col1:
    img_id = st.text_input(
        "图像ID",
        placeholder="例如: 294",
        help="输入要加载的图像ID",
        label_visibility="collapsed",
        key="eval_img_id_input",
    )
with col2:
    load_btn = st.button("📁 加载图像", use_container_width=True, type="primary")

if load_btn:
    if not img_id:
        st.error("请输入图像ID")
    else:
        # 构建图像目录路径
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data", img_id)
        
        # 检查目录是否存在
        if not os.path.exists(data_dir):
            st.error(f"未找到图像目录: {data_dir}")
        else:
            # 加载原始图像
            original_path = os.path.join(data_dir, "original.png")
            adversarial_path = os.path.join(data_dir, "adversarial.png")
            perturbation_path = os.path.join(data_dir, "perturbation_vis.png")
            
            # 检查文件是否存在
            missing_files = []
            if not os.path.exists(original_path):
                missing_files.append("original.png")
            if not os.path.exists(adversarial_path):
                missing_files.append("adversarial.png")
            
            if missing_files:
                st.error(f"目录中缺少以下文件: {', '.join(missing_files)}")
            else:
                # 加载图像
                st.session_state.eval_original_image = Image.open(original_path)
                st.session_state.eval_adversarial_image = Image.open(adversarial_path)
                st.session_state.eval_img_id = img_id
                
                # 尝试加载扰动可视化（可选）
                if os.path.exists(perturbation_path):
                    st.session_state.eval_perturbation_image = Image.open(perturbation_path)
                else:
                    st.session_state.eval_perturbation_image = None
                
                st.session_state.eval_original_answer = None
                st.session_state.eval_adversarial_answer = None
                
                st.success(f"✅ 图像 {img_id} 加载成功！")

# 显示图像
if st.session_state.get("eval_original_image") and st.session_state.get("eval_adversarial_image"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🖼️ 原始图像**")
        st.image(st.session_state.eval_original_image, use_container_width=True)
    
    with col2:
        st.markdown("**🛡️ 对抗图像**")
        st.image(st.session_state.eval_adversarial_image, use_container_width=True)
    
    with col3:
        st.markdown("**🔍 扰动可视化**")
        if st.session_state.get("eval_perturbation_image"):
            st.image(st.session_state.eval_perturbation_image, use_container_width=True)
        else:
            st.info("无扰动可视化")


# ==================== 步骤2: 输入问题 ====================
if st.session_state.get("eval_original_image") and st.session_state.get("eval_adversarial_image"):
    st.divider()
    st.markdown("### ❓ 步骤 2: 输入测试问题")
    
    question = st.text_area(
        "请输入问题",
        placeholder="例如: What is in the image?",
        height=80,
        key="eval_question",
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        test_btn = st.button("🔍 开始测试", type="primary", use_container_width=True)
    
    if test_btn:
        if not question:
            st.warning("请输入测试问题")
        else:
            with st.spinner("正在加载模型并获取回答..."):
                # 加载BLIP2模型（device 需为 int：0=GPU, -1=CPU，与 blip2.py / cma.py 一致）
                device = 0 if torch.cuda.is_available() else -1
                module = importlib.import_module("models.blip2")
                eval_model = load_model(device, module, "blip2")
                
                # 准备输入（与 test_model_loading.py 一致：Question + Answer 前缀）
                question_part, answer_part = eval_model.get_vqa_prompt(question)
                prompt = question_part + answer_part
                
                # 获取原始图像回答
                ori_output = extract_vqa_answer(
                    eval_model.get_outputs(
                        batch_images=[st.session_state.eval_original_image],
                        batch_text=[prompt],
                        max_generation_length=50,
                        num_beams=3,
                        length_penalty=0,
                    )[0]
                )
                
                # 获取对抗图像回答
                adv_output = extract_vqa_answer(
                    eval_model.get_outputs(
                        batch_images=[st.session_state.eval_adversarial_image],
                        batch_text=[prompt],
                        max_generation_length=50,
                        num_beams=3,
                        length_penalty=0,
                    )[0]
                )
                
                st.session_state.eval_original_answer = ori_output
                st.session_state.eval_adversarial_answer = adv_output
                
                st.success("✅ 测试完成！")
    
    # ==================== 步骤3: 显示结果 ====================
    if st.session_state.get("eval_original_answer") and st.session_state.get("eval_adversarial_answer"):
        st.divider()
        st.markdown("### 📊 步骤 3: 测试结果对比")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🖼️ 原始图像回答**")
            st.info(st.session_state.eval_original_answer)
        
        with col2:
            st.markdown("**🛡️ 对抗图像回答**")
            st.info(st.session_state.eval_adversarial_answer)
        
        # 攻击成功判断
        st.markdown("---")
        result_col1, result_col2, result_col3 = st.columns([1, 2, 1])
        with result_col2:
            if st.session_state.eval_original_answer != st.session_state.eval_adversarial_answer:
                st.success("🎯 攻击成功！")
            else:
                st.error("❌ 攻击失败！")
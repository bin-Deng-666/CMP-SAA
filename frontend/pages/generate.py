import streamlit as st
import sys
import os
from PIL import Image
import torch

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.attack_tool import load_dataset, get_subset

st.set_page_config(page_title="对抗图像生成", layout="wide", initial_sidebar_state="collapsed")

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
</style>
""", unsafe_allow_html=True)

# 返回按钮 + 标题
header_col1, header_col2, header_col3 = st.columns([1, 3, 1])
with header_col1:
    if st.button("← 返回", key="back_btn"):
        st.switch_page("home.py")
with header_col2:
    st.title("🛡️ 对抗图像生成")
    st.caption("选择攻击方法并配置参数，生成对抗样本")

st.divider()

# 初始化会话状态
if 'page_initialized' not in st.session_state:
    st.session_state.original_image = None
    st.session_state.adversarial_image = None
    st.session_state.perturbation = None
    st.session_state.img_id = None
    st.session_state.page_initialized = True


# ==================== 步骤1: 选择攻击方法 ====================
st.markdown("### 📋 步骤 1: 选择攻击方法")

method = st.selectbox(
    "选择攻击方法",
    ["请选择...", "跨模态辅助攻击方法", "多维度语义攻击方法"],
    help="跨模态辅助攻击方法: 基于跨模态辅助的对抗攻击\n多维度语义攻击方法: 基于多维度语义最大化的对抗攻击",
    label_visibility="collapsed"
)

# ==================== 步骤2: 配置参数 ====================
if method != "请选择...":
    st.divider()
    st.markdown("### ⚙️ 步骤 2: 配置攻击参数")
    
    # 方法特定参数
    if method == "跨模态辅助攻击方法":
        
        col1, col2 = st.columns(2)
        with col1:
            target_text = st.text_input("指定攻击文本", value="Unknown", help="指定攻击的目标文本")
        with col2:
            prompt_num = st.selectbox("用于训练的文本提示数量", [25, 50, 70, 100], index=0, help="选择训练时使用的文本提示数量")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            adv_length = st.slider("文本辅助后缀长度", min_value=0, max_value=40, value=10, help="对抗文本后缀的长度")
        with col2:
            iters = st.number_input("迭代轮次", min_value=100, max_value=2000, value=500, step=50, help="训练迭代轮次")
        with col3:
            fraction = st.slider("扰动大小限制", min_value=16/255, max_value=32/255, value=16/255, step=1/255, format="%.4f", help="扰动的最大大小限制")
        
        # 高级选项
        with st.expander("高级选项"):
            col1, col2 = st.columns(2)
            with col1:
                img_lr = st.slider("图像扰动学习率", min_value=1/255, max_value=4/255, value=1/255, step=0.5/255, format="%.4f", help="图像扰动的学习率")
            with col2:
                suffix_lr = st.number_input("文本辅助后缀学习率", min_value=0.001, max_value=0.1, value=0.01, step=0.001, format="%.3f", help="文本后缀的学习率")
    
    elif method == "多维度语义攻击方法":
        
        backbones = st.multiselect(
            "选择特征提取器（可多选）",
            options=["B16", "B32", "L336", "Laion"],
            default=["B16"],
            help="选择用于特征提取的CLIP模型"
        )
        
        # 基础参数
        col1, col2, col3 = st.columns(3)
        with col1:
            iterations = st.number_input("迭代周期", min_value=10, max_value=500, value=80, step=10)
        with col2:
            alpha = st.number_input("学习率", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        with col3:
            epsilon = st.number_input("扰动上限", min_value=1, max_value=64, value=16, step=1, help="最大扰动值（像素级）")
        
        # 裁剪类型（多选）
        st.markdown("**裁剪类型**")
        crop_types = st.multiselect(
            "选择裁剪类型（可多选）",
            options=["random", "yolo", "center", "grid", "edge"],
            default=["random", "yolo"],
            help="图像裁剪策略，可多选"
        )
        
        # random 裁剪参数
        if "random" in crop_types:
            st.markdown("*Random 裁剪参数*")
            col1, col2 = st.columns(2)
            with col1:
                num_crops = st.number_input("裁剪数量", min_value=1, max_value=20, value=6, step=1)
            with col2:
                min_crop_size = st.slider("最小裁剪长度", min_value=0.1, max_value=1.0, value=0.5, step=0.05, 
                                         help="相对于图像长宽最小值的裁剪比例")
        
        # yolo 裁剪参数
        if "yolo" in crop_types:
            st.markdown("*YOLO 裁剪参数*")
            col1, col2 = st.columns(2)
            with col1:
                yolo_confidence = st.slider("最小置信度", min_value=0.1, max_value=0.9, value=0.5, step=0.05)
            with col2:
                yolo_min_area = st.slider("最小裁剪面积", min_value=0.01, max_value=0.5, value=0.05, step=0.01,
                                         help="相对于图像整体面积的最小比例")
    
    st.divider()
    
    # ==================== 步骤3: 加载原始图像 ====================
    st.divider()
    st.markdown("### 🖼️ 步骤 3: 加载原始图像")
    
    col1, col2 = st.columns([4, 1])
    with col1:
        img_id = st.text_input("图像ID", placeholder="请输入图像ID，例如: 294", label_visibility="collapsed")
    with col2:
        load_btn = st.button("📂 加载图像", use_container_width=True, type="secondary")
    
    @st.cache_data
    def get_test_dataset():
        """缓存加载测试数据集"""
        _, test_dataset = load_dataset()
        return test_dataset
    
    def load_image_by_id(image_id, dataset):
        """根据图像ID从数据集加载图像"""
        for item in dataset:
            if str(item["image_id"]) == str(image_id):
                return item["image"]
        return None
    
    if load_btn and img_id:
        with st.spinner("正在加载图像..."):
            # 加载数据集
            test_dataset = get_test_dataset()
            
            # 根据ID加载图像
            loaded_image = load_image_by_id(img_id, test_dataset)
            
            if loaded_image is None:
                st.error(f"未找到图像ID: {img_id}")
            else:
                st.session_state.original_image = loaded_image
                st.session_state.img_id = img_id
                st.success(f"✅ 图像 {img_id} 加载成功")
    elif load_btn and not img_id:
        st.warning("请输入图像ID")
    
    # 显示原始图像
    if st.session_state.original_image:
        st.image(st.session_state.original_image, caption="原始图像", use_container_width=True)
    else:
        st.info("请输入图像ID并点击加载按钮")
    
    # ==================== 步骤4: 生成对抗图像 ====================
    if st.session_state.original_image:
        st.divider()
        st.markdown("### 🚀 步骤 4: 生成对抗图像")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            generate_btn = st.button("✨ 开始生成", type="primary", use_container_width=True)
        
        if generate_btn:
            with st.spinner("正在生成对抗图像，请稍候..."):
                # TODO: 调用实际的攻击算法
                # 这里先模拟生成过程
                import time
                time.sleep(2)
                
                # 模拟结果（实际应调用 maximize.py 或 cma.py）
                st.session_state.adversarial_image = st.session_state.original_image
                st.session_state.perturbation = None
                
                st.success("✅ 对抗图像生成完成！")
        
        
        # ==================== 步骤5: 显示结果 ====================
        if st.session_state.adversarial_image:
            st.divider()
            st.markdown("### 📊 步骤 5: 生成结果")
            
            result_col1, result_col2, result_col3 = st.columns(3)
            
            with result_col1:
                st.markdown("**🖼️ 原始图像**")
                st.image(st.session_state.original_image, use_container_width=True)
            
            with result_col2:
                st.markdown("**🛡️ 对抗图像**")
                st.image(st.session_state.adversarial_image, use_container_width=True)
            
            with result_col3:
                st.markdown("**🔍 扰动可视化**")
                if st.session_state.perturbation is not None:
                    st.image(st.session_state.perturbation, use_container_width=True)
                else:
                    st.info("暂不可用")
            
            # 下载按钮
            st.markdown("---")
            download_col1, download_col2, download_col3 = st.columns([1, 2, 1])
            with download_col2:
                # 保存图像到缓冲区
                import io
                buf = io.BytesIO()
                st.session_state.adversarial_image.save(buf, format='PNG')
                buf.seek(0)
                
                # 获取图像ID（如果未输入则使用默认值）
                img_id_val = img_id if img_id else 'image'
                
                st.download_button(
                    label="💾 下载对抗图像",
                    data=buf,
                    file_name=f"adversarial_{img_id_val}.png",
                    mime="image/png",
                    use_container_width=True
                )
            

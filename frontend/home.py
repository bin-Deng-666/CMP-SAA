import streamlit as st
import torch

st.set_page_config(page_title="对抗图像系统", layout="centered", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {padding-top: 3rem; padding-bottom: 3rem; max-width: 900px;}
    
    .title {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        background: #f1f5f9;
        border-radius: 20px;
        font-size: 0.85rem;
        color: #475569;
    }
    
    .badge-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #22c55e;
    }
    
    .card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        transition: all 0.2s;
    }
    
    .card:hover {
        border-color: #667eea;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.1);
        transform: translateY(-2px);
    }
    
    .card-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    .card-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 0.5rem;
    }
    
    .card-desc {
        color: #64748b;
        font-size: 0.95rem;
        line-height: 1.5;
        margin-bottom: 1.5rem;
    }

    /* 按钮与卡片间距 */
    .stButton {
        margin-top: 1rem;
    }
    
    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.8rem;
        margin-top: 4rem;
        padding-top: 2rem;
        border-top: 1px solid #f1f5f9;
    }
</style>
""", unsafe_allow_html=True)

# 标题区
st.markdown('<div class="title">对抗图像系统</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">对抗样本生成与评估平台</div>', unsafe_allow_html=True)

# 状态徽章
cuda_available = torch.cuda.is_available()
status_text = f"GPU 可用 ({torch.cuda.device_count()} 设备)" if cuda_available else "CPU 模式"
st.markdown(f'<div style="text-align: center; margin-bottom: 3rem;"><span class="badge"><span class="badge-dot" style="background: {"#22c55e" if cuda_available else "#f59e0b"};"></span>{status_text}</span></div>', unsafe_allow_html=True)

# 功能卡片
col1, col2 = st.columns(2, gap="medium")

with col1:
    st.markdown("""
    <div class="card">
        <div class="card-icon">🛡️</div>
        <div class="card-title">生成对抗样本</div>
        <div class="card-desc">使用CMA和Maximize方法生成对抗图像</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("开始生成", key="gen", use_container_width=True):
        st.switch_page("pages/generate.py")

with col2:
    st.markdown("""
    <div class="card">
        <div class="card-icon">📊</div>
        <div class="card-title">评估攻击效果</div>
        <div class="card-desc">对比原始图像和对抗图像的模型输出，评估攻击效果</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("开始评估", key="eval", use_container_width=True):
        st.switch_page("pages/evaluate.py")

# 页脚
st.markdown('<div class="footer">毕业设计项目 · ZY2306335 邓彬 · 支持 BLIP2 / InstructBLIP</div>', unsafe_allow_html=True)
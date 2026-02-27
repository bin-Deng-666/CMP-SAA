import streamlit as st

# 设置页面标题和布局
st.set_page_config(page_title="对抗图像系统", layout="wide")

# 创建自定义CSS样式
st.markdown("""
<style>
    .home-container {
        max-width: 1000px;
        margin: 0 auto;
        padding: 60px 40px;
        text-align: center;
    }

    .home-title {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 20px;
        background: linear-gradient(135deg, #4361ee 0%, #3a0ca3 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
    }

    .home-subtitle {
        font-size: 1.2rem;
        color: #4a5568;
        margin-bottom: 60px;
        max-width: 600px;
        margin-left: auto;
        margin-right: auto;
    }

    .option-cards {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 40px;
        margin-bottom: 60px;
    }

    .option-card {
        background: white;
        padding: 40px 30px;
        border-radius: 16px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
        cursor: pointer;
        border: 2px solid transparent;
    }

    .option-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.12);
        border-color: #4361ee;
    }

    .option-card-icon {
        font-size: 3rem;
        margin-bottom: 20px;
        background: linear-gradient(135deg, #4361ee 0%, #3a0ca3 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .option-card-title {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 15px;
        color: #1a1a2e;
    }

    .option-card-description {
        font-size: 1rem;
        color: #4a5568;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# 主页内容
st.markdown('<div class="home-container">', unsafe_allow_html=True)

# 标题和副标题
st.markdown('<h1 class="home-title" style="text-align: center;">对抗图像系统</h1>', unsafe_allow_html=True)

# 选项卡片 - 使用Streamlit的columns实现双栏
col1, col2 = st.columns([1, 1])

# 对抗图像生成卡片
with col1:
    st.markdown("""
    <div class="option-card">
        <div class="option-card-icon">🎨</div>
        <h2 class="option-card-title">对抗图像生成</h2>
        <p class="option-card-description">生成对抗图像，测试模型的鲁棒性。上传原始图像，选择攻击算法，生成对抗样本。</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("进入生成页面", key="btn_generate", use_container_width=True):
        # 使用Streamlit的页面切换功能
        st.switch_page("pages/1-对抗图像生成.py")

# 对抗图像测试卡片
with col2:
    st.markdown("""
    <div class="option-card">
        <div class="option-card-icon">🔬</div>
        <h2 class="option-card-title">对抗图像测试</h2>
        <p class="option-card-description">测试对抗图像的效果，对比原始图像和对抗图像的模型预测结果，分析模型性能。</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("进入测试页面", key="btn_test", use_container_width=True):
        # 使用Streamlit的页面切换功能
        st.switch_page("pages/2-对抗图像测试.py")

# 页脚信息
st.markdown('<p style="color: #94a3b8; font-size: 0.9rem;">© 2026 对抗图像系统</p>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
"""Streamlit 多页面会话状态：进入页面时清理，避免换页后残留。"""

import streamlit as st

PAGE_HOME = "home"
PAGE_GENERATE = "generate"
PAGE_EVALUATE = "evaluate"

ACTIVE_PAGE_KEY = "active_page"

# 生成页（含旧版共用 key，便于迁移）
GENERATE_STATE_KEYS = (
    "gen_original_image",
    "gen_adversarial_image",
    "gen_perturbation",
    "gen_img_id",
    "gen_saved_dir",
    "original_image",
    "adversarial_image",
    "perturbation",
    "img_id",
)

GENERATE_WIDGET_KEYS = (
    "gen_method",
    "gen_img_id_input",
)

EVALUATE_STATE_KEYS = (
    "eval_original_image",
    "eval_adversarial_image",
    "eval_perturbation_image",
    "eval_original_answer",
    "eval_adversarial_answer",
    "eval_img_id",
    "original_image",
    "adversarial_image",
    "perturbation_image",
    "original_answer",
    "adversarial_answer",
    "img_id",
)

EVALUATE_WIDGET_KEYS = (
    "eval_img_id_input",
    "eval_question",
)


def clear_generate_state() -> None:
    for key in GENERATE_STATE_KEYS + GENERATE_WIDGET_KEYS:
        st.session_state.pop(key, None)


def clear_evaluate_state() -> None:
    for key in EVALUATE_STATE_KEYS + EVALUATE_WIDGET_KEYS:
        st.session_state.pop(key, None)
    st.session_state.pop("page_initialized", None)


def on_page_enter(page: str, clear_fn) -> None:
    """仅在从其他页面进入时执行清理，本页内交互不清理。"""
    if st.session_state.get(ACTIVE_PAGE_KEY) != page:
        clear_fn()
        st.session_state[ACTIVE_PAGE_KEY] = page


def mark_home_page() -> None:
    st.session_state[ACTIVE_PAGE_KEY] = PAGE_HOME

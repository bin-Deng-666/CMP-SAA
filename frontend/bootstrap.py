"""
Streamlit 从 frontend/ 启动时，先把项目根目录加入 sys.path，
并避免 frontend/utils 与项目根目录 utils 命名冲突。
"""

import os
import sys

FRONTEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(FRONTEND_DIR)


def setup_project_path() -> str:
    """将 GraduationProject 根目录置于 sys.path 最前，并清理错误的 utils 缓存。"""
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    if not _is_project_utils_loaded():
        _purge_utils_modules()
    return PROJECT_ROOT


def _is_project_utils_loaded() -> bool:
    """当前已加载的 utils 是否指向项目根目录下的 utils（含 attack_tool）。"""
    utils_mod = sys.modules.get("utils")
    if utils_mod is None:
        return True

    attack_name = "attack_tool.py"
    mod_file = getattr(utils_mod, "__file__", None)
    if mod_file:
        if os.path.isfile(os.path.join(os.path.dirname(mod_file), attack_name)):
            return True

    for path_entry in getattr(utils_mod, "__path__", []) or []:
        if os.path.isfile(os.path.join(path_entry, attack_name)):
            return True

    return False


def _purge_utils_modules() -> None:
    for key in list(sys.modules):
        if key == "utils" or key.startswith("utils."):
            del sys.modules[key]

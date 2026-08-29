
import os
import shutil
import streamlit as st
import yaml
import sys

# 找到项目根目录（config.py 所在目录的上一级）
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
from utils.file_utils import read_yaml, save_yaml

languages = {'zh-CN': "简体中文", 'en': "english", 'zh-TW': "繁體中文"}
exclude_keys = ['01_first_visit', '02_first_visit', '03_first_visit', 'all_first_visit',
                'image_chat_history', 'image_answer_history', 'video_chat_history', 'video_answer_history', 'image_question_input', 'vdeio_question_input',
                'rag_chat_history', 'rag_answer_history', 'rag_search_link', 'rag_question_input',
                'image_detect_chat_history', 'image_detect_answer_history', 'target_img_path_list', 'image_detect_question_input',
                'video_detect_chat_history', 'video_detect_answer_history', 'target_video_save_path', 'vdeio_detect_question_input',
                ]

# exclude_keys = ['01_first_visit', '02_first_visit', '03_first_visit', 'all_first_visit',

#                 ]

script_path = os.path.abspath(__file__)
# 脚本所在的目录
script_dir = os.path.dirname(script_path)
config_file_name = "config.yml"
config_file = os.path.join(script_dir, config_file_name)


session_file_name = "session.yml"
session_file = os.path.join(script_dir, session_file_name)


def load_config():
    """加载并返回配置文件内容

    该函数负责从本地文件中读取或创建配置文件。

    参数: None

    返回值: 配置文件的内容，格式为字典。

    实现细节：
        1. 首先检查`config_file`是否存在。如果不存在，使用`shutil.copy()`将`config_example_file`复制到`config_file`。
        2. 如果配置文件已经存在，则调用`read_yaml(config_file)`读取并返回其内容。
    """
    print("load_config")
    # 加载配置文件
    if not os.path.exists(config_file):
        # 复制示例配置文件到实际配置文件
        shutil.copy(config_example_file, config_file)
    if os.path.exists(config_file):
        # 如果配置文件存在，读取并返回其内容
        return read_yaml(config_file)


def test_config(todo_config, *args):
    """测试配置结构的完整性和可访问性

    该函数用于验证给定的配置数据结构是否符合预期，并确保所有指定路径都存在。

    参数:
        todo_config (dict): 需要测试的初始配置数据
        *args (str, ...): 需要检查的配置路径

    返回值:
        修改后的配置数据结构，保持原有数据不变，只修改指定路径的值。

    实现细节：
        1. 创建临时配置副本`temp_config`，以避免修改原始数据
        2. 遍历每个指定路径`arg`
            a. 如果路径不在当前配置中，添加空字典作为占位符
            b. 更新`temp_config`为当前路径下的值，准备处理下一个路径
        3. 返回修改后的临时配置副本
    """
    temp_config = todo_config
    for arg in args:
        if arg not in temp_config:
            temp_config[arg] = {}
        temp_config = temp_config[arg]


def save_config():
    # 保存配置文件
    if os.path.exists(config_file):
        save_yaml(config_file, my_config)



def save_session_state_to_yaml():
    """
    保存当前 Streamlit 的 session_state（会话状态）到本地 YAML 文件中，
    同时排除预设的不需要保存的键。

    功能流程：
    1. 遍历当前 session_state，排除 `exclude_keys` 中的键。
    2. 将剩余的键值对写入到 YAML 文件中进行本地持久化保存。
    """
    # 创建一个字典，包含所有未被排除的 session_state 键值对
    state_to_save = {key: value for key, value in st.session_state.items() if key not in exclude_keys}

    """将 Streamlit session_state 中的所有值保存到 YAML 文件"""
    with open(session_file, 'w') as file:
        yaml.dump(dict(state_to_save), file)


def delete_first_visit_session_state(first_visit):
    """
    删除除当前 first_visit 以外的所有 first_visit 标志键。

    参数:
        first_visit (str): 当前访问页面的标志键（如 '01_first_visit'）

    功能流程：
    1. 遍历所有预定义的 exclude_keys。
    2. 如果当前键是 first_visit，则保留；
       否则，从 session_state 中删除，防止状态混淆。
    """
    # 从session_state中删除其他first_vist标记
    for key in exclude_keys:
        if key != first_visit and key in st.session_state:
            # 删除不等于当前 first_visit 的键
            del st.session_state[key]


def load_session_state_from_yaml(first_visit):
    """
    加载 YAML 文件中的会话状态信息并更新 Streamlit 的 session_state。
    仅在首次访问页面时加载，后续访问不重复加载。

    参数:
        first_visit (str): 当前页面的首次访问标志键（如 '01_first_visit'）

    功能流程：
    1. 调用 delete_first_visit_session_state 删除其他页面的访问标志。
    2. 如果当前页面是第一次访问（session_state 中无该键），
       则从 YAML 文件中加载状态写入 session_state。
    3. 设置 first_visit 键为 True（首次访问）或 False（后续访问）。
    """

    # 删除不属于当前页面的 first_visit 键，避免标志混淆
    delete_first_visit_session_state(first_visit)

    # 判断是否为首次访问当前页面
    if first_visit not in st.session_state:
        # 第一次进入页面，设置标志为 True
        st.session_state[first_visit] = True
        """从 YAML 文件中读取数据并更新 session_state"""
        if os.path.exists(session_file):
            try:
                with open(session_file, 'r') as file:
                    # data = yaml.safe_load(file) # 加载 YAML 文件为字典
                    data = yaml.load(file, Loader=yaml.UnsafeLoader)  # ⚠️ 注意安全风险
                    print(f"load {session_file}")
                    for key, value in data.items():
                        st.session_state[key] = value  # 将加载的数据写入 session_state
            except FileNotFoundError:
                # 文件不存在时给出提示
                st.warning(f"File {session_file} not found.")
    else:
        # 如果不是首次访问，设置为 False
        st.session_state[first_visit] = False
        
# def load_session_state_from_yaml(first_visit):
#     """
#     加载 YAML 文件中的会话状态信息并更新 Streamlit 的 session_state。
#     仅在首次访问页面时加载，后续访问不重复加载。

#     参数:
#         first_visit (str): 当前页面的首次访问标志键（如 '01_first_visit'）

#     功能流程：
#     1. 调用 delete_first_visit_session_state 删除其他页面的访问标志。
#     2. 如果当前页面是第一次访问（session_state 中无该键），
#        则从 YAML 文件中加载状态写入 session_state。
#     3. 设置 first_visit 键为 True（首次访问）或 False（后续访问）。
#     """


#     # # 第一次进入页面，设置标志为 True
#     # st.session_state[first_visit] = True
#     print(f"load {session_file}")
#     """从 YAML 文件中读取数据并更新 session_state"""
#     if os.path.exists(session_file):
#         try:
#             with open(session_file, 'r') as file:
#                 # data = yaml.safe_load(file) # 加载 YAML 文件为字典
#                 data = yaml.load(file, Loader=yaml.UnsafeLoader)  # ⚠️ 注意安全风险
#                 for key, value in data.items():
#                     st.session_state[key] = value  # 将加载的数据写入 session_state
#         except FileNotFoundError:
#             # 文件不存在时给出提示
#             st.warning(f"File {session_file} not found.")



my_config = load_config()
print(my_config)

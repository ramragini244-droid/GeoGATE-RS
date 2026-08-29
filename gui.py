# 导入标准库和第三方库
import streamlit as st    # 用于构建Web应用界面

# 导入配置相关模块
from config.config import (
    my_config,            # 主配置对象
    save_config,          # 保存配置函数
    languages,           # 支持的语言列表
    test_config,         # 配置测试函数
    delete_first_visit_session_state, # 删除首次访问状态函数
)


def tr(str):
    return str 

############################################ 渲染组件、样式 ##################################################
from utils.custom_css import  inject_tab_style, render_header, set_app_config

set_app_config()
render_header()
inject_tab_style()

# 删除所有首次访问状态
delete_first_visit_session_state("all_first_visit")

# 初始化UI语言设置
if 'ui_language' not in st.session_state:
    st.session_state['ui_language'] = 'zh-CN - 简体中文'


def set_ui_language():
    """
    设置UI界面语言
    
    从session state获取选择的语言并更新配置
    """
    print('set_ui_language:', st.session_state['ui_language'])
    # 更新配置中的语言设置
    my_config['ui']['language'] = st.session_state['ui_language'].split(" - ")[0].strip()
    print('set ui language:', my_config['ui']['language'])
    # 保存配置
    save_config()



def set_llm_sk(provider, key):
    """
    设置LLM服务密钥
    
    Args:
        provider (str): 服务提供商名称
        key (str): session state中的密钥键名
    """
    my_config['llm'][provider]['secret_key'] = st.session_state[key]
    save_config()


def set_llm_key(provider, key):
    """
    设置LLM API密钥
    
    Args:
        provider (str): 服务提供商名称
        key (str): session state中的API密钥键名
    """
    my_config['llm'][provider]['api_key'] = st.session_state[key]
    save_config()


def set_llm_base_url(provider, key):
    """
    设置LLM基础URL
    
    Args:
        provider (str): 服务提供商名称
        key (str): session state中的URL键名
    """
    my_config['llm'][provider]['base_url'] = st.session_state[key]
    save_config()


def set_llm_model_name(provider, key):
    """
    设置LLM模型名称
    
    Args:
        provider (str): 服务提供商名称
        key (str): session state中的模型名称键名
    """
    if provider not in my_config['llm']:
        my_config['llm'][provider] = {}
    my_config['llm'][provider]['model_name'] = st.session_state[key]
    save_config()
    

def set_llm_provider():
    """
    设置LLM提供商
    
    从session state获取选择的LLM提供商并更新配置
    """
    my_config['llm']['provider'] = st.session_state['llm_provider']
    save_config()
    
def set_test_provider():
    """
    设置LLM提供商
    
    从session state获取选择的LLM提供商并更新配置
    """
    # my_config['llm']['provider'] = st.session_state['llm_provider']
    my_config['is_test'] = st.session_state['test_provider']
    save_config()
    
def set_rag_vector_provider():
    my_config['rag']['vector']['provider'] = st.session_state['rag_vector_provider']
    save_config()

def set_rag_vector_port(provider, key):
    my_config['rag']['vector'][provider] = st.session_state[key]
    save_config()

def set_rag_vector_name(provider, key):
    my_config['rag']['vector'][provider]['vector_name'] = st.session_state[key]
    save_config()
    

def set_rag_doc_path():
    my_config['rag']['doc_path'] = st.session_state['rag_doc_path']
    save_config()
    
def set_rag_embedding_model_path():
    my_config['rag']['embedding_model_path'] = st.session_state['rag_embedding_model_path']
    save_config()
    
    
############################################ 首页配置界面--语言选择 ############################################

# 设置语言选择器
display_languages = []  # 初始化显示语言列表
selected_index = 0     # 初始化选中索引
# 遍历所有支持的语言代码
for i, code in enumerate(languages.keys()):
    # 将语言代码和名称组合添加到显示列表
    display_languages.append(f"{code} - {languages[code]}")
    # 如果是当前选中的语言，记录其索引
    if f"{code} - {languages[code]}" == st.session_state['ui_language']:
        selected_index = i
        
# 创建语言选择下拉框，设置回调函数
# print("selected_index:", selected_index)
selected_language = st.selectbox(tr("Language"), options=display_languages,
                                 index=selected_index, key='ui_language', on_change=set_ui_language)
############################################ 首页配置界面--LLM资源配置 ############################################

# 创建LLM配置容器
llm_container = st.container(border=True)
with (llm_container):
    st.subheader("LLM 配置")
    # 定义可用的LLM提供商列表
    llm_providers = ['FM9G4BV']
    # 获取当前选中的LLM提供商
    saved_llm_provider = my_config['llm']['provider']
    # 查找当前提供商的索引
    saved_llm_provider_index = 0
    for i, provider in enumerate(llm_providers):
        if provider == saved_llm_provider:
            saved_llm_provider_index = i
            break

    # 创建LLM提供商选择下拉框
    llm_provider = st.selectbox(tr("LLM Provider"), options=llm_providers, index=saved_llm_provider_index,
                                key='llm_provider', on_change=set_llm_provider)
    print(llm_provider)

    # 创建LLM提供商特定配置面板
    with st.expander(llm_provider, expanded=True):
        tips = f"""
               ##### {llm_provider} 配置信息
               """
        st.info(tips)

        # Azure、DeepSeek和Ollama需要Base URL配置
        if llm_provider == 'FM9G4BV':
            st_llm_base_url = st.text_input(tr("Base Url"),
                                            value=my_config['llm'].get(llm_provider, {}).get('base_url', ''),
                                            key=llm_provider + '_base_url',
                                            on_change=set_llm_base_url,
                                            args=(llm_provider, llm_provider + '_base_url'))

        # 所有提供商都需要模型名称配置
        st_llm_model_name = st.text_input(tr("Model Name"),
                                          value=my_config['llm'].get(llm_provider, {}).get('model_name', ''),
                                          key=llm_provider + '_model_name', on_change=set_llm_model_name,
                                          args=(llm_provider, llm_provider + '_model_name'))

# 创建LLM配置容器
rag_container = st.container(border=True)
with (rag_container):
    st.subheader("RAG 配置")
    st_rag_doc_path = st.text_input(tr("RAG document path"),
                            value=my_config['rag'].get('doc_path', ''),
                            key='rag_doc_path',
                            on_change=set_rag_doc_path,
                            )
    
    st_rag_embedding_model_path = st.text_input(tr("RAG document path"),
                        value=my_config['rag'].get('embedding_model_path', ''),
                        key='rag_embedding_model_path',
                        on_change=set_rag_embedding_model_path,
                        )
    
    # 定义可用的LLM提供商列表
    vector_providers = ['chromadb', 'milvus']
    # 获取当前选中的LLM提供商
    saved_vector_provider = my_config['rag']['vector']['provider']
    # 查找当前提供商的索引
    saved_vector_provider_index = 0
    for i, provider in enumerate(vector_providers):
        if provider == saved_vector_provider:
            saved_provider_provider_index = i
            break

    # 创建LLM提供商选择下拉框
    rag_vector_provider = st.selectbox(tr("RAG vector Provider"), options=vector_providers, index=saved_provider_provider_index,
                                key='rag_vector_provider', on_change=set_rag_vector_provider)
    print(rag_vector_provider)

    # 创建LLM提供商特定配置面板
    with st.expander(rag_vector_provider, expanded=True):
        tips = f"""
               ##### {rag_vector_provider} 配置信息
               """
        st.info(tips)

        # Azure、DeepSeek和Ollama需要Base URL配置
        if rag_vector_provider == 'milvus':
            st_rag_vector_port = st.text_input(tr("port"),
                                            value=my_config['rag'].get('vector', {}).get(rag_vector_provider, '').get("port", 3306),
                                            key=rag_vector_provider + '_port',
                                            on_change=set_rag_vector_port,
                                            args=(rag_vector_provider, rag_vector_provider + '_port'))

        # 所有提供商都需要模型名称配置
        st_rag_vector_name = st.text_input(tr("Vector Name"),
                                          value=my_config['rag'].get('vector', {}).get(rag_vector_provider, '').get("vector_name", 'none'),
                                          key=rag_vector_provider + '_name',
                                          on_change=set_rag_vector_name,
                                          args=(rag_vector_provider, rag_vector_provider + '_name'))
        
############################################ 首页配置界面--LLM资源配置 ############################################

# 创建LLM配置容器
test_container = st.container(border=True)
with (llm_container):
    st.subheader("是否TEST")
    # 定义可用的LLM提供商列表
    test_providers = ['TRUE', 'FALSE']
    # 获取当前选中的LLM提供商
    saved_test_provider = my_config['is_test']
    # 查找当前提供商的索引
    saved_test_provider_index = 0
    for i, provider in enumerate(test_providers):
        if provider == saved_test_provider:
            saved_test_provider_index = i
            break

    # 创建LLM提供商选择下拉框
    test_provider = st.selectbox(tr("是否TEST"), options=test_providers, index=saved_test_provider_index,
                                key='test_provider', on_change=set_test_provider)
        

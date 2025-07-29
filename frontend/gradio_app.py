"""
Gradio前端界面
提供聊天界面和系统管理功能
"""
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import time
import uuid
import logging
from typing import List, Tuple, Optional, Dict, Any

# 处理依赖包可能未安装的情况
try:
    import gradio as gr
    import requests
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    print("Warning: Gradio dependencies not available. Please install requirements.txt")

from config.settings import get_settings


class ECAgentUI:
    """ECAgent前端界面类"""
    
    def __init__(self, api_url: str = None):
        self.settings = get_settings()
        self.api_url = api_url or f"http://{self.settings.api.api_host}:{self.settings.api.api_port}"
        self.session_id = str(uuid.uuid4())
        
        # 设置日志
        from config.logging_config import get_logger
        self.logger = get_logger(__name__)
        
        # 界面状态
        self.conversation_history = []
        self.system_stats = {}
        
    def chat_fn(self, message: str, history: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]]]:
        """聊天函数"""
        if not message.strip():
            return "", history
        
        try:
            # 发送请求到API
            response = requests.post(
                f"{self.api_url}/chat",
                json={
                    "message": message,
                    "session_id": self.session_id
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                bot_response = result["response"]
                
                # 添加来源信息
                if result.get("sources"):
                    sources_info = "\n\n📚 **参考来源：**\n" + "\n".join([
                        f"• {source}" for source in result["sources"][:3]
                    ])
                    bot_response += sources_info
                
                # 添加意图和过滤信息
                if result.get("intent"):
                    intent_info = f"\n\n🎯 **识别意图：** {result['intent']}"
                    bot_response += intent_info
                
                if result.get("filtered"):
                    bot_response += "\n\n⚠️ *回复已经过安全过滤*"
                
                # 更新历史记录
                history.append((message, bot_response))
                
                return "", history
            else:
                error_msg = f"❌ API请求失败: {response.status_code}"
                history.append((message, error_msg))
                return "", history
                
        except requests.exceptions.ConnectionError:
            error_msg = "❌ 无法连接到服务器，请检查API服务是否启动"
            history.append((message, error_msg))
            return "", history
        except requests.exceptions.Timeout:
            error_msg = "❌ 请求超时，请稍后再试"
            history.append((message, error_msg))
            return "", history
        except Exception as e:
            error_msg = f"❌ 连接错误: {str(e)}"
            history.append((message, error_msg))
            return "", history
    
    def clear_chat(self):
        """清除聊天记录"""
        try:
            # 清除服务器端会话
            requests.delete(f"{self.api_url}/session/{self.session_id}")
            
            # 生成新的会话ID
            self.session_id = str(uuid.uuid4())
            
        except Exception as e:
            self.logger.error(f"Error clearing session: {e}")
        
        return [], ""
    
    def get_session_info(self):
        """获取会话信息"""
        try:
            response = requests.get(f"{self.api_url}/session/{self.session_id}")
            if response.status_code == 200:
                info = response.json()
                return f"""
**会话信息：**
• 会话ID: {info.get('session_id', 'N/A')}
• 用户ID: {info.get('user_id', 'N/A')}
• 创建时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(info.get('created_at', 0)))}
• 最后活跃: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(info.get('last_active', 0)))}
• 对话轮数: {info.get('turn_count', 0)}
• 消息数量: {info.get('message_count', 0)}
• 状态: {'活跃' if info.get('is_active') else '非活跃'}
• 持续时间: {info.get('duration', 0):.1f}秒
"""
            else:
                return "❌ 无法获取会话信息"
        except Exception as e:
            return f"❌ 连接错误: {str(e)}"
    
    def get_system_stats(self):
        """获取系统统计信息"""
        try:
            response = requests.get(f"{self.api_url}/stats")
            if response.status_code == 200:
                stats = response.json()
                self.system_stats = stats
                return f"""
**系统状态：**
• API状态: {stats.get('api_status', 'N/A')}
• LLM可用: {'✅' if stats.get('llm_available') else '❌'}
• 向量存储可用: {'✅' if stats.get('vectorstore_available') else '❌'}
• 安全过滤启用: {'✅' if stats.get('security_enabled') else '❌'}
• 总会话数: {stats.get('total_sessions', 0)}
• 活跃会话数: {stats.get('active_sessions', 0)}
"""
            else:
                return "❌ 无法获取系统统计信息"
        except Exception as e:
            return f"❌ 连接错误: {str(e)}"
    
    def get_knowledge_base_stats(self):
        """获取知识库统计信息"""
        try:
            response = requests.get(f"{self.api_url}/knowledge-base/stats")
            if response.status_code == 200:
                stats = response.json()
                return f"""
**知识库统计：**
• 文档数量: {stats.get('document_count', 0)}
• 集合名称: {stats.get('collection_name', 'N/A')}
• 持久化目录: {stats.get('persist_directory', 'N/A')}
• LLM可用: {'✅' if stats.get('llm_available') else '❌'}
• 检索器可用: {'✅' if stats.get('retriever_available') else '❌'}
• RAG链可用: {'✅' if stats.get('rag_chain_available') else '❌'}
• 对话链可用: {'✅' if stats.get('conversation_chain_available') else '❌'}
"""
            else:
                return "❌ 无法获取知识库统计信息"
        except Exception as e:
            return f"❌ 连接错误: {str(e)}"
    
    def search_knowledge_base(self, query: str, top_k: int = 5):
        """搜索知识库"""
        if not query.strip():
            return "请输入搜索查询"
        
        try:
            response = requests.post(
                f"{self.api_url}/knowledge-base/search",
                params={"query": query, "top_k": top_k}
            )
            if response.status_code == 200:
                result = response.json()
                results = result.get("results", [])
                
                if not results:
                    return "❌ 没有找到相关文档"
                
                output = f"**搜索结果 ({len(results)}个)：**\n\n"
                for i, doc in enumerate(results, 1):
                    content = doc.get("content", "")[:200] + "..." if len(doc.get("content", "")) > 200 else doc.get("content", "")
                    source = doc.get("source", "未知来源")
                    output += f"**{i}. {source}**\n{content}\n\n"
                
                return output
            else:
                return "❌ 搜索失败"
        except Exception as e:
            return f"❌ 连接错误: {str(e)}"
    
    def get_security_stats(self):
        """获取安全统计信息"""
        try:
            response = requests.get(f"{self.api_url}/security/stats")
            if response.status_code == 200:
                stats = response.json()
                
                # 获取敏感词信息
                words_response = requests.get(f"{self.api_url}/security/sensitive-words")
                words_count = 0
                if words_response.status_code == 200:
                    words_count = words_response.json().get("count", 0)
                
                return f"""
**安全统计：**
• 敏感词数量: {words_count}
• 审计日志启用: {'✅' if stats.get('audit_enabled') else '❌'}
• 内容审核链可用: {'✅' if stats.get('chains_available', {}).get('content_review') else '❌'}
• 格式化链可用: {'✅' if stats.get('chains_available', {}).get('format_chain') else '❌'}
• 意图识别链可用: {'✅' if stats.get('chains_available', {}).get('intent_chain') else '❌'}
• 审计日志条数: {stats.get('audit_log_count', 0)}
"""
            else:
                return "❌ 无法获取安全统计信息"
        except Exception as e:
            return f"❌ 连接错误: {str(e)}"
    
    def check_api_health(self):
        """检查API健康状态"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                return "🟢 API服务正常"
            else:
                return "🔴 API服务异常"
        except Exception:
            return "🔴 API服务连接失败"
    
    def add_sensitive_word(self, word: str):
        """添加敏感词"""
        if not word.strip():
            return "请输入要添加的敏感词"
        
        try:
            response = requests.post(
                f"{self.api_url}/security/sensitive-words",
                params={"word": word.strip()}
            )
            if response.status_code == 200:
                result = response.json()
                return f"✅ {result.get('message', '操作完成')}"
            else:
                return "❌ 添加敏感词失败"
        except Exception as e:
            return f"❌ 连接错误: {str(e)}"
    
    def cleanup_sessions(self):
        """清理过期会话"""
        try:
            response = requests.post(f"{self.api_url}/session/cleanup")
            if response.status_code == 200:
                result = response.json()
                return f"✅ 清理完成，清除了 {result.get('cleaned_count', 0)} 个过期会话"
            else:
                return "❌ 清理失败"
        except Exception as e:
            return f"❌ 连接错误: {str(e)}"
    
    def create_interface(self):
        """创建Gradio界面"""
        
        # 自定义CSS
        custom_css = """
        .gradio-container {
            max-width: 1200px !important;
        }
        .chat-container {
            height: 500px;
        }
        .stat-box {
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 10px;
            margin: 5px 0;
        }
        """
        
        # 创建界面
        demo = gr.Blocks(title="ECAgent - 电商客服助手", theme=gr.themes.Default())
        
        with demo:
            
            # 标题和描述
            gr.Markdown(f"# {self.settings.frontend.ui_title}")
            gr.Markdown(f"*{self.settings.frontend.ui_description}*")
            
            with gr.Tabs():
                
                # 聊天界面
                with gr.TabItem("💬 智能客服"):
                    with gr.Row():
                        with gr.Column(scale=3):
                                                          # 主聊天界面
                              chatbot = gr.Chatbot(
                                  [],
                                  elem_id="chatbot",
                                  height=500,
                                  type="messages",
                                  avatar_images=("👤", "🤖"),
                                  show_copy_button=True
                              )
                            
                            with gr.Row():
                                msg = gr.Textbox(
                                    placeholder="请输入您的问题...",
                                    container=False,
                                    scale=4,
                                    max_lines=3
                                )
                                submit_btn = gr.Button("发送", scale=1, variant="primary")
                                clear_btn = gr.Button("清除", scale=1)
                            
                            # 示例问题
                            with gr.Row():
                                gr.Examples(
                                    examples=[
                                        "如何申请退货？",
                                        "订单什么时候发货？",
                                        "如何查看物流信息？",
                                        "有什么优惠活动吗？",
                                        "如何联系人工客服？",
                                        "支付失败怎么办？",
                                        "如何修改收货地址？"
                                    ],
                                    inputs=msg,
                                    label="💡 常见问题示例"
                                )
                        
                        with gr.Column(scale=1):
                            # 侧边栏
                            gr.Markdown("### 📊 会话信息")
                            session_info = gr.Textbox(
                                label="当前会话",
                                interactive=False,
                                lines=8,
                                max_lines=8
                            )
                            
                            info_btn = gr.Button("刷新信息", variant="secondary")
                            
                            gr.Markdown("### 🔧 系统状态")
                            health_status = gr.Textbox(
                                label="API状态",
                                value="检查中...",
                                interactive=False
                            )
                
                # 系统管理
                with gr.TabItem("⚙️ 系统管理"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### 📈 系统统计")
                            system_stats_display = gr.Textbox(
                                label="系统状态",
                                interactive=False,
                                lines=10
                            )
                            system_stats_btn = gr.Button("获取系统统计", variant="primary")
                            
                            gr.Markdown("### 📚 知识库管理")
                            kb_stats_display = gr.Textbox(
                                label="知识库状态",
                                interactive=False,
                                lines=10
                            )
                            kb_stats_btn = gr.Button("获取知识库统计", variant="primary")
                        
                        with gr.Column():
                            gr.Markdown("### 🔍 知识库搜索")
                            search_query = gr.Textbox(
                                label="搜索查询",
                                placeholder="输入搜索关键词..."
                            )
                            search_top_k = gr.Slider(
                                minimum=1,
                                maximum=20,
                                value=5,
                                step=1,
                                label="返回结果数量"
                            )
                            search_btn = gr.Button("搜索知识库", variant="primary")
                            search_results = gr.Textbox(
                                label="搜索结果",
                                interactive=False,
                                lines=15
                            )
                
                # 安全管理
                with gr.TabItem("🛡️ 安全管理"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### 🔒 安全统计")
                            security_stats_display = gr.Textbox(
                                label="安全状态",
                                interactive=False,
                                lines=10
                            )
                            security_stats_btn = gr.Button("获取安全统计", variant="primary")
                            
                            gr.Markdown("### 🚫 敏感词管理")
                            sensitive_word_input = gr.Textbox(
                                label="添加敏感词",
                                placeholder="输入要添加的敏感词..."
                            )
                            add_word_btn = gr.Button("添加敏感词", variant="primary")
                            word_result = gr.Textbox(
                                label="操作结果",
                                interactive=False
                            )
                        
                        with gr.Column():
                            gr.Markdown("### 🧹 会话管理")
                            cleanup_result = gr.Textbox(
                                label="清理结果",
                                interactive=False
                            )
                            cleanup_btn = gr.Button("清理过期会话", variant="primary")
                            
                            gr.Markdown("### 📊 API信息")
                            api_info = gr.Textbox(
                                label="API地址",
                                value=self.api_url,
                                interactive=False
                            )
                            
                            gr.Markdown("### 🔄 操作日志")
                            operation_log = gr.Textbox(
                                label="操作记录",
                                interactive=False,
                                lines=5
                            )
            
            # 事件绑定
            
            # 聊天事件
            submit_btn.click(
                self.chat_fn,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot],
                show_progress=True
            )
            
            msg.submit(
                self.chat_fn,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot],
                show_progress=True
            )
            
            clear_btn.click(
                self.clear_chat,
                outputs=[chatbot, msg]
            )
            
            info_btn.click(
                self.get_session_info,
                outputs=session_info
            )
            
            # 系统管理事件
            system_stats_btn.click(
                self.get_system_stats,
                outputs=system_stats_display
            )
            
            kb_stats_btn.click(
                self.get_knowledge_base_stats,
                outputs=kb_stats_display
            )
            
            search_btn.click(
                self.search_knowledge_base,
                inputs=[search_query, search_top_k],
                outputs=search_results
            )
            
            # 安全管理事件
            security_stats_btn.click(
                self.get_security_stats,
                outputs=security_stats_display
            )
            
            add_word_btn.click(
                self.add_sensitive_word,
                inputs=sensitive_word_input,
                outputs=word_result
            )
            
            cleanup_btn.click(
                self.cleanup_sessions,
                outputs=cleanup_result
            )
            
            # 启动时获取会话信息
            demo.load(
                self.get_session_info,
                outputs=session_info
            )
        
        return demo
    
    def launch(self, **kwargs):
        """启动界面"""
        if not DEPENDENCIES_AVAILABLE:
            print("Gradio dependencies not available. Please install requirements.txt")
            return
        
        demo = self.create_interface()
        
        # 默认启动参数
        default_kwargs = {
            "server_name": self.settings.frontend.gradio_host,
            "server_port": self.settings.frontend.gradio_port,
            "share": self.settings.frontend.gradio_share,
            "inbrowser": True,
            "show_error": True
        }
        
        # 合并用户提供的参数
        launch_kwargs = {**default_kwargs, **kwargs}
        
        self.logger.info(f"Starting Gradio UI on {launch_kwargs['server_name']}:{launch_kwargs['server_port']}")
        demo.launch(**launch_kwargs)


def create_ui(api_url: str = None) -> ECAgentUI:
    """创建UI实例"""
    return ECAgentUI(api_url=api_url)


if __name__ == "__main__":
    # 创建UI实例
    ui = create_ui()
    
    # 启动界面
    ui.launch() 
#!/usr/bin/env python3
"""
简单前端界面 - 用于测试聊天功能
"""

import gradio as gr
import requests
import json

# API配置
API_BASE_URL = "http://localhost:8000"

def chat_with_api(message, history):
    """与API进行聊天"""
    try:
        # 发送请求到API
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json={"message": message},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["response"]
        else:
            return f"API错误：{response.status_code}"
            
    except requests.exceptions.ConnectionError:
        return "❌ 无法连接到API服务，请确保API服务已启动"
    except Exception as e:
        return f"❌ 错误：{str(e)}"

def check_api_status():
    """检查API服务状态"""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            return "✅ API服务正常"
        else:
            return f"❌ API服务异常：{response.status_code}"
    except:
        return "❌ API服务未启动"

# 创建Gradio界面
with gr.Blocks(title="ECAgent 电商客服助手") as demo:
    gr.Markdown("# 🛒 ECAgent 电商客服助手")
    gr.Markdown("智能客服系统演示界面")
    
    with gr.Row():
        with gr.Column(scale=3):
            # 聊天界面
            chatbot = gr.Chatbot(
                label="聊天记录",
                height=400,
                show_label=True,
                avatar_images=("🙋‍♀️", "🤖")
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    label="输入消息",
                    placeholder="请输入您的问题...",
                    lines=1,
                    scale=4
                )
                send_btn = gr.Button("发送", variant="primary", scale=1)
                
            with gr.Row():
                clear_btn = gr.Button("清空对话", variant="secondary")
                
        with gr.Column(scale=1):
            # 状态面板
            gr.Markdown("## 📊 系统状态")
            status_display = gr.Textbox(
                label="API状态",
                value="检查中...",
                interactive=False,
                lines=2
            )
            status_btn = gr.Button("刷新状态", variant="secondary")
            
            # 功能说明
            gr.Markdown("""
            ## 📝 功能说明
            - 输入"你好"进行问候
            - 输入"帮助"获取帮助信息
            - 支持中英文对话
            - 实时响应用户询问
            """)
            
            # 示例问题
            gr.Markdown("## 💡 示例问题")
            example_btns = [
                gr.Button("你好", size="sm"),
                gr.Button("帮助", size="sm"),
                gr.Button("产品介绍", size="sm"),
                gr.Button("售后服务", size="sm")
            ]

    # 事件处理
    def user_message(user_input, history):
        return "", history + [[user_input, None]]
    
    def bot_response(history):
        user_input = history[-1][0]
        bot_reply = chat_with_api(user_input, history)
        history[-1][1] = bot_reply
        return history
    
    def clear_chat():
        return []
    
    def set_example_input(example_text):
        return example_text
    
    # 绑定事件
    msg.submit(user_message, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_response, chatbot, chatbot
    )
    send_btn.click(user_message, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_response, chatbot, chatbot
    )
    clear_btn.click(clear_chat, None, chatbot, queue=False)
    status_btn.click(check_api_status, None, status_display)
    
    # 示例按钮事件
    for btn in example_btns:
        btn.click(set_example_input, btn, msg)
    
    # 页面加载时检查API状态
    demo.load(check_api_status, None, status_display)

if __name__ == "__main__":
    print("🎨 启动ECAgent前端界面...")
    print("🌐 访问地址: http://localhost:7860")
    print("📱 请确保API服务已启动 (http://localhost:8000)")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False
    ) 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的Gradio测试应用
用于验证前端功能和系统状态
"""

import gradio as gr
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def chat_fn(message, history):
    """简单的聊天回复函数"""
    if not message:
        return history, ""
    
    # 简单的回复逻辑
    response = f"您好！我收到了您的消息：「{message}」。这是一个简化的测试版本。"
    
    # 添加到历史记录
    history.append((message, response))
    return history, ""

def get_system_status():
    """获取系统状态"""
    try:
        from config.settings import get_settings
        settings = get_settings()
        return f"✅ 系统配置加载成功\n配置文件: {settings.log.log_file}"
    except Exception as e:
        return f"❌ 系统配置加载失败: {e}"

def main():
    """主函数"""
    
    with gr.Blocks(title="ECAgent 测试界面", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🤖 ECAgent 电商客服助手 (测试版)")
        gr.Markdown("这是一个简化的测试界面，用于验证系统基本功能。")
        
        with gr.Tab("💬 聊天测试"):
            chatbot = gr.Chatbot(
                [],
                height=400,
                type="tuples"  # 使用兼容的格式
            )
            
            msg = gr.Textbox(
                placeholder="请输入您的问题...",
                label="消息输入",
                lines=1
            )
            
            with gr.Row():
                send_btn = gr.Button("发送", variant="primary")
                clear_btn = gr.Button("清空对话")
            
            # 绑定事件
            send_btn.click(
                chat_fn,
                inputs=[msg, chatbot],
                outputs=[chatbot, msg]
            )
            
            msg.submit(
                chat_fn,
                inputs=[msg, chatbot],
                outputs=[chatbot, msg]
            )
            
            clear_btn.click(
                lambda: ([], ""),
                outputs=[chatbot, msg]
            )
        
        with gr.Tab("🔧 系统状态"):
            status_display = gr.Textbox(
                label="系统状态",
                value="点击刷新按钮获取状态",
                lines=10
            )
            
            refresh_btn = gr.Button("刷新状态")
            refresh_btn.click(
                get_system_status,
                outputs=status_display
            )
        
        # 启动时获取状态
        demo.load(
            get_system_status,
            outputs=status_display
        )
    
    print("🚀 启动ECAgent测试界面...")
    print("📊 界面地址: http://localhost:7860")
    print("⏹️  按 Ctrl+C 停止服务")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=False
    )

if __name__ == "__main__":
    main()
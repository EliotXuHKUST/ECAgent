#!/usr/bin/env python3
"""
简单启动脚本 - 用于快速测试系统基础功能
"""

import os
import sys
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

# 添加项目路径到系统路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 创建简单的FastAPI应用
app = FastAPI(title="ECAgent API", description="电商客服助手API")

@app.get("/")
async def root():
    return {"message": "ECAgent API is running", "status": "ok"}

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "ECAgent"}

@app.post("/chat")
async def chat(message: dict):
    """简单的聊天接口"""
    user_input = message.get("message", "")
    
    # 简单的回复逻辑
    if "你好" in user_input or "hello" in user_input.lower():
        response = "您好！我是ECAgent智能客服助手，很高兴为您服务！"
    elif "帮助" in user_input or "help" in user_input.lower():
        response = "我可以帮助您解答关于产品、订单、售后等问题。请问您需要什么帮助？"
    else:
        response = f"您提到了：{user_input}。我正在为您查找相关信息..."
    
    return {
        "response": response,
        "session_id": "test_session",
        "timestamp": "2024-01-01T12:00:00Z"
    }

if __name__ == "__main__":
    print("🚀 启动ECAgent简单测试服务...")
    print("📖 API文档: http://localhost:8000/docs")
    print("🔍 健康检查: http://localhost:8000/health")
    print("💬 聊天接口: POST http://localhost:8000/chat")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info") 
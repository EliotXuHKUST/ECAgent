#!/bin/bash

# ECAgent 系统停止脚本
# 用于停止所有服务

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 停止本地服务
stop_local_services() {
    print_info "停止本地服务..."
    
    # 停止API服务
    if [ -f ".api_pid" ]; then
        API_PID=$(cat .api_pid)
        if kill -0 $API_PID 2>/dev/null; then
            kill $API_PID
            print_success "API服务已停止 (PID: $API_PID)"
        else
            print_warning "API服务进程不存在"
        fi
        rm -f .api_pid
    else
        print_warning "未找到API服务PID文件"
    fi
    
    # 停止前端服务
    if [ -f ".frontend_pid" ]; then
        FRONTEND_PID=$(cat .frontend_pid)
        if kill -0 $FRONTEND_PID 2>/dev/null; then
            kill $FRONTEND_PID
            print_success "前端服务已停止 (PID: $FRONTEND_PID)"
        else
            print_warning "前端服务进程不存在"
        fi
        rm -f .frontend_pid
    else
        print_warning "未找到前端服务PID文件"
    fi
    
    # 清理可能残留的进程
    pkill -f "api.main:app" 2>/dev/null || true
    pkill -f "frontend/gradio_app.py" 2>/dev/null || true
    
    print_success "本地服务停止完成"
}

# 停止Docker服务
stop_docker_services() {
    print_info "停止Docker服务..."
    
    cd deploy
    
    # 停止并删除容器
    docker-compose down
    
    print_success "Docker服务停止完成"
}

# 清理临时文件
cleanup_temp_files() {
    print_info "清理临时文件..."
    
    # 清理PID文件
    rm -f .api_pid
    rm -f .frontend_pid
    
    # 清理日志文件（可选）
    if [ "$1" = "--clean-logs" ]; then
        print_info "清理日志文件..."
        rm -rf logs/*.log
        print_success "日志文件已清理"
    fi
    
    print_success "临时文件清理完成"
}

# 主函数
main() {
    echo "============================================"
    echo "      ECAgent 电商客服助手 停止脚本"
    echo "============================================"
    echo ""
    
    # 检查参数
    CLEAN_LOGS=false
    if [ "$1" = "--clean-logs" ]; then
        CLEAN_LOGS=true
    fi
    
    # 检查是否有Docker服务在运行
    if command -v docker &> /dev/null && docker ps | grep -q "ecagent"; then
        print_info "检测到Docker服务正在运行"
        stop_docker_services
    fi
    
    # 停止本地服务
    stop_local_services
    
    # 清理临时文件
    if [ "$CLEAN_LOGS" = true ]; then
        cleanup_temp_files --clean-logs
    else
        cleanup_temp_files
    fi
    
    print_success "所有服务已停止"
    echo ""
    print_info "如需重新启动服务，请运行: ./scripts/start.sh"
    
    if [ "$CLEAN_LOGS" = false ]; then
        print_info "如需清理日志文件，请运行: ./scripts/stop.sh --clean-logs"
    fi
}

# 运行主函数
main "$@" 
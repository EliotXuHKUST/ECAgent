#!/bin/bash

# ECAgent 系统启动脚本
# 用于快速启动整个系统

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

# 检查依赖
check_dependencies() {
    print_info "检查系统依赖..."
    
    # 检查Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 未安装，请先安装 Python 3.9+"
        exit 1
    fi
    
    # 检查pip
    if ! command -v pip3 &> /dev/null; then
        print_error "pip3 未安装，请先安装 pip"
        exit 1
    fi
    
    # 检查Docker（可选）
    if command -v docker &> /dev/null; then
        print_success "Docker 已安装"
        DOCKER_AVAILABLE=true
    else
        print_warning "Docker 未安装，将使用本地模式"
        DOCKER_AVAILABLE=false
    fi
    
    print_success "依赖检查完成"
}

# 创建必要的目录
create_directories() {
    print_info "创建必要的目录..."
    
    mkdir -p chroma_db
    mkdir -p logs
    mkdir -p data/knowledge_base
    mkdir -p data/sessions
    mkdir -p models/weights
    mkdir -p models/fine_tuned
    
    print_success "目录创建完成"
}

# 创建示例数据
create_sample_data() {
    print_info "创建示例数据..."
    
    # 创建示例FAQ文件
    if [ ! -f "data/knowledge_base/faq.txt" ]; then
        cat > data/knowledge_base/faq.txt << EOF
# 电商客服FAQ

## 退换货相关

### 如何申请退货？
您好！申请退货很简单：
1. 登录您的账户
2. 找到相应订单
3. 点击申请退货
4. 填写退货原因
5. 提交申请
我们会在1-2个工作日内审核您的申请。

### 退货需要多长时间？
一般情况下，退货处理时间为3-7个工作日：
- 审核时间：1-2个工作日
- 物流时间：1-3个工作日
- 退款时间：1-2个工作日

## 物流相关

### 如何查看物流信息？
您可以通过以下方式查看物流信息：
1. 登录账户进入"我的订单"
2. 找到相应订单点击"查看物流"
3. 或者直接在快递公司官网输入运单号查询

### 订单什么时候发货？
订单发货时间根据商品类型而定：
- 现货商品：通常在付款后24小时内发货
- 预售商品：按照商品页面显示的发货时间
- 定制商品：5-7个工作日

## 支付相关

### 支付失败怎么办？
支付失败可能的原因和解决方法：
1. 银行卡余额不足 - 请确认余额充足
2. 网络问题 - 请检查网络连接
3. 银行系统维护 - 请稍后重试
4. 支付限额 - 请联系银行调整限额

### 支持哪些支付方式？
我们支持以下支付方式：
- 微信支付
- 支付宝
- 银行卡支付
- 信用卡支付

## 优惠活动

### 有什么优惠活动吗？
我们经常举办各种优惠活动：
1. 新用户注册即享受优惠券
2. 会员专享折扣
3. 节假日促销活动
4. 满减优惠
建议您关注我们的官方公众号获取最新活动信息。
EOF
        print_success "示例FAQ文件已创建"
    fi
    
    # 创建示例产品信息
    if [ ! -f "data/knowledge_base/products.txt" ]; then
        cat > data/knowledge_base/products.txt << EOF
# 产品信息

## 热销产品

### 智能手机系列
- iPhone 15 Pro: 最新款苹果手机，A17 Pro芯片，钛合金材质
- 小米14: 徕卡影像，骁龙8 Gen3处理器
- 华为Mate60 Pro: 卫星通信，麒麟9000S芯片

### 笔记本电脑系列
- MacBook Pro M3: 苹果最新M3芯片，专业级性能
- 联想ThinkPad X1: 商务办公首选，轻薄便携
- 戴尔XPS 13: 窄边框设计，高清显示屏

### 家电产品
- 海尔冰箱：变频节能，大容量设计
- 美的空调：智能控制，静音运行
- 格力电风扇：多档调速，遥控操作

## 服务保障

### 售后服务
- 7天无理由退换货
- 全国联保服务
- 24小时客服热线

### 配送服务
- 全国包邮
- 当日达服务（部分城市）
- 预约配送时间

### 质量保证
- 正品保证
- 质量问题包换
- 延保服务可选
EOF
        print_success "示例产品信息文件已创建"
    fi
    
    print_success "示例数据创建完成"
}

# 安装Python依赖
install_dependencies() {
    print_info "安装Python依赖..."
    
    if [ -f "requirements.txt" ]; then
        pip3 install -r requirements.txt
        print_success "依赖安装完成"
    else
        print_error "requirements.txt 文件不存在"
        exit 1
    fi
}

# 启动服务（本地模式）
start_local_services() {
    print_info "启动本地服务..."
    
    # 启动API服务
    print_info "启动API服务..."
    python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000 &
    API_PID=$!
    
    # 等待API服务启动
    sleep 5
    
    # 检查API服务是否启动成功
    if curl -f http://localhost:8000/health &> /dev/null; then
        print_success "API服务启动成功 (PID: $API_PID)"
    else
        print_error "API服务启动失败"
        kill $API_PID 2>/dev/null || true
        exit 1
    fi
    
    # 启动前端服务
    print_info "启动前端服务..."
    python3 frontend/gradio_app.py &
    FRONTEND_PID=$!
    
    # 等待前端服务启动
    sleep 5
    
    # 保存PID到文件
    echo $API_PID > .api_pid
    echo $FRONTEND_PID > .frontend_pid
    
    print_success "所有服务启动完成!"
    print_info "服务信息:"
    print_info "  API服务: http://localhost:8000"
    print_info "  前端服务: http://localhost:7860"
    print_info "  API文档: http://localhost:8000/docs"
    print_info ""
    print_info "要停止服务，请运行: ./scripts/stop.sh"
}

# 启动服务（Docker模式）
start_docker_services() {
    print_info "启动Docker服务..."
    
    cd deploy
    
    # 构建并启动服务
    docker-compose up -d --build
    
    print_success "Docker服务启动完成!"
    print_info "服务信息:"
    print_info "  前端服务: http://localhost"
    print_info "  API服务: http://localhost/api"
    print_info "  监控面板: http://localhost:3000 (用户名: admin, 密码: admin)"
    print_info ""
    print_info "要停止服务，请运行: docker-compose down"
}

# 主函数
main() {
    echo "============================================"
    echo "      ECAgent 电商客服助手 启动脚本"
    echo "============================================"
    echo ""
    
    # 检查依赖
    check_dependencies
    
    # 创建目录
    create_directories
    
    # 创建示例数据
    create_sample_data
    
    # 询问启动模式
    echo ""
    print_info "请选择启动模式:"
    echo "1) 本地模式 (直接运行Python服务)"
    echo "2) Docker模式 (使用Docker容器)"
    echo ""
    read -p "请输入选择 (1/2): " choice
    
    case $choice in
        1)
            print_info "选择本地模式"
            install_dependencies
            start_local_services
            ;;
        2)
            if [ "$DOCKER_AVAILABLE" = true ]; then
                print_info "选择Docker模式"
                start_docker_services
            else
                print_error "Docker未安装，无法使用Docker模式"
                exit 1
            fi
            ;;
        *)
            print_error "无效选择，请重新运行脚本"
            exit 1
            ;;
    esac
}

# 处理中断信号
cleanup() {
    print_info "正在关闭服务..."
    if [ -f ".api_pid" ]; then
        kill $(cat .api_pid) 2>/dev/null || true
        rm -f .api_pid
    fi
    if [ -f ".frontend_pid" ]; then
        kill $(cat .frontend_pid) 2>/dev/null || true
        rm -f .frontend_pid
    fi
    print_success "服务已关闭"
    exit 0
}

trap cleanup INT TERM

# 运行主函数
main "$@" 
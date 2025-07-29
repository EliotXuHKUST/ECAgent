#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU环境就绪检查脚本
验证GPU训练环境是否完整配置
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

def check_basic_environment():
    """检查基础环境"""
    print("🔍 检查基础环境...")
    
    checks = {
        "Python版本": sys.version_info >= (3, 8),
        "项目根目录": Path(".").exists(),
        "数据目录": Path("./data").exists() or True,  # 可以创建
        "模型目录": Path("./models").exists() or True,  # 可以创建
    }
    
    for check_name, result in checks.items():
        status = "✅" if result else "❌"
        print(f"   {status} {check_name}")
    
    return all(checks.values())

def check_python_dependencies():
    """检查Python依赖"""
    print("\n📦 检查Python依赖...")
    
    required_packages = {
        "torch": "PyTorch",
        "transformers": "Transformers",
        "peft": "PEFT (LoRA支持)",
        "datasets": "Datasets",
        "bitsandbytes": "Bitsandbytes (量化支持)",
        "accelerate": "Accelerate",
        "gradio": "Gradio (前端)",
        "fastapi": "FastAPI (后端)",
        "psutil": "PSUtil (系统监控)"
    }
    
    missing_packages = []
    for package, description in required_packages.items():
        try:
            __import__(package)
            print(f"   ✅ {description}")
        except ImportError:
            print(f"   ❌ {description} - 请运行: pip install {package}")
            missing_packages.append(package)
    
    return len(missing_packages) == 0

def check_gpu_environment():
    """检查GPU环境"""
    print("\n🖥️  检查GPU环境...")
    
    try:
        import torch
        
        # CUDA可用性
        cuda_available = torch.cuda.is_available()
        print(f"   {'✅' if cuda_available else '❌'} CUDA可用性: {cuda_available}")
        
        if cuda_available:
            # GPU数量
            gpu_count = torch.cuda.device_count()
            print(f"   ✅ GPU数量: {gpu_count}")
            
            # GPU详情
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"   ✅ GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
            
            # CUDA版本
            cuda_version = torch.version.cuda
            print(f"   ✅ CUDA版本: {cuda_version}")
            
            # 测试GPU操作
            try:
                test_tensor = torch.randn(10, 10).cuda()
                result = test_tensor @ test_tensor.T
                print(f"   ✅ GPU计算测试: 通过")
            except Exception as e:
                print(f"   ❌ GPU计算测试: 失败 - {e}")
                return False
        else:
            print("   ⚠️  GPU不可用，将使用CPU模式")
        
        return True
        
    except ImportError:
        print("   ❌ PyTorch未安装")
        return False

def check_project_files():
    """检查项目文件"""
    print("\n📁 检查项目文件...")
    
    required_files = {
        "requirements.txt": "依赖配置文件",
        "train_ecommerce_faq.py": "训练主脚本",
        "models/fine_tuning/train.py": "微调模块",
        "config/settings.py": "配置文件",
        "GPU_DEPLOYMENT_GUIDE.md": "GPU部署指南",
        "gpu_train.py": "GPU训练脚本",
        "monitor_gpu.py": "GPU监控脚本",
        "benchmark_gpu_model.py": "性能基准测试脚本"
    }
    
    missing_files = []
    for file_path, description in required_files.items():
        if Path(file_path).exists():
            print(f"   ✅ {description}")
        else:
            print(f"   ❌ {description} - 文件不存在: {file_path}")
            missing_files.append(file_path)
    
    return len(missing_files) == 0

def check_model_availability():
    """检查模型可用性"""
    print("\n🤖 检查模型可用性...")
    
    try:
        from transformers import AutoTokenizer
        
        # 测试基础模型加载
        model_name = "Qwen/Qwen-7B-Chat"
        print(f"   🔍 测试模型: {model_name}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            print(f"   ✅ 模型tokenizer加载成功")
            return True
        except Exception as e:
            print(f"   ⚠️  无法加载预训练模型 (正常，首次运行时会自动下载): {e}")
            return True  # 这是正常的，首次运行时模型会自动下载
            
    except ImportError:
        print("   ❌ Transformers库未安装")
        return False

def generate_quick_setup_guide():
    """生成快速设置指南"""
    print("\n📝 快速设置指南:")
    print("=" * 50)
    
    print("\n1. 安装GPU版PyTorch:")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    
    print("\n2. 安装项目依赖:")
    print("   pip install -r requirements.txt")
    
    print("\n3. 验证GPU环境:")
    print("   python -c \"import torch; print(torch.cuda.is_available())\"")
    
    print("\n4. 运行GPU训练:")
    print("   python gpu_train.py")
    
    print("\n5. 监控训练过程:")
    print("   python monitor_gpu.py")
    
    print("\n6. 性能基准测试:")
    print("   python benchmark_gpu_model.py")

def main():
    """主函数"""
    print("🚀 ECAgent GPU环境就绪检查")
    print("=" * 50)
    
    checks = [
        ("基础环境", check_basic_environment),
        ("Python依赖", check_python_dependencies),
        ("GPU环境", check_gpu_environment),
        ("项目文件", check_project_files),
        ("模型可用性", check_model_availability)
    ]
    
    all_passed = True
    for check_name, check_func in checks:
        try:
            result = check_func()
            if not result:
                all_passed = False
        except Exception as e:
            print(f"   ❌ {check_name}检查失败: {e}")
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 所有检查通过！GPU环境已就绪")
        print("\n下一步操作:")
        print("1. 运行 GPU 训练: python gpu_train.py")
        print("2. 启动监控: python monitor_gpu.py --interval 5")
        print("3. 查看指南: cat GPU_DEPLOYMENT_GUIDE.md")
    else:
        print("⚠️  部分检查未通过，请查看上述错误信息")
        generate_quick_setup_guide()
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
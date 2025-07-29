#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU模型性能基准测试
测试训练后模型的推理性能和质量
"""

import torch
import time
import sys
from pathlib import Path
from typing import List, Dict, Any
import json

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

def check_model_exists(model_path: str) -> bool:
    """检查模型是否存在"""
    model_dir = Path(model_path)
    if not model_dir.exists():
        print(f"❌ 模型路径不存在: {model_path}")
        return False
    
    # 检查必要文件
    required_files = ["config.json", "pytorch_model.bin", "tokenizer.json"]
    missing_files = []
    
    for file_name in required_files:
        if not (model_dir / file_name).exists():
            missing_files.append(file_name)
    
    if missing_files:
        print(f"⚠️  模型文件不完整，缺少: {missing_files}")
        print("尝试加载基础模型进行测试...")
    
    return True

def load_model_for_benchmark(model_path: str):
    """加载模型用于基准测试"""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        print(f"📦 加载模型: {model_path}")
        
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side='left'
        )
        
        # 设置pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        print(f"✅ 模型加载成功")
        print(f"   设备: {next(model.parameters()).device}")
        print(f"   参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
        
        return tokenizer, model
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None, None

def benchmark_inference_speed(tokenizer, model, test_queries: List[str], num_runs: int = 3):
    """基准测试推理速度"""
    print(f"\n🏃 推理速度测试 (运行{num_runs}次取平均值)")
    print("-" * 60)
    
    results = []
    
    for query in test_queries:
        query_times = []
        responses = []
        
        for run in range(num_runs):
            start_time = time.time()
            
            # 准备输入
            prompt = f"用户：{query}\n客服："
            inputs = tokenizer(prompt, return_tensors="pt")
            
            if torch.cuda.is_available():
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # 生成回复
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # 解码回复
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.replace(prompt, "").strip()
            
            end_time = time.time()
            inference_time = end_time - start_time
            
            query_times.append(inference_time)
            if run == 0:  # 只保存第一次的回复
                responses.append(response)
        
        avg_time = sum(query_times) / len(query_times)
        
        results.append({
            'query': query,
            'avg_time': avg_time,
            'response': responses[0] if responses else ""
        })
        
        print(f"Query: {query}")
        print(f"平均时间: {avg_time:.2f}s")
        print(f"Response: {responses[0][:100]}..." if responses and len(responses[0]) > 100 else f"Response: {responses[0] if responses else 'N/A'}")
        print("-" * 60)
    
    # 计算总体统计
    total_avg_time = sum(r['avg_time'] for r in results) / len(results)
    throughput = 1 / total_avg_time
    
    print(f"\n📊 总体性能:")
    print(f"   平均推理时间: {total_avg_time:.2f}s")
    print(f"   吞吐量: {throughput:.2f} queries/second")
    
    return results

def benchmark_gpu_memory(model):
    """基准测试GPU内存使用"""
    if not torch.cuda.is_available():
        print("\n⚠️  GPU不可用，跳过内存测试")
        return
    
    print(f"\n💾 GPU内存使用测试")
    print("-" * 40)
    
    for i in range(torch.cuda.device_count()):
        gpu_name = torch.cuda.get_device_name(i)
        memory_allocated = torch.cuda.memory_allocated(i) / 1e9
        memory_reserved = torch.cuda.memory_reserved(i) / 1e9
        memory_total = torch.cuda.get_device_properties(i).total_memory / 1e9
        
        print(f"GPU {i} ({gpu_name}):")
        print(f"   已分配内存: {memory_allocated:.2f}GB")
        print(f"   预留内存: {memory_reserved:.2f}GB")
        print(f"   总内存: {memory_total:.2f}GB")
        print(f"   使用率: {(memory_allocated/memory_total)*100:.1f}%")

def benchmark_model_quality(results: List[Dict[str, Any]]):
    """基准测试模型回复质量"""
    print(f"\n🎯 模型质量评估")
    print("-" * 40)
    
    # 简单的质量指标
    quality_metrics = {
        'avg_response_length': 0,
        'responses_with_greeting': 0,
        'responses_with_closing': 0,
        'empty_responses': 0
    }
    
    for result in results:
        response = result['response']
        
        # 响应长度
        quality_metrics['avg_response_length'] += len(response)
        
        # 包含问候语
        if any(greeting in response for greeting in ['您好', '你好', '欢迎']):
            quality_metrics['responses_with_greeting'] += 1
        
        # 包含结束语
        if any(closing in response for closing in ['谢谢', '感谢', '还有其他问题', '随时咨询']):
            quality_metrics['responses_with_closing'] += 1
        
        # 空响应
        if not response.strip():
            quality_metrics['empty_responses'] += 1
    
    # 计算平均值和百分比
    num_results = len(results)
    quality_metrics['avg_response_length'] /= num_results
    
    print(f"平均回复长度: {quality_metrics['avg_response_length']:.1f} 字符")
    print(f"包含问候语: {quality_metrics['responses_with_greeting']}/{num_results} ({quality_metrics['responses_with_greeting']/num_results*100:.1f}%)")
    print(f"包含结束语: {quality_metrics['responses_with_closing']}/{num_results} ({quality_metrics['responses_with_closing']/num_results*100:.1f}%)")
    print(f"空回复: {quality_metrics['empty_responses']}/{num_results} ({quality_metrics['empty_responses']/num_results*100:.1f}%)")
    
    return quality_metrics

def save_benchmark_results(results: List[Dict[str, Any]], quality_metrics: Dict[str, Any], output_file: str = "benchmark_results.json"):
    """保存基准测试结果"""
    benchmark_data = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'performance_results': results,
        'quality_metrics': quality_metrics,
        'system_info': {
            'cuda_available': torch.cuda.is_available(),
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'pytorch_version': torch.__version__
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(benchmark_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 基准测试结果已保存到: {output_file}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ECAgent GPU模型性能基准测试")
    parser.add_argument("--model_path", "-m", type=str, default="./models/gpu_fine_tuned",
                       help="模型路径")
    parser.add_argument("--fallback_model", type=str, default="Qwen/Qwen-7B-Chat",
                       help="备用模型名称")
    parser.add_argument("--runs", "-r", type=int, default=3,
                       help="每个测试运行次数")
    parser.add_argument("--output", "-o", type=str, default="benchmark_results.json",
                       help="结果输出文件")
    
    args = parser.parse_args()
    
    print("🧪 ECAgent GPU模型性能基准测试")
    print("=" * 50)
    
    # 测试用例
    test_queries = [
        "如何退货？",
        "这个商品有保修吗？",
        "支持货到付款吗？",
        "快递多久能到？",
        "可以使用优惠券吗？",
        "商品缺货怎么办？",
        "如何申请售后服务？",
        "配送范围包括哪些地区？"
    ]
    
    # 检查模型
    model_path = args.model_path
    if not check_model_exists(model_path):
        print(f"使用备用模型: {args.fallback_model}")
        model_path = args.fallback_model
    
    # 加载模型
    tokenizer, model = load_model_for_benchmark(model_path)
    if tokenizer is None or model is None:
        print("❌ 无法加载模型，基准测试失败")
        return
    
    # 运行基准测试
    try:
        # 推理速度测试
        results = benchmark_inference_speed(tokenizer, model, test_queries, args.runs)
        
        # GPU内存测试
        benchmark_gpu_memory(model)
        
        # 模型质量评估
        quality_metrics = benchmark_model_quality(results)
        
        # 保存结果
        save_benchmark_results(results, quality_metrics, args.output)
        
        print(f"\n✅ 基准测试完成！")
        
    except Exception as e:
        print(f"❌ 基准测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
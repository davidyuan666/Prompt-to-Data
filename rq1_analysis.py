import re
import numpy as np
import tempfile
import subprocess
import os
import json
import argparse

def extract_python_code(text):
    """从 ChatML 标签中提取代码块"""
    pattern = r"```python\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""

def run_ds1000_test_safe(solution_code, context_code, timeout=10):
    """
    安全地运行 DS-1000 测试逻辑
    """
    try:
        # 构建完整执行代码
        full_exec_code = context_code.replace("[insert]", solution_code)
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(full_exec_code)
            temp_file = f.name
        
        try:
            # 使用子进程运行，限制时间和资源
            result = subprocess.run(
                ['python', temp_file],
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, 'PYTHONPATH': '.'}
            )
            
            # 检查是否成功执行（返回码为0）
            return result.returncode == 0
            
        except subprocess.TimeoutExpired:
            print(f"  Timeout expired for code execution")
            return False
        finally:
            # 清理临时文件
            if os.path.exists(temp_file):
                os.unlink(temp_file)
                
    except Exception as e:
        print(f"  Error in test execution: {str(e)}")
        return False

def evaluate_rq1(filtered_data, use_safe_execution=True, timeout=10):
    """
    RQ1: 评估代码正确性 (Pass@1)
    """
    total_tested = 0
    total_passed = 0
    no_code_count = 0
    execution_errors = 0
    
    print(f"RQ1: Evaluating code correctness for {len(filtered_data)} items...")
    
    for idx, item in enumerate(filtered_data, 1):
        code = extract_python_code(item['full_response'])
        
        if not code:
            no_code_count += 1
            if idx <= 10:  # 只显示前10个无代码的警告
                print(f"  Item {idx}: No Python code found in response")
            continue
        
        # 选择执行方式
        if use_safe_execution:
            is_correct = run_ds1000_test_safe(code, item['code_context'], timeout)
        else:
            # 使用原始的 exec 方式（不安全，仅用于测试）
            try:
                full_exec_code = item['code_context'].replace("[insert]", code)
                local_env = {}
                exec(full_exec_code, local_env)
                local_env['test_execution'](code)
                is_correct = True
            except Exception as e:
                is_correct = False
        
        total_tested += 1
        if is_correct:
            total_passed += 1
        else:
            execution_errors += 1
        
        if idx % 20 == 0:
            print(f"  Processed {idx}/{len(filtered_data)} items...")
    
    # 打印结果
    print(f"\n{'='*60}")
    print("RQ1: Code Correctness Evaluation Results")
    print(f"{'='*60}")
    print(f"Total items analyzed: {len(filtered_data)}")
    print(f"Items with no code: {no_code_count} ({no_code_count/len(filtered_data):.2%})")
    print(f"Items tested: {total_tested}")
    print(f"Items passed: {total_passed}")
    print(f"Execution errors: {execution_errors}")
    print(f"\nPass@1 Score: {total_passed}/{total_tested} = {total_passed/total_tested:.2%}")
    
    # 返回详细结果
    return {
        'total_items': len(filtered_data),
        'no_code_count': no_code_count,
        'total_tested': total_tested,
        'total_passed': total_passed,
        'execution_errors': execution_errors,
        'pass_rate': total_passed / total_tested if total_tested > 0 else 0
    }

def main():
    """主函数：运行 RQ1 分析"""
    parser = argparse.ArgumentParser(description='RQ1 Analysis: Code correctness evaluation')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to filtered data JSONL file')
    parser.add_argument('--unsafe', action='store_true',
                       help='Use unsafe exec() instead of subprocess (not recommended)')
    parser.add_argument('--timeout', type=int, default=10,
                       help='Timeout for code execution in seconds (default: 10)')
    parser.add_argument('--max_items', type=int, default=None,
                       help='Maximum number of items to evaluate (for testing)')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save detailed results JSON file')
    
    args = parser.parse_args()
    
    # 加载数据
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist")
        return
    
    filtered_data = []
    with open(args.input, 'r') as f:
        for line in f:
            filtered_data.append(json.loads(line))
    
    if args.max_items:
        filtered_data = filtered_data[:args.max_items]
        print(f"Evaluating limited to {args.max_items} items")
    
    print(f"Loaded {len(filtered_data)} items from {args.input}")
    
    # 运行 RQ1 评估
    results = evaluate_rq1(
        filtered_data, 
        use_safe_execution=not args.unsafe,
        timeout=args.timeout
    )
    
    # 保存详细结果
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to: {args.output}")
    
    print(f"\nRQ1 analysis completed!")

if __name__ == "__main__":
    main()
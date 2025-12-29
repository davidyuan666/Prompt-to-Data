import json
import tempfile
import subprocess
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any

def extract_code_from_context(code_context: str, reference_code: str) -> str:
    """
    将reference code插入到code context中
    """
    # 找到[insert]标记并替换
    if "[insert]" in code_context:
        return code_context.replace("[insert]", reference_code)
    else:
        # 如果没有[insert]标记，尝试在适当位置插入
        lines = code_context.split('\n')
        for i, line in enumerate(lines):
            if 'test_execution' in line and '(' in line:
                # 在test_execution函数调用前插入
                insert_point = i
                lines.insert(insert_point, reference_code)
                return '\n'.join(lines)
        # 如果找不到合适位置，直接追加
        return code_context + '\n' + reference_code

def run_test_safe(full_code: str, timeout: int = 30) -> Dict[str, Any]:
    """
    安全地运行测试代码
    """
    result = {
        'passed': False,
        'error': None,
        'stdout': '',
        'stderr': '',
        'returncode': -1
    }
    
    try:
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(full_code)
            temp_file = f.name
        
        try:
            # 运行代码
            process = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, 'PYTHONPATH': '.'}
            )
            
            result['stdout'] = process.stdout
            result['stderr'] = process.stderr
            result['returncode'] = process.returncode
            result['passed'] = process.returncode == 0
            
        except subprocess.TimeoutExpired:
            result['error'] = f"Timeout expired after {timeout} seconds"
        except Exception as e:
            result['error'] = f"Execution error: {str(e)}"
        finally:
            # 清理临时文件
            if os.path.exists(temp_file):
                os.unlink(temp_file)
                
    except Exception as e:
        result['error'] = f"File creation error: {str(e)}"
    
    return result

def validate_ds1000_item(item: Dict, timeout: int = 30) -> Dict[str, Any]:
    """
    验证单个DS-1000数据项
    """
    item_id = item.get('metadata', {}).get('problem_id', 'unknown')
    library = item.get('metadata', {}).get('library', 'unknown')
    
    print(f"Validating item {item_id} ({library})...")
    
    # 获取reference code和code context
    reference_code = item.get('reference_code', '')
    code_context = item.get('code_context', '')
    
    if not reference_code:
        return {
            'item_id': item_id,
            'library': library,
            'valid': False,
            'error': 'No reference code found',
            'details': {}
        }
    
    if not code_context:
        return {
            'item_id': item_id,
            'library': library,
            'valid': False,
            'error': 'No code context found',
            'details': {}
        }
    
    # 构建完整代码
    full_code = extract_code_from_context(code_context, reference_code)
    
    # 运行测试
    test_result = run_test_safe(full_code, timeout)
    
    # 分析结果
    if test_result['passed']:
        return {
            'item_id': item_id,
            'library': library,
            'valid': True,
            'error': None,
            'details': {
                'stdout': test_result['stdout'][:500],  # 截断输出
                'stderr': test_result['stderr'][:500],
                'returncode': test_result['returncode']
            }
        }
    else:
        error_msg = test_result['error'] or f"Test failed with return code {test_result['returncode']}"
        if test_result['stderr']:
            error_msg += f"\nStderr: {test_result['stderr'][:200]}"
        
        return {
            'item_id': item_id,
            'library': library,
            'valid': False,
            'error': error_msg,
            'details': {
                'stdout': test_result['stdout'][:500],
                'stderr': test_result['stderr'][:500],
                'returncode': test_result['returncode']
            }
        }

def validate_ds1000_dataset(input_file: str, output_file: str = None, 
                           max_items: int = None, timeout: int = 30) -> Dict[str, Any]:
    """
    验证整个DS-1000数据集
    """
    print(f"Loading dataset from {input_file}...")
    
    # 读取数据
    items = []
    with open(input_file, 'r') as f:
        for line in f:
            try:
                items.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line: {e}")
    
    if max_items:
        items = items[:max_items]
        print(f"Limited to {max_items} items")
    
    print(f"Loaded {len(items)} items")
    
    # 验证每个项目
    results = []
    valid_count = 0
    library_stats = {}
    
    for i, item in enumerate(items, 1):
        result = validate_ds1000_item(item, timeout)
        results.append(result)
        
        if result['valid']:
            valid_count += 1
        
        # 更新库统计
        library = result['library']
        if library not in library_stats:
            library_stats[library] = {'total': 0, 'valid': 0}
        library_stats[library]['total'] += 1
        if result['valid']:
            library_stats[library]['valid'] += 1
        
        # 进度报告
        if i % 10 == 0:
            print(f"Processed {i}/{len(items)} items...")
    
    # 计算总体统计
    total_items = len(items)
    overall_valid_rate = valid_count / total_items if total_items > 0 else 0
    
    # 打印结果摘要
    print(f"\n{'='*60}")
    print("DS-1000 Dataset Validation Results")
    print(f"{'='*60}")
    print(f"Total items: {total_items}")
    print(f"Valid items: {valid_count}")
    print(f"Invalid items: {total_items - valid_count}")
    print(f"Overall validity rate: {overall_valid_rate:.2%}")
    print(f"\nLibrary-wise statistics:")
    
    for library, stats in library_stats.items():
        valid_rate = stats['valid'] / stats['total'] if stats['total'] > 0 else 0
        print(f"  {library}: {stats['valid']}/{stats['total']} = {valid_rate:.2%}")
    
    # 收集详细结果
    summary = {
        'total_items': total_items,
        'valid_count': valid_count,
        'invalid_count': total_items - valid_count,
        'overall_valid_rate': overall_valid_rate,
        'library_stats': library_stats,
        'detailed_results': results
    }
    
    # 保存结果
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nDetailed results saved to: {output_file}")
    
    return summary

'''
python validate_ds1000.py --input ds1000.jsonl --max_items 50 --output results/validation.json
'''
def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Validate DS-1000 dataset by running reference code in context'
    )
    parser.add_argument('--input', type=str, required=True,
                       help='Path to DS-1000 JSONL file')
    parser.add_argument('--output', type=str, default='validation_results.json',
                       help='Path to save validation results (default: validation_results.json)')
    parser.add_argument('--max_items', type=int, default=None,
                       help='Maximum number of items to validate (for testing)')
    parser.add_argument('--timeout', type=int, default=30,
                       help='Timeout for each test execution in seconds (default: 30)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed error messages for failed tests')
    
    args = parser.parse_args()
    
    # 验证数据集
    results = validate_ds1000_dataset(
        input_file=args.input,
        output_file=args.output,
        max_items=args.max_items,
        timeout=args.timeout
    )
    
    # 如果启用详细模式，打印失败详情
    if args.verbose:
        print(f"\n{'='*60}")
        print("Failed Items Details:")
        print(f"{'='*60}")
        
        for result in results['detailed_results']:
            if not result['valid']:
                print(f"\nItem {result['item_id']} ({result['library']}):")
                print(f"  Error: {result['error']}")
                if result['details'].get('stderr'):
                    print(f"  Stderr: {result['details']['stderr'][:200]}...")
    
    print(f"\nValidation completed!")

if __name__ == "__main__":
    main()
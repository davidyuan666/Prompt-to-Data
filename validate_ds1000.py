import json
import tempfile
import subprocess
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any
import shutil

def create_output_directories():
    """创建输出目录结构"""
    directories = [
        "debug_codes",           # 完整代码
        "debug_codes/passed",    # 通过的代码
        "debug_codes/failed",    # 失败的代码
        "debug_codes/errors",    # 有错误的代码
        "validation_results"     # 验证结果
    ]
    
    for dir_name in directories:
        Path(dir_name).mkdir(parents=True, exist_ok=True)
    
    print("Created output directories:")
    for dir_name in directories:
        print(f"  ✓ {dir_name}/")

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

def save_full_code(full_code: str, item_id: str, library: str, status: str = "unknown") -> str:
    """
    保存完整代码到文件
    """
    # 清理文件名中的非法字符
    safe_library = library.replace("/", "_").replace("\\", "_").replace(":", "_")
    
    # 根据状态选择目录
    if status == "passed":
        dir_path = Path("debug_codes/passed")
    elif status == "failed":
        dir_path = Path("debug_codes/failed")
    elif status == "error":
        dir_path = Path("debug_codes/errors")
    else:
        dir_path = Path("debug_codes")
    
    dir_path.mkdir(parents=True, exist_ok=True)
    
    # 生成文件名
    filename = dir_path / f"item_{item_id}_{safe_library}.py"
    
    # 保存代码
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(full_code)
    
    return str(filename)

def run_test_safe(full_code: str, timeout: int = 30) -> Dict[str, Any]:
    """
    安全地运行测试代码
    """
    result = {
        'passed': False,
        'error': None,
        'stdout': '',
        'stderr': '',
        'returncode': -1,
        'execution_time': 0
    }
    
    try:
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(full_code)
            temp_file = f.name
        
        try:
            import time
            start_time = time.time()
            
            # 运行代码
            process = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, 'PYTHONPATH': '.'},
                encoding='utf-8'
            )
            
            end_time = time.time()
            result['execution_time'] = end_time - start_time
            
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

def check_dependencies():
    """
    检查必要的依赖是否已安装
    """
    print("Checking dependencies...")
    
    dependencies = [
        ('pandas', 'pd'),
        ('numpy', 'np'),
        ('scipy', 'scipy'),
        ('sklearn', 'sklearn'),
        ('matplotlib', 'matplotlib'),
        ('tensorflow', 'tensorflow'),
        ('torch', 'torch'),
        ('seaborn', 'seaborn')
    ]
    
    missing_deps = []
    
    for package_name, import_name in dependencies:
        try:
            __import__(import_name if import_name != 'sklearn' else 'sklearn')
            print(f"  ✓ {package_name}")
        except ImportError:
            missing_deps.append(package_name)
            print(f"  ✗ {package_name} (missing)")
    
    if missing_deps:
        print(f"\nMissing dependencies: {', '.join(missing_deps)}")
        print("Please install them using:")
        print(f"  pip install {' '.join(missing_deps)}")
        return False
    
    print("All dependencies are installed!")
    return True

def validate_ds1000_item(item: Dict, timeout: int = 30, save_code: bool = True) -> Dict[str, Any]:
    """
    验证单个DS-1000数据项
    """
    item_id = item.get('metadata', {}).get('problem_id', 'unknown')
    library = item.get('metadata', {}).get('library', 'unknown')
    
    print(f"\n{'='*60}")
    print(f"Validating item {item_id} ({library})...")
    
    # 获取reference code和code context
    reference_code = item.get('reference_code', '')
    code_context = item.get('code_context', '')
    
    if not reference_code:
        print(f"  ✗ No reference code found")
        return {
            'item_id': item_id,
            'library': library,
            'valid': False,
            'error': 'No reference code found',
            'details': {},
            'code_file': None
        }
    
    if not code_context:
        print(f"  ✗ No code context found")
        return {
            'item_id': item_id,
            'library': library,
            'valid': False,
            'error': 'No code context found',
            'details': {},
            'code_file': None
        }
    
    # 构建完整代码
    full_code = extract_code_from_context(code_context, reference_code)
    
    # 保存代码到文件
    code_file = None
    if save_code:
        # 先保存到临时位置，运行后再根据结果移动到相应目录
        temp_file = save_full_code(full_code, item_id, library, "temp")
        code_file = temp_file
    
    # 打印代码摘要
    print(f"  Reference code: {len(reference_code)} chars")
    print(f"  Code context: {len(code_context)} chars")
    print(f"  Full code: {len(full_code)} chars")
    
    # 运行测试
    test_result = run_test_safe(full_code, timeout)
    
    # 分析结果
    if test_result['passed']:
        status = "passed"
        print(f"  ✓ Test passed in {test_result['execution_time']:.2f}s")
        
        # 如果保存了代码，移动到passed目录
        if save_code and code_file:
            final_file = save_full_code(full_code, item_id, library, "passed")
            # 删除临时文件
            if os.path.exists(code_file):
                os.remove(code_file)
            code_file = final_file
        
        return {
            'item_id': item_id,
            'library': library,
            'valid': True,
            'error': None,
            'details': {
                'stdout': test_result['stdout'][:500],
                'stderr': test_result['stderr'][:500],
                'returncode': test_result['returncode'],
                'execution_time': test_result['execution_time']
            },
            'code_file': code_file,
            'full_code': full_code if save_code else None
        }
    else:
        error_msg = test_result['error'] or f"Test failed with return code {test_result['returncode']}"
        print(f"  ✗ Test failed: {error_msg}")
        
        # 确定失败类型
        if test_result['returncode'] == 0 and test_result['stderr']:
            status = "error"
        else:
            status = "failed"
        
        # 如果保存了代码，移动到相应目录
        if save_code and code_file:
            final_file = save_full_code(full_code, item_id, library, status)
            # 保存错误信息到单独文件
            error_file = Path(final_file).with_suffix('.error.txt')
            with open(error_file, 'w', encoding='utf-8') as f:
                f.write(f"Error: {error_msg}\n\n")
                f.write("STDERR:\n")
                f.write(test_result['stderr'])
                f.write("\n\nSTDOUT:\n")
                f.write(test_result['stdout'])
            
            # 删除临时文件
            if os.path.exists(code_file):
                os.remove(code_file)
            code_file = final_file
        
        # 打印详细错误
        if test_result['stderr']:
            print(f"  Error details (first 5 lines):")
            error_lines = test_result['stderr'].split('\n')[:5]
            for line in error_lines:
                if line.strip():
                    print(f"    {line}")
        
        return {
            'item_id': item_id,
            'library': library,
            'valid': False,
            'error': error_msg,
            'details': {
                'stdout': test_result['stdout'][:500],
                'stderr': test_result['stderr'],
                'returncode': test_result['returncode'],
                'execution_time': test_result['execution_time']
            },
            'code_file': code_file,
            'full_code': full_code if save_code else None
        }

def validate_ds1000_dataset(input_file: str, output_file: str = None, 
                           max_items: int = None, timeout: int = 30,
                           save_code: bool = True) -> Dict[str, Any]:
    """
    验证整个DS-1000数据集
    """
    print(f"Loading dataset from {input_file}...")
    
    # 创建输出目录
    create_output_directories()
    
    # 读取数据
    items = []
    with open(input_file, 'r', encoding='utf-8') as f:
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
        result = validate_ds1000_item(
            item, 
            timeout=timeout, 
            save_code=save_code
        )
        results.append(result)
        
        if result['valid']:
            valid_count += 1
        
        # 更新库统计
        library = result['library']
        if library not in library_stats:
            library_stats[library] = {'total': 0, 'valid': 0, 'failed': 0, 'errors': 0}
        library_stats[library]['total'] += 1
        if result['valid']:
            library_stats[library]['valid'] += 1
        elif result.get('error', '').startswith('Test failed'):
            library_stats[library]['failed'] += 1
        else:
            library_stats[library]['errors'] += 1
        
        # 进度报告
        if i % 5 == 0:
            current_rate = valid_count / i if i > 0 else 0
            print(f"\nProgress: {i}/{len(items)} | Valid: {valid_count} | Rate: {current_rate:.2%}")
    
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
    
    # 打印库统计
    print(f"\nLibrary-wise statistics:")
    print(f"{'Library':<15} {'Total':<8} {'Valid':<8} {'Failed':<8} {'Errors':<8} {'Rate':<8}")
    print(f"{'-'*60}")
    
    for library, stats in sorted(library_stats.items()):
        valid_rate = stats['valid'] / stats['total'] if stats['total'] > 0 else 0
        print(f"{library:<15} {stats['total']:<8} {stats['valid']:<8} {stats['failed']:<8} {stats['errors']:<8} {valid_rate:.2%}")
    
    # 收集详细结果
    summary = {
        'total_items': total_items,
        'valid_count': valid_count,
        'invalid_count': total_items - valid_count,
        'overall_valid_rate': overall_valid_rate,
        'library_stats': library_stats,
        'detailed_results': [
            {
                'item_id': r['item_id'],
                'library': r['library'],
                'valid': r['valid'],
                'error': r['error'],
                'code_file': r.get('code_file')
            }
            for r in results
        ]
    }
    
    # 保存结果
    if output_file:
        output_path = Path("validation_results") / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\nDetailed results saved to: {output_path}")
    
    # 保存完整的详细结果
    full_results_file = Path("validation_results") / "full_validation_results.json"
    with open(full_results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return summary

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Validate DS-1000 dataset by running reference code in context'
    )
    parser.add_argument('--input', type=str, required=True,
                       help='Path to DS-1000 JSONL file')
    parser.add_argument('--output', type=str, default='validation_summary.json',
                       help='Path to save validation summary (default: validation_summary.json)')
    parser.add_argument('--max_items', type=int, default=None,
                       help='Maximum number of items to validate (for testing)')
    parser.add_argument('--timeout', type=int, default=30,
                       help='Timeout for each test execution in seconds (default: 30)')
    parser.add_argument('--no_save_code', action='store_true',
                       help='Do not save full code to files')
    parser.add_argument('--check_deps', action='store_true',
                       help='Check dependencies before validation')
    
    args = parser.parse_args()
    
    # 检查依赖
    if args.check_deps:
        if not check_dependencies():
            print("\nPlease install missing dependencies and try again.")
            return
    
    # 验证数据集
    results = validate_ds1000_dataset(
        input_file=args.input,
        output_file=args.output,
        max_items=args.max_items,
        timeout=args.timeout,
        save_code=not args.no_save_code
    )
    
    # 打印目录结构
    print(f"\n{'='*60}")
    print("Output Directory Structure:")
    print(f"{'='*60}")
    print("debug_codes/")
    print("  ├── passed/      # 测试通过的代码")
    print("  ├── failed/      # 测试失败的代码")
    print("  └── errors/      # 运行出错的代码")
    print("validation_results/")
    print("  ├── validation_summary.json  # 验证摘要")
    print("  └── full_validation_results.json  # 完整结果")
    
    print(f"\nValidation completed!")
    print(f"Summary: {results['valid_count']}/{results['total_items']} valid ({results['overall_valid_rate']:.2%})")

if __name__ == "__main__":
    main()
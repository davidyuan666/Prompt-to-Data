import json
import numpy as np
import os
import argparse
from collections import defaultdict

def analyze_library_performance(filtered_data):
    """
    RQ3: 分析不同库的表现差异
    """
    # 初始化统计
    lib_stats = defaultdict(lambda: {
        'total': 0,
        'passed': 0,
        'etd_ratios': [],
        'avg_entropies': []
    })
    
    # 收集数据
    for item in filtered_data:
        lib = item['library']
        stats = lib_stats[lib]
        
        stats['total'] += 1
        
        # 检查是否通过测试（如果有测试结果）
        if 'test_passed' in item:
            if item['test_passed']:
                stats['passed'] += 1
        
        # 收集 ETD 相关数据
        if 'etd_ratio' in item:
            stats['etd_ratios'].append(item['etd_ratio'])
        
        if 'avg_entropy' in item:
            stats['avg_entropies'].append(item['avg_entropy'])
    
    # 计算统计指标
    results = {}
    for lib, stats in lib_stats.items():
        if stats['total'] > 0:
            pass_rate = stats['passed'] / stats['total'] if 'test_passed' in filtered_data[0] else None
            
            results[lib] = {
                'total_items': stats['total'],
                'pass_rate': pass_rate,
                'avg_etd_ratio': np.mean(stats['etd_ratios']) if stats['etd_ratios'] else None,
                'std_etd_ratio': np.std(stats['etd_ratios']) if stats['etd_ratios'] else None,
                'avg_entropy': np.mean(stats['avg_entropies']) if stats['avg_entropies'] else None,
                'std_entropy': np.std(stats['avg_entropies']) if stats['avg_entropies'] else None
            }
    
    return results

def print_rq3_results(results):
    """打印 RQ3 分析结果"""
    print(f"\n{'='*70}")
    print("RQ3: Library Performance Analysis")
    print(f"{'='*70}")
    
    # 按通过率排序（如果有测试结果）
    sorted_libs = sorted(results.items(), 
                        key=lambda x: x[1]['pass_rate'] if x[1]['pass_rate'] is not None else 0, 
                        reverse=True)
    
    print(f"{'Library':<15} {'Items':<8} {'Pass Rate':<12} {'Avg ETD':<10} {'Avg Entropy':<12}")
    print(f"{'-'*15} {'-'*8} {'-'*12} {'-'*10} {'-'*12}")
    
    for lib, stats in sorted_libs:
        pass_rate_str = f"{stats['pass_rate']:.2%}" if stats['pass_rate'] is not None else "N/A"
        avg_etd_str = f"{stats['avg_etd_ratio']:.3f}" if stats['avg_etd_ratio'] is not None else "N/A"
        avg_entropy_str = f"{stats['avg_entropy']:.3f}" if stats['avg_entropy'] is not None else "N/A"
        
        print(f"{lib:<15} {stats['total_items']:<8} {pass_rate_str:<12} {avg_etd_str:<10} {avg_entropy_str:<12}")
    
    # 总体统计
    print(f"\n{'='*70}")
    print("Overall Statistics:")
    
    total_items = sum(stats['total_items'] for stats in results.values())
    print(f"Total libraries analyzed: {len(results)}")
    print(f"Total items across all libraries: {total_items}")
    
    # 如果有测试结果，计算总体通过率
    if all(stats['pass_rate'] is not None for stats in results.values()):
        total_passed = sum(stats['pass_rate'] * stats['total_items'] for stats in results.values())
        overall_pass_rate = total_passed / total_items
        print(f"Overall pass rate: {overall_pass_rate:.2%}")
    
    # ETD 和熵的总体统计
    all_etd_ratios = []
    all_entropies = []
    
    for stats in results.values():
        if stats['avg_etd_ratio'] is not None:
            all_etd_ratios.extend([stats['avg_etd_ratio']] * stats['total_items'])
        if stats['avg_entropy'] is not None:
            all_entropies.extend([stats['avg_entropy']] * stats['total_items'])
    
    if all_etd_ratios:
        print(f"Overall average ETD ratio: {np.mean(all_etd_ratios):.3f} ± {np.std(all_etd_ratios):.3f}")
    
    if all_entropies:
        print(f"Overall average entropy: {np.mean(all_entropies):.3f} ± {np.std(all_entropies):.3f}")

def main():
    """主函数：运行 RQ3 分析"""
    parser = argparse.ArgumentParser(description='RQ3 Analysis: Library performance comparison')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to filtered data JSONL file')
    parser.add_argument('--rq1_results', type=str, default=None,
                       help='Path to RQ1 results JSON file (for test pass rates)')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save RQ3 results JSON file')
    parser.add_argument('--plot', action='store_true',
                       help='Generate visualization plots')
    
    args = parser.parse_args()
    
    # 加载数据
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist")
        return
    
    filtered_data = []
    with open(args.input, 'r') as f:
        for line in f:
            filtered_data.append(json.loads(line))
    
    print(f"Loaded {len(filtered_data)} items from {args.input}")
    
    # 如果提供了 RQ1 结果，合并测试通过信息
    if args.rq1_results and os.path.exists(args.rq1_results):
        # 这里可以添加代码来合并 RQ1 的测试结果
        # 例如，可以从 RQ1 结果中读取每个项目的测试状态
        print(f"Note: RQ1 results merging not implemented in this version")
    
    # 运行 RQ3 分析
    results = analyze_library_performance(filtered_data)
    
    # 打印结果
    print_rq3_results(results)
    
    # 保存结果
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nRQ3 results saved to: {args.output}")
    
    # 生成可视化（如果请求）
    if args.plot:
        generate_visualizations(results, filtered_data)
    
    print(f"\nRQ3 analysis completed!")

def generate_visualizations(results, filtered_data):
    """生成可视化图表"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        print("\nGenerating visualizations...")
        
        # 设置样式
        sns.set_style("whitegrid")
        plt.figure(figsize=(15, 10))
        
        # 1. 库分布条形图
        plt.subplot(2, 2, 1)
        libraries = list(results.keys())
        counts = [results[lib]['total_items'] for lib in libraries]
        
        bars = plt.bar(libraries, counts)
        plt.title('Distribution of Items by Library')
        plt.xlabel('Library')
        plt.ylabel('Number of Items')
        plt.xticks(rotation=45, ha='right')
        
        # 添加数值标签
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom', fontsize=9)
        
        # 2. 通过率条形图（如果有测试结果）
        if all(results[lib]['pass_rate'] is not None for lib in libraries):
            plt.subplot(2, 2, 2)
            pass_rates = [results[lib]['pass_rate'] for lib in libraries]
            
            bars = plt.bar(libraries, pass_rates)
            plt.title('Pass Rate by Library')
            plt.xlabel('Library')
            plt.ylabel('Pass Rate')
            plt.xticks(rotation=45, ha='right')
            plt.ylim(0, 1.1)
            
            # 添加百分比标签
            for bar, rate in zip(bars, pass_rates):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{rate:.1%}', ha='center', va='bottom', fontsize=9)
        
        # 3. ETD 比率箱线图
        plt.subplot(2, 2, 3)
        etd_data = []
        etd_labels = []
        
        for lib in libraries:
            if results[lib]['avg_etd_ratio'] is not None:
                # 收集该库的所有 ETD 比率
                lib_etd_ratios = [item['etd_ratio'] for item in filtered_data 
                                 if item['library'] == lib and 'etd_ratio' in item]
                if lib_etd_ratios:
                    etd_data.append(lib_etd_ratios)
                    etd_labels.append(lib)
        
        if etd_data:
            plt.boxplot(etd_data, labels=etd_labels)
            plt.title('ETD Ratio Distribution by Library')
            plt.xlabel('Library')
            plt.ylabel('ETD Ratio')
            plt.xticks(rotation=45, ha='right')
        
        # 4. 熵值散点图
        plt.subplot(2, 2, 4)
        for lib in libraries:
            if results[lib]['avg_entropy'] is not None:
                # 收集该库的所有熵值
                lib_entropies = [item['avg_entropy'] for item in filtered_data 
                                if item['library'] == lib and 'avg_entropy' in item]
                if lib_entropies:
                    x_pos = libraries.index(lib) + 1
                    plt.scatter([x_pos] * len(lib_entropies), lib_entropies, 
                               alpha=0.6, label=lib)
        
        plt.title('Entropy Distribution by Library')
        plt.xlabel('Library')
        plt.ylabel('Average Entropy')
        plt.xticks(range(1, len(libraries) + 1), libraries, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig('rq3_visualization.png', dpi=300, bbox_inches='tight')
        print("Visualization saved as 'rq3_visualization.png'")
        
    except ImportError:
        print("Visualization requires matplotlib and seaborn. Install with: pip install matplotlib seaborn")

if __name__ == "__main__":
    main()
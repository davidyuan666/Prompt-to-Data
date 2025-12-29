import json
import numpy as np
import argparse
import os

def calculate_token_entropy(token_logprobs):
    """计算单个 Token 的 Shannon 熵"""
    # 转换为概率分布
    probs = np.exp([lp['logprob'] for lp in token_logprobs])
    probs = probs / np.sum(probs)
    return -np.sum(probs * np.log2(probs + 1e-9))

def process_rq2_and_filter(synthesis_path, tau=1.5):
    """
    对应论文 RQ2: 过滤幻觉并统计格式错误
    """
    valid_data = []
    fer_count = 0
    total = 0
    
    with open(synthesis_path, 'r') as f:
        for line in f:
            total += 1
            item = json.loads(line)
            
            # 1. 检查格式 (FER)
            if "<|im_start|>" not in item['full_response'] or "<|im_end|>" not in item['full_response']:
                fer_count += 1
                continue
            
            # 2. 计算平均熵 (ETD)
            # 注意：现在 logprobs 是序列化后的字典格式
            if item['logprobs'] is None:
                continue  # 跳过没有 logprobs 的数据
                
            # 计算每个 token 的熵
            token_entropies = []
            for token_data in item['logprobs']:
                if 'top_logprobs' in token_data and token_data['top_logprobs']:
                    entropy = calculate_token_entropy(token_data['top_logprobs'])
                    token_entropies.append(entropy)
            
            if not token_entropies:  # 如果没有有效的熵值，跳过
                continue
                
            avg_entropy = np.mean(token_entropies)
            ref_entropy = 1.0  # 基准参考熵
            etd_ratio = avg_entropy / ref_entropy
            
            # 只有熵比值在阈值 tau 以下的才被视为"聚焦"的对齐数据
            if etd_ratio < tau:
                item['etd_ratio'] = etd_ratio
                item['avg_entropy'] = avg_entropy
                valid_data.append(item)
                
    print(f"RQ2 Analysis Results:")
    print(f"  Total items processed: {total}")
    print(f"  Format Error Rate (FER): {fer_count}/{total} = {fer_count/total:.2%}")
    print(f"  ETD Passed: {len(valid_data)}/{total} = {len(valid_data)/total:.2%}")
    print(f"  Filtered out: {total - len(valid_data) - fer_count} items (high entropy)")
    
    return valid_data

def save_filtered_data(filtered_data, output_path):
    """保存过滤后的数据"""
    with open(output_path, 'w') as f:
        for item in filtered_data:
            f.write(json.dumps(item) + "\n")
    print(f"Filtered data saved to: {output_path}")
    print(f"Total filtered items: {len(filtered_data)}")

def analyze_entropy_distribution(filtered_data):
    """分析熵值分布"""
    if not filtered_data:
        print("No data to analyze")
        return
    
    etd_ratios = [item['etd_ratio'] for item in filtered_data]
    avg_entropies = [item['avg_entropy'] for item in filtered_data]
    
    print("\nEntropy Distribution Analysis:")
    print(f"  ETD Ratio - Min: {np.min(etd_ratios):.3f}, Max: {np.max(etd_ratios):.3f}, Mean: {np.mean(etd_ratios):.3f}")
    print(f"  Avg Entropy - Min: {np.min(avg_entropies):.3f}, Max: {np.max(avg_entropies):.3f}, Mean: {np.mean(avg_entropies):.3f}")
    
    # 统计不同阈值下的通过率
    thresholds = [1.0, 1.2, 1.5, 2.0]
    print("\nPass rate at different thresholds:")
    for threshold in thresholds:
        passed = sum(1 for ratio in etd_ratios if ratio < threshold)
        print(f"  tau={threshold}: {passed}/{len(etd_ratios)} = {passed/len(etd_ratios):.2%}")

def main():
    """主函数：运行 RQ2 分析"""
    parser = argparse.ArgumentParser(description='RQ2 Analysis: Filter hallucinations and format errors')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to synthesis output JSONL file')
    parser.add_argument('--output', type=str, default='filtered_data.jsonl',
                       help='Path to save filtered data (default: filtered_data.jsonl)')
    parser.add_argument('--tau', type=float, default=1.5,
                       help='ETD threshold (default: 1.5)')
    parser.add_argument('--analyze', action='store_true',
                       help='Show detailed entropy distribution analysis')
    parser.add_argument('--no_save', action='store_true',
                       help='Do not save filtered data, just show statistics')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist")
        return
    
    print(f"Starting RQ2 Analysis...")
    print(f"  Input file: {args.input}")
    print(f"  ETD threshold (tau): {args.tau}")
    
    # 运行 RQ2 分析
    filtered_data = process_rq2_and_filter(args.input, tau=args.tau)
    
    # 分析熵值分布
    if args.analyze and filtered_data:
        analyze_entropy_distribution(filtered_data)
    
    # 保存过滤后的数据
    if not args.no_save and filtered_data:
        save_filtered_data(filtered_data, args.output)
    
    print("\nRQ2 Analysis completed!")


'''
python rq2_analysis.py --input synthesis_output.jsonl --output filtered_data.jsonl --tau 1.5 --analyze
'''
if __name__ == "__main__":
    main()
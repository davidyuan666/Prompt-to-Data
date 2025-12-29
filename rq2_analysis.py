def calculate_token_entropy(token_logprobs):
    """计算单个 Token 的 Shannon 熵"""
    # 转换为概率分布
    probs = np.exp([lp.logprob for lp in token_logprobs])
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
            # 这里的参考熵 H_orig 实际操作中通常取 DS-1000 均值 (~0.8-1.2)
            avg_entropy = np.mean([calculate_token_entropy(tp.top_logprobs) for tp in item['logprobs']])
            ref_entropy = 1.0  # 基准参考熵
            etd_ratio = avg_entropy / ref_entropy
            
            # 只有熵比值在阈值 tau 以下的才被视为“聚焦”的对齐数据
            if etd_ratio < tau:
                item['etd_ratio'] = etd_ratio
                valid_data.append(item)
                
    print(f"RQ2 Result: FER = {fer_count/total:.2%}, ETD Passed = {len(valid_data)}/{total}")
    return valid_data
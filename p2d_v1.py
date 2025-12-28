import os
import json
import re
import pandas as pd
from typing import List, Dict
from openai import OpenAI
from tqdm import tqdm

# =================配置区域=================
API_KEY = os.environ.get("DEEPSEEK_API_KEY")
BASE_URL = "https://api.deepseek.com"
TEACHER_MODEL = "deepseek-coder" 

# Qwen/ChatML 格式 Token
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

# 实验设定的三种引导风格
STYLES = {
    "Direct": "Provide a concise code solution.",
    "CoT": "Provide a step-by-step reasoning process before the code.",
    "Debugging": "Identify potential errors and provide a robust fix."
}
# ==========================================

class P2DSynthesisReporter:
    def __init__(self, api_key: str, base_url: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.report_data = []

    def parse_ds1000(self, line: str) -> Dict:
        data = json.loads(line)
        # 提取核心问题描述
        raw_prompt = data.get("prompt", "")
        clean_desc = raw_prompt.split("A:\n\n")[0].replace("Problem:\n\n", "").strip()
        return {
            "id": data.get("metadata", {}).get("problem_id"),
            "library": data.get("metadata", {}).get("library"),
            "description": clean_desc,
            "reference": data.get("reference_code", "")
        }

    def analyze_token_metrics(self, gen_text: str, ref_code: str, style: str):
        """核心指标计算：Token 权重比例与变化范围"""
        # 提取生成的代码块
        code_match = re.search(r"```python\n(.*?)```", gen_text, re.DOTALL)
        gen_code = code_match.group(1).strip() if code_match else ""
        
        # 计算 Token 数 (按空格粗略计算，或使用 tiktoken)
        gen_tokens = len(gen_text.split())
        ref_tokens = len(ref_code.split())
        
        # Token Weight Ratio (TWR)
        twr = round(gen_tokens / ref_tokens, 2) if ref_tokens > 0 else 0
        
        return {
            "style": style,
            "gen_len": gen_tokens,
            "ref_len": ref_tokens,
            "twr": twr,
            "code_alignment": 1 if gen_code.strip() == ref_code.strip() else 0
        }

    def generate_and_save(self, input_file: str, sft_output: str, report_output: str):
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        sft_samples = []
        
        print(f"🚀 Starting P2D synthesis and report generation...")
        for line in tqdm(lines[:20]): # 示例处理前20条
            seed = self.parse_ds1000(line)
            
            for style_name, style_prompt in STYLES.items():
                meta_p = (
                    f"{style_prompt}\nContext: {seed['description']}\n"
                    f"Format: {IM_START}user\n[Instruction]\n{IM_END}\n"
                    f"{IM_START}assistant\n[Solution]\n{IM_END}"
                )
                
                # 调用模型
                response = self.client.chat.completions.create(
                    model=TEACHER_MODEL,
                    messages=[{"role": "user", "content": meta_p}],
                    temperature=0.7
                )
                raw_res = response.choices[0].message.content
                
                # 分析数据
                metrics = self.analyze_token_metrics(raw_res, seed['reference'], style_name)
                metrics['problem_id'] = seed['id']
                self.report_data.append(metrics)
                
                # 保存为微调格式
                sft_samples.append({"messages": [{"role": "system", "content": style_prompt}, 
                                               {"role": "user", "content": seed['description']},
                                               {"role": "assistant", "content": raw_res}]})

        # --- 保存文件 ---
        # 1. 保存微调数据集 (JSONL)
        with open(sft_output, 'w', encoding='utf-8') as f:
            for sample in sft_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        # 2. 保存实验报告 (JSON & CSV 为方便分析)
        df_report = pd.DataFrame(self.report_data)
        df_report.to_csv(report_output.replace('.json', '.csv'), index=False)
        with open(report_output, 'w') as f:
            json.dump(self.report_data, f, indent=4)

        print(f"\n✅ SFT Data saved to: {sft_output}")
        print(f"✅ Experiment Report saved to: {report_output}")

# 执行
if __name__ == "__main__":
    reporter = P2DSynthesisReporter(API_KEY, BASE_URL)
    reporter.generate_and_save("ds1000.jsonl", "p2d_sft_train.jsonl", "p2d_experiment_report.json")
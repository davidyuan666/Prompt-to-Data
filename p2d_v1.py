import os
import json
import re
import ast
from typing import List, Dict, Optional
from openai import OpenAI
from tqdm import tqdm

# =================CONFIGURATION=================
API_KEY = "sk-834575b2a7414832bd59f6a60117999f" 
BASE_URL = "https://api.deepseek.com"
TEACHER_MODEL = "deepseek-coder" 

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

# 定义不同的遵循风格 (Section 3.4 提到的 Persona)
PERSONAS = {
    "concise": "Act as a direct software engineer. Provide a minimal and clean solution.",
    "tutorial": "Act as a technical instructor. Use Chain-of-Thought to explain the code steps.",
    "robust": "Act as a QA engineer. Ensure the code handles edge cases and is robust."
}
# ===============================================

class DS1000P2DAnalyzer:
    def __init__(self, api_key: str, base_url: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def parse_line(self, line: str) -> Dict:
        """解析 DS-1000 原始行，提取核心 Prompt 和 Reference"""
        data = json.loads(line)
        # 提取原始描述部分
        raw_prompt = data.get("prompt", "")
        # DS-1000 通常在 "Problem:\n\n" 之后描述逻辑
        description = raw_prompt.split("A:\n\n")[0].replace("Problem:\n\n", "").strip()
        
        return {
            "description": description,
            "reference": data.get("reference_code", ""),
            "library": data.get("metadata", {}).get("library", "Python"),
            "problem_id": data.get("metadata", {}).get("problem_id", "0")
        }

    def construct_styled_prompt(self, seed: Dict, style: str) -> str:
        """根据风格构建 Meta-Prompt"""
        persona = PERSONAS.get(style)
        return (
            f"{persona}\n\n"
            f"Context: {seed['description']}\n"
            f"Library: {seed['library']}\n\n"
            f"CRITICAL FORMATTING REQUIREMENT:\n"
            f"Use strict ChatML tokens:\n"
            f"{IM_START}user\n[Synthesize a natural instruction for this problem]\n{IM_END}\n"
            f"{IM_START}assistant\n[Provide the Python solution]\n{IM_END}"
        )

    def get_completion(self, prompt: str) -> str:
        try:
            res = self.client.chat.completions.create(
                model=TEACHER_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            return res.choices[0].message.content
        except: return ""

    def calculate_metrics(self, gen_text: str, ref_code: str):
        """计算 Token 比例和代码变化范围"""
        # 提取 assistant 中的代码部分
        code_match = re.search(r"assistant\n(.*?)(?=<\|im_end\|>|$)", gen_text, re.DOTALL)
        gen_code = code_match.group(1).strip() if code_match else gen_text
        
        gen_tokens = len(gen_code.split())
        ref_tokens = len(ref_code.split())
        
        # Token 比例 (Weight Ratio)
        ratio = round(gen_tokens / ref_tokens if ref_tokens > 0 else 0, 2)
        
        return {
            "gen_code": gen_code,
            "ratio": ratio,
            "len_diff": gen_tokens - ref_tokens
        }

    def process_file(self, input_path: str, limit: int = 5):
        final_report = []
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()[:limit]

        for line in tqdm(lines, desc="Analyzing DS-1000 P2D"):
            seed = self.parse_line(line)
            entry = {"problem_id": seed['problem_id'], "styles": {}}
            
            for style in PERSONAS.keys():
                meta_p = self.construct_styled_prompt(seed, style)
                raw_response = self.get_completion(meta_p)
                metrics = self.calculate_metrics(raw_response, seed['reference'])
                
                entry["styles"][style] = {
                    "token_ratio": metrics['ratio'],
                    "generated_code": metrics['gen_code'],
                    "reference_code": seed['reference']
                }
            final_report.append(entry)
            
        return final_report

if __name__ == "__main__":
    analyzer = DS1000P2DAnalyzer(API_KEY, BASE_URL)
    # 运行分析并打印结果
    report = analyzer.process_file("ds1000.jsonl")
    print(json.dumps(report, indent=2))
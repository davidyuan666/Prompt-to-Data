#!/usr/bin/env python3
"""
RQ1: Format Compliance Rate Analysis
实验目的：评估P2D框架生成的指令对在目标格式（ChatML/Qwen）上的合规率
"""

import os
import json
import re
from typing import List, Dict, Tuple
from openai import OpenAI
from tqdm import tqdm

class RQ1FormatCompliance:
    """
    RQ1: 分析P2D框架生成的指令对在ChatML/Qwen格式上的合规率
    """
    
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.teacher_model = "deepseek-coder"
        self.im_start = "<|im_start|>"
        self.im_end = "<|im_end|>"
        
    def construct_meta_prompt(self, problem_description: str, library: str = "python") -> str:
        """
        构建元提示，强制模型充当'格式工程师'
        """
        return (
            f"You are an expert Data Science Instruction Architect.\n"
            f"Your goal is to synthesize a high-quality instruction-response pair "
            f"based on the following problem context using the {library} library.\n\n"
            f"Context: {problem_description}\n\n"
            f"CRITICAL FORMATTING REQUIREMENT (Strict ChatML):\n"
            f"The output must be strictly wrapped in specific tokens exactly as follows:\n"
            f"{self.im_start}user\n[Synthesized Instruction]\n{self.im_end}\n"
            f"{self.im_start}assistant\n[Python Code Solution with CoT explanation]\n{self.im_end}\n\n"
            f"Ensure the code is self-contained, executable, and uses {library} best practices."
        )
    
    def generate_with_retry(self, prompt: str, max_retries: int = 3) -> str:
        """生成响应，带重试机制"""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.teacher_model,
                    messages=[
                        {"role": "system", "content": "You are a strict Format Engineer."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=2048
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    return ""
        return ""
    
    def verify_chatml_format(self, text: str) -> Tuple[bool, Dict]:
        """验证ChatML格式合规性"""
        pattern = re.compile(
            r"<\|im_start\|>user\s*(.*?)<\|im_end\|>\s*<\|im_start\|>assistant\s*(.*?)<\|im_end\|>", 
            re.DOTALL
        )
        match = pattern.search(text)
        
        if match:
            return True, {
                "instruction": match.group(1).strip(),
                "response": match.group(2).strip(),
                "raw_output": text
            }
        return False, {}
    
    def run_experiment(self, seed_problems: List[Dict], output_file: str = "rq1_results.json"):
        """运行RQ1实验"""
        results = {
            "total_samples": len(seed_problems),
            "compliant_samples": 0,
            "compliance_rate": 0.0,
            "detailed_results": [],
            "error_analysis": {
                "format_errors": 0,
                "generation_errors": 0,
                "empty_responses": 0
            }
        }
        
        print(f"🚀 Starting RQ1 Experiment: Format Compliance Analysis")
        print(f"Testing {len(seed_problems)} seed problems...")
        
        for problem in tqdm(seed_problems, desc="Processing"):
            problem_id = problem.get("id", "unknown")
            library = problem.get("library", "python")
            description = problem.get("description", "")
            
            # 1. 构建元提示
            meta_prompt = self.construct_meta_prompt(description, library)
            
            # 2. 生成响应
            raw_output = self.generate_with_retry(meta_prompt)
            
            if not raw_output:
                results["error_analysis"]["empty_responses"] += 1
                results["detailed_results"].append({
                    "problem_id": problem_id,
                    "status": "ERROR_EMPTY_RESPONSE",
                    "library": library
                })
                continue
            
            # 3. 验证格式
            is_compliant, parsed_data = self.verify_chatml_format(raw_output)
            
            if is_compliant:
                results["compliant_samples"] += 1
                status = "COMPLIANT"
                
                # 提取代码块进行进一步分析
                code_blocks = re.findall(r"```python\n(.*?)```", parsed_data["response"], re.DOTALL)
                has_code = len(code_blocks) > 0
                
                parsed_data["has_code_block"] = has_code
                parsed_data["code_block_count"] = len(code_blocks)
                
            else:
                results["error_analysis"]["format_errors"] += 1
                status = "NON_COMPLIANT"
                
                # 分析可能的格式错误
                has_user_tag = self.im_start + "user" in raw_output
                has_assistant_tag = self.im_start + "assistant" in raw_output
                has_end_tags = self.im_end in raw_output
                
                parsed_data = {
                    "format_analysis": {
                        "has_user_tag": has_user_tag,
                        "has_assistant_tag": has_assistant_tag,
                        "has_end_tags": has_end_tags,
                        "raw_output_preview": raw_output[:200] + "..." if len(raw_output) > 200 else raw_output
                    }
                }
            
            results["detailed_results"].append({
                "problem_id": problem_id,
                "status": status,
                "library": library,
                "data": parsed_data
            })
        
        # 计算合规率
        results["compliance_rate"] = results["compliant_samples"] / results["total_samples"] if results["total_samples"] > 0 else 0
        
        # 保存结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 打印摘要
        print(f"\n📊 RQ1 Experiment Results:")
        print(f"   Total Samples: {results['total_samples']}")
        print(f"   Compliant Samples: {results['compliant_samples']}")
        print(f"   Compliance Rate: {results['compliance_rate']:.2%}")
        print(f"\n🔍 Error Analysis:")
        print(f"   Format Errors: {results['error_analysis']['format_errors']}")
        print(f"   Generation Errors: {results['error_analysis']['generation_errors']}")
        print(f"   Empty Responses: {results['error_analysis']['empty_responses']}")
        print(f"\n💾 Results saved to: {output_file}")
        
        return results

def load_ds1000_samples(file_path: str, sample_count: int = 50) -> List[Dict]:
    """从DS-1000数据集加载样本"""
    samples = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= sample_count:
                    break
                
                data = json.loads(line)
                
                # 提取问题描述
                raw_prompt = data.get("prompt", "")
                clean_desc = raw_prompt.split("A:\n\n")[0].replace("Problem:\n\n", "").strip()
                
                samples.append({
                    "id": data.get("metadata", {}).get("problem_id", f"ds1000_{i}"),
                    "library": data.get("metadata", {}).get("library", "python"),
                    "description": clean_desc,
                    "original_data": data
                })
    except FileNotFoundError:
        print(f"⚠️  Warning: {file_path} not found. Using mock data.")
        # 使用模拟数据
        samples = [
            {
                "id": "mock_pandas_01",
                "library": "pandas",
                "description": "Given a DataFrame df, filter rows where column 'A' is greater than 5 and calculate the mean of column 'B'."
            },
            {
                "id": "mock_numpy_02",
                "library": "numpy",
                "description": "Create a 3x3 identity matrix and multiply it by a random 3x1 vector."
            },
            {
                "id": "mock_matplotlib_03",
                "library": "matplotlib",
                "description": "Create a line plot with x values from 0 to 10 and y values as the square of x."
            }
        ]
    
    return samples

if __name__ == "__main__":
    # 配置
    API_KEY = os.environ.get("DEEPSEEK_API_KEY", "your_api_key_here")
    
    if API_KEY == "your_api_key_here":
        print("⚠️  Please set DEEPSEEK_API_KEY environment variable or edit this script.")
        print("   You can create a .env file with: DEEPSEEK_API_KEY=your_key_here")
        exit(1)
    
    # 加载数据
    print("📂 Loading DS-1000 samples...")
    seed_problems = load_ds1000_samples("ds1000.jsonl", sample_count=30)
    
    # 运行实验
    experiment = RQ1FormatCompliance(API_KEY)
    results = experiment.run_experiment(seed_problems, "experiment_rq1_results.json")
    
    # 生成可视化摘要
    print("\n📈 Summary Visualization:")
    print("   " + "█" * int(results["compliance_rate"] * 50) + " " * (50 - int(results["compliance_rate"] * 50)))
    print(f"   {'Compliance Rate:':<20} {results['compliance_rate']:.2%}")
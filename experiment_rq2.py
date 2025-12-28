#!/usr/bin/env python3
"""
RQ2: Token Weight Ratio (TWR) and Response Quality Analysis
实验目的：分析不同提示风格对生成响应长度、质量和Token权重比例的影响
"""

import os
import json
import re
import pandas as pd
from typing import List, Dict, Tuple
from openai import OpenAI
from tqdm import tqdm
import numpy as np

class RQ2TokenAnalysis:
    """
    RQ2: 分析不同提示风格下的Token权重比例和响应质量
    """
    
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.teacher_model = "deepseek-coder"
        self.im_start = "<|im_start|>"
        self.im_end = "<|im_end|>"
        
        # 定义三种提示风格
        self.prompt_styles = {
            "Direct": "Provide a concise code solution.",
            "CoT": "Provide a step-by-step reasoning process before the code.",
            "Debugging": "Identify potential errors and provide a robust fix."
        }
    
    def construct_style_prompt(self, problem_description: str, style: str, library: str = "python") -> str:
        """构建不同风格的提示"""
        style_instruction = self.prompt_styles.get(style, "Provide a solution.")
        
        return (
            f"{style_instruction}\n"
            f"Context: {problem_description}\n"
            f"Library: {library}\n"
            f"Format: {self.im_start}user\n[Instruction]\n{self.im_end}\n"
            f"{self.im_start}assistant\n[Solution]\n{self.im_end}"
        )
    
    def generate_response(self, prompt: str) -> str:
        """生成响应"""
        try:
            response = self.client.chat.completions.create(
                model=self.teacher_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2048
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Generation error: {e}")
            return ""
    
    def calculate_token_metrics(self, generated_text: str, reference_code: str = "") -> Dict:
        """计算Token相关指标"""
        # 简单分词（实际中可以使用tiktoken）
        gen_tokens = len(generated_text.split())
        ref_tokens = len(reference_code.split()) if reference_code else 0
        
        # Token Weight Ratio (TWR)
        twr = round(gen_tokens / ref_tokens, 3) if ref_tokens > 0 else 0
        
        # 提取代码块
        code_match = re.search(r"```python\n(.*?)```", generated_text, re.DOTALL)
        gen_code = code_match.group(1).strip() if code_match else ""
        
        # 代码对齐度（简单版本）
        code_alignment = 1.0 if gen_code.strip() == reference_code.strip() else 0.0
        
        # 计算代码比例
        code_ratio = len(gen_code.split()) / gen_tokens if gen_tokens > 0 else 0
        
        return {
            "generated_tokens": gen_tokens,
            "reference_tokens": ref_tokens,
            "token_weight_ratio": twr,
            "has_code_block": bool(code_match),
            "code_alignment": code_alignment,
            "code_ratio": round(code_ratio, 3),
            "extracted_code": gen_code[:100] + "..." if len(gen_code) > 100 else gen_code
        }
    
    def analyze_response_quality(self, generated_text: str) -> Dict:
        """分析响应质量"""
        # 检查是否包含解释
        has_explanation = any(keyword in generated_text.lower() 
                             for keyword in ["explanation", "reasoning", "step", "because", "therefore"])
        
        # 检查是否包含错误处理
        has_error_handling = any(keyword in generated_text.lower() 
                                for keyword in ["error", "exception", "try", "except", "check", "validate"])
        
        # 检查代码结构
        has_imports = "import " in generated_text
        has_functions = "def " in generated_text or "lambda " in generated_text
        
        # 计算可读性指标（简单版本）
        lines = generated_text.split('\n')
        avg_line_length = sum(len(line) for line in lines) / len(lines) if lines else 0
        
        return {
            "has_explanation": has_explanation,
            "has_error_handling": has_error_handling,
            "has_imports": has_imports,
            "has_functions": has_functions,
            "line_count": len(lines),
            "avg_line_length": round(avg_line_length, 1),
            "readability_score": min(10, max(1, 10 - (avg_line_length - 80) / 20))  # 简单评分
        }
    
    def run_experiment(self, seed_problems: List[Dict], output_prefix: str = "rq2") -> Dict:
        """运行RQ2实验"""
        all_results = []
        
        print(f"🚀 Starting RQ2 Experiment: Token Weight Ratio Analysis")
        print(f"Testing {len(seed_problems)} problems with {len(self.prompt_styles)} styles...")
        
        for problem in tqdm(seed_problems, desc="Problems"):
            problem_id = problem.get("id", "unknown")
            library = problem.get("library", "python")
            description = problem.get("description", "")
            reference_code = problem.get("reference_code", "")
            
            for style_name in self.prompt_styles.keys():
                # 1. 构建风格化提示
                prompt = self.construct_style_prompt(description, style_name, library)
                
                # 2. 生成响应
                generated_text = self.generate_response(prompt)
                
                if not generated_text:
                    continue
                
                # 3. 计算Token指标
                token_metrics = self.calculate_token_metrics(generated_text, reference_code)
                
                # 4. 分析响应质量
                quality_metrics = self.analyze_response_quality(generated_text)
                
                # 5. 记录结果
                result = {
                    "problem_id": problem_id,
                    "library": library,
                    "prompt_style": style_name,
                    "generated_text_preview": generated_text[:200] + "..." if len(generated_text) > 200 else generated_text,
                    **token_metrics,
                    **quality_metrics
                }
                
                all_results.append(result)
        
        # 转换为DataFrame进行统计分析
        df = pd.DataFrame(all_results)
        
        # 计算总体统计
        summary_stats = {
            "total_samples": len(all_results),
            "problems_tested": len(seed_problems),
            "styles_tested": list(self.prompt_styles.keys()),
            "by_style": {},
            "by_library": {}
        }
        
        # 按风格统计
        for style in self.prompt_styles.keys():
            style_df = df[df["prompt_style"] == style]
            if not style_df.empty:
                summary_stats["by_style"][style] = {
                    "avg_tokens": round(style_df["generated_tokens"].mean(), 1),
                    "avg_twr": round(style_df["token_weight_ratio"].mean(), 3),
                    "avg_code_ratio": round(style_df["code_ratio"].mean(), 3),
                    "explanation_rate": round(style_df["has_explanation"].mean(), 3),
                    "error_handling_rate": round(style_df["has_error_handling"].mean(), 3),
                    "sample_count": len(style_df)
                }
        
        # 按库统计
        libraries = df["library"].unique()
        for lib in libraries:
            lib_df = df[df["library"] == lib]
            if not lib_df.empty:
                summary_stats["by_library"][lib] = {
                    "avg_tokens": round(lib_df["generated_tokens"].mean(), 1),
                    "avg_twr": round(lib_df["token_weight_ratio"].mean(), 3),
                    "sample_count": len(lib_df)
                }
        
        # 保存详细结果和摘要
        detailed_file = f"{output_prefix}_detailed_results.json"
        summary_file = f"{output_prefix}_summary_stats.json"
        csv_file = f"{output_prefix}_results.csv"
        
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_stats, f, indent=2, ensure_ascii=False)
        
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        # 打印结果摘要
        self.print_summary(summary_stats, df)
        
        return {
            "detailed_results": all_results,
            "summary_stats": summary_stats,
            "dataframe": df
        }
    
    def print_summary(self, summary_stats: Dict, df: pd.DataFrame):
        """打印实验摘要"""
        print(f"\n📊 RQ2 Experiment Results:")
        print(f"   Total Generated Samples: {summary_stats['total_samples']}")
        print(f"   Problems Tested: {summary_stats['problems_tested']}")
        print(f"   Prompt Styles: {', '.join(summary_stats['styles_tested'])}")
        
        print(f"\n🔍 Analysis by Prompt Style:")
        print("   Style           | Avg Tokens | Avg TWR | Code Ratio | Explanation | Error Handling")
        print("   " + "-" * 80)
        
        for style, stats in summary_stats["by_style"].items():
            print(f"   {style:<15} | {stats['avg_tokens']:>10} | {stats['avg_twr']:>7.3f} | "
                  f"{stats['avg_code_ratio']:>10.3f} | {stats['explanation_rate']:>11.3f} | {stats['error_handling_rate']:>13.3f}")
        
        print(f"\n📈 Key Findings:")
        
        # 找到最佳风格
        if summary_stats["by_style"]:
            best_style = max(summary_stats["by_style"].items(), 
                           key=lambda x: x[1]["explanation_rate"] + x[1]["error_handling_rate"])
            print(f"   1. Best overall style: {best_style[0]} "
                  f"(Explanation: {best_style[1]['explanation_rate']:.1%}, "
                  f"Error Handling: {best_style[1]['error_handling_rate']:.1%})")
            
            # Token效率分析
            most_efficient = min(summary_stats["by_style"].items(), key=lambda x: x[1]["avg_tokens"])
            print(f"   2. Most token-efficient: {most_efficient[0]} "
                  f"({most_efficient[1]['avg_tokens']:.0f} tokens on average)")
            
            # 代码质量分析
            highest_code_ratio = max(summary_stats["by_style"].items(), key=lambda x: x[1]["avg_code_ratio"])
            print(f"   3. Highest code-to-text ratio: {highest_code_ratio[0]} "
                  f"({highest_code_ratio[1]['avg_code_ratio']:.1%} code content)")
        
        print(f"\n💾 Results saved to:")
        print(f"   Detailed JSON: rq2_detailed_results.json")
        print(f"   Summary JSON: rq2_summary_stats.json")
        print(f"   CSV: rq2_results.csv")

def load_ds1000_with_references(file_path: str, sample_count: int = 20) -> List[Dict]:
    """从DS-1000加载带参考代码的样本"""
    samples = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= sample_count:
                    break
                
                data = json.loads(line)
                
                # 提取问题描述和参考代码
                raw_prompt = data.get("prompt", "")
                clean_desc = raw_prompt.split("A:\n\n")[0].replace("Problem:\n\n", "").strip()
                reference_code = data.get("reference_code", "")
                
                samples.append({
                    "id": data.get("metadata", {}).get("problem_id", f"ds1000_{i}"),
                    "library": data.get("metadata", {}).get("library", "python"),
                    "description": clean_desc,
                    "reference_code": reference_code,
                    "original_data": data
                })
    except FileNotFoundError:
        print(f"⚠️  Warning: {file_path} not found. Using mock data.")
        # 使用模拟数据
        samples = [
            {
                "id": "mock_pandas_01",
                "library": "pandas",
                "description": "Given a DataFrame df, filter rows where column 'A' is greater than 5.",
                "reference_code": "df[df['A'] > 5]"
            },
            {
                "id": "mock_numpy_02",
                "library": "numpy",
                "description": "Create a 3x3 identity matrix.",
                "reference_code": "import numpy as np\nnp.eye(3)"
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
    print("📂 Loading DS-1000 samples with references...")
    seed_problems = load_ds1000_with_references("ds1000.jsonl", sample_count=15)
    
    # 运行实验
    experiment = RQ2TokenAnalysis(API_KEY)
    results = experiment.run_experiment(seed_problems, "experiment_rq2")
    
    print("\n✅ RQ2 Experiment completed successfully!")
import os
import json
import re
import ast
from typing import List, Dict, Optional
from openai import OpenAI
from tqdm import tqdm

# =================CONFIGURATION=================
# 模拟论文中提到的设置 (Section 5.1)
API_KEY = os.environ.get("DEEPSEEK_API_KEY")
BASE_URL = "https://api.deepseek.com" # DeepSeek V2 API 地址
TEACHER_MODEL = "deepseek-coder" 

# Qwen/ChatML 格式的特殊 Token
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
# ===============================================

class P2DGenerator:
    """
    Implementation of the Prompt-to-Data (P2D) Framework.
    Corresponds to Section 3 of the paper.
    """
    
    def __init__(self, api_key: str, base_url: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        
    def construct_meta_prompt(self, seed_problem: Dict) -> str:
        """
        构建 Meta-Prompt (对应论文 Section 3.2 & 3.4).
        强制 Teacher Model 充当 'Format Engineer'。
        """
        library = seed_problem.get('library', 'python')
        description = seed_problem.get('description', '')
        
        # 定义目标 Schema (ChatML)
        schema_instruction = (
            f"You are an expert Data Science Instruction Architect.\n"
            f"Your goal is to synthesize a high-quality instruction-response pair "
            f"based on the following problem context using the {library} library.\n\n"
            f"Context: {description}\n\n"
            f"CRITICAL FORMATTING REQUIREMENT (Strict ChatML):\n"
            f"The output must be strictly wrapped in specific tokens exactly as follows:\n"
            f"{IM_START}user\n[Synthesized Instruction]\n{IM_END}\n"
            f"{IM_START}assistant\n[Python Code Solution with CoT explanation]\n{IM_END}\n\n"
            f"Ensure the code is self-contained, executable, and uses {library} best practices."
        )
        return schema_instruction

    def generate_raw_data(self, prompt: str) -> str:
        """调用 DeepSeek 生成原始数据"""
        try:
            response = self.client.chat.completions.create(
                model=TEACHER_MODEL,
                messages=[
                    {"role": "system", "content": "You are a strict Format Engineer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7, # 增加多样性 (RQ1)
                max_tokens=2048
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API Error: {e}")
            return ""

    def verify_schema(self, raw_text: str) -> Optional[Dict]:
        """
        验证格式合规性 (RQ2: Format Adherence).
        使用 Regex 解析 ChatML 结构。
        """
        # 正则匹配 Qwen/ChatML 格式
        pattern = re.compile(
            r"<\|im_start\|>user\s*(.*?)<\|im_end\|>\s*<\|im_start\|>assistant\s*(.*?)<\|im_end\|>", 
            re.DOTALL
        )
        match = pattern.search(raw_text)
        
        if match:
            return {
                "instruction": match.group(1).strip(),
                "output": match.group(2).strip()
            }
        return None

    def unit_test_proxy(self, code_content: str) -> bool:
        """
        逻辑正确性检查 (Section 3.5).
        在实际生产中应使用 sandbox 运行 (e.g. Docker).
        这里使用 AST 解析作为轻量级语法检查。
        """
        try:
            # 提取代码块 (假设代码在 ```python ... ``` 中)
            code_blocks = re.findall(r"```python\n(.*?)```", code_content, re.DOTALL)
            if not code_blocks:
                # 尝试直接解析（如果回复纯粹是代码）
                ast.parse(code_content)
            else:
                for block in code_blocks:
                    ast.parse(block)
            return True
        except SyntaxError:
            return False

    def run_pipeline(self, seed_dataset: List[Dict], output_file: str):
        """执行完整的 P2D 流程 (Algorithm 1)"""
        aligned_data = []
        
        print(f"Starting P2D Synthesis on {len(seed_dataset)} seeds...")
        
        for seed in tqdm(seed_dataset):
            # 1. Construct Prompt
            meta_prompt = self.construct_meta_prompt(seed)
            
            # 2. Generate (Attempt until valid or max retries)
            valid = False
            retries = 0
            while not valid and retries < 3:
                raw_output = self.generate_raw_data(meta_prompt)
                
                # 3. Format Alignment Check
                parsed = self.verify_schema(raw_output)
                
                if parsed:
                    # 4. Logical/Syntax Check
                    if self.unit_test_proxy(parsed['output']):
                        # 构建最终的训练数据样本
                        training_sample = {
                            "messages": [
                                {"role": "user", "content": parsed['instruction']},
                                {"role": "assistant", "content": parsed['output']}
                            ],
                            "source": "P2D_DeepSeek_Synthesized"
                        }
                        aligned_data.append(training_sample)
                        valid = True
                    else:
                        retries += 1
                else:
                    retries += 1
        
        # 5. Save to JSONL
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in aligned_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"\nSynthesis Complete. {len(aligned_data)} aligned samples saved to {output_file}")
        print(f"Format Compliance Rate: {len(aligned_data)/len(seed_dataset):.1%}")

# =================USAGE EXAMPLE=================
if __name__ == "__main__":
    # 模拟 DS-1000 Seed Data (Section 3.3)
    mock_seeds = [
        {
            "id": "ds1000_pandas_01",
            "library": "pandas",
            "description": "Given a DataFrame df, filter rows where column 'A' is greater than 5 and calculate the mean of column 'B'."
        },
        {
            "id": "ds1000_numpy_02",
            "library": "numpy",
            "description": "Create a 3x3 identity matrix and multiply it by a random 3x1 vector."
        }
    ]

    # 初始化生成器
    # 注意：运行前请设置环境变量或直接替换 API Key
    api_key = os.getenv("DEEPSEEK_API_KEY", "sk-placeholder") 
    
    p2d = P2DGenerator(api_key=api_key, base_url=BASE_URL)
    
    # 运行合成流程
    p2d.run_pipeline(mock_seeds, "qwen_finetune_data.jsonl")
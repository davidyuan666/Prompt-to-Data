import json
import openai
import numpy as np
from tqdm import tqdm
import os  # Added missing import

# Fixed environment variable access and added error handling
api_key = os.environ.get('DEEPSEEK_API_KEY')
if not api_key:
    raise ValueError("DEEPSEEK_API_KEY environment variable is not set")

client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

def get_p2d_synthesis(problem_prompt, library):
    """
    Non-agentic Single-pass Synthesis
    """
    meta_prompt = f"Act as an Instruction Architect. Convert this {library} problem into a ChatML training pair. \nSeed: {problem_prompt}"
    
    # 必须开启 logprobs 以支持后续的 ETD 计算
    response = client.chat.completions.create(
        model="deepseek-coder",
        messages=[{"role": "user", "content": meta_prompt}],
        temperature=0.3,
        logprobs=True,
        top_logprobs=5
    )
    return response

def run_synthesis(input_path, output_path):
    with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line in tqdm(f_in, desc="P2D Synthesis"):
            item = json.loads(line)
            res = get_p2d_synthesis(item['prompt'], item['metadata']['library'])
            
            # 保存合成文本及对应的 logprobs 用于 RQ2
            output_item = {
                "id": item['metadata']['problem_id'],
                "library": item['metadata']['library'],
                "full_response": res.choices[0].message.content,
                "logprobs": res.choices[0].logprobs.content, # 关键：存储 Token 级概率
                "reference_code": item['reference_code'],
                "code_context": item['code_context']
            }
            f_out.write(json.dumps(output_item) + "\n")
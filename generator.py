import json
import openai
import numpy as np
from tqdm import tqdm
import os
import sys
import argparse
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

# Fixed environment variable access and added error handling
api_key = os.environ.get('DEEPSEEK_API_KEY')
if not api_key:
    # 尝试其他可能的变量名
    api_key = os.environ.get('DEEPSEEK_API_KEY') or os.environ.get('DEEPSEEK_KEY')
    
if not api_key:
    raise ValueError("DEEPSEEK_API_KEY environment variable is not set. "
                     "Please set it in your environment or create a .env file.")

client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

def serialize_logprobs(logprobs):
    """
    Convert logprobs object to JSON serializable format
    """
    if logprobs is None:
        return None
    
    serialized = []
    for logprob in logprobs:
        if hasattr(logprob, 'token'):
            item = {
                'token': logprob.token,
                'logprob': logprob.logprob,
                'bytes': logprob.bytes if hasattr(logprob, 'bytes') else None,
                'top_logprobs': []
            }
            
            # Handle top_logprobs if present
            if hasattr(logprob, 'top_logprobs') and logprob.top_logprobs:
                for top_logprob in logprob.top_logprobs:
                    top_item = {
                        'token': top_logprob.token,
                        'logprob': top_logprob.logprob,
                        'bytes': top_logprob.bytes if hasattr(top_logprob, 'bytes') else None
                    }
                    item['top_logprobs'].append(top_item)
            
            serialized.append(item)
    return serialized

def get_p2d_synthesis(problem_prompt, library, reference_code, code_context):
    """
    Non-agentic Single-pass Synthesis for DS1000 format
    """
    # Create a prompt specifically for DS1000 format problems
    meta_prompt = f"""Act as an Instruction Architect. Convert this {library} programming problem from DS1000 dataset into a ChatML training pair.

Problem Description:
{problem_prompt}

Reference Solution Code:
{reference_code}

Code Context (for testing):
{code_context}

Generate a training example in the following format:
1. A clear instruction based on the problem description
2. The correct solution code (same as reference_code)
3. Brief explanation of the solution

The training example should be suitable for fine-tuning a code assistant model."""

    # 必须开启 logprobs 以支持后续的 ETD 计算
    response = client.chat.completions.create(
        model="deepseek-coder",
        messages=[{"role": "user", "content": meta_prompt}],
        temperature=0.3,
        logprobs=True,
        top_logprobs=5
    )
    return response

def run_synthesis(input_path, output_path, max_items=None):
    """
    Run synthesis with optional limit on number of items
    """
    with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
        lines = list(f_in)
        
        if max_items:
            lines = lines[:max_items]
            print(f"Processing limited to {max_items} items")
        
        for line in tqdm(lines, desc="P2D Synthesis", total=len(lines)):
            try:
                item = json.loads(line)
                
                # Extract all necessary fields from DS1000 format
                problem_prompt = item['prompt']
                library = item['metadata']['library']
                reference_code = item['reference_code']
                code_context = item['code_context']
                
                res = get_p2d_synthesis(
                    problem_prompt=problem_prompt,
                    library=library,
                    reference_code=reference_code,
                    code_context=code_context
                )
                
                # 序列化 logprobs 以便 JSON 存储
                serialized_logprobs = serialize_logprobs(res.choices[0].logprobs.content)
                
                # 保存合成文本及对应的 logprobs 用于 RQ2
                output_item = {
                    "id": item['metadata']['problem_id'],
                    "library": library,
                    "full_response": res.choices[0].message.content,
                    "logprobs": serialized_logprobs,  # 使用序列化后的 logprobs
                    "reference_code": reference_code,
                    "code_context": code_context,
                    "metadata": item['metadata'],
                    "original_prompt": problem_prompt
                }
                f_out.write(json.dumps(output_item) + "\n")
                
            except Exception as e:
                print(f"Error processing item: {str(e)}")
                continue

def main():
    """
    Main function to run the synthesis process
    """
    parser = argparse.ArgumentParser(description='Generate P2D synthesis from DS1000 dataset')
    parser.add_argument('--input', type=str, required=True, 
                       help='Path to input DS1000.jsonl file')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to output JSONL file')
    parser.add_argument('--api_key', type=str, default=None,
                       help='DeepSeek API key (optional, uses DEEPSEEK_API_KEY env var by default)')
    parser.add_argument('--max_items', type=int, default=None,
                       help='Maximum number of items to process (for testing)')
    parser.add_argument('--env_file', type=str, default='.env',
                       help='Path to .env file (default: .env)')
    
    args = parser.parse_args()
    
    # 加载指定的环境文件
    if os.path.exists(args.env_file):
        load_dotenv(args.env_file)
        print(f"Loaded environment variables from: {args.env_file}")
    
    # Override API key if provided via command line
    if args.api_key:
        global api_key, client
        api_key = args.api_key
        os.environ['DEEPSEEK_API_KEY'] = args.api_key
        client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        print("Using API key from command line argument")
    else:
        # 重新检查环境变量（可能在 .env 文件中）
        api_key = os.environ.get('DEEPSEEK_API_KEY') or os.environ.get('DEEPSEEK_KEY')
        if not api_key:
            print("Error: DEEPSEEK_API_KEY not found in environment variables or .env file")
            print("Please provide API key via --api_key argument or set it in .env file")
            sys.exit(1)
        print("Using API key from environment variables")
    
    print(f"Processing DS1000 dataset from: {args.input}")
    print(f"Output will be saved to: {args.output}")
    
    try:
        # Check if input file exists
        if not os.path.exists(args.input):
            print(f"Error: Input file '{args.input}' does not exist")
            sys.exit(1)
        
        # Process the dataset
        run_synthesis(args.input, args.output, args.max_items)
        
        print("Synthesis completed successfully!")
        
    except Exception as e:
        print(f"Error during synthesis: {str(e)}")
        sys.exit(1)


'''
python generator.py --input ds1000.jsonl --output test_output.jsonl --max_items 2
'''
if __name__ == "__main__":
    main()
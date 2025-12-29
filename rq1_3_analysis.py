import re

def extract_python_code(text):
    """从 ChatML 标签中提取代码块"""
    pattern = r"```python\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""

def run_ds1000_test_logic(solution_code, context_code):
    """
    利用 DS-1000 自带的 test_execution 逻辑进行验证
    """
    try:
        # 构建执行环境
        full_exec_code = context_code.replace("[insert]", solution_code)
        
        # 准备沙盒环境
        # 注意：在服务器运行建议使用 multiprocessing 限制时长
        local_env = {}
        exec(full_exec_code, local_env)
        
        # 调用 DS-1000 内部定义的 test_execution 函数
        # solution_code 会被传递给该函数进行 assertion 检查
        local_env['test_execution'](solution_code)
        return True
    except Exception as e:
        return False

def evaluate_rq1(filtered_data):
    lib_results = {}
    for item in filtered_data:
        code = extract_python_code(item['full_response'])
        is_correct = run_ds1000_test_logic(code, item['code_context'])
        
        lib = item['library']
        if lib not in lib_results: lib_results[lib] = []
        lib_results[lib].append(is_correct)
        
    for lib, results in lib_results.items():
        print(f"Library {lib} Pass@1: {np.mean(results):.2%}")
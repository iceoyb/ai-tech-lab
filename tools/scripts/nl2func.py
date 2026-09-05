#!/usr/bin/env python3
"""
自然语言文本函数生成器 (Compile-by-Training Demo)

灵感来自论文 "Compile by Training: Turning Natural-Language Specifications 
into Local Neural Functions"。

用大模型把自然语言描述的文本处理需求，生成一个独立可用的 Python 函数。

用法:
    python3 nl2func.py "把输入字符串中的邮箱地址全部脱敏" --input "test@example.com"
    python3 nl2func.py "提取文本中所有的手机号码" --file input.txt
    python3 nl2func.py "把中文日期转换成 YYYY-MM-DD 格式" --save converter.py
"""

import argparse
import sys
import os
import json
import subprocess
import tempfile
from pathlib import Path


def load_env():
    env_path = Path.home() / ".hermes" / ".env"
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, val = line.split('=', 1)
            val = val.split(' #')[0].strip().strip('"').strip("'")
            if key not in os.environ or not os.environ[key]:
                os.environ[key] = val


def get_ark_key():
    load_env()
    return os.environ.get('ARK_API_KEY', '')


ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/plan/v3"
DEFAULT_MODEL = "kimi-k3"


def call_llm(prompt, max_tokens=2000, temperature=0.3):
    """调用大模型返回文本"""
    api_key = get_ark_key()
    if not api_key:
        raise RuntimeError("ARK_API_KEY is not configured")
    
    payload = {
        "model": DEFAULT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "thinking": {"type": "disabled"}
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(payload, f)
        tmpfile = f.name
    
    try:
        cmd = [
            "curl", "-s",
            ARK_BASE_URL + "/chat/completions",
            "-H", "Authorization: Bearer " + api_key,
            "-H", "Content-Type: application/json",
            "--data-binary", "@" + tmpfile
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        data = json.loads(r.stdout)
        
        if 'choices' in data and data['choices']:
            content = data['choices'][0]['message'].get('content', '')
            if not content:
                content = data['choices'][0]['message'].get('reasoning_content', '')
            return content.strip()
        else:
            raise RuntimeError("API error: " + str(data.get('error', r.stdout[:200])))
    finally:
        os.unlink(tmpfile)


def generate_function(spec):
    """根据自然语言描述生成 Python 函数"""
    lines = [
        "You are a senior Python engineer. Generate a standalone Python function based on the user's natural language description.",
        "",
        "## User Requirement",
        spec,
        "",
        "## Requirements",
        "1. Generate a function named process_text that takes a string input and returns the processed result",
        "2. The function must be robust, handling edge cases (empty string, invalid input, encoding issues)",
        "3. Use only Python standard library, no external dependencies",
        "4. Include detailed comments and docstring",
        "5. Include test cases in if __name__ == '__main__' block at the end",
        "",
        "## Output Format",
        "Output ONLY Python code, no explanation text. Wrap code in ```python ``` blocks.",
    ]
    prompt = "\n".join(lines)
    
    result = call_llm(prompt, max_tokens=2000, temperature=0.3)
    
    # Extract code from markdown code blocks
    if "```python" in result:
        code = result.split("```python")[1].split("```")[0].strip()
    elif "```" in result:
        code = result.split("```")[1].split("```")[0].strip()
    else:
        code = result.strip()
    
    return code


def main():
    parser = argparse.ArgumentParser(
        description="Natural Language to Function Generator (Compile-by-Training Demo)"
    )
    parser.add_argument("spec", help="Natural language description of the function")
    parser.add_argument("--input", "-i", help="Input text to process directly")
    parser.add_argument("--file", "-f", help="Read input text from file")
    parser.add_argument("--save", "-s", help="Save generated function to Python file")
    parser.add_argument("--show-code", action="store_true", help="Show the generated code")
    
    args = parser.parse_args()
    
    print("Generating function: " + args.spec)
    print()
    
    code = generate_function(args.spec)
    
    if args.show_code or args.save:
        print("Generated code:")
        print("-" * 60)
        print(code)
        print("-" * 60)
        print()
    
    if args.save:
        with open(args.save, 'w') as f:
            f.write(code)
        print("Saved to: " + args.save)
        print()
    
    input_text = None
    if args.input:
        input_text = args.input
    elif args.file:
        with open(args.file) as f:
            input_text = f.read()
    
    if input_text:
        print("Running test:")
        print("=" * 60)
        
        local_ns = {}
        exec(code, local_ns)
        
        if 'process_text' in local_ns:
            result = local_ns['process_text'](input_text)
            print("Input (" + str(len(input_text)) + " chars):")
            print(input_text[:200] + ('...' if len(input_text) > 200 else ''))
            print()
            print("Output (" + str(len(result)) + " chars):")
            print(result[:200] + ('...' if len(result) > 200 else result))
        else:
            print("Warning: process_text function not found in generated code")
    
    else:
        print("Running built-in test cases:")
        print("=" * 60)
        exec(code)


if __name__ == "__main__":
    main()

# -*- coding = utf-8 -*-
# @Time : 2023/8/31 23:36
# @Author: Tan Qiyuan
# @File: implementation
import ast
import re

from utils.LLMs.LLMsAdapter import LLMsAdapter, Charactor
from utils.prompts.Prompter import Prompter


def crop_string(input_string):
    index1 = input_string.find('\ndef')
    index2 = input_string.find('\nif')
    if index2 > index1 > -1:
        return input_string[:index1]
    return input_string[:index2]


def extract_function(text):
    # 使用正则表达式找到以"def"开始，后面跟着函数名的部分
    pattern = r"```python\s+(.*?)\s+```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        function_block = match.group(1).strip()
        # 再次使用正则表达式找到以"def"开始的部分，以定位函数体
        function_start = function_block.find("def ")
        if function_start != -1:
            return extract_function_body((function_block[function_start:]))
    return text


def extract_function_body(func_str):
    parsed_ast = ast.parse(func_str)
    function_defs = [node for node in parsed_ast.body if isinstance(node, ast.FunctionDef)]
    if len(function_defs) == 1:
        body = function_defs[0].body
        original_source_lines = func_str.splitlines()
        start_line = body[0].lineno - 1
        end_line = body[-1].end_lineno
        body_lines = original_source_lines[start_line:end_line]
        body_str = '\n'.join(body_lines)

        return body_str
    else:
        return "Error: Input should contain exactly one function definition."


def self_planning(llm_adapter: LLMsAdapter, prompt, prompter: Prompter = Prompter()) -> str:
    planning_prompt: str = planning(llm_adapter, prompt, prompter)
    planning_prompt = planning_prompt[:planning_prompt.find('"""')] + '"""'
    prompt = prompt[:-4] + 'Let’s think step by step.' + planning_prompt
    # print(f"start of writing prompt \n{prompt} \n end of writing prompt")
    return crop_string(llm_adapter.completion(prompt))


def planning(llm_adapter: LLMsAdapter, prompt: str, prompter: Prompter = Prompter()) -> str:
    prompt: str = prompter.PLANNING_PROMPT + prompt[:-4] + 'Let’s think step by step.'
    return llm_adapter.completion(prompt)


def self_collaboration(llm_adapter: LLMsAdapter, prompt: str, prompter: Prompter = Prompter()) -> str:
    analyst: Charactor = llm_adapter.get_charactor(prompter.ANALYST, 'analyst')
    developer: Charactor = llm_adapter.get_charactor(prompter.DEVELOPER, 'developer')
    tester: Charactor = llm_adapter.get_charactor(prompter.TESTER, 'tester')
    code: str = ''

    user_requirements: dict[str, str] = {'role': 'user',
                                         'content': f"The user requests the team to write a complete implementation "
                                                    f"for this function:\n {prompt}"}
    messages: list[dict[str, str]] = [user_requirements,
                                      {'role': 'user', 'content': f'The analyst says: '
                                                                  f'{analyst.converse([user_requirements])}'}]

    for i in range(3):
        code = extract_function((developer.converse(messages)))
        messages.append({'role': 'user', 'content': f"The developer's code is:\n{code}"})
        feedback = tester.converse(messages)
        print(messages)
        if 'no problems found' in feedback.lower():
            break
        messages.append({'role': 'user', 'content': f"The tester's feedback is:\n{feedback}"})
    return code

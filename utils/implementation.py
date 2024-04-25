# -*- coding = utf-8 -*-
# @Time : 2023/8/31 23:36
# @Author: Tan Qiyuan
# @File: implementation
import ast
import json
import re

from utils.LLMs.LLMsAdapter import LLMsAdapter, Charactor
from utils.prompts.Prompter import Prompter


def crop_string(input_string):
    index1 = input_string.find('\ndef')
    index2 = input_string.find('\nif')
    if index2 > index1 > -1:
        return input_string[:index1]
    return input_string[:index2]


def extract_function_body(text: str):
    text = process_signs(text)
    # 在文本中找到所有可能的 Python 函数定义
    function_defs = []
    start = 0
    while True:
        start = text.find("def ", start)
        if start == -1:
            break

        s1 = start
        while True:
            index_return = text.find("    return ", s1)
            if index_return == -1:
                index_return = s1
                break
            s1 = index_return + 1
        # 确定函数定义的结束位置
        end = text.find("\n", index_return)
        if end == -1:
            end = len(text)
        function_def = text[start:end]

        function_defs.append(function_def)
        start = end

    # 尝试解析每个函数定义
    for function_def in function_defs:
        # 使用 AST 解析 Python 代码
        try:
            tree = ast.parse(function_def)
        except:
            return text

        # 找到所有的函数定义节点
        function_defs_ast = [node for node in tree.body if isinstance(node, ast.FunctionDef)]

        if function_defs_ast:
            # 取第一个函数定义
            first_function_def = function_defs_ast[0]

            # 获取函数名
            function_name = first_function_def.name

            # 获取函数参数
            args = ', '.join(arg.arg for arg in first_function_def.args.args)

            # 获取函数体
            function_body = ast.unparse(first_function_def.body)

            # 生成函数的字符串表示
            function_string = function_body

            return function_string

    return text


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
                                                                  f'{process_signs(analyst.converse([user_requirements]))}'}]

    for i in range(3):
        code = extract_function_body(process_signs(developer.converse(messages)))
        messages.append({'role': 'user', 'content': f"The developer's code is:\n" + code})
        feedback = tester.converse(messages)
        if 'no problems found' in feedback.lower():
            # print(messages)
            # print(feedback)
            print(code)
            break
        messages.append({'role': 'user', 'content': f"The tester's feedback is:\n{feedback}"})
        print(messages)
    return code


def process_json_file(file_path):
    # 用于存储处理后的数据
    processed_data = []
    with open(file_path, 'r') as file:
        for line in file:
            entry = json.loads(line)
            if 'completion' in entry:
                completion_str = entry['completion']
                # 使用 extract_function 函数处理字符串
                processed_str = extract_function_body(completion_str)
                # 删除连续出现的 '/'，保留一个 '/'
                processed_str = process_signs(processed_str)
                print(processed_str)
                entry['completion'] = processed_str
                # print(entry)
            processed_data.append(entry)

    with open(file_path + "-new", 'w') as file:
        for entry in processed_data:
            json.dump(entry, file)
            file.write('\n')  # 每个 JSON 对象之间添加换行


def process_signs(s: str):
    s = s.replace('\\n', '换行符')
    s = s.replace('\\', '')
    s = s.replace('换行符', '\n')
    return s

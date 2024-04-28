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
    text = process_n(text)
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

    for function_def in function_defs:
        # 使用 AST 解析 Python 代码
        try:
            tree = ast.parse(function_def)
        except:
            return function_def

        function_defs_ast = [node for node in tree.body if isinstance(node, ast.FunctionDef)]

        if function_defs_ast:
            first_function_def = function_defs_ast[0]
            function_body = ast.unparse(first_function_def.body)
            return function_body

    return text


def self_planning(llm_adapter: LLMsAdapter, prompt, prompter: Prompter = Prompter()) -> str:
    planning_prompt: str = planning(llm_adapter, prompt, prompter)
    planning_prompt = planning_prompt[:planning_prompt.find('"""')] + '"""'
    prompt = prompt[:-4] + planning_prompt
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
                                                                  f'{process_n(analyst.converse([user_requirements]))}'}]

    for i in range(3):
        code = extract_function_body(process_n(developer.converse(messages)))
        messages.append({'role': 'user', 'content': f"The developer's code is:\n" + code})
        feedback = tester.converse(messages)
        if 'no problems found' in feedback.lower():
            break
        messages.append({'role': 'user', 'content': f"The tester's feedback is:\n{feedback}"})
    code = '    ' + code.replace('\n', '    \n').replace('\\\\', "\\")
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
                processed_str = process_n(processed_str)
                print(processed_str)
                entry['completion'] = processed_str
                # print(entry)
            processed_data.append(entry)

    with open(file_path + "-new", 'w') as file:
        for entry in processed_data:
            json.dump(entry, file)
            file.write('\n')  # 每个 JSON 对象之间添加换行


def process_json_file2(file_path: str):
    # 用于存储处理后的数据
    processed_data = []
    with open(file_path, 'r') as file:
        for line in file:
            entry = json.loads(line)
            if 'completion' in entry:
                completion_str = entry['completion']
                completion_str = process_n(completion_str)
                entry['completion'] = "    " + completion_str.replace('\n', "\n    ")
                print(entry)
            processed_data.append(entry)

    with open(file_path.replace(".json", "-new.json"), 'w') as file:
        for entry in processed_data:
            json.dump(entry, file)
            file.write('\n')  # 每个 JSON 对象之间添加换行


def process_json_file3(file_path: str):
    # 用于存储处理后的数据
    processed_data = []
    with open(file_path, 'r') as file:
        for line in file:
            entry = json.loads(line)
            if 'completion' in entry:
                completion_str = entry['completion']
                entry['completion'] = crop_string(completion_str)
                print(entry)
            processed_data.append(entry)

    with open(file_path.replace(".json", "-new.json"), 'w') as file:
        for entry in processed_data:
            json.dump(entry, file)
            file.write('\n')  # 每个 JSON 对象之间添加换行


def process_n(s: str):
    s = s.replace('\\n', '换行符')
    s = s.replace('\\', '')
    s = s.replace('换行符', '\n')
    return s

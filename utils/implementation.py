# -*- coding = utf-8 -*-
# @Time : 2023/8/31 23:36
# @Author: Tan Qiyuan
# @File: implementation
import re

from utils.LLMs.LLMsAdapter import LLMsAdapter, Charactor
from utils.prompts.Prompter import Prompter


def crop_string(input_string):
    index1 = input_string.find('\ndef')
    index2 = input_string.find('\nif')
    if index2 > index1 > -1:
        return input_string[:index1]
    return input_string[:index2]


def extract_function_body(function_string):
    pattern = r"def\s+\w+\s*\([^)]*\)\s*:\s*([\s\S]*?)(?=def|\Z)"
    match = re.search(pattern, function_string)
    if match:
        return match.group(1).strip()
    else:
        return function_string


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
                                         'content': f"The user requests the team to write a complete implementation for "
                                                    f"this function:\n {prompt}"}
    messages: list[dict[str, str]] = [user_requirements,
                                      {'role': 'user', 'content': f'Analyst: {analyst.converse([user_requirements])}'}]

    for i in range(3):
        code = extract_function_body((developer.converse(messages)))
        messages.append({'role': 'user', 'content': f"The developer's code is:\n{code}"})
        feedback = tester.converse(messages)
        print(messages)
        if 'no problems found' in feedback.lower():
            break
        messages.append({'role': 'user', 'content': f"The tester's feedback is:\n{feedback}"})
    return code

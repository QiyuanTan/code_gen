# -*- coding = utf-8 -*-
# @Time : 2023/8/31 23:36
# @Author: Tan Qiyuan
# @File: implementation
import string

from utils.LLMs.LLMsAdapter import LLMsAdapter
from utils.prompts.Prompter import Prompter


def crop_string(input_string):
    index1 = input_string.find('\ndef')
    index2 = input_string.find('\nif')
    if index2 > index1 > -1:
        return input_string[:index1]
    return input_string[:index2]


def self_planning(llm_adapter: LLMsAdapter, prompt, prompter: Prompter = Prompter()) -> string:
    planning_prompt = planning(llm_adapter, prompt, prompter)
    planning_prompt = planning_prompt[:planning_prompt.find('"""')] + '"""'
    prompt = prompter.WRITING_PROMPT + prompt[:-4] + 'Let’s think step by step.' + planning_prompt
    # print(f"start of writing prompt \n{prompt} \n end of writing prompt")
    return crop_string(llm_adapter.completion(prompt))


def planning(llm_adapter: LLMsAdapter, prompt, prompter: Prompter = Prompter()) -> string:
    prompt = prompter.PLANNING_PROMPT + prompt[:-4] + 'Let’s think step by step.'
    # print(f"start of planning prompt \n{prompt} \n end of planning prompt")
    return llm_adapter.completion(prompt)


def self_collaboration(llm_adapter: LLMsAdapter, prompt, prompter: Prompter = Prompter()):
    analyst = llm_adapter.get_charactor(prompter.ANALYST)
    developer = llm_adapter.get_charactor(prompter.DEVELOPER)
    tester = llm_adapter.get_charactor(prompter.TESTER)
    code = ''

    user_requirements = {'role': 'user',
                         'content': f"please write a complete implementation for this function:\n {prompt}"}
    messages = [user_requirements,
                {'role': 'system', 'content': 'Analyst: ' + analyst.converse(user_requirements)}]
    for i in range(3):
        code = developer.converse(messages)
        messages.append({'role': 'system', 'content': f"The developer's code is:\n{code}"})
        feedback = tester.converse(messages)
        if 'no problems found' in feedback.lower():
            break
        messages.append({'role': 'system', 'content': f"The test's feedback is:\n{feedback}"})
        print(messages)

    return code

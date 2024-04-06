# -*- coding = utf-8 -*-
# @Time : 2023/8/31 23:36
# @Author : Tan Qiyuan
# @File : implementation
import string

import openai
import zhipuai
from transformers import AutoTokenizer, AutoModel
from human_eval.data import read_problems

from utils.LLMs.LLMsAdapter import LLMsAdapter
from utils.prompts.Prompter import Prompter


def crop_string(input_string):
    index1 = input_string.find('\ndef')
    index2 = input_string.find('\nif')
    if index2 > index1 > -1:
        return input_string[:index1]
    return input_string[:index2]


def self_planning(nlp_adapter: LLMsAdapter, prompt, prompter: Prompter = Prompter) -> string:
    planning_prompt = planning(nlp_adapter, prompter, prompt)
    planning_prompt = planning_prompt[:planning_prompt.find('"""')] + '"""'
    prompt = prompt[:-4] + planning_prompt
    # print(prompt)
    return crop_string(nlp_adapter.completion(prompt))


def planning(nlp_adapter: LLMsAdapter, prompt, prompter: Prompter = Prompter) -> string:
    prompt = prompter.PLANNING_PROMPT + prompt[:-4] + 'Let’s think step by step.'
    return nlp_adapter.completion(prompt)


def self_collaboration(nlp_adapter: LLMsAdapter, prompt, prompter: Prompter = Prompter):
    code = ''
    white_board = 'analyst:'
    white_board += (nlp_adapter.chat_completion(
        prompter.TEAM_DESCRIPTION +
        f"\n the user's requirement is to complete the following function{prompt}" +
        prompter.ANALYST))

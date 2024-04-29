# -*- coding = utf-8 -*-
# @File: implementation

from utils.LLMs import *
from utils.prompts.Prompter import Prompter


def crop_string(input_string):
    """
    Crops the input strings to the first occurrence of the keyword "def" or "if".
    :param input_string: The input string to be cropped.
    :return: The cropped string.
    """
    index1 = input_string.find('\ndef')
    index2 = input_string.find('\nif')
    if index2 > index1 > -1:
        return input_string[:index1]
    return input_string[:index2]


def extract_function_body(text: str):
    """
    Extracts the function body from the given text.
    :param text: The text containing the function definition.
    :return: The function body as a string.
    """
    # remove repeated escape characters produced by the model
    text = process_escape_character(text)

    # find the first appearance of def
    start = text.find("def ")
    start = text.find("\n", start) + 1

    # find the last appearance of return
    s1 = start
    while True:
        index_return = text.find("    return ", s1)
        if index_return == -1:
            index_return = s1
            break
        s1 = index_return + 1

    # find the new line after the last return keyword
    end = text.find("\n", index_return)
    if end == -1:
        end = len(text)

    return text[start:end]


def self_planning(llm_adapter: LLMsAdapter, prompt, prompter: Prompter = Prompter()) -> str:
    """
    The implementation of the self-planning framework
    :param llm_adapter: The LLMsAdapter instance
    :param prompt: The prompt for generating the code
    :param prompter: The Prompter instance
    :return: The generated code with self-planning
    """
    # get the planning prompt
    planning_prompt: str = planning(llm_adapter, prompt, prompter)

    # joint the planning prompt
    planning_prompt = planning_prompt[:planning_prompt.find('"""')] + '"""'
    prompt = prompt[:-4] + planning_prompt
    return crop_string(llm_adapter.completion(prompt))


def planning(llm_adapter: LLMsAdapter, prompt: str, prompter: Prompter = Prompter()) -> str:
    """
    Generates the planning prompt for the self-planning framework
    :param llm_adapter: The LLMsAdapter instance
    :param prompt: The prompt for generating the planning prompt
    :param prompter: The Prompter instance
    :return: The generated planning prompt
    """
    prompt: str = prompter.PLANNING_PROMPT + prompt[:-4] + 'Let’s think step by step.'
    return llm_adapter.completion(prompt)


def self_collaboration(llm_adapter: LLMsAdapter,
                       prompt: str,
                       prompter: Prompter = Prompter(),
                       max_retries: int = 3) -> str:
    """
    The implementation of tbe self-collaboration framework
    :param llm_adapter: The LLMsAdapter instance
    :param prompt: The prompt for generating the code
    :param prompter: The Prompter instance
    :param max_retries: The maximum number of retries for generating the code
    :return: The generated code with self-collaboration
    """

    # initialize the characters
    analyst: Charactor = llm_adapter.get_charactor(prompter.ANALYST, 'analyst')
    developer: Charactor = llm_adapter.get_charactor(prompter.DEVELOPER, 'developer')
    tester: Charactor = llm_adapter.get_charactor(prompter.TESTER, 'tester')

    # generates the answer from the analyst who analyzes the user's requirements
    code: str = ''
    user_requirements: dict[str, str] = {'role': 'user',
                                         'content': f"The user requests the team to write a complete implementation "
                                                    f"for this function:\n {prompt}"}
    messages: list[dict[str, str]] = [user_requirements, {'role': 'user', 'content': f'The analyst says: {analyst.converse([user_requirements])}'}]

    # the coder and the test collaborate to complete the code
    for i in range(max_retries):
        code = extract_function_body(developer.converse(messages))
        messages.append({'role': 'user', 'content': f"The developer's code is:\n" + code})
        feedback = tester.converse(messages)
        if 'no problems found' in feedback.lower():
            break
        messages.append({'role': 'user', 'content': f"The tester's feedback is:\n{feedback}"})

    return code


def process_escape_character(s: str):
    """
    Removes the repeated escape characters from the given string.
    :param s: The string to process.
    :return: The processed string with the repeated escape characters removed.
    """
    s = s.replace('\\n', 'NEW_LINE')
    s = s.replace('\\', '')
    s = s.replace('NEW_LINE', '\n')
    return s

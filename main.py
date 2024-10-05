# -*- coding = utf-8 -*-
# @File: main

from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime
from human_eval.data import write_jsonl, read_problems
from tqdm import tqdm

from utils.implementation import *
from utils.LLMs import *

problems = read_problems()

def generate_samples(model_adapter: LLMsAdapter, keys, experiment_name,
                     completion, num_samples_per_task=1, max_workers=5):
    """
    generate samples for a set of experiment
    :param model_adapter: the model adapter
    :param keys: keys of tasks to generate
    :param experiment_name: experiment name, appears in the generated file
    :param completion: the completion for code generation
    :param num_samples_per_task: number of samples per task
    :param max_workers: max workers
    :return: none, write samples to disk
    """
    # initialize variables
    model_adapter.recount_tokens()
    samples = []
    total_iterations = num_samples_per_task * len(keys)
    futures = []

    # generate the samples with multi-threading
    with tqdm(total=total_iterations, desc=f'Generating samples: {model_adapter}-{experiment_name}') as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for _ in range(num_samples_per_task):
                for task_id in keys:
                    futures.append(executor.submit(add_sample, samples, task_id, completion, model_adapter, pbar))

    # wait for all threads to end
    wait(futures)

    # sort the samples by task_id
    samples.sort(key=lambda i: int(i['task_id'].split('/')[1]))

    # generate the file name
    used_tokens = model_adapter.get_token()
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"{current_datetime}-{model_adapter}-{experiment_name}-{used_tokens}tokens.jsonl"

    # write the results to disk
    write_jsonl(file_name, samples)


def add_sample(samples, task_id, completion, model_adapter, pbar):
    """
    generate and add a sample to the list samples
    :param samples: list of samples for the generated samples to add in
    :param task_id: task id
    :param completion: the completion for code generation
    :param model_adapter: model adapter
    :param pbar: progress bar to be updated
    :return: none
    """
    samples.append(dict(task_id=task_id,
                        completion=completion(prompt=problems[task_id]["prompt"],
                        llm_adapter=model_adapter)))
    pbar.update(1)


def completion_for_completion_models(llm_adapter, prompt):
    """
    generate a completion for the given prompt using the completion mode of the llm_adapter
    :param llm_adapter: llm adapter
    :param prompt: Prompt for generating the completion
    :return: the generated completion
    """
    return llm_adapter.completion(prompt)


def completion_for_chat_models(llm_adapter: LLMsAdapter, prompt):
    """
    generate a completion for the given prompt using the chat completion mode of the llm_adapter
    :param llm_adapter: llm adapter
    :param prompt: Prompt for generating the completion
    :return: the generated completion
    """
    return '    ' + process_escape_character(extract_function_body(llm_adapter.chat_completion([{'role': 'user',
                                                                                                 'content': 'Please write a complete '
                                                                                                            'implementation for this '
                                                                                                            'function. Remember, '
                                                                                                            'do not include the '
                                                                                                            'function header, '
                                                                                                            'and do not write '
                                                                                                            'anything but the code '
                                                                                                            'for implementation.'
                                                                                                            + prompt}])))


if __name__ == '__main__':
    problem_keys = list(problems.keys())

    # Set up the models
    glm4 = ZhipuModelsAdapter('glm-4', api_key='API_KEY')

    # Generate samples
    generate_samples(glm4, problem_keys, "self_collaboration", self_collaboration, max_workers=1)
    generate_samples(glm4, problem_keys, "dirct_com", completion_for_completion_models, max_workers=5)

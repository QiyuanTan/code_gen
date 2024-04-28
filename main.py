from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime

from human_eval.data import write_jsonl, read_problems
from tqdm import tqdm

from utils.LLMs.ChatGLMAdapter import ZhipuModelsAdapter
from utils.LLMs.LocalLLMsAdapter import LocalLLMsAdapter
from utils.implementation import *

problems = read_problems()


def generate_samples(model_adapter: LLMsAdapter, keys, experiment_name,
                     completion, num_samples_per_task=1, max_workers=5):
    model_adapter.recount_tokens()
    samples = []

    total_iterations = num_samples_per_task * len(keys)

    futures = []
    with tqdm(total=total_iterations, desc=f'Generating samples: {model_adapter}-{experiment_name}') as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for _ in range(num_samples_per_task):
                for task_id in keys:
                    futures.append(executor.submit(add_sample, samples, task_id, completion, model_adapter, pbar))

    wait(futures)

    samples.sort(key=lambda i: int(i['task_id'].split('/')[1]))

    used_tokens = model_adapter.get_token()
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"{current_datetime}-{model_adapter}-{experiment_name}-{used_tokens}tokens.jsonl"
    write_jsonl(file_name, samples)


def add_sample(samples, task_id, completion, model_adapter, pbar):
    samples.append(dict(task_id=task_id,
                        completion=completion(prompt=problems[task_id]["prompt"],
                                              llm_adapter=model_adapter)))
    pbar.update(1)


def completion_for_completion_models(llm_adapter, prompt):
    return crop_string(llm_adapter.completion(prompt))


def completion_for_chat_models(llm_adapter: LLMsAdapter, prompt):
    return process_n(extract_function_body(llm_adapter.chat_completion([{'role': 'user',
                                                                         'content': 'Please write a complete '
                                                                                    'implementation for this '
                                                                                    'function. Remember, '
                                                                                    'do not include the function '
                                                                                    'header, and do not write '
                                                                                    'anything but the code for '
                                                                                    'implementation. '
                                                                                    + prompt}])))


if __name__ == '__main__':
    problem_keys = list(problems.keys())
    glm4 = LocalLLMsAdapter('chatglm2-6b')
    # glm4 = ZhipuModelsAdapter('glm-4', api_key='0b4dfa49fd18b4b01a9bdaed106e1a8a.Hv5dwBnO1rp8k7P0')

    generate_samples(glm4, problem_keys, "self_plan", self_planning, max_workers=5)
    generate_samples(glm4, problem_keys, "dirct_com[", completion_for_completion_models, max_workers=5)

from datetime import datetime

from human_eval.data import write_jsonl, read_problems
from tqdm import tqdm

from utils.LLMs.ChatGLMAdapter import *
from utils.LLMs.LLMsAdapter import LLMsAdapter
from utils.LLMs.LocalLLMsAdapter import LocalLLMsAdapter
from utils.implementation import *

problems = read_problems()


def self_planning_experiment(model_adapter: LLMsAdapter):
    model_adapter.recount_tokens()
    samples = []

    num_samples_per_task = 1
    keys = list(problems.keys())[10:]
    total_iterations = num_samples_per_task * len(keys) * 3

    with tqdm(total=total_iterations, desc='Generating samples') as pbar:
        for _ in range(num_samples_per_task):
            for task_id in keys:
                samples.append(
                    dict(task_id=task_id, completion=self_planning(model_adapter, problems[task_id]["prompt"])))
                pbar.update(2)

    used_tokens = model_adapter.get_token()
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    write_jsonl(f"{current_datetime}-{model_adapter}-selfplanning-{used_tokens}tokens.jsonl", samples)


def control_group_experiment(model_adapter: LLMsAdapter):
    model_adapter.recount_tokens()
    samples = []

    num_samples_per_task = 1
    keys = list(problems.keys())[10:]
    total_iterations = num_samples_per_task * len(keys) * 3

    with tqdm(total=total_iterations, desc='Generating samples') as pbar:
        for _ in range(num_samples_per_task):
            for task_id in keys:
                samples.append(dict(task_id=task_id, completion=model_adapter.completion(problems[task_id]["prompt"])))
                pbar.update()

    used_tokens = model_adapter.get_token()

    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    write_jsonl(f"{current_datetime}-{model_adapter}-direct-{used_tokens}tokens.jsonl", samples)


if __name__ == '__main__':
    model = LocalLLMsAdapter('vicuna-7b-v1.5')
    self_planning_experiment(model)
    control_group_experiment(model)

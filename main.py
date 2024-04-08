from datetime import datetime

from human_eval.data import write_jsonl, read_problems
from tqdm import tqdm

from utils.LLMs.LocalLLMsAdapter import LocalLLMsAdapter
from utils.implementation import *

problems = read_problems()


def self_planning_experiment(model_adapter: LLMsAdapter, keys):
    model_adapter.recount_tokens()
    samples = []

    num_samples_per_task = 1
    total_iterations = num_samples_per_task * len(keys) * 3

    with tqdm(total=total_iterations, desc='Generating samples') as pbar:
        for _ in range(num_samples_per_task):
            for task_id in keys:
                samples.append(
                    dict(task_id=task_id, completion=self_planning(model_adapter, problems[task_id]["prompt"])))
                pbar.update(2)

    used_tokens = model_adapter.get_token()
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    write_jsonl(f"{current_datetime}-{model_adapter}-self-planning-{used_tokens}tokens.jsonl", samples)


def control_group_experiment(model_adapter: LLMsAdapter, keys):
    model_adapter.recount_tokens()
    samples = []

    num_samples_per_task = 1
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
    keys_non_training = list(problems.keys())
    # keys_non_training.remove('HumanEval/37')
    # keys_non_training.remove('HumanEval/137')
    # keys_non_training.remove('HumanEval/69')
    # keys_non_training.remove('HumanEval/39')
    # keys_non_training.remove('HumanEval/67')
    # keys_non_training.remove('HumanEval/141')
    # keys_non_training.remove('HumanEval/134')
    # keys_non_training.remove('HumanEval/89')
    model = LocalLLMsAdapter('vicuna-13b-v1.5')
    self_planning_experiment(model, keys_non_training)
    control_group_experiment(model, keys_non_training)

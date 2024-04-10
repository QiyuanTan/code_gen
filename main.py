from datetime import datetime

from human_eval.data import write_jsonl, read_problems
from human_eval.evaluate_functional_correctness import entry_point
from tqdm import tqdm

from utils.LLMs.LocalLLMsAdapter import LocalLLMsAdapter
from utils.implementation import *

problems = read_problems()


def generate_samples(model_adapter: LLMsAdapter, keys, experiment_name, completion, num_samples_per_task=1):
    model_adapter.recount_tokens()
    samples = []

    total_iterations = num_samples_per_task * len(keys) * 3

    with tqdm(total=total_iterations, desc='Generating samples') as pbar:
        for _ in range(num_samples_per_task):
            for task_id in keys:
                samples.append(
                    dict(task_id=task_id, completion=completion(prompt=problems[task_id]["prompt"],
                                                                llm_adapter=model_adapter)))
                pbar.update(2)

    used_tokens = model_adapter.get_token()
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"{current_datetime}-{model_adapter}-{experiment_name}-{used_tokens}tokens.jsonl"
    write_jsonl(file_name, samples)

    entry_point(file_name)


def completion_for_completion_models(model_adapter, prompt):
    return model_adapter.completion(prompt)


def completion_for_chat_models(model_adapter: LLMsAdapter, prompt):
    return model_adapter.chat_completion({'role': 'user',
                                          'content': 'Please write a complete implementation for this function. Do '
                                                     'not include the function header, and do not write anything but '
                                                     'the code for implementation. ' + prompt})


if __name__ == '__main__':
    problem_keys = list(problems.keys())
    models = [LocalLLMsAdapter('vicuna-13b-v1.5')]
    for model in models:
        generate_samples(model, problem_keys, "self_planning", self_planning)
        generate_samples(model, problem_keys, "self_collaboration", self_collaboration)
        generate_samples(model, problem_keys, "direct", completion_for_completion_models)

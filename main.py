from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from human_eval.data import write_jsonl, read_problems, HUMAN_EVAL
from human_eval.evaluation import evaluate_functional_correctness
from tqdm import tqdm

from utils.LLMs.ChatGLMAdapter import ZhipuModelsAdapter, CharactorGLMAdapter
from utils.LLMs.LocalLLMsAdapter import LocalLLMsAdapter
from utils.implementation import *

problems = read_problems()


def generate_samples(model_adapter: LLMsAdapter, keys, experiment_name, completion, num_samples_per_task=1):
    model_adapter.recount_tokens()
    samples = []

    total_iterations = num_samples_per_task * len(keys)

    with ThreadPoolExecutor(max_workers=5) as executor, tqdm(total=total_iterations, desc='Generating samples') as pbar:
        for _ in range(num_samples_per_task):
            for task_id in keys:
                executor.submit(add_sample, samples, task_id, completion, model_adapter, pbar)

    used_tokens = model_adapter.get_token()
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"{current_datetime}-{model_adapter}-{experiment_name}-{used_tokens}tokens.jsonl"
    write_jsonl(file_name, samples)

    k = "1,10,100"
    k = list(map(int, k.split(",")))
    evaluate_functional_correctness(file_name,
                                    k=k,
                                    n_workers=10,
                                    timeout=3.0,
                                    problem_file=HUMAN_EVAL)


def add_sample(samples, task_id, completion, model_adapter, pbar):
    samples.append(dict(task_id=task_id,
                        completion=completion(prompt=problems[task_id]["prompt"],
                                              llm_adapter=model_adapter)))
    pbar.update(1)


def completion_for_completion_models(llm_adapter, prompt):
    return llm_adapter.completion(prompt)


def completion_for_chat_models(llm_adapter: LLMsAdapter, prompt):
    return extract_function_body(llm_adapter.chat_completion([{'role': 'user',
                                                               'content': 'Please write a complete implementation for '
                                                                          'this function. Remember, do not include '
                                                                          'the function header, and do not write '
                                                                          'anything but the code for implementation. '
                                                                          + prompt}]))


if __name__ == '__main__':
    problem_keys = list(problems.keys())
    glm3 = ZhipuModelsAdapter('glm-3-Turbo')
    glm4 = ZhipuModelsAdapter('glm-4')

    generate_samples(glm3, problem_keys, "self_collaboration", self_collaboration)
    generate_samples(glm3, problem_keys, "direct_chat", completion_for_chat_models)

    generate_samples(glm4, problem_keys, "self_collaboration", self_collaboration)
    generate_samples(glm4, problem_keys, "direct_chat", completion_for_chat_models)

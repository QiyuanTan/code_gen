from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime

from human_eval.data import write_jsonl, read_problems, HUMAN_EVAL
from human_eval.evaluation import evaluate_functional_correctness
from tqdm import tqdm

from utils.LLMs.ChatGLMAdapter import ZhipuModelsAdapter, CodeGeeXAdapter
from utils.implementation import *

problems = read_problems()


def generate_samples(model_adapter: LLMsAdapter, keys, experiment_name,
                     completion, num_samples_per_task=1, max_workers=5):
    model_adapter.recount_tokens()
    samples = []

    total_iterations = num_samples_per_task * len(keys)

    with ThreadPoolExecutor(max_workers=max_workers) as executor, \
            tqdm(total=total_iterations, desc=f'Generating samples: {model_adapter}-{experiment_name}') as pbar:
        for _ in range(num_samples_per_task):
            for task_id in keys:
                executor.submit(add_sample, samples, task_id, completion, model_adapter, pbar)

        executor.shutdown(wait=True)

    samples.sort(key=lambda i: int(i['task_id'].split('/')[1]))

    used_tokens = model_adapter.get_token()
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"{current_datetime}-{model_adapter}-{experiment_name}-{used_tokens}tokens.jsonl"
    write_jsonl(file_name, samples)

    k = "1,10,100"
    k = list(map(int, k.split(",")))
    print(evaluate_functional_correctness(file_name,
                                          k=k,
                                          n_workers=10,
                                          timeout=3.0,
                                          problem_file=HUMAN_EVAL))


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
    glm3 = ZhipuModelsAdapter('glm-3-Turbo', api_key='0b4dfa49fd18b4b01a9bdaed106e1a8a.Hv5dwBnO1rp8k7P0')
    glm4 = ZhipuModelsAdapter('glm-4', api_key='0b4dfa49fd18b4b01a9bdaed106e1a8a.Hv5dwBnO1rp8k7P0')
    # codegeex = CodeGeeXAdapter()

    # generate_samples(codegeex, problem_keys, "self_panning", self_planning, max_workers=1)
    # generate_samples(codegeex, problem_keys, "dirct_completion", completion_for_completion_models, max_workers=1)

    generate_samples(glm3, problem_keys, "self_collaboration", self_collaboration, max_workers=1)
    # generate_samples(glm3, problem_keys, "direct_chat", completion_for_chat_models)

    # generate_samples(glm4, problem_keys, "self_collaboration", self_collaboration)
    # generate_samples(glm4, problem_keys, "direct_chat", completion_for_chat_models)

    # process_json_file("2024-04-24_16-38-50-glm-4-self_collaboration-1114253tokens.jsonl")

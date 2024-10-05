# code_gen

This project is an attempt to implement the self-planning code generation framework ([arXiv:2303.06689](https://arxiv.org/abs/2303.06689)) and the self-collaboration code generation framework ([arXiv:2304.07590](https://arxiv.org/abs/2304.07590)). These frameworks aim to improve the process of large language model (LLM) code generation. The program can generate samples that can be evaluated using Human-Eval.

## Table of Contents
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/your-repo-name.git
    cd code_gen
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

3. Clone the Human-Eval repository to the same directory and set it up following its instructions:
    ```sh
    git clone https://github.com/openai/human-eval.git
    cd human-eval
    # Follow the setup instructions in the Human-Eval repository
    ```

## Project Structure
- `requirements.txt`: Lists the dependencies required for the project.
- `human-eval/`: Contains the Human-Eval repository.
- `main.py`: Main script to generate samples using the self-planning and self-collaboration frameworks.
- `utils/LLMs.py`: Contains the adapters for different LLMs and their respective methods.
- `utils/implementation.py`: Contains utility functions for the project.
- `requirements.txt`: Lists the dependencies required for the project.

## Usage

1. Go to `main.py` and set up the model with your API key:

2. Use `generate_samples` to generate samples with the desired framework and models. The function takes the following parameters:
    - `model`: The model adapter to use.
    - `problem_keys`: The keys of the problems to generate samples for.
    - `experiment_name`: Experiment name, appears in the generated file
    - `completion`: The way of completion to use. Existed functions for this parameter are:
        - `self_planning`: The self-planning framework.
        - `self_collaboration`: The self-collaboration framework.
        - `completion_for_completion_models`Generating the samples directly from the completion models.: 
        - `completion_for_chat_models`: Generating the samples directly from the chat models.
     - `num_samples_per_task`: Number of samples per task
     - `max_workers`: The number of workers to use for generating samples.
   
   An example for the setup goes as follows:
   ```python
   if __name__ == '__main__':
   problem_keys = list(problems.keys())

   # Set up the models
   glm4 = ZhipuModelsAdapter('glm-4', api_key='API_KEY')

   # Generate samples
   generate_samples(glm4, problem_keys, "self_collaboration", self_collaboration, max_workers=1)
   generate_samples(glm4, problem_keys, "dirct_com", completion_for_completion_models, max_workers=5)
   ```

3. Run the main script to generate samples:
    ```sh
    python main.py
    ```

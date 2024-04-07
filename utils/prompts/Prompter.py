from human_eval.data import read_problems


class Prompter:
    def __init__(self):
        problems = read_problems()

        with open('utils/prompts/planning_prompt.txt', 'r', encoding='utf-8') as file:
            # 2. 读取文件内容
            self.PLANNING_PROMPT = file.read()

        with open('utils/prompts/team_description.txt', 'r', encoding='utf-8') as file:
            # 2. 读取文件内容
            self.TEAM_DESCRIPTION = file.read()

        with open('utils/prompts/coder.txt', 'r', encoding='utf-8') as file:
            # 2. 读取文件内容
            self.CODER = file.read()

        with open('utils/prompts/analyst.txt', 'r', encoding='utf-8') as file:
            # 2. 读取文件内容
            self.ANALYST = file.read()

        with open('utils/prompts/tester.txt', 'r', encoding='utf-8') as file:
            # 2. 读取文件内容
            self.TESTER = file.read()

        with open('utils/prompts/writing_prompt.txt', 'r', encoding='utf-8') as file:
            # 2. 读取文件内容
            self.WRITING_PROMPT = file.read()

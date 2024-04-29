# -*- coding = utf-8 -*-
# @File: Prompter

class Prompter:
    """
    A class to handle the prompts
    """
    def __init__(self):
        with open('utils/prompts/planning_prompt.txt', 'r', encoding='utf-8') as file:
            self.PLANNING_PROMPT = file.read()

        with open('utils/prompts/team_description.txt', 'r', encoding='utf-8') as file:
            self.TEAM_DESCRIPTION = file.read()

        with open('utils/prompts/developer.txt', 'r', encoding='utf-8') as file:
            self.DEVELOPER = self.TEAM_DESCRIPTION + file.read()

        with open('utils/prompts/analyst.txt', 'r', encoding='utf-8') as file:
            self.ANALYST = self.TEAM_DESCRIPTION + file.read()

        with open('utils/prompts/tester.txt', 'r', encoding='utf-8') as file:
            self.TESTER = self.TEAM_DESCRIPTION + file.read()

        with open('utils/prompts/writing_prompt.txt', 'r', encoding='utf-8') as file:
            self.WRITING_PROMPT = file.read()

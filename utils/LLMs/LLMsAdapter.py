class LLMsAdapter:
    def __init__(self, model_name):
        self.token_count = 0
        self.model_name = model_name

    def completion(self, prompt, max_tokens=100) -> str:
        raise NotImplementedError

    def chat_completion(self, prompt, max_tokens=100) -> str:
        raise NotImplementedError

    def update_token(self, token_delta):
        self.token_count += token_delta

    def get_token(self):
        return self.token_count

    def recount_tokens(self):
        self.token_count = 0

    def get_charactor(self, role_prompt):
        return self.Charactor(self, role_prompt)

    def get_model_name(self):
        return self.model_name

    def __str__(self):
        return self.model_name

    class Charactor:
        def __init__(self, llm_adapter, role_prompt):
            self.llm_adapter = llm_adapter
            self.role_prompt = {'role': 'system', 'content': role_prompt}

        def converse(self, prompt):
            results = self.llm_adapter.chat_completion([self.role_prompt] + prompt)
            return results

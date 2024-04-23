import openai
from utils.LLMs.LLMsAdapter import LLMsAdapter


class OpenaiLLMsAdapter(LLMsAdapter):

    def __init__(self, model_name, api_key='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'):
        super().__init__(model_name=model_name)
        openai.api_key = api_key

    def completion(self,
                   prompt,
                   max_tokens=100,
                   max_length=300,
                   top_p=0.9,
                   temperature=0.0,
                   api_base="https://api.openai.com/v1"):
        openai.api_base = api_base
        chat = openai.Completion.create(
            model=self.get_model_name(),
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_length
        )
        try:
            self.update_token(chat.usage.total_tokens)
        except AttributeError:
            self.update_token(-1)
        return chat.choices[0].text

    def chat_completion(self,
                        messages,
                        max_tokens=100,
                        max_length=300,
                        top_p=0.9,
                        temperature=0.0,
                        api_base="https://api.openai.com/v1"):
        openai.api_base = api_base
        chat = openai.ChatCompletion.create(
            model=self.get_model_name(),
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_length
        )
        try:
            self.update_token(chat.usage.total_tokens)
        except AttributeError:
            self.update_token(-1)
        return chat.choices[0].message.content

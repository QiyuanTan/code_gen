from utils.LLMs.OpenaiLLMsAdapter import OpenaiLLMsAdapter


class LocalLLMsAdapter(OpenaiLLMsAdapter):

    def __init__(self, model_name, api_base = "http://localhost:8000/v1"):
        super().__init__(model_name=model_name)
        self.api_base = api_base

    def completion(self,
                   prompt,
                   max_tokens=100,
                   max_length=300,
                   top_p=0.9,
                   temperature=0.0,
                   api_base="http://localhost:8000/v1"):
        return super().completion(prompt=prompt,
                                  max_tokens=max_tokens,
                                  max_length=max_length,
                                  top_p=top_p,
                                  temperature=temperature,
                                  api_base=self.api_base)

    def chat_completion(self,
                        messages,
                        max_tokens=100,
                        max_length=300,
                        top_p=0.9,
                        temperature=0.0,
                        api_base="http://localhost:8000/v1"):
        return super().chat_completion(messages=messages,
                                       max_tokens=max_tokens,
                                       max_length=max_length,
                                       top_p=top_p,
                                       temperature=temperature,
                                       api_base=self.api_base)

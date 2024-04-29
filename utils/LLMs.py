# -*- coding = utf-8 -*-
# @File: LLMs

import openai
import zhipuai
from tenacity import retry, stop_after_attempt, wait_fixed


class Charactor:
    """
    The Charactor class represents a character in the self-collaboration code generation framework.
    """

    def __init__(self, llm_adapter, role_prompt, role):
        """
        Initializes the Charactor object with the given prompts
        :param llm_adapter: An LLM adapter object.
        :param role_prompt: The role prompt.
        :param role: The role of the character.
        """
        self.llm_adapter = llm_adapter
        self.role_prompt = {'role': 'system', 'content': role_prompt}
        self.role = role

    def converse(self, prompt: list[dict[str, str]]) -> str:
        """
        Let the Charactor generate responses based on the given prompt
        :param prompt: A list of prompts.
        :return: The generated response.
        """
        results = self.llm_adapter.chat_completion([self.role_prompt] + prompt)
        return results


class LLMsAdapter:
    """
    The LLMsAdapter class provides methods for interacting with various LLMs.
    This class is a superclass that contains abstract methods and cannot be used directly
    """

    def __init__(self, model_name):
        """
        Initializes the LLMsAdapter object with the given model name.
        :param model_name: The name of the LLM
        """
        self.token_count: int = 0
        self.model_name = model_name

    def completion(self, prompt: str) -> str:
        """
        Generates a completion for the given prompt.
        :param prompt: The prompt for the LLM.
        :return: The generated completion.
        """
        raise NotImplementedError

    def chat_completion(self, prompt: list[dict[str, str]]) -> str:
        """
        Generates a chat completion for the given prompt.
        :param prompt: The prompt for the LLM.
        :return: The generated completion.
        """
        raise NotImplementedError

    def update_token(self, token_delta: int):
        """
        Updates the token count for the LLM.
        :param token_delta: The amount to update the token count by.
        :return: None
        """
        self.token_count += token_delta

    def get_token(self) -> int:
        """
        Returns the current token count for the LLM.
        :return: The current token counts.
        """
        return self.token_count

    def recount_tokens(self):
        """
        Resets the token count for the LLM.
        :return: None
        """
        self.token_count = 0

    def get_charactor(self, role_prompt: str, role: str) -> Charactor:
        """
        Returns a Charactor object based on the LLMAdapter for the given role and role prompt.
        :param role_prompt: The role prompt for the character.
        :param role: The role of the character.
        :return: A character object.
        """
        return Charactor(self, role_prompt, role)

    def get_model_name(self) -> str:
        """
        Returns the model name for the LLM.
        :return: The model name.
        """
        return self.model_name

    def __str__(self):
        return self.model_name


class OpenaiLLMsAdapter(LLMsAdapter):
    """
    An LLMsAdapter implementation for OpenAI models.
    """

    def __init__(self, model_name, api_key='default api key not shown, replace this string with your own one'):
        """
        Initializes the OpenaiLLMsAdapter with the specified model name and API key.
        It is important to not that the default API key is shown, you can claim
        your own API key from OpenAI.
        :param model_name: The name of the OpenAI model to use.
        :param api_key: The API key for OpenAI.
        """
        super().__init__(model_name=model_name)
        openai.api_key = api_key

    @retry(stop=(stop_after_attempt(3)), wait=wait_fixed(2))
    def completion(self,
                   prompt,
                   max_tokens=300,
                   top_p=0.9,
                   temperature=0.0,
                   api_base="https://api.openai.com/v1"):
        """
        Generates a completion for the given prompt.
        :param prompt: The prompt for the completion.
        :param max_tokens: The maximum number of tokens to generate.
        :param top_p: The top_p parameter for the completion.
        :param temperature: The temperature parameter for the completion.
        :param api_base: The API base URL for OpenAI.
        :return: The generated completion.
        """
        # set api base
        openai.api_base = api_base

        # generate response
        chat = openai.Completion.create(
            model=self.get_model_name(),
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )

        # Try to update token usage and handles the exception if fail to get the token usage
        try:
            self.update_token(chat.usage.total_tokens)
        except AttributeError:
            self.update_token(-1)
        return chat.choices[0].text

    @retry(stop=(stop_after_attempt(3)), wait=wait_fixed(2))
    def chat_completion(self,
                        messages,
                        top_p=0.9,
                        temperature=0.0,
                        api_base="https://api.openai.com/v1",
                        **kwargs):
        """
        Generates a completion for the given chat messages.
        :param messages: The chat messages to use for completion.
        :param top_p: The top_p parameter for the completion.
        :param temperature: The temperature parameter for the completion.
        :param api_base: The API base URL for OpenAI.
        :return: The generated completion.
        """
        # set api base
        openai.api_base = api_base

        # generate response
        chat = openai.ChatCompletion.create(
            model=self.get_model_name(),
            messages=messages,
            temperature=temperature,
            top_p=top_p,
        )

        # Try to update token usage and handles the exception if fail to get the token usage
        try:
            self.update_token(chat.usage.total_tokens)
        except AttributeError:
            self.update_token(-1)
        return chat.choices[0].message.content


class LocalLLMsAdapter(OpenaiLLMsAdapter):
    """
    Adapter for local LLMs.
    """

    def __init__(self, model_name, api_base="http://localhost:8000/v1"):
        """
        Initializes the adapter.
        :param model_name: The name of the LLM model.
        :param api_base: The API base URL for the local LLM.
        """
        super().__init__(model_name=model_name)
        self.api_base = api_base

    def completion(self, prompt, max_tokens=300, top_p=0.9, temperature=0.0, **kwargs):
        """
        Generates a completion for the given prompt.
        :param prompt: The prompt for the LLM.
        :param max_tokens: The maximum number of tokens to generate.
        :param top_p: The top_p parameter for the LLM.
        :param temperature: The temperature parameter for the LLM.
        :return: The generated completion.
        """
        return super().completion(prompt=prompt, top_p=top_p, temperature=temperature, api_base=self.api_base)

    def chat_completion(self,
                        messages,
                        top_p=0.9,
                        temperature=0.0,
                        **kwargs):
        """
        Generates a chat completion for the given messages.
        :param messages: The messages for the chat completion.
        :param top_p: The top_p parameter for the LLM.
        :param temperature: The temperature parameter for the LLM.
        :return: The generated chat completion.
        """
        return super().chat_completion(messages=messages,
                                       top_p=top_p,
                                       temperature=temperature,
                                       api_base=self.api_base)


class ZhipuModelsAdapter(LLMsAdapter):
    """
    Adapter for models using the API form ZhipuAI.
    """

    def __init__(self, model_name: str, api_key: str = 'default api key not shown'):
        """
        Initializes the adapter for the given model.
        It is important to not that the default API key is shown, you can claim
        your own API key from the ZhipuAI website(https://open.bigmodel.cn/).
        :param model_name: The name of the model.
        :param api_key: The API key for the model.
        """
        super().__init__(model_name)
        zhipuai.api_key = api_key

    def completion(self, prompt, max_length=300, top_p=0.9, temperature=0.0):
        """
        ChatGLM series from ZhiPuAI does not support completion mode
        calling this method would result in a NotImplementedError
        """
        raise NotImplementedError('ChatGLM series from ZhiPuAI does not support completion')

    @retry(stop=(stop_after_attempt(3)), wait=wait_fixed(2))
    def chat_completion(self, prompt, max_length=300, top_p=0.9, temperature=0.0):
        """
        Chat completion using the ZhipuAI API.
        :param prompt: The prompt for the chat completion.
        :param max_length: The maximum length of the response, which is the maximum token.
        :param top_p: The top p value for the sampling.
        :param temperature: The temperature for the sampling.
        :return: The response from the ZhipuAI API.
        """
        response = zhipuai.model_api.invoke(
            model=self.get_model_name(),
            prompt=prompt,
            top_p=top_p,
            temperature=temperature + 0.1,
            max_tokens=max_length
        )

        self.update_token(response['data']['usage']['total_tokens'])
        return response['data']['choices'][0]['content']


class CharactorGLMAdapter(ZhipuModelsAdapter):
    """
    Adapter for the CharactorGLM model.
    """
    def __init__(self):
        super().__init__("charglm-3")

    def completion(self, **kwargs):
        raise NotImplementedError('CharactorGLM does not support completion')

    def get_charactor(self, role_prompt, role) -> Charactor:
        """
        Get a Charactor instance for the CharactorGLM model.
        :param role_prompt: The role prompt for the character.
        :param role: The role of the character.
        """
        return CharGLMCharactor(self, role_prompt, role)


class CharGLMCharactor(Charactor):
    """
    Character class for the CharactorGLM model.
    """

    def __init__(self, llmadapter, role_prompt, role):
        super().__init__(llmadapter, role_prompt, role)
        self.roles_remain = ['analyst', 'developer', 'tester']
        self.roles_remain.remove(self.role)

    def converse(self, prompt):
        """
        Converse with the character using the CharactorGLM model.
        :param prompt: The prompt for the conversation.
        :return: The response from the character.
        """

        # Modification to the prompt to adapt the format CharactorGLM supports
        for i in prompt:
            i['role'].replace('system', 'assistant')

        # generates response
        response = zhipuai.model_api.invoke(
            model="charglm-3",
            meta={
                "user_info": f"I am a team leader who runs a software development team. I will pass messages from "
                             f"the {self.roles_remain[0]} and the {self.roles_remain[1]} to you to make sure "
                             f"this team runs well",
                "bot_info": self.role_prompt,
                "bot_name": self.role,
                "user_name": "team leader"
            },
            prompt=prompt
        )
        self.llm_adapter.update_token(response['data']['usage']['total_tokens'])
        return response['data']['choices'][0]['content']

import zhipuai
from transformers import AutoTokenizer, AutoModel
from tenacity import retry

from utils.LLMs.LLMsAdapter import LLMsAdapter


class ZhipuModelsAdapter(LLMsAdapter):

    def __init__(self, model_name, api_key='2c700bf3ba6419b8ea37f0602baf527c.7vBfCGkpob2Y8Qzr'):
        super().__init__(model_name)
        zhipuai.api_key = api_key

    def completion(self, prompt, max_tokens=100, max_length=300, top_p=0.9, temperature=0.0):
        # response = zhipuai.model_api.invoke(
        #     model=self.get_model_name(),
        #     prompt=prompt,
        #     top_p=top_p,
        #     temperature=temperature,
        #     max_length=max_length
        # )
        # self.update_token(response['usage']['total_tokens'])
        # return response['data']['choices'][0]['content']
        raise NotImplementedError()

    @retry
    def chat_completion(self, prompt, max_length=300, top_p=0.9, temperature=0.0):
        response = zhipuai.model_api.invoke(
            model=self.get_model_name(),
            prompt=prompt,
            top_p=top_p,
            temperature=temperature+0.1,
            max_length=max_length
        )
        self.update_token(response['data']['usage']['total_tokens'])
        return response['data']['choices'][0]['content']


class CharactorGLMAdapter(ZhipuModelsAdapter):
    def __init__(self):
        super().__init__("charglm-3")

    def completion(self, **kwargs):
        raise NotImplementedError('CharactorGLM does not support completion')

    def get_charactor(self, role_prompt, role):
        return self.Charactor(self, role_prompt, role)

    class Charactor(LLMsAdapter.Charactor):

        def __init__(self, llmadapter, role_prompt, role):
            super().__init__(llmadapter, role_prompt, role)
            self.roles_remain = ['analyst', 'developer', 'tester']
            self.roles_remain.remove(self.role)

        def converse(self, prompt):
            for i in prompt:
                i['role'].replace('system', 'assistant')

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
            print(response)
            self.llm_adapter.update_token(response['data']['usage']['total_tokens'])
            return response['data']['choices'][0]['content']


class CodeGeeXAdapter(LLMsAdapter):
    def __init__(self):
        super().__init__("codegeex2-6b")

    def completion(self, prompt, max_tokens=100, max_length=300, top_p=0.9, temperature=0.0):
        tokenizer = AutoTokenizer.from_pretrained("THUDM/codegeex2-6b", trust_remote_code=True)
        model = AutoModel.from_pretrained("THUDM/codegeex2-6b", trust_remote_code=True).half().cuda()
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(inputs, max_length=max_length, top_p=top_p, temperature=temperature)
        self.update_token(tokenizer.vocab_size)
        return tokenizer.decode(outputs[0])

    def chat_completion(self, prompt, max_tokens=100):
        raise NotImplementedError("CodeGeeX does not support chat completion.")

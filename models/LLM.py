import openai
import tiktoken


from .constants import ChatModel, MODEL_INPUT_PRICE, MODEL_OUTPUT_PRICE
from .prompts import (
    FILTER_PROMPT_SYSTEM,
    FILTER_PROMPT_USER,
    FILTER_PROMPT_USER_EXAMPLE,
    FILTER_PROMPT_ASSISTANT_EXAMPLE,
    FILTER_PROMPT_CODE_BLOCKS,
    MAIN_PROMPT_SYSTEM,
)
from .Embedder import RepoEmbedder


class LLM:
    def __init__(
        self,
        filter_with_llm=False,
        model=ChatModel.GPT_3_5_TURBO_0125,
        need_confirmation=True,
    ):
        self.main_model = model
        self.filter_with_llm = filter_with_llm
        self.filter_model = ChatModel.GPT_3_5_TURBO_0125
        self.running_cost = 0
        self.has_key = False
        self.repo_embeddings = None
        self.need_confirmation = need_confirmation

    def set_api_key(self, api_key):
        openai.api_key = api_key
        self.has_key = True

    def set_main_model(self, model_name):
        if "GPT-3.5" in model_name:
            self.main_model = ChatModel.GPT_3_5_TURBO_0125
        elif "GPT-4" in model_name:
            self.main_model = ChatModel.GPT_4
        else:
            self.main_model = ChatModel.GPT_3_5_TURBO_0125

    def set_repo(self, repo_url):
        self.repo_url = repo_url
        self.repo_name = repo_url.split("/")[-1].split(".")[0]
        self.repo_author = repo_url.split("/")[-2]

        self.repo_embeddings = RepoEmbedder(repo_url, self.repo_name)

    def ask(self, question):
        assert self.repo_embeddings is not None, "No repository set"
        candidates = self.repo_embeddings.search_repo(question)

        if self.filter_with_llm:
            ### Currently not as helpful ###
            candidates = self.filter_candidates(candidates, question)
            if candidates is None:
                return "No relevant code blocks found"

        messages = self.generate_main_input(candidates, question)
        price_estimate = self.estimate_price(messages, self.main_model)
        if self.need_confirmation:
            confirm = input(f"Ask for an estimated cost of {price_estimate}? (y/n)")
            if confirm != "y":
                print("Request not submitted")
                return
        self.running_cost += price_estimate
        response = openai.ChatCompletion.create(
            model=self.main_model, messages=messages
        )
        self.running_cost += self.estimate_price(
            [response["choices"][0]["message"]], self.main_model, output=True
        )
        return response["choices"][0]["message"]["content"]

    def filter_candidates(self, candidates, prompt):
        filter_prompt = self.generate_filter_input(candidates, prompt)
        price_estimate = self.estimate_price(filter_prompt, self.filter_model)
        if self.need_confirmation:
            confirm = input(
                f"Continue with candidate filtering? \n Estimated Price: {price_estimate} (y/n)"
            )
            if confirm != "y":
                print("Request not submitted")
                return
        self.running_cost += price_estimate
        response = openai.ChatCompletion.create(
            model=self.filter_model, messages=filter_prompt
        )
        items = response["choices"][0]["message"]["content"].split("\n")
        choices = []
        for item in items:
            try:
                num = item.split(": ")[0]
                if item.split(": ")[1] == "yes":
                    choices.append(int(num))
            except IndexError:
                continue
        self.running_cost += self.estimate_price(
            [response["choices"][0]["message"]], self.filter_model, output=True
        )
        filtered_candidates = [candidates.iloc[i] for i in choices]
        return filtered_candidates

    def generate_filter_input(self, candidates, question):
        messages = [
            {
                "role": "system",
                "content": FILTER_PROMPT_SYSTEM,
            },
            {
                "role": "user",
                "content": FILTER_PROMPT_USER_EXAMPLE,
            },
            {
                "role": "assistant",
                "content": FILTER_PROMPT_ASSISTANT_EXAMPLE,
            },
            {
                "role": "user",
                "content": FILTER_PROMPT_USER.format(
                    question=question,
                    code_blocks=self.format_code(candidates),
                ),
            },
        ]
        return messages

    def generate_main_input(self, code_items, question):
        formatted_code_blocks = self.format_code(code_items)
        messages = [
            {
                "role": "system",
                "content": MAIN_PROMPT_SYSTEM.format(
                    repo_name=self.repo_name,
                    dir_tree=self.repo_embeddings.get_dir_tree(),
                    formatted_code_blocks=formatted_code_blocks,
                ),
            },
            {
                "role": "user",
                "content": question,
            },
        ]
        return messages

    def format_code(self, code_items):
        formatted_code_blocks = ""
        for i, row in code_items.iterrows():
            block = FILTER_PROMPT_CODE_BLOCKS.format(
                index=i,
                file_path=row["filepath"],
                code=row["code"],
            )
            formatted_code_blocks += block
        return formatted_code_blocks

    def estimate_price(self, messages, engine, output=False):
        encoder = tiktoken.encoding_for_model(engine)
        tokens = []
        for message in messages:
            tokens += encoder.encode(message["content"])
        if output:
            return len(tokens) * MODEL_OUTPUT_PRICE[engine]
        else:
            return len(tokens) * MODEL_INPUT_PRICE[engine]

    def get_running_cost(self):
        if self.repo_embeddings is None:
            return self.running_cost
        return self.running_cost + self.repo_embeddings.running_cost

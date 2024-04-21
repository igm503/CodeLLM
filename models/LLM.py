import json

import openai
import tiktoken
import pandas as pd

from .Embedder import RepoEmbedder
from .constants import ChatModel
from .prompts import (
    FILTER_PROMPT_SYSTEM,
    FILTER_PROMPT_USER,
    FILTER_PROMPT_USER_EXAMPLE,
    FILTER_PROMPT_ASSISTANT_EXAMPLE,
    FILTER_PROMPT_CODE_BLOCKS,
    MAIN_PROMPT_SYSTEM,
)


class LLM:
    def __init__(
        self,
        filter_with_llm=False,
        model=ChatModel.GPT_3_5_TURBO_0125,
    ):
        self.main_model = model
        self.filter_with_llm = filter_with_llm
        self.filter_model = ChatModel.GPT_3_5_TURBO_0125
        self.has_key = False
        self.repo_embeddings = None

    def set_api_key(self, api_key: str):
        openai.api_key = api_key
        self.has_key = True

    def set_main_model(self, model_name: str):
        self.main_model = model_name

    def set_repo(self, repo_url: str):
        self.repo_url = repo_url
        self.repo_name = repo_url.split("/")[-1].split(".")[0]
        author = repo_url.split("/")[-2]

        self.repo_embeddings = RepoEmbedder(repo_url, f"{author}_{self.repo_name}")

    def ask(self, question: str):
        messages = self.get_messages(question, self.filter_with_llm)
        completion = openai.ChatCompletion.create(
            model=self.main_model,
            messages=messages,
        )
        response = completion.choices[0].message.content
        num_input_tokens = completion.usage.prompt_tokens
        num_output_tokens = completion.usage.completion_tokens
        return response, num_input_tokens, num_output_tokens

    def get_messages(self, question: str, filter: bool = False):
        assert self.repo_embeddings is not None, "no repository set"
        candidates = self.repo_embeddings.search_repo(question)
        if filter:
            ### Currently not as helpful ###
            candidates = self.filter_candidates(candidates, question)
        return self.generate_main_input(candidates, question)

    def filter_candidates(self, candidates: pd.DataFrame, prompt: str) -> pd.DataFrame:
        filter_prompt = self.generate_filter_input(candidates, prompt)
        response = openai.ChatCompletion.create(
            model=self.filter_model,
            messages=filter_prompt,
        )
        try:
            items = json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            print("error decoding filter response")
            return candidates
        choices = [int(k) for k, v in items.items() if v]
        if not choices:
            return candidates
        filtered_candidates = candidates.iloc[choices]
        return filtered_candidates

    def generate_filter_input(self, candidates: pd.DataFrame, question: str):
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

    def generate_main_input(self, code_items: pd.DataFrame, question: str):
        assert self.repo_embeddings is not None, "No repository set"
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

    def format_code(self, code_items: pd.DataFrame):
        formatted_code_blocks = ""
        for i, row in code_items.iterrows():
            block = FILTER_PROMPT_CODE_BLOCKS.format(
                index=i,
                file_path=row["filepath"],
                code=row["code"],
            )
            formatted_code_blocks += block
        return formatted_code_blocks

    def get_message_length(self, question: str):
        messages = self.get_messages(question)
        return self.num_tokens_from_messages(messages, self.main_model)

    def num_tokens_from_messages(self, messages: list[dict[str, str]], model: str):
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        num_tokens = 0
        for message in messages:
            num_tokens += 4
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += -1
        num_tokens += 2
        return num_tokens

    def get_repo_tokens(self):
        assert self.repo_embeddings is not None, "No repository set"
        return self.repo_embeddings.get_code_tokens()

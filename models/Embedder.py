from os.path import abspath, dirname, join, isfile
from os import makedirs
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import tiktoken
import numpy as np
import pandas as pd
from tqdm import tqdm
from openai.embeddings_utils import get_embedding

from .Parser import RepoParser
from .constants import EmbeddingModel


class RepoEmbedder:
    DIR = join(dirname(abspath(__file__)), "..", "repos")

    def __init__(
        self,
        repo_url: str,
        repo_name: str,
        model=EmbeddingModel.EMBEDDING_3_SMALL,
    ):
        self.data_dir = join(RepoEmbedder.DIR, repo_name)
        makedirs(self.data_dir, exist_ok=True)

        embeddings_path = join(self.data_dir, "embeddings.npy")
        code_path = join(self.data_dir, "code")
        repo_path = join(self.data_dir, repo_name)
        print(embeddings_path, code_path, repo_path)

        self.parser = RepoParser(repo_url, repo_path, code_path)

        self.model = model
        self.embed = partial(get_embedding, engine=self.model)
        self.running_cost = 0

        self.code = self.load_code(code_path)
        self.embeddings = self.load_embeddings(embeddings_path)

    def load_code(self, code_path: str):
        if isfile(code_path):
            code = pd.read_csv(code_path)
            assert isinstance(code, pd.DataFrame), "Code not loaded"
        else:
            code = self.parser.generate_code_df()
        return code

    def load_embeddings(self, embeddings_path: str):
        if isfile(embeddings_path):
            print("Embeddings found.")
            return np.load(embeddings_path)
        else:
            print("Embeddings not found.")
            return self.generate_embeddings(embeddings_path)

    def generate_embeddings(self, embeddings_path: str):
        text_list = self.code["code"].tolist()
        print("Generating embeddings...")
        with ThreadPoolExecutor() as executor:
            embedding_list = list(
                tqdm(executor.map(self.embed, text_list), total=len(text_list))
            )
        embeddings = np.array(embedding_list)
        np.save(embeddings_path, embeddings)
        return embeddings

    def search_repo(self, prompt: str, k: int = 20) -> pd.DataFrame:
        query = np.array(self.embed(prompt))
        match_indices = self.filter_embeddings(query, k=k)
        matches = self.code.iloc[match_indices]
        return matches

    def filter_embeddings(self, query: np.ndarray, k: int = 20):
        _cosine_similarities = np.dot(self.embeddings, query.T)
        cosine_similarities = np.squeeze(_cosine_similarities)
        k = min(k, len(cosine_similarities))
        big_indices = np.argpartition(cosine_similarities, -k)[-k:]
        return big_indices

    def get_dir_tree(self):
        return self.parser.get_dir_tree()

    def get_code_tokens(self):
        encoding = tiktoken.encoding_for_model(self.model)
        num_tokens = 0
        for code in self.code["code"].tolist():
            num_tokens += len(encoding.encode(code))
        return num_tokens

from os.path import abspath, dirname, join, isfile
from os import makedirs
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np
import pandas as pd
import tiktoken
from tqdm import tqdm
from openai.embeddings_utils import get_embedding

from .Parser import RepoParser
from .constants import MODEL_INPUT_PRICE


class RepoEmbedder:
    DIR = join(dirname(abspath(__file__)), "repos")

    def __init__(self, repo_url: str, repo_name: str, engine="text-embedding-3-small"):
        self.data_dir = join(RepoEmbedder.DIR, repo_name)
        makedirs(self.data_dir, exist_ok=True)

        embeddings_path = join(self.data_dir, "embeddings.npy")
        code_path = join(self.data_dir, "code")
        repo_path = join(self.data_dir, repo_name)

        self.parser = RepoParser(repo_url, repo_path, code_path)

        self.engine = engine
        self.embed = partial(get_embedding, engine=self.engine)
        self.running_cost = 0

        self.code = self.load_code(code_path)
        self.embeddings = self.load_embeddings(embeddings_path)

    def load_code(self, code_path):
        if isfile(code_path):
            return pd.read_csv(code_path)
        else:
            return self.parser.generate_code_df()

    def load_embeddings(self, embeddings_path):
        if isfile(embeddings_path):
            print("Embeddings found.")
            return np.load(embeddings_path)
        else:
            print("Embeddings not found.")
            return self.generate_embeddings(embeddings_path)

    def generate_embeddings(self, embeddings_path):
        text_list = self.code["code"].tolist()
        cost = self.estimate_price(text_list)
        confirm = input(f"Generate for an estimated cost of {cost}? (y/n)")
        if confirm != "y":
            print("No Embeddings Generated")
            return
        self.running_cost += cost
        embedding_list = []
        print("Generating embeddings...")

        with ThreadPoolExecutor() as executor:
            embedding_list = list(
                tqdm(executor.map(self.embed, text_list), total=len(text_list))
            )
        embeddings = np.array(embedding_list)
        np.save(embeddings_path, embeddings)
        return embeddings

    def search_repo(self, prompt, k=20):
        query = np.array(self.embed(prompt))
        match_indices = self.filter_embeddings(query, k=k)
        matches = self.code.iloc[match_indices]
        self.running_cost += self.estimate_price([prompt])
        return matches

    def filter_embeddings(self, query, k=20):
        _cosine_similarities = np.dot(self.embeddings, query.T)
        cosine_similarities = np.squeeze(_cosine_similarities)
        k = min(k, len(cosine_similarities))
        big_indices = np.argpartition(cosine_similarities, -k)[-k:]
        return big_indices

    def estimate_price(self, text):
        encoding = tiktoken.encoding_for_model(self.engine)
        tokens = [encoding.encode(t) for t in text]
        return len(tokens) * MODEL_INPUT_PRICE[self.engine]

    def get_dir_tree(self):
        return self.parser.get_dir_tree()

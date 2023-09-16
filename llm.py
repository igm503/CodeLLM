import os
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm
import numpy as np
import pandas as pd
from openai.embeddings_utils import get_embedding
import openai
from tree_sitter import Language, Parser
import tiktoken



EMBEDDING_PRICE = {
    'text-embedding-ada-002': 0.0001,
}

CHAT_PRICE = {
    'gpt-3.5-turbo': 0.003,
    'gpt-4': 0.06,
}

ENDINGS = ['.py']

PY_LANGUAGE = Language('build/my-languages.so', 'python')

class CodeParser():
    def __init__(self, repo_path: str, save_path: str):

        self.repo_path = repo_path
        self.save_path = save_path

        self.parser = Parser()
        self.parser.set_language(PY_LANGUAGE)

    def get_files(self):
        files = []
        for root, _, filenames in os.walk(self.repo_path):
            for filename in filenames:
                if filename.endswith('.py'):
                    files.append(os.path.join(root, filename))
        return files
    
    def get_code(self, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def generate_code_df(self):
        filepaths = self.get_files()
        data = []
        for filepath in filepaths:
            filename = filepath.split('/')[-1]
            code_string = self.get_code(filepath)
            blocks = self.extract_blocks_from_code(code_string)
            for i, block in enumerate(blocks):
                clean_block = block.replace('<|endoftext|>', '<endoftext>')
                data.append({'code': clean_block, 'filename': filename, 'filepath': filepath, 'num_in_file': i})
            
        self.code = pd.DataFrame(data)
        self.code.to_csv(self.save_path, index=False) 
        return self.code
    
    def extract_blocks_from_code(self, code: str):
        tree = self.parser.parse(bytes(code, "utf8"))

        blocks = []
        last_end_byte = 0
        def find_blocks(node):
            nonlocal last_end_byte
            if node.type == 'function_definition':
                if node.start_byte > last_end_byte:
                    outside_function = code[last_end_byte:node.start_byte].strip()
                    if outside_function:
                        blocks.append(outside_function)
                
                start_byte = node.start_byte
                end_byte = node.end_byte
                blocks.append(code[start_byte:end_byte])
                
                last_end_byte = end_byte
            for child in node.children:
                find_blocks(child)

        find_blocks(tree.root_node)

        if last_end_byte < len(code):
            remaining_code = code[last_end_byte:].strip()
            if remaining_code:
                blocks.append(remaining_code)

        return blocks

class RepoEmbeddings():

    DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

    def __init__(self, repo_path):

        self.repo_path = repo_path
        self.repo_name = repo_path.split('/')[-1]

        self.data_dir = os.path.join(RepoEmbeddings.DIR, self.repo_name)

        if not os.path.isdir(self.data_dir):
            os.mkdir(self.data_dir)

        self.code_path = os.path.join(self.data_dir, 'code')
        self.embeddings_path = os.path.join(self.data_dir, 'embeddings')

        self.code_parser = CodeParser(self.repo_path, self.code_path) 

        self.code = None
        self.embeddings = None

        self.engine = "text-embedding-ada-002"

    def load_code(self):
        if os.path.isfile(self.code_path):
            self.code = pd.read_csv(self.code_path)
        else:
            self.code = self.code_parser.generate_code_df()
            
    def load_embeddings(self):
        if os.path.isfile(self.embeddings_path):
            self.embeddings = np.load(self.embeddings_path)
        else:
            self.generate_embeddings()

    def generate_embeddings(self):
        text_list = self.code['code'].tolist()
        confirm = input(f"Embeddings not found. Would you like to generate {len(self.code)} embeddings? \n Estimated Cost: {self.estimate_price(text_list)} (y/n)")
        if confirm != 'y':
            print('Exiting...')
            exit()
        embedding_list = []
        print('Generating embeddings...')

        def worker(text):
            return get_embedding(text, engines=self.engine)

        with ThreadPoolExecutor() as executor:
            embedding_list = list(tqdm(executor.map(worker, text_list), total=len(text_list)))

        self.embeddings = np.array(embedding_list)
        np.save(self.embeddings_path, self.embeddings)

    def search_repo(self, prompt, k=20): 
        query = get_embedding(prompt, engine=self.engine)
        candidate_indices = self.filter_embeddings(query, k=k)
        candidates = self.code.iloc[candidate_indices]
        return candidates

    def filter_embeddings(self, query, k=20):
        _cosine_similarities = np.dot(self.embeddings, query.T)
        cosine_similarities = np.squeeze(_cosine_similarities)
        k = min(k, len(cosine_similarities))
        big_indices = np.argpartition(cosine_similarities, -k)[k:]
        return big_indices
    
    def estimate_price(self, text):
        encoding = tiktoken.encoding_for_model(self.engine)
        text = ('\n').join(text)
        text = encoding.encode(text)
        return len(text) / 1000 * EMBEDDING_PRICE[self.engine]


class LLM():
    def __init__(self):
        self.filter_model = 'gpt-3.5-turbo'
        self.main_model = 'gpt-3.5-turbo'

    def set_repo(self, repo_path):
        self.repo_embeddings = RepoEmbeddings(repo_path)
        self.repo_embeddings.load_code()
        self.repo_embeddings.load_embeddings()

    def ask(self, prompt):
        candidates = self.repo_embeddings.search_repo(prompt)
        filtered_candidates = self.filter_candidates(candidates, prompt)
        final_prompt = self.generate_main_prompt(filtered_candidates, prompt)
        confirm = input(f'Continue to ask the chat model? \n Estimated Price: {self.estimate_price(final_prompt, self.main_model)} (y/n)')
        if confirm != 'y':
            print('Request not submitted')
            return
        response = openai.ChatCompletion.create(
            model=self.main_model,
            messages=final_prompt
        ) 
        return response

    def filter_candidates(self, candidates, prompt):
        filter_prompt = self.generate_filter_prompt(candidates, prompt)
        confirm = input(f'Continue with candidate filtering? \n Estimated Price: {self.estimate_price(filter_prompt, self.filter_model)} (y/n)')
        if confirm != 'y':
            print('Request not submitted')
            return
        response = openai.ChatCompletion.create(
            model=self.filter_model,
            prompt=filter_prompt
        )
        
        filtered_candidates = candidates[response.choices]
        return filtered_candidates 

    def generate_filter_prompt(self, candidates, prompt):
        formatted_code_blocks = self.format_code(candidates)
        messages=[
            {
                "role": "system",
                "content": 
                    "You are an experienced, intelligent software developer \
                    who is explaining how a codebase works to an intelligent \
                    developer who isn't yet familiar with it. The user would \
                    like to know which of these code blocks are relevant to \
                    the topic he is asking. Your job is to label each code \
                    block as relevant or irrelevant to the user's question. \
                    You should not provide any other information to the user."
            },
            {
                "role": "user",
                ### NEED TO WRITE EXAMPLE PROMPT HERE'''
                "content": 
                    f"Question: \n\n \
                    Code Bocks: "
            },
            {
                "role": "assitant",
                "content": None 
                     # NEED TO WRITE EXAMPLE RESPONSE HERE 
            },
            {
                "role": "user",
                "content": 
                    f"Question: {prompt} \n\n \
                    Code Blocks: {formatted_code_blocks}"
            }
        ] 




    def generate_main_prompt(self, code_items, prompt):

        formatted_code_blocks = self.format_code(code_items)

        messages=[
            {
                "role": "system", 
                "content": 
                    "You are an experienced, intelligent software developer \
                    who is explaining how a codebase works to an intelligent \
                    developer who isn't yet familiar with it. When the user \
                    asks a question, the system will provide you with several \
                    code blocks from the codebase that might be relevant to \
                    the user's question. You will also receive a directory \
                    listing of the codebase. Your job is to respond to the \
                    question, using the code blocks as context. However \
                    you should not refer to the code blocks in your response, \
                    except by quoting them, since the user has not seen them."
            },
            {
                "role": "user", 
                "content": prompt
            },
            {
                "role": "system",
                "content": 
                    f"Here are some code blocks that might be relevant to the \
                    user's question: {formatted_code_blocks}"
            }
        ]
        return messages

    def format_code(self, code_items):
        formatted_code_list = []
        for i, row in enumerate(code_items):
            formatted_code_list.append(f"Code block {i} \n from file: {row['filepath']} \n {row['code']}")
        formatted_code_blocks = '\n\n'.join(formatted_code_list)
        return formatted_code_blocks

    def estimate_price(self, messages, engine):
        encoder = tiktoken.encoding_for_model(engine)
        text = []
        for message in messages:
            text += encoder.encode(message['content'])
            return len(text) / 1000 * CHAT_PRICE[self.engine]

if __name__ == '__main__':
    openai.api_key = os.getenv("OPENAI_API_KEY")
    llm = LLM()
    llm.set_repo('tinygrad')
    response = llm.ask('How does the autograd work?')

import os
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm
import numpy as np
import pandas as pd
from openai.embeddings_utils import get_embedding
import openai
from tree_sitter import Language, Parser
import tiktoken
from git import Repo

MODEL_INPUT_PRICE = {
    'text-embedding-ada-002': 0.0001,
    'gpt-3.5-turbo-16k': 0.003,
    'gpt-4': 0.06,
}

MODEL_OUTPUT_PRICE = {
    'gpt-3.5-turbo-16k': 0.004,
    'gpt-4': 0.12,
}

ENDINGS = ['.py']

PY_LANGUAGE = Language('build/my-languages.so', 'python')

class CodeParser():
    def __init__(self, repo_url: str, repo_name: str, save_path: str):

        self.repo_url = repo_url
        self.repo_name = repo_name

        self.repo_path = os.path.join(RepoEmbedder.DIR, self.repo_name, self.repo_name)
        self.save_path = save_path

        if not os.path.isdir(self.repo_path):
            self.repo = Repo.clone_from(self.repo_url, self.repo_path)
        else:
            self.repo = Repo(self.repo_path)

        self.parser = Parser()
        self.parser.set_language(PY_LANGUAGE)

        self.create_tree()
    
    def create_tree(self):

        def get_repo_tree(root, level=0):
            tree_string = ''
            for entry in root:
                tree_string += f'{"-" * 4 * level}| {entry.path}, {entry.type}\n'
                if entry.type == "tree" and level < 2:
                    tree_string += get_repo_tree(entry, level + 1)
            return tree_string
        
        self.dir_tree = get_repo_tree(self.repo.head.commit.tree)

    def get_files(self):
        files = []
        rel_files = []
        for root, _, filenames in os.walk(self.repo_path):
            for filename in filenames:
                if filename.endswith('.py'):
                    rel_dir = os.path.relpath(root, self.repo_path)
                    rel_file = os.path.join(rel_dir, filename)
                    rel_files.append(rel_file)
                    files.append(os.path.join(root, filename))
        return files, rel_files
    
    def get_code(self, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def generate_code_df(self):
        filepaths, rel_filepaths = self.get_files()
        data = []
        for filepath, rel_filepath in zip(filepaths, rel_filepaths):
            filename = filepath.split('/')[-1]
            code_string = self.get_code(filepath)
            blocks = self.extract_blocks_from_code(code_string)
            for i, block in enumerate(blocks):
                clean_block = block.replace('<|endoftext|>', '<endoftext>')
                data.append({'code': clean_block, 'filename': filename, 'filepath': rel_filepath, 'num_in_file': i})
            
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


class RepoEmbedder():

    DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

    def __init__(self, repo_url: str, repo_name: str, get_confirm=True):

        self.get_confirm = get_confirm
        self.repo_name = repo_name     
        self.data_dir = os.path.join(RepoEmbedder.DIR, self.repo_name)
        
        if not os.path.isdir(RepoEmbedder.DIR):
            os.mkdir(RepoEmbedder.DIR)
        if not os.path.isdir(self.data_dir):
            os.mkdir(self.data_dir)

        self.code_path = os.path.join(self.data_dir, 'code')
        self.embeddings_path = os.path.join(self.data_dir, 'embeddings.npy')

        self.code_parser = CodeParser(repo_url, repo_name, self.code_path) 

        self.code = None
        self.embeddings = None

        self.running_cost = 0

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
        cost = self.estimate_price(text_list)
        if self.get_confirm:
            confirm = input(f"Embeddings not found. Would you like to generate {len(self.code)} embeddings? \n Estimated Cost: {cost} (y/n)")
        else:
            confirm = 'y'
        if confirm != 'y':
            print('No Embeddings Generated')
            return
        self.running_cost += cost
        embedding_list = []
        print('Generating embeddings...')

        def worker(text):
            return get_embedding(text, engine=self.engine)

        with ThreadPoolExecutor() as executor:
            embedding_list = list(tqdm(executor.map(worker, text_list), total=len(text_list)))

        self.embeddings = np.array(embedding_list)
        np.save(self.embeddings_path, self.embeddings)

    def search_repo(self, prompt, k=20): 
        query = np.array(get_embedding(prompt, engine=self.engine))
        candidate_indices = self.filter_embeddings(query, k=k)
        candidates = self.code.iloc[candidate_indices]
        self.running_cost += self.estimate_price([prompt])
        return candidates

    def filter_embeddings(self, query, k=20):
        _cosine_similarities = np.dot(self.embeddings, query.T)
        cosine_similarities = np.squeeze(_cosine_similarities)
        k = min(k, len(cosine_similarities))
        big_indices = np.argpartition(cosine_similarities, -k)[-k:]
        return big_indices
    
    def estimate_price(self, text):
        encoding = tiktoken.encoding_for_model(self.engine)
        tokens = []
        for t in text:
            tokens += encoding.encode(t)
        return len(tokens) / 1000 * MODEL_INPUT_PRICE[self.engine]


class LLM():
    def __init__(self, ask_for_confirmation=True, filter_with_llm=False, model='gpt-3.5-turbo-16k'):
        self.get_confirm = ask_for_confirmation
        self.main_model = model
        self.filter_with_llm = filter_with_llm
        self.filter_model = 'gpt-3.5-turbo-16k'
        self.running_cost = 0
        self.has_key = False

        self.repo_embeddings = None
    
    def set_api_key(self, api_key):
        openai.api_key = api_key
        self.has_key = True
    
    def set_main_model(self, model_name):
        if 'GPT-3.5' in model_name:
            self.main_model = 'gpt-3.5-turbo-16k'
        elif 'GPT-4' in model_name:
            self.main_model = 'gpt-4'
        else:
            self.main_model = 'gpt-3.5-turbo-16k'

    def set_repo(self, repo_url):
        self.repo_url = repo_url
        self.repo_name = repo_url.split('/')[-1].split('.')[0]
        self.repo_author = repo_url.split('/')[-2]
        
        self.repo_embeddings = RepoEmbedder(repo_url, self.repo_name, get_confirm=self.get_confirm)
        self.repo_embeddings.load_code()
        self.repo_embeddings.load_embeddings()

    def ask(self, prompt):
        candidates = self.repo_embeddings.search_repo(prompt)

        if self.filter_with_llm:
        ### Currently not as helpful ###
            candidates = self.filter_candidates(candidates, prompt)
            if candidates is None:
                return 'No relevant code blocks found'
        
        final_prompt = self.generate_main_prompt(candidates, prompt)
        price_estimate = self.estimate_price(final_prompt, self.main_model)
        if self.get_confirm:
            confirm = input(f'Continue to ask the chat model? \n Estimated Price: {price_estimate} (y/n)')
        else:
            confirm = 'y'
        if confirm != 'y':
            print('Request not submitted')
            return
        self.running_cost += price_estimate 
        response = openai.ChatCompletion.create(
            model=self.main_model,
            messages=final_prompt
        ) 
        self.running_cost += self.estimate_price([response['choices'][0]['message']], self.main_model, output=True)
        return response['choices'][0]['message']['content']

    def filter_candidates(self, candidates, prompt):
        filter_prompt = self.generate_filter_prompt(candidates, prompt)
        price_estimate = self.estimate_price(filter_prompt, self.filter_model)
        if self.get_confirm:
            confirm = input(f'Continue with candidate filtering? \n Estimated Price: {price_estimate} (y/n)')
        else:
            confirm = 'y'
        if confirm != 'y':
            print('Request not submitted')
            return
        self.running_cost += price_estimate
        response = openai.ChatCompletion.create(
            model=self.filter_model,
            messages=filter_prompt
        )
        items = response['choices'][0]['message']['content'].split('\n')
        choices = []
        for item in items:
            try:
                num = item.split(': ')[0]
                if item.split(': ')[1] == 'yes':
                    choices.append(int(num))
            except:
                continue
        self.running_cost += self.estimate_price([response['choices'][0]['message']], self.filter_model, output=True)

        filtered_candidates = [candidates.iloc[i] for i in choices]

        return filtered_candidates 

    def generate_filter_prompt(self, candidates, prompt):
        messages=[
            {
                "role": "system",
                "content": 
                    """You are an experienced, intelligent software developer 
                    who is explaining how a codebase works to an intelligent 
                    developer who isn't yet familiar with it. The user would 
                    like to know which of these code blocks are relevant to 
                    the topic he is asking about. Your job is to label each 
                    code block as relevant or irrelevant to the user's question.
                    You may include things if you think they are relevant, but
                    don't include anything that is clearly irrelevant.
                    You should not provide any other information to the user.
                    Simply label each code block with "yes" or "no". For example,
                    if code block 1, 2, and 4 are relevant, but 3 is not, you 
                    would respond with 
"1: yes\n2: yes\n3: no\n4: yes" """
            },
            {
                "role": "user",
                
                "content": 
                    f"""Question: How is the backwards version of the lgamma
                    function implemented in the mps backend?\n\n 
                    Code Bocks:
\n\n
Code block 1
\n
from file: Reinforcement Learning/PPOProcCNN.py
\n 
def forward(self, x):\n        x = self.pool(F.relu(self.conv1(x)))\n        x = self.batch_norm_conv1(x)\n        x = self.pool(F.relu(self.conv2(x)))\n        x = self.batch_norm_conv2(x)\n        x = torch.flatten(x, 1) \n        x = F.relu(self.fc1(x))\n        x = F.relu(self.fc2(x))\n        x = self.fc3(x)\n        return x
\n\n
Code block 2
\n
from file: aten/src/ATen/native/mps/operations/gamma.mm
\n
TORCH_IMPL_FUNC(digamma_out_mps)(const Tensor& self, const Tensor& output_) {{
  TORCH_CHECK(self.scalar_type() != ScalarType::Double, "MPS does not support digamma_out op with scalar type: Double");

  Tensor output = output_;
  bool needs_output_copy = false;
  uint32_t length = output.numel();
  if (length == 0) {{
    return;
  }}

  if (!self.is_contiguous()) {{
    output = output.contiguous();
    needs_output_copy = true;
  }}

  using namespace mps;

  std::string input_type = scalarToMetalTypeString(self.scalar_type());
  std::string output_type = scalarToMetalTypeString(output.scalar_type());

  @autoreleasepool {{
    id<MTLDevice> device = MPSDevice::getInstance()->device();
    id<MTLComputePipelineState> cplState = getCPLState(device, input_type, output_type, "digamma");

    MPSStream* mpsStream = getCurrentMPSStream();
    dispatch_sync(mpsStream->queue(), ^() {{
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      id<MTLBuffer> outBuf = getMTLBufferStorage(output);
      id<MTLBuffer> selfBuf = getMTLBufferStorage(self);

      getMPSProfiler().beginProfileKernel(cplState, "digamma_out", {self});

      [computeEncoder setComputePipelineState:cplState];
      [computeEncoder setBuffer:selfBuf offset:self.storage_offset() * self.element_size() atIndex:0];
      [computeEncoder setBuffer:outBuf offset:output.storage_offset() * output.element_size() atIndex:1];

      mps::dispatch1DJob(computeEncoder, cplState, static_cast<uint32_t>(length));

      getMPSProfiler().endProfileKernel(cplState);
    }});
  }}
  if (needs_output_copy) {{
    output_.copy_(output);
  }}
}}
                    """
            },
            {
                "role": "assistant",
                "content": '1: no\n2: yes'
            },
            {
                "role": "user",
                "content": 
                    f"""Question: {prompt} \n\n 
                    Code Blocks: {self.format_code(candidates)}"""
            }
        ] 
        return messages


    def generate_main_prompt(self, code_items, prompt):

        formatted_code_blocks = self.format_code(code_items)

        messages=[
            {
                "role": "system", 
                "content": 
                    f"""You are an experienced, intelligent software developer 
                    who is explaining how a codebase works to an intelligent 
                    developer who isn't yet familiar with it. Before you see 
                    the user's question, I will provide you with several 
                    code blocks from the codebase that might be relevant to 
                    the user's question. I will also give you a directory 
                    listing of the codebase. Your job is to respond to the 
                    question, using the code blocks as context. However 
                    you should not refer to the code blocks in your response, 
                    except by quoting them, since the user has not seen them. 
                    If you do quote them, you should quote them exactly, and 
                    include the filepath. Finally, do not ask the user to be
                    shown more code snippets. It is not the user's job to provide
                    code snippets.\n\n\n
                    The user is asking about the git repository {self.repo_name}. 
                    Here is the high level directory structure of the repo:
                    \n\n
                    {self.repo_embeddings.code_parser.dir_tree}
                    \n\n
                    Here are some code blocks that might be relevant to the 
                    user's question:
                    \n\n
                    {formatted_code_blocks}"""
            },
            {
                "role": "user", 
                "content": prompt
            },
        ]
        return messages

    def format_code(self, code_items):
        formatted_code_list = []
        for i, row in code_items.iterrows():
            formatted_code_list.append(f"Code block {i} \n from file: {row['filepath']} \n {row['code']}")
        formatted_code_blocks = '\n\n'.join(formatted_code_list)
        return formatted_code_blocks

    def estimate_price(self, messages, engine, output=False):
        encoder = tiktoken.encoding_for_model(engine)
        tokens = []
        for message in messages:
            tokens += encoder.encode(message['content'])
        if output:
            return len(tokens) / 1000 * MODEL_OUTPUT_PRICE[engine]
        else:
            return len(tokens) / 1000 * MODEL_INPUT_PRICE[engine]

    def get_running_cost(self):
        if self.repo_embeddings is None:
            return self.running_cost
        return self.running_cost + self.repo_embeddings.running_cost

if __name__ == '__main__':
    openai.api_key = os.getenv("OPENAI_API_KEY")
    llm = LLM()
    llm.set_repo('https://github.com/igm503/Deep-Learning-Replications-and-Experiments.git')
    response = llm.ask('What sort of data is being generated in data_gen.py? What would it be for?')

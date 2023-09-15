import os
import pandas as pd
from tree_sitter import Language, Parser

ENDINGS = ['.py']

PY_LANGUAGE = Language('build/my-languages.so', 'python')

class CodeParser():
    def __init__(self, repo_path: str):

        self.repo_path = repo_path

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
                data.append({'code': block, 'filename': filename, 'filepath': filepath, 'num_in_file': i})
            
        self.code = pd.DataFrame(data)
        self.code.to_csv(os.path.join(self.repo_path, 'code.csv'), index=False) 
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

parse = CodeParser('torch')
data = parse.generate_code_df()
total_chars = 0
for code_block in data['code']:
    total_chars += len(code_block)
print(len(data))
print(total_chars / 2)
print((total_chars / 2) * 0.0001 /1000)
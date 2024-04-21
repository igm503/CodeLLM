import os

import pandas as pd
from git import Repo
from tree_sitter import Parser

from .constants import PY_LANGUAGE


class RepoParser:
    def __init__(self, repo_url: str, repo_path: str, save_path: str):
        self.repo_path = repo_path
        self.save_path = save_path

        if os.path.isdir(self.repo_path):
            self.repo = Repo(self.repo_path)
        else:
            self.repo = Repo.clone_from(repo_url, self.repo_path)

        self.parser = Parser()
        self.parser.set_language(PY_LANGUAGE)

        self.dir_tree = self.create_tree(self.repo.head.commit.tree)

    def create_tree(self, root, level=0):
        tree_string = ""
        for entry in root:
            tree_string += f"{entry.path}, {entry.type}\n"
            if entry.type == "tree" and level < 2:
                tree_string += self.create_tree(entry, level + 1)
        return tree_string

    def generate_code_df(self):
        paths, rel_paths = self.get_file_paths()
        data = []
        for path, rel_path in zip(paths, rel_paths):
            file_name = os.path.basename(path)
            code_string = self.get_code(path)
            code_blocks = self.extract_blocks_from_code(code_string)
            for i, block in enumerate(code_blocks):
                clean_block = block.replace("<|endoftext|>", "<endoftext>")
                data.append(
                    {
                        "code": clean_block,
                        "filename": file_name,
                        "filepath": rel_path,
                        "num_in_file": i,
                    }
                )
        self.code = pd.DataFrame(data)
        self.code.to_csv(self.save_path, index=False)
        return self.code

    def get_file_paths(self):
        paths = []
        rel_paths = []
        for root, _, files in os.walk(self.repo_path):
            for file in files:
                if file.endswith(".py"):
                    path = os.path.join(root, file)
                    rel_path = os.path.relpath(path, self.repo_path)
                    paths.append(os.path.join(root, file))
                    rel_paths.append(rel_path)
        return paths, rel_paths

    def get_code(self, file_path: str):
        with open(file_path, "r", encoding="utf-8") as f:
            code_string = f.read()
        return code_string

    def extract_blocks_from_code(self, code: str):
        tree = self.parser.parse(bytes(code, "utf-8"))

        blocks = []
        last_end_byte = 0

        def find_blocks(node):
            nonlocal last_end_byte
            nonlocal blocks
            if node.type == "function_definition":
                start_byte = node.start_byte
                if start_byte > last_end_byte:
                    outside_function = code[last_end_byte:start_byte].strip()
                    if outside_function:
                        blocks.append(outside_function)
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

    def get_dir_tree(self):
        return self.dir_tree

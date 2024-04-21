import os

from tree_sitter import Language

DIR = os.path.dirname(os.path.abspath(__file__))
build_path = os.path.join(DIR, "build", "my-languages.so")
repo_path = os.path.join(DIR, "tree-sitter-python")

Language.build_library(
    build_path,
    [repo_path],
)

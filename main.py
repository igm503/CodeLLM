import requests
import os

from streamlit_ace import st_ace
import streamlit as st
from git import Repo

from llm import LLM

st.set_page_config(layout="wide")

model = LLM()

if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "repo_url" not in st.session_state:
    st.session_state["repo_url"] = None

def process_text():
    user_input = st.session_state["text"]
    clear_text()
    if user_input == 'help':
        st.session_state["messages"].append('BOT: Once you have loaded a repo, you can ask the chatbot questions about the code.')
    else:
        

def clear_text():
    st.session_state["messages"].append('YOU: ' + st.session_state["text"])
    st.session_state["text"] = ''

def get_repo(url=None, author=None, name=None):
    if url is not None:
        st.session_state['repo_url'] = url
        st.session_state['repo_author'] = url.split('/')[3].split('.')[0]
        st.session_state['repo_name'] = url.split('/')[4]
        st.session_state["messages"].append(f'BOT: Loading repo {st.session_state["repo_name"]} from URL')
        
    elif author is not None and name is not None:
        st.session_state['repo_author'] = author
        st.session_state['repo_name'] = name
        st.session_state['repo_url'] = f'https://github.com/{author}/{name}.git'
        st.session_state["messages"].append(f'BOT: Loading repo {name} from Author: ' + author )
    else:
        st.session_state["messages"].append('BOT: Please specify either a URL or an Author and Name')
        return
    
    repo = Repo.clone_from(repo_url, os.path.dirname(__file__))
    model.set_repo()

st.title('Context Aware Code LLM')

for item in st.session_state["messages"]:
    st.write(item)
input = st.text_input("Input window", key="text", on_change=process_text)


with st.sidebar:
    with st.expander("Set Repo by URL"):
        repo_url = st.text_input("Repository URL", value="")
        if st.button("Load", key='load_by_url'):
            get_repo(url=repo_url)
    with st.expander("Set Repo by Author and Name"):
        repo_author = st.text_input("Repository Author", value="")
        repo_name = st.text_input("Repository Name", value="")
        if st.button("Load", key='load_by_author_name'):
            get_repo(author=repo_author, name=repo_name)



@st.cache_data()
def get_github_content(user, repo, path=''):
    url = f'https://api.github.com/repos/{user}/{repo}/contents/{path}'
    response = requests.get(url)
    return response.json()

def print_directory_structure(user, repo, path=''):
    contents = get_github_content(user, repo, path)
    for item in contents:
        if item['type'] == 'dir':
            st.write(f'Directory: {item["path"]}')
            print_directory_structure(user, repo, item['path'])
        else:
            st.write(f'File: {item["path"]}')



css='''
<style>
    section.main>div {
        padding-bottom: 1rem;
    }
    [data-testid="column"] {
        overflow: auto;
        height: 70vh;
    }
</style>
'''

st.markdown(css, unsafe_allow_html=True)



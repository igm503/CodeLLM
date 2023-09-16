import streamlit as st

from llm import LLM

st.set_page_config(layout="wide")

if 'llm' not in st.session_state:
    st.session_state.llm = LLM(ask_for_confirmation=False, model='gpt-4')  # Replace with your actual LLM initialization logic

model = st.session_state.llm

if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "repo_url" not in st.session_state:
    st.session_state["repo_url"] = None
if "repo_author" not in st.session_state:
    st.session_state["repo_author"] = None
if "repo_name" not in st.session_state:
    st.session_state["repo_name"] = None

def process_text():
    user_input = st.session_state["text"]
    clear_text()
    if user_input == 'help':
        st.session_state["messages"].append('System: Once you have loaded a repo, you can ask the chatbot questions about the code.')
    else:
        response = model.ask(user_input)
        st.session_state["messages"].append('CodeLLM:\n\n' + response)

def clear_text():
    st.session_state["messages"].append('You:\n\n' + st.session_state["text"])
    st.session_state["text"] = ''

def get_repo(url=None, author=None, name=None):
    if url is not None:
        st.session_state['repo_url'] = url
        st.session_state['repo_author'] = url.split('/')[3]
        st.session_state['repo_name'] = url.split('/')[4].split('.')[0]
        st.session_state["messages"].append(f'System: Loading repo {st.session_state["repo_name"]} from URL')
    elif author is not None and name is not None:
        st.session_state['repo_author'] = author
        st.session_state['repo_name'] = name
        st.session_state['repo_url'] = f'https://github.com/{author}/{name}.git'
        st.session_state["messages"].append(f'System: Loading repo {name} from Author: ' + author )
    else:
        st.session_state["messages"].append('System: Please specify either a URL or an Author and Name')
        return
    
    model.set_repo(st.session_state['repo_url'])
    st.session_state["messages"].append(f'System: Repo loaded successfully')

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

if st.session_state['repo_url'] is None:
    st.title('Context Aware Code LLM', help='Select a Repository to Load')
else:
    st.title(f'Context Aware Code LLM', help=f'Repository: {st.session_state["repo_author"]}/{st.session_state["repo_name"]}')

for item in st.session_state["messages"]:
    st.write(item)
input = st.text_input("Input window", key="text", on_change=process_text)



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



import streamlit as st
from llm import LLM

st.set_page_config(layout="wide")

if 'llm' not in st.session_state:
    st.session_state.llm = LLM(ask_for_confirmation=False)

model = st.session_state.llm

if "cost" not in st.session_state:
    st.session_state["cost"] = 0
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "repo_url" not in st.session_state:
    st.session_state["repo_url"] = None
if "repo_author" not in st.session_state:
    st.session_state["repo_author"] = None
if "repo_name" not in st.session_state:
    st.session_state["repo_name"] = None
if "access_key" not in st.session_state:
    st.session_state["access_key"] = None

def process_text():
    user_input = st.session_state["text"]
    clear_text()
    if user_input == 'help':
        st.session_state["messages"].append('#### System:\nOnce you have loaded a repo, you can ask the chatbot questions about the code.\n')
    elif st.session_state['repo_url'] is None:
        st.session_state["messages"].append('#### System:\nPlease load a repo first\n') 
    elif not model.has_key:
        st.session_state["messages"].append('#### System:\nPlease set an access key first\n')
    else:
        response = model.ask(user_input)
        st.session_state["messages"].append('#### CodeLLM:\n' + response + '\n')

def clear_text():
    st.session_state["messages"].append('#### You:\n' + st.session_state["text"] + '\n')
    st.session_state["text"] = ''

def reset_repo_info():
    st.session_state['repo_url'] = None
    st.session_state['repo_author'] = None
    st.session_state['repo_name'] = None

def get_repo(url=None, author=None, name=None):
    if not model.has_key:
        st.session_state["messages"].append('#### System:\nPlease set an access key first\n')
        return
    
    if url is not None:
        try:
            st.session_state['repo_url'] = url
            st.session_state['repo_author'] = url.split('/')[3]
            st.session_state['repo_name'] = url.split('/')[4].split('.')[0]
            st.session_state["messages"].append(f'#### System:\nLoading repo {st.session_state["repo_name"]} from URL {url}\n')
        except:
            st.session_state["messages"].append(f'#### System:\n"{url}" is an invalid url\n')
            reset_repo_info()
            return
    elif author is not None and name is not None:
        st.session_state['repo_author'] = author
        st.session_state['repo_name'] = name
        st.session_state['repo_url'] = f'https://github.com/{author}/{name}.git'
        st.session_state["messages"].append(f'#### System:\nLoading repo {name} from ' + author + '\n')
    else:
        st.session_state["messages"].append('#### System:\nPlease specify either a URL or an Author and Name\n')
        return
    try: 
        model.set_repo(st.session_state['repo_url'])
    except:
        st.session_state["messages"].append(f'#### System:\nFailed to fetch "https://github.com/{author}/{name}.git"\n')
        reset_repo_info()
        return
    st.session_state["messages"].append(f'#### System:\nRepo loaded successfully\n')

with st.sidebar:

    st.write("## Set Access Key")
    if model.has_key:
        with st.expander("Change Access Key"):
            access_key = st.text_input("Access Key", type="password")
            if st.button("Submit"):
                if access_key != '':
                    model.set_api_key(access_key)
                    st.session_state["messages"].append(f'#### System:\nAccess Key Changed Successfully\n')
    else:
        access_key = st.text_input("Access Key", type="password")
        if st.button("Submit"):
            if access_key != '':
                model.set_api_key(access_key)
                st.session_state["messages"].append(f'#### System:\nAccess Key Set Successfully\n')

    st.write("## Set Repository")
    with st.expander("Set Repo by URL"):
        repo_url = st.text_input("Repository URL", value="")
        if st.button("Load", key='load_by_url'):
            get_repo(url=repo_url)
    with st.expander("Set Repo by Author and Name"):
        repo_author = st.text_input("Repository Author", value="")
        repo_name = st.text_input("Repository Name", value="")
        if st.button("Load", key='load_by_author_name'):
            get_repo(author=repo_author, name=repo_name)

    st.write('## Estimated Cost')
    st.write(f'${model.running_cost:.2f} USD')

if st.session_state['repo_url'] is None:
    st.title('Repository Aware Code LLM', help='Select a Repository to Load')
else:
    st.title(f'Repository Aware Code LLM', help=f'Repository: {st.session_state["repo_author"]}/{st.session_state["repo_name"]}')

for item in st.session_state["messages"]:
    st.write(item)

input = st.text_input("Input", key="text", on_change=process_text)



import streamlit as st

from llm import LLM

def create_message(text, author):
    message = f'#### {author}:\n{text}\n'
    st.session_state.messages.append(message)

def process_text():
    user_input = st.session_state.text
    clear_text()
    if user_input == 'help':
        create_message('Enter an OpenAI API key, specify a repo, and then ask the chatbot questions about the code.', 'System')
    elif not st.session_state.llm.has_key:
        create_message('Please set an access key first', 'System')
    elif st.session_state.repo_url is None:
        create_message('Please load a repo first', 'System')
    else:
        response = st.session_state.llm.ask(user_input)
        create_message(response, 'CodeLLM')

def clear_text():
    create_message(st.session_state.text, 'You')
    st.session_state.text = ''

def reset_repo_info():
    st.session_state.repo_url = None
    st.session_state.repo_author = None
    st.session_state.repo_name = None

def get_repo(url=None, author=None, name=None):
    if not st.session_state.llm.has_key:
        create_message('Please set an access key first', 'System')
        return
    if url is not None:
        try:
            st.session_state.repo_url = url
            st.session_state.repo_author = url.split('/')[3]
            st.session_state.repo_name = url.split('/')[4].split('.')[0]
            create_message(f'Loading repo {st.session_state.repo_name} from URL {url}', 'System')
        except:
            create_message(f'"{url}" is an invalid url', 'System')
            reset_repo_info()
            return
    elif author is not None and name is not None:
        st.session_state.repo_author = author
        st.session_state.repo_name = name
        st.session_state.repo_url = f'https://github.com/{author}/{name}.git'
        create_message(f'Loading repo {name} from {author}', 'System')
    else:
        create_message('Please specify either a URL or an Author and Name', 'System')
        return
    try: 
        st.session_state.llm.set_repo(st.session_state.repo_url)
    except:
        create_message(f'Failed to fetch "https://github.com/{author}/{name}.git"', 'System}')
        reset_repo_info()
        return
    create_message('Repo loaded successfully', 'System')

def init_session_state():

    st.set_page_config(layout="wide")
    
    DEFAULTS = {
        'cost': 0,
        'messages': [],
        'repo_url': None,
        'repo_author': None,
        'repo_name': None,
        'access_key': None
    }

    for key, default_value in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

    if 'llm' not in st.session_state:
        st.session_state.llm = LLM(ask_for_confirmation=False)

    st.session_state.llm = st.session_state.llm

def access_key():
    st.write("## Set Access Key")
    if st.session_state.llm.has_key:
        with st.expander("Change Access Key"):
            access_key = st.text_input("Access Key", type="password")
            if st.button("Submit"):
                if access_key != '':
                    st.session_state.llm.set_api_key(access_key)
                    create_message(f'Access Key Changed Successfully', 'System')
    else:
        access_key = st.text_input("Access Key", type="password")
        if st.button("Submit"):
            if access_key != '':
                st.session_state.llm.set_api_key(access_key)
                create_message(f'Access Key Set Successfully', 'System')

def repository_selection():
    if st.session_state.repo_url is None:
        st.write("## Set Repository")
    else:
        st.write("## Repository")
        st.write(f'{st.session_state.repo_author}/{st.session_state.repo_name}')
        st.write("## Change Repository")
    with st.expander("Set Repo by URL"):
        repo_url = st.text_input("Repository URL", value="")
        if st.button("Load", key='load_by_url'):
            get_repo(url=repo_url)
    with st.expander("Set Repo by Author and Name"):
        repo_author = st.text_input("Repository Author", value="")
        repo_name = st.text_input("Repository Name", value="")
        if st.button("Load", key='load_by_author_name'):
            get_repo(author=repo_author, name=repo_name)

def model_selection():
    st.write("## Set LLM Model")
    model_name = st.selectbox("Model", ["GPT-3.5", "GPT-4"], index=0)
    st.session_state.llm.set_main_model(model_name)

    st.write('## Estimated Cost')
    st.write(f'${st.session_state.llm.get_running_cost():.2f} USD')
            
def sidebar():
    with st.sidebar:
        access_key()
        repository_selection()
        model_selection() 

def title():
    if st.session_state.repo_url is None:
        st.title('Repository Aware Code LLM', help='Enter an OpenAI API Key and Select a Repository to Load')
    else:
        st.title(f'Repository Aware Code LLM', help=f'Repository: {st.session_state.repo_author}/{st.session_state.repo_name}')

def chat_interface():
    for item in st.session_state.messages:
        st.write(item)
    st.text_input("Input", key="text", on_change=process_text)

def init_main():
    title()
    chat_interface()

init_session_state()
sidebar()
init_main()

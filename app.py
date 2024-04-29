import streamlit as st

from models.LLM import LLM
from models.Cost import CostTracker
from models.constants import ChatModel


def render():
    init_session_state()
    sidebar()
    title()
    chat_interface()


def init_session_state():
    st.set_page_config(layout="wide")

    DEFAULTS = {
        "messages": [],
        "repo_url": None,
        "repo_author": None,
        "repo_name": None,
        "access_key": None,
        "need_confirmation": False,
    }

    for key, default_value in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

    if "llm" not in st.session_state:
        st.session_state.llm = LLM()

    if "cost" not in st.session_state:
        st.session_state.cost = CostTracker()


def sidebar():
    with st.sidebar:
        access_key()
        repository_selection()
        model_selection()
        cost_display()
        confirmation_ticker()


def access_key():
    st.write("## Set Access Key")
    if st.session_state.llm.has_key:
        with st.expander("Change Access Key"):
            access_key = st.text_input("Access Key", type="password")
            if st.button("Submit"):
                if access_key != "":
                    st.session_state.llm.set_api_key(access_key)
                    create_message("Access Key Changed Successfully")
    else:
        access_key = st.text_input("Access Key", type="password")
        if st.button("Submit"):
            if access_key != "":
                st.session_state.llm.set_api_key(access_key)
                create_message("Access Key Set Successfully")


def create_message(text: str, author: str = "System"):
    message = f"#### {author}:\n{text}\n"
    st.session_state.messages.append(message)


def repository_selection():
    if st.session_state.repo_url is None:
        st.write("## Set Repository")
    else:
        st.write("## Repository")
        st.write(f"{st.session_state.repo_author}/{st.session_state.repo_name}")
        st.write("## Change Repository")
    with st.expander("Set Repo by URL"):
        repo_url = st.text_input("Repository URL", value="")
        if st.button("Load", key="load_by_url"):
            set_repo(url=repo_url)
    with st.expander("Set Repo by Author and Name"):
        repo_author = st.text_input("Repository Author", value="")
        repo_name = st.text_input("Repository Name", value="")
        if st.button("Load", key="load_by_author_name"):
            set_repo(author=repo_author, name=repo_name)


def set_repo(
    url: str | None = None,
    author: str | None = None,
    name: str | None = None,
):
    if not st.session_state.llm.has_key:
        create_message("Please set an access key first")
        return
    if url is not None:
        try:
            st.session_state.repo_url = url
            st.session_state.repo_author = url.split("/")[3]
            st.session_state.repo_name = url.split("/")[4].split(".")[0]
            create_message(f"Loading repo {st.session_state.repo_name} from URL {url}")
        except IndexError:
            create_message(f'"{url}" is an invalid url')
            reset_repo_info()
            return
    elif author is not None and name is not None:
        st.session_state.repo_author = author
        st.session_state.repo_name = name
        st.session_state.repo_url = f"https://github.com/{author}/{name}.git"
        create_message(f"Loading repo {st.session_state.repo_name} from URL {url}")
    else:
        create_message("Please specify either a URL or an Author and Name")
        return
    try:
        st.session_state.llm.set_repo(st.session_state.repo_url)
        st.session_state.cost.update_embedding_cost(st.session_state.llm.get_repo_tokens())
    except Exception as e:
        create_message(f'Failed to fetch and embed "github.com/{author}/{name}.git"')
        create_message(f"Error: {e}")
        reset_repo_info()
        return
    create_message("Repo loaded successfully")


def reset_repo_info():
    st.session_state.repo_url = None
    st.session_state.repo_author = None
    st.session_state.repo_name = None


def model_selection():
    st.write("## Set LLM Model")
    model_name = st.selectbox("Model", ["GPT-3.5", "GPT-4"], index=0)
    model_map = {
        "GPT-3.5": ChatModel.GPT_3_5_TURBO_0125,
        "GPT-4": ChatModel.GPT_4,
    }
    try:
        assert model_name is not None
        st.session_state.llm.set_main_model(model_map[model_name])
        st.session_state.cost.set_main_model(model_map[model_name])
    except KeyError:
        create_message(f"Model {model_name} not found")


def cost_display():
    st.write("## Estimated Cost")
    st.write(f"Chat Cost:       ${st.session_state.cost.get_chat_cost():.2f}")
    st.write(f"Embedding Cost:  ${st.session_state.cost.get_embedding_cost():.2f}")
    st.write(f"Total Cost:      ${st.session_state.cost.get_cost():.2f}")


def confirmation_ticker():
    st.session_state.need_confirmation = st.checkbox(
        "Require Confirmation",
        st.session_state.need_confirmation,
    )


def title():
    if st.session_state.repo_url is None:
        st.title(
            "Repository Aware Code LLM",
            help="Enter an OpenAI API Key and Select a Repository to Load",
        )
    else:
        st.title(
            "Repository Aware Code LLM",
            help=f"Repository: {st.session_state.repo_author}/{st.session_state.repo_name}",
        )


def chat_interface():
    for item in st.session_state.messages:
        st.write(item)
    st.text_input("Input", key="text", on_change=process_text)


def process_text():
    user_input = st.session_state.text
    clear_text()
    if user_input == "help":
        create_message(
            "Enter an OpenAI API key, specify a repo, and then ask the chatbot questions about the code."
        )
    elif not st.session_state.llm.has_key:
        create_message("Please set an access key first")
    elif st.session_state.repo_url is None:
        create_message("Please load a repo first")
    else:
        response, num_inputs, num_outputs = st.session_state.llm.ask(user_input)
        st.session_state.cost.update_chat_cost(
            num_inputs, num_outputs, filter_model=False
        )
        create_message(response, "CodeLLM")


def clear_text():
    create_message(st.session_state.text, "You")
    st.session_state.text = ""


render()

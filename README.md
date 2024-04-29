# CodeLLM

This repository contains an LLM assistant for developers that leverages OpenAI's API and retrieval-augmented-generation techniques to help you navigate and understand an unfamiliar codebase or get advice about your code that depends on repo-wide understanding. It integrates this functionality in an easy-to-use Streamlit app.

You can try it out here: https://code-llm-server-icuu5twmmq-uc.a.run.app/

## What this means for you

- No more copy/pasting code into the ChatGPT interface!
- ChatGPT can consider your entire codebase when responding!

## How the system works
After you add an OpenAI API key and specify a github repo, the program will clone the repo to the 'repos/' directory, parse the python files for class and function definitions and generates embeddings for each of the code blocks. When you ask a question about the codebase, your question will be turned into an embedding, and code blocks whose embeddings are similar to your question's embedding will be sent, along with your question, to an OpenAI LLM.

## Getting Started

- Clone the repository onto your local machine and navigate to its root directory:
```
git clone https://github.com/igm503/CodeLLM.git
cd CodeLLM
```

### Run Directly with Streamlit 

- Install the project dependencies:
```
pip install -r requirements.txt
```
- You're now ready to start up the Streamlit application. Run the following command:
```
streamlit run app.py
```
The app will be available at http://localhost:8502.

### Run with Docker

To build and then run the docker image, run the run_docker.sh script:
```
./scripts/run_docker.sh
```
If you can't run the file, give it executable permissions:
```
chmod +x run_docker.sh
```
In a web browser, navigate to the url http://localhost:8080

## To Do

- Add support for languages other than Python
- Break up LLM filtering step into separate requests to shorter context models to decrease cost
- Increase the number of embeddings retrieved when using LLM filtering step
- Generate dependency tree for functions and provide that info to LLM model
- Include doc files (e.g. Readme.md) in the embeddings
- Use OpenAI’s functions feature to allow the model to request more information
- Allow user to see the estimated cost of generating embeddings for a new repo before they confirm

## Community Contributions
I welcome community contributions and pull requests. Feel free to explore the code and provide any enhancements, bug fixes, or features you think might be helpful.

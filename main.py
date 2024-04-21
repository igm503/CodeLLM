import os

import openai

from models.LLM import LLM


openai.api_key = os.getenv("OPENAI_API_KEY")
llm = LLM()
llm.set_repo("https://github.com/igm503/Deep-Learning-Replications-and-Experiments.git")
response = llm.ask(
    "What sort of data is being generated in data_gen.py? What would it be for?"
)
print(response)

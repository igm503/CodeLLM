NUM_TOKENS = 1_000_000


class ChatModel:
    GPT_4 = "gpt-4"
    GPT_4_32K = "gpt-4-32k"
    GPT_3_5_TURBO_0125 = "gpt-3.5-turbo-0125"
    GPT_3_5_TURBO_INSTRUCT = "gpt-3.5-turbo-instruct"


class EmbeddingModel:
    EMBEDDING_ADA = "text-embedding-ada-002"
    EMBEDDING_3_SMALL = "text-embedding-3-small"
    EMBEDDING_3_LARGE = "text-embedding-3-large"


MODEL_INPUT_COST = {
    EmbeddingModel.EMBEDDING_ADA: 0.1 / NUM_TOKENS,
    EmbeddingModel.EMBEDDING_3_SMALL: 0.02 / NUM_TOKENS,
    EmbeddingModel.EMBEDDING_3_LARGE: 0.13 / NUM_TOKENS,
    ChatModel.GPT_4: 30.0 / NUM_TOKENS,
    ChatModel.GPT_4_32K: 60.0 / NUM_TOKENS,
    ChatModel.GPT_3_5_TURBO_0125: 0.5 / NUM_TOKENS,
    ChatModel.GPT_3_5_TURBO_INSTRUCT: 1.5 / NUM_TOKENS,
}

MODEL_OUTPUT_COST = {
    ChatModel.GPT_4: 60.0 / NUM_TOKENS,
    ChatModel.GPT_4_32K: 120.0 / NUM_TOKENS,
    ChatModel.GPT_3_5_TURBO_0125: 1.5 / NUM_TOKENS,
    ChatModel.GPT_3_5_TURBO_INSTRUCT: 2.0 / NUM_TOKENS,
}

LANGUAGES = {
    "java": "java",
    "py": "python",
    "c": "c",
    "h": "c",
}

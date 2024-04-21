from .constants import MODEL_INPUT_COST, MODEL_OUTPUT_COST, ChatModel, EmbeddingModel


class CostTracker:
    def __init__(self):
        self.chat_cost = 0
        self.embedding_cost = 0

        self.main_model = ChatModel.GPT_3_5_TURBO_0125
        self.filter_model = ChatModel.GPT_3_5_TURBO_0125
        self.embedding_model = EmbeddingModel.EMBEDDING_3_SMALL

    def set_main_model(self, model: str):
        self.main_model = model

    def estimate_chat_cost(self, num_input: int, filter_model: bool = False):
        input_cost = num_input * MODEL_INPUT_COST[self.main_model]
        if filter_model:
            input_cost += num_input * MODEL_INPUT_COST[self.filter_model]
        output_cost = 1000 * MODEL_OUTPUT_COST[self.main_model]  # rough estimate
        return input_cost + output_cost

    def update_chat_cost(
        self,
        num_input: int,
        num_output: int = 0,
        filter_model: bool = False,
    ):
        input_cost = num_input * MODEL_INPUT_COST[self.main_model]
        if filter_model:
            input_cost += num_input * MODEL_INPUT_COST[self.filter_model]
        output_cost = num_output * MODEL_OUTPUT_COST[self.main_model]
        self.chat_cost += input_cost + output_cost

    def estimate_embedding_cost(self, num_tokens: int):
        return num_tokens * MODEL_INPUT_COST[self.embedding_model]

    def update_embedding_cost(self, num_tokens: int):
        self.embedding_cost += num_tokens * MODEL_INPUT_COST[self.embedding_model]

    def get_chat_cost(self):
        return self.chat_cost

    def get_embedding_cost(self):
        return self.embedding_cost

    def get_cost(self):
        return self.chat_cost + self.embedding_cost

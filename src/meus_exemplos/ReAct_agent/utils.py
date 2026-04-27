from langchain.chat_models import init_chat_model, BaseChatModel

def load_llm() -> BaseChatModel:
    llm = init_chat_model("ollama:gpt-oss:latest")
    return llm

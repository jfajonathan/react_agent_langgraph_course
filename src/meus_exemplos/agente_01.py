from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.graph.state import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from rich import print
from rich.pretty import pprint
from rich.markdown import Markdown
import threading

llm = init_chat_model("ollama:llama3.2:3b")

# Definindo o estado do agente
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# definindo os nodes
def call_llm(state: AgentState) -> AgentState:
    # print(state)
    llm_result = llm.invoke(state["messages"])
    # llm_result = AIMessage("Olá, humano! Eu sou um agente simples.")
    return {'messages': [llm_result]}


# Criando o grafo de estados
builder = StateGraph(
    AgentState, context_schema=None, input_schema=AgentState, output_schema=AgentState
)

builder.add_node("call_llm", call_llm)
builder.add_edge(START, "call_llm")
builder.add_edge("call_llm", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)
config = RunnableConfig(configurable={"thread_id": threading.get_ident()})

if __name__ == "__main__":
    # Executando o grafo
    # human_message = HumanMessage(content="Olá, agente!")
    # result = graph.invoke({"messages":[human_message]})
    # # pprint(result, expand_all=True)
    # pprint(result['messages'][-1].content)
    # print(Markdown("---"))
    current_menssages: Sequence[BaseMessage] = []

    while True:
        user_input = input("Digite sua mensagem: ")
        print(Markdown("---"))

        if user_input.lower() in ["q", "quit","sair","exit"]:
            print("Bye 👋")
            print(Markdown("---"))
            break

        human_message = HumanMessage(user_input)
        current_menssages = [*current_menssages, human_message]

        result = graph.invoke({
            "messages": current_menssages},
            config=config
            )
        current_menssages = result["messages"]

        print((Markdown(result['messages'][-1].content)))
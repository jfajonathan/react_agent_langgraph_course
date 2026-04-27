from langchain_core.messages import HumanMessage
from langgraph.graph.state import RunnableConfig
from rich import print
from rich.markdown import Markdown

from meus_exemplos.ReAct_agent.graph import build_graph

def main() -> None:
    config = RunnableConfig(configurable={'thread_id': 1})
    graph = build_graph()
    
    user_input = "Pode me dizer quanto é 2.75 multiplicado por 3.05? pegue o resultado e multiplique por 2, e me diga o resultado final?"
    human_message = HumanMessage(user_input)
    current_messages = [human_message]
    result = graph.invoke({'messages': current_messages}, config=config)
    print(Markdown(result['messages'][-1].content))

if __name__ == '__main__':
    main()
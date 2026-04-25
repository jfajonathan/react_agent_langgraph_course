from typing import Literal

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import START
from langgraph.graph.state import CompiledStateGraph, StateGraph
from pydantic import ValidationError


from meus_exemplos.ReAct_agent.state import State
from meus_exemplos.ReAct_agent.tools import TOOLS, TOOLS_BY_NAME
from meus_exemplos.ReAct_agent.utils import load_llm

SYSTEM_MESSAGE = SystemMessage(
    "You are a helpful assistant. Only use tools when the user explicitly asks "
    "for an operation that requires them. Do not call tools on your own initiative."
)

def call_llm(state: State) -> State:
    print("> call llm")
    llm = load_llm().bind_tools(TOOLS)
    result = llm.invoke([SYSTEM_MESSAGE] + list(state["messages"]))
    return {"messages": [result]}

def tool_node(state: State) -> State:
    print("> tool node")
    llm_response = state['messages'][-1]

    if not isinstance(llm_response, AIMessage) or not getattr(
        llm_response, "tool_calls",None
        ):
        return state
    
    call = llm_response.tool_calls[-1]
    name, args, id_ = call["name"], call["args"], call["id"]

    try: 
        content = TOOLS_BY_NAME[name].invoke(args)
        status = 'success'
    except (KeyError, IndexError, TypeError, ValueError,ValidationError) as error:
        content = f"Please, fix your mistakes: {error}"
        status = 'error'
    
    tool_message = ToolMessage(content=content, tool_call_id=id_, status = status)

    return {"messages": [tool_message]}

def router(state: State) -> Literal['tool_node', '__end__']:
    print("> router")
    llm_response = state['messages'][-1]
    if getattr(llm_response, "tool_calls", None):
        return "tool_node"
    return '__end__'

def build_graph() -> CompiledStateGraph[State, None, State, State]:
    builder = StateGraph(State)
    
    builder.add_node("call_llm", call_llm)
    builder.add_node("tool_node", tool_node)

    builder.add_edge(START, "call_llm")
    builder.add_conditional_edges("call_llm", router, ["tool_node", "__end__"])
    builder.add_edge("tool_node", "call_llm")    

    return builder.compile(checkpointer=InMemorySaver())


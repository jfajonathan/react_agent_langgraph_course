from langchain.chat_models import init_chat_model
from langchain.tools import BaseTool, tool
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from pydantic import ValidationError
from rich import print
from rich.pretty import pprint
from rich.markdown import Markdown

@tool
def multiply(a: float, b: float) -> float:
    """
    Multiplies a * b and returns the result.

    Args:
        a (float): The first number to multiply (float multiplicand).
        b (float): The second number to multiply (float multiplier).
    
    Return:
        The resulting float of the equation a * b

    """
    return a * b

llm = init_chat_model("ollama:llama3.2:3b")
messages: list[BaseMessage] = []
system_message = SystemMessage(
    "You are a helpful assistant. You have access to tools. When the user asks for something, first look and always use if you have a tool that solves that problem")
messages.append(system_message)
human_message = HumanMessage("What is the result of multiplying 3.5 by 2.3?")
messages.append(human_message)

tools: list[BaseTool] = [multiply]
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools=tools)
llm_response = llm_with_tools.invoke([system_message, human_message])
if isinstance(llm_response, AIMessage) and getattr(llm_response, "tool_calls",None):
    call = llm_response.tool_calls[-1]
    name, args, id_ = call["name"], call["args"], call["id"]

    try: 
        content = str(tools_by_name[name].invoke(args))
        status = 'success'
    except (KeyError, IndexError, TypeError, ValueError,ValidationError) as error:
        content = f"Please, fix your mistakes: {error}"
        status = 'error'
    
    tool_message = ToolMessage(content=content, tool_call_id=id_, status = status)
    # print(tool_message)
    messages.append(tool_message)

    llm_response = llm_with_tools.invoke([system_message, human_message, tool_message])
    messages.append(llm_response)
    pprint(messages)
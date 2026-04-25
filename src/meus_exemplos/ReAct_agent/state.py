# Definindo o estado do agente
from collections.abc import Sequence
from typing import TypedDict, Annotated
from langgraph.graph.message import BaseMessage, add_messages

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List

class State(TypedDict):
    messages: List[dict]

def simple_response(state: State) -> State:
    return {"messages": state["messages"] + [{"role": "assistant", "content": "Hello!"}]}

checkpointer = InMemorySaver()

builder = StateGraph(State)
builder.add_node("respond", simple_response)
builder.add_edge(START, "respond")
builder.add_edge("respond", END)
graph = builder.compile(checkpointer=checkpointer)

result = graph.invoke(
    {"messages": [{"role": "user", "content": "hi! i am Bob"}]},
    {"configurable": {"thread_id": "1"}},
)
print(result["messages"])
import getpass
import os

from langchain.tools import tool
from langchain.chat_models import init_chat_model

# oaiKey="xxxxx"

def get_openai_api_key():
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

get_openai_api_key()
# global variable
model = init_chat_model(
    "gpt-5-nano",
    model_provider="openai",
    temperature=0.7,
)



# define a tool
@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Add `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """Subtract `b` from `a`.

    Args:
        a: First int
        b: Second int
    """
    return a - b

@tool
def divide(a: int, b: int) -> int:
    """Divide `a` by `b`.

    Args:
        a: First int
        b: Second int
    """
    return a / b

# Augument the llm with tools

tools = [add, multiply, subtract, divide]
tools_by_name = {tool.name: tool for tool in tools}

# print the tools_by_name for debugging
print("tools_by_name: ",tools_by_name)


print(tools_by_name)

model_with_tools = model.bind_tools(tools)

from langchain.messages import AnyMessage
from typing_extensions import Annotated,TypedDict
import operator

class MessagesState(TypedDict):
    """State of the messages."""
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


from langchain.messages import SystemMessage

def llm_call(state: dict):
    """LLM decides whether to call a toll or not"""

    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."),
                ]
                + state["messages"]
                )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }

from langchain.messages import ToolMessage

def tool_node(state: dict):
    """Perform the tool call"""
    
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(
            ToolMessage(
                content=observation,
                tool_call_id=tool_call["id"],
            )
        )
    return {
        "messages": result,
    }


from typing import Literal
from langgraph.graph import StateGraph, START, END

def should_continue(state: dict) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    # if the LLM makes a tool call or not, if it does, we should continue the loop and proform the tool call
    if state["messages"][-1].tool_calls:
        return "tool_node"
    return END



# define the graph
cowoker_builder = StateGraph(MessagesState)


# add the nodes to the graph
cowoker_builder.add_node("tool_node", tool_node)
cowoker_builder.add_node("llm_call", llm_call)


# add edges to connect ndoes
cowoker_builder.add_edge(START, "llm_call")
cowoker_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END],
)
cowoker_builder.add_edge("tool_node", "llm_call")

# Compile the agent
agent = cowoker_builder.compile()

# Show the agent
from IPython.display import Image
Image(agent.get_graph().draw_mermaid_png())

# invoke
from langchain.messages import HumanMessage
messages = [HumanMessage(content="Add 2 and 3")]
messages = agent.invoke({"messages": messages})

# print messages for debugging
print("messages: ",messages)

for message in messages["messages"]:
    message.pretty_print()
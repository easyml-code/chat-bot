from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from pydantic import BaseModel
from typing import List
from langchain_core.messages import BaseMessage
from dotenv import load_dotenv
import os

# Load .env variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
                model="llama3-70b-8192",
                api_key=groq_api_key,
                temperature=0.5,
                max_tokens=1024
)

class ChatState(BaseModel):
    input: str
    response: str
    messages: List[BaseMessage]

def chat(state: ChatState) -> ChatState:
    state.messages.append(HumanMessage(content=state.input))
    result = llm.invoke(state.messages)
    state.response = result.content
    state.messages.append(AIMessage(content=state.response))
    return state

builder = StateGraph(ChatState)
builder.add_node("chat", chat)
builder.set_entry_point("chat")

builder.add_edge("chat", END)
graph = builder.compile()
print(graph)

def main():
    messages=[
        SystemMessage(content="you are an intelligent chat-bot")
    ]
    while True:
        query = input("User: ")
        if query=="quit":
            break
        state = ChatState(input=query, response="", messages=messages)
        result = graph.invoke(state)
        # print(result)
        print("AI: ", result['response'])
        messages=result['messages']

if __name__ == "__main__":
    main()

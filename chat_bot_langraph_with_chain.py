from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langgraph.graph import StateGraph, START, END
from langchain.chains import LLMChain
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

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="you are a helpful chat-bot"),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessage(content="{input}")
])

memory = ConversationBufferMemory(
    return_messages=True,
    memory_key="chat_history"
)

chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory
)

class ChatState(BaseModel):
    input: str
    response: str
    # messages: List[BaseMessage]

def chat(state: ChatState) -> ChatState:
    print("state: ", state)
    result = chain.invoke({"input": state.input})
    print("\n\n\n")
    print(result)
    print("\n\n\n")
    state.response = result['text']
    return state

builder = StateGraph(ChatState)
builder.add_node("chat", chat)
builder.set_entry_point("chat")

builder.add_edge("chat", END)
graph = builder.compile()
print(graph)

def main():
    # messages=[
    #     SystemMessage(content="you are an intelligent chat-bot")
    # ]
    while True:
        query = input("User: ")
        if query=="quit":
            break
        state = ChatState(input=query, response="")
        result = graph.invoke(state)
        print(result)
        print("AI: ", result['response'])
        # messages=result['messages']

if __name__ == "__main__":
    main()

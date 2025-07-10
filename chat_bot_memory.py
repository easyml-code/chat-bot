from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import pydantic
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
    # ("system", "you are a helpful chat-bot"),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessage(content="{input}"),
    # ("human", "{input}")
])

# memory = ConversationBufferMemory(
#     return_messages=True,
#     memory_key="chat_history"
# )

# chain = LLMChain(
#     llm=llm,
#     prompt=prompt,
#     memory=memory
# )

# Chain through runnable 
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

# LCEL style pipeline
runnable = prompt | llm

# Adding session id to the chat
import uuid
user_id = "vikram"
chat_id = str(uuid.uuid4())
session_id = f"{user_id}_{chat_id}"

print("Session id: ", session_id)


memory_store = {}

memory_factory = lambda session_id: memory_store.setdefault(
    session_id, InMemoryChatMessageHistory()
)

# Wrap with memory
chain_with_memory = RunnableWithMessageHistory(
    runnable,
    memory_factory,
    input_messages_key="input",
    history_messages_key="chat_history"
)


def main():
    while True:
        query = input("User: ")
        if query=="quit":
            print("Session History: ", memory_factory)
            print("Session History Store: ", memory_store)
            break
        # result=chain.invoke(input=query)
        result=chain_with_memory.invoke(
            {"input": query},
            config={"configurable": {"session_id": session_id}}
        )
        print("AI: ", result.content, "\n")
        # print("\n\n",result, "\n\n")

if __name__ == "__main__":
    main()

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel
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

messages=[
    SystemMessage(content="you are an intelligent chat-bot")
]

def main():
    while True:
        query = input("User: ")
        if query=="quit":
            print("\n\nHistory\n", messages)
            break

        messages.append(HumanMessage(content=query))
        result=llm.invoke(messages)
        print("AI: ", result.content)
        messages.append(AIMessage(content=result.content))

if __name__ == "__main__":
    main()

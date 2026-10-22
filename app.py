from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_agent
from langchain_core.tools import Tool

search = GoogleSerperAPIWrapper()
llm = ChatGroq(model="openai/gpt-oss-20b")

tools = [
    Tool(
        name="google_search",
        func=search.run,
        description="Search Google for current information.",
    )
]

system_prompt = """You are a helpful assistant that can answer questions using Google Search.
Keep every response under 100 words. Use this format:
Answer: one or two direct sentences.
Key points: up to three short bullets, only when useful.
Sources: list the URLs returned by search, only when available.
Do not include lengthy background or repeat the question. Never invent sources."""

agent = create_agent(model=llm, tools=tools, system_prompt=system_prompt)

while True:
   query = input("USER: ")
   if query.lower() == "quit":
       break

   response = agent.invoke({"messages": [{"role": "user", "content": query}]})
   answer = response["messages"][-1].content
   print(f"AI: {answer}")

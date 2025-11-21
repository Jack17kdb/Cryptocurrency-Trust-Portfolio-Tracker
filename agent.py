import os
import requests
from dotenv import load_dotenv
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()
API_KEY = os.getenv("CRYPTONEWS_API_KEY")

@dataclass
class Context:
	user_id: str

@dataclass
class ResponseFormat:
	symbol: str
	price: float
	companyName: str
	country: str
	currency: str
	website: str

@tool("get_user", description="Get coin from the user_id", return_direct=False)
def get_user(runtime: ToolRuntime[Context]) -> str:
	userid = runtime.context.user_id
	return {"jack": "AAPL", "jace": "TSLA", "jake": "MSFT"}.get(userid, "GOOGL")

@tool("get_stock", description="Fetches financial and operational information for a specific stock symbol, including the company's market capitalization, stock price, industry, and much more.", return_direct=False)
def get_stock(symbol: str) -> dict:
	url = f"https://financialmodelingprep.com/stable/profile?symbol={symbol}&apikey={API_KEY}"
	data = requests.get(url).json()
	return data[0] if data else {}

model = init_chat_model("gpt-4.1-mini", temperature=0.1)
checkpointer = InMemorySaver()

agent = create_agent(
	model=model,
	tools=[get_user, get_stock],
	system_prompt=  "You are a stock sentiment portfolio advisor"
			"Get the responses from the get_stock tool"
			"Return the response exactly as stated",
	context_schema=Context,
	response_format = ResponseFormat,
	checkpointer=checkpointer
)

config = {"configurable": {"thread_id": 1}}

response = agent.invoke(
	{"messages": [{"role": "user", "content": "What is my stock?"}]},
	context=Context(user_id="jake"),
	config=config
)

print(response['structured_response'])

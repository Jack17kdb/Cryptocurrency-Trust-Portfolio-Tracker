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
	price_in_company: float
	companyName: str
	country: str
	currency: str
	website: str
	crypto_coin: str
	coin_price_in_user_currency: float

users = {
	"jack": {"stock": "AAPL", "coin": "bitcoin", "currency": "USD"},
	"jace": {"stock": "TSLA", "coin": "ethereum", "currency": "EUR"},
	"jake": {"stock": "MSFT", "coin": "solana", "currency": "GBP"}
}

@tool("get_user_stock", description="Get stock from the user_id", return_direct=False)
def get_user_stock(user_id: str) -> str:
	user_data = users.get(user_id.lower())
	if user_data:
		return user_data["stock"]
	else:
		return "GOOGL"

@tool("get_user_coin", description="Get coin from the user_id", return_direct=False)
def get_user_coin(user_id: str) -> str:
	user_data = users.get(user_id.lower())
	if user_data:
		return user_data["coin"]
	else:
		return "tether"

@tool("get_user_currency", description="Get the user's currency from the user_id", return_direct=False)
def get_user_currency(user_id: str) -> str:
	user_data = users.get(user_id.lower())
	if user_data:
		return user_data["currency"]
	else:
		return "JPY"

@tool("get_stock_data", description="Fetches financial and operational information for a specific stock symbol, including the company's market capitalization, stock price, industry, and much more.", return_direct=False)
def get_stock_data(symbol: str) -> dict:
	url = f"https://financialmodelingprep.com/stable/profile?symbol={symbol}&apikey={API_KEY}"
	data = requests.get(url).json()
	return data[0] if data else {}

@tool("get_coin_data", description="Fetches coin value in the user's currency", return_direct=False)
def get_coin_data(coin: str, currency: str) -> dict:
	url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin}&vs_currencies={currency}&include_market_cap=true&include_24hr_vol=true&include_24hr_change=true"
	data = requests.get(url).json()
	return data

model = init_chat_model("gpt-4.1-mini", temperature=0.1)
checkpointer = InMemorySaver()

agent = create_agent(
	model=model,
	tools=[get_user_stock, get_user_coin, get_user_currency, get_stock_data, get_coin_data],
	system_prompt=  "You are a stock sentiment portfolio advisor"
			"Use the get_user_stock, get_user_coin, get_user_currency to get user's information"
			"Get the responses from the get_stock_data and get_coin_data tools"
			"Return the response exactly as stated",
	context_schema=Context,
	response_format = ResponseFormat,
	checkpointer=checkpointer
)

config = {"configurable": {"thread_id": 1}}

response1 = agent.invoke(
	{"messages": [{"role": "user", "content": "What are my stock and cryptocurrency investments?"}]},
	context=Context(user_id="jake"),
	config=config
)

print("FIRST RESPONSE\n")
print(response1['structured_response'])
print("\n" + "=" * 50 + "\n")

response2 = agent.invoke(
    {"messages": [{"role": "user", "content": "Which one performed better today?"}]},
    context=Context(user_id="jake"),
    config=config
)

print("SECOND RESPONSE\n")
print(response2['structured_response'])

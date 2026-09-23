import asyncio
import sys

# Windows defaults to ProactorEventLoop, which psycopg's async driver can't
# use. Force SelectorEventLoop on Windows before any asyncio code runs.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import json
import os
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from State import TravelState

from dotenv import load_dotenv
load_dotenv()  # loads .env file into os.environ, if present

# ---- Config ----
POSTGRES_URI = "postgresql://postgres:postgres@localhost:5432/travelmate"

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")

MCP_SERVERS = {
    # Pre-built, published server — spawned locally via uvx over stdio
    # (still "local MCP" per the diagram's terminology, just not
    # hand-written by us).
    "flight": {
        "command": "uvx",
        "args": ["aviationstack-mcp"],
        "transport": "stdio",
        "env": {"AVIATION_STACK_API_KEY": os.environ.get("AVIATION_STACK_API_KEY", "")},
    },
    "weather": {"url": "http://127.0.0.1:8000/mcp", "transport": "streamable_http"},
    # Tavily's officially hosted remote MCP server — genuinely remote,
    # nothing spawned locally at all.
    "tavily": {
        "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={TAVILY_API_KEY}",
        "transport": "streamable_http",
    },
}

llm = ChatGroq(model="openai/gpt-oss-safeguard-20b")


def _find_tool(tools, name):
    for t in tools:
        if t.name == name:
            return t
    raise ValueError(f"Tool '{name}' not found among discovered MCP tools: {[t.name for t in tools]}")


async def build_graph():
    mcp_client = MultiServerMCPClient(MCP_SERVERS)
    tools = await mcp_client.get_tools()
    print("Discovered tools:", [t.name for t in tools])

    # NOTE: this published server's tools differ from our own custom one —
    # it doesn't have a plain "search_flights(origin, destination)" tool.
    # Leaving tool lookup out here deliberately; wire flight_agent after
    # you've seen the real 'Discovered tools' output below, since the
    # published package's exact tool names/schemas can change between
    # versions and I don't want to guess wrong.
    get_weather_forecast = _find_tool(tools, "get_weather_forecast")
    tavily_search = _find_tool(tools, "tavily_search")
    future_flights = _find_tool(tools, "future_flights_arrival_departure_schedule")

    # ---- Agent nodes: each calls its own MCP tool directly (fixed pipeline,
    # matching the diagram, rather than an LLM deciding which tool to call) ----

    async def flight_agent(state: TravelState) -> dict:
        # This tool's args are flat (no nested "input" wrapper), unlike our
        # own servers. schedule_type="arrival" shows flights landing at the
        # destination airport around the departure date.
        result = await future_flights.ainvoke({
            "airport_iata_code": state["destination_airport"],
            "schedule_type": "arrival",
            "date": state["departure_date"],
            "number_of_flights": 5,
        })
        parsed = json.loads(result) if isinstance(result, str) else result
        return {"flight_results": parsed}

    async def hotel_agent(state: TravelState) -> dict:
        # Tavily's hosted tool takes a plain "query" string, unlike our own
        # servers which wrap a validated Pydantic "input" object.
        result = await tavily_search.ainvoke({
            "query": f"best hotels to stay in {state['destination']}"
        })
        parsed = json.loads(result) if isinstance(result, str) else result
        return {"hotel_results": parsed}

    async def weather_agent(state: TravelState) -> dict:
        result = await get_weather_forecast.ainvoke({"input": {"city": state["destination"]}})
        parsed = json.loads(result) if isinstance(result, str) else result
        return {"weather_results": parsed}

    async def itinerary_agent(state: TravelState) -> dict:
        prompt = f"""You are a travel itinerary planner. Given this data, write a
day-by-day itinerary for a trip to {state['destination']} from
{state.get('departure_date')} to {state.get('return_date')}.

Flight info: {json.dumps(state.get('flight_results'))}
Hotel options: {json.dumps(state.get('hotel_results'))}
Weather forecast: {json.dumps(state.get('weather_results'))}

Keep it concise: a short flight/hotel summary, then a day-by-day plan that
accounts for the weather (suggest indoor activities on rainy days)."""
        response = await llm.ainvoke(prompt)
        return {"itinerary": response.content}

    async def final_agent(state: TravelState) -> dict:
        prompt = f"""Rewrite this itinerary as a friendly, well-formatted final
response for the traveler. Keep all factual details, just make it read
naturally:

{state['itinerary']}"""
        response = await llm.ainvoke(prompt)
        return {"final_response": response.content}

    # ---- Build the fixed pipeline: Flight -> Hotel -> Weather -> Itinerary -> Final ----
    graph = StateGraph(TravelState)
    graph.add_node("flight_agent", flight_agent)
    graph.add_node("hotel_agent", hotel_agent)
    graph.add_node("weather_agent", weather_agent)
    graph.add_node("itinerary_agent", itinerary_agent)
    graph.add_node("final_agent", final_agent)

    graph.set_entry_point("flight_agent")
    graph.add_edge("flight_agent", "hotel_agent")
    graph.add_edge("hotel_agent", "weather_agent")
    graph.add_edge("weather_agent", "itinerary_agent")
    graph.add_edge("itinerary_agent", "final_agent")
    graph.add_edge("final_agent", END)

    return graph  # uncompiled — caller wraps it with the checkpointer


async def main():
    graph = await build_graph()

    initial_state: TravelState = {
        "origin": "BLR",
        "destination": "Goa",
        "destination_airport": "GOI",
        "departure_date": "2026-11-10",
        "return_date": "2026-11-13",
        "travelers": 2,
    }
    config = {"configurable": {"thread_id": "trip-001"}}  # ties this run to Postgres storage

    # ---- Postgres checkpointer: this is the 'PostgreSQL long-term memory'
    # box in the diagram. Using it as `async with` keeps the connection
    # alive for exactly the duration of compile+run, and closes it cleanly
    # afterward — the earlier manual __aenter__-without-__aexit__ approach
    # was what caused the connection to be torn down mid-run. ----
    async with AsyncPostgresSaver.from_conn_string(POSTGRES_URI) as checkpointer:
        await checkpointer.setup()  # creates the checkpoint tables if missing
        app = graph.compile(checkpointer=checkpointer)
        result = await app.ainvoke(initial_state, config=config)

    print("\n=== FINAL RESPONSE ===\n")
    print(result["final_response"])


if __name__ == "__main__":
    asyncio.run(main())
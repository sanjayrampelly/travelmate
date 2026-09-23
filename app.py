import asyncio
import sys
import uuid

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from TravelMategraph import build_graph, POSTGRES_URI

# ---- App state populated once at startup, reused for every request ----
app_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Connect to all MCP servers (flight via uvx/stdio, weather over HTTP,
    # tavily remote) and open the Postgres checkpointer ONCE here, instead
    # of per-request — reconnecting uvx/HTTP/remote servers on every call
    # would be slow and wasteful.
    graph = await build_graph()
    checkpointer_cm = AsyncPostgresSaver.from_conn_string(POSTGRES_URI)
    checkpointer = await checkpointer_cm.__aenter__()
    await checkpointer.setup()

    app_state["compiled_graph"] = graph.compile(checkpointer=checkpointer)
    app_state["checkpointer_cm"] = checkpointer_cm

    yield  # server runs here

    # Clean shutdown — closes the Postgres connection when the app stops
    await checkpointer_cm.__aexit__(None, None, None)


app = FastAPI(title="TravelMate API", lifespan=lifespan)

# Allow the browser-served HTML page to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- Request/response models ----

class TripRequest(BaseModel):
    origin: str = Field(description="Origin airport IATA code, e.g. 'BLR'", min_length=3, max_length=3)
    destination: str = Field(description="Destination city name, e.g. 'Goa'", min_length=1)
    destination_airport: str = Field(description="Destination airport IATA code, e.g. 'GOI'", min_length=3, max_length=3)
    departure_date: str = Field(description="YYYY-MM-DD")
    return_date: str = Field(description="YYYY-MM-DD")
    travelers: int = Field(default=1, ge=1, le=20)


class TripResponse(BaseModel):
    success: bool
    thread_id: str
    final_response: str | None = None
    error: str | None = None


# ---- Endpoints ----

@app.post("/plan-trip", response_model=TripResponse)
async def plan_trip(request: TripRequest) -> TripResponse:
    """Run the full TravelMate pipeline (flight -> hotel -> weather ->
    itinerary -> final response) for a single trip request."""
    thread_id = str(uuid.uuid4())
    try:
        result = await app_state["compiled_graph"].ainvoke(
            request.model_dump(),
            config={"configurable": {"thread_id": thread_id}},
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Trip planning failed: {exc}")

    return TripResponse(success=True, thread_id=thread_id, final_response=result.get("final_response"))


@app.get("/")
async def serve_ui():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn
    # uvicorn.run() creates its own event loop internally and does not
    # respect the WindowsSelectorEventLoopPolicy set above. Running the
    # Server directly inside our own asyncio.run() ensures our policy
    # (and therefore psycopg's async compatibility) actually takes effect.
    config = uvicorn.Config(app, host="127.0.0.1", port=8080)
    server = uvicorn.Server(config)
    asyncio.run(server.serve())
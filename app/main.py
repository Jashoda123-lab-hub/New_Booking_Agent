"""FastAPI service.

POST /message     {"sender": "...", "text": "..."}  ->  reply + action taken
GET  /bookings    all current (active) bookings with property and time
GET  /properties  properties with their free slots (for the web page)
GET  /            a small web page to chat with the agent and watch bookings
"""
import os
from datetime import timedelta
from pathlib import Path
from typing import List, Literal, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import sessionmaker

from .agent import ViewingAgent, make_llm
from .db import init_db, make_engine
from .tools import BookingTools

load_dotenv()

STATIC_DIR = Path(__file__).parent / "static"


def make_checkpointer():
    """LangGraph checkpointer that stores each sender's conversation in a SQLite file."""
    import sqlite3

    from langgraph.checkpoint.sqlite import SqliteSaver

    path = os.getenv("CONVERSATIONS_DB", "conversations.db")
    saver = SqliteSaver(sqlite3.connect(path, check_same_thread=False))
    saver.setup()
    return saver


class MessageIn(BaseModel):
    sender: str = Field(..., min_length=1, description="Who sent it (phone, email or name).",
                        examples=["alice@example.com"])
    text: str = Field(..., min_length=1, examples=["Hi, can I see the flat at 14 Oak Road on Saturday morning?"])


class BookingOut(BaseModel):
    booking_id: int
    property_id: int
    address: str
    starts_at: str
    ends_at: str
    contact: Optional[str] = None


class MessageOut(BaseModel):
    reply: str
    action: Literal["booked", "cancelled", "clarify", "no_availability", "answered", "none"]
    booking: Optional[BookingOut] = None


class BookingListItem(BookingOut):
    status: str
    created_at: Optional[str] = None


def create_app(agent: Optional[ViewingAgent] = None, database_url: Optional[str] = None) -> FastAPI:
    """Build the app. Tests pass their own agent (fake LLM, temp DB)."""
    if agent is None:
        engine = make_engine(database_url)
        init_db(engine)
        agent = ViewingAgent(BookingTools(sessionmaker(bind=engine)), make_llm(),
                             checkpointer=make_checkpointer())

    app = FastAPI(title="Viewing Booking Agent", version="1.0")
    app.state.agent = agent

    def get_agent(request: Request) -> ViewingAgent:
        return request.app.state.agent

    @app.post("/message", response_model=MessageOut)
    def post_message(msg: MessageIn, agent: ViewingAgent = Depends(get_agent)):
        return agent.handle(msg.sender.strip(), msg.text.strip())

    @app.get("/bookings", response_model=List[BookingListItem])
    def get_bookings(agent: ViewingAgent = Depends(get_agent)):
        return agent.tools.list_bookings()

    @app.get("/properties")
    def get_properties(days: int = 7, agent: ViewingAgent = Depends(get_agent)):
        """Each property with its free slots over the next `days` days (used by the web page)."""
        now = agent.clock()
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        days = max(1, min(days, 31))
        return [
            {**p, "free_slots": agent.tools.get_free_slots(p["id"], start, start + timedelta(days=days + 1), now)}
            for p in agent.tools.list_properties()
        ]

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    return app


def get_app() -> FastAPI:
    """Entry point for `uvicorn app.main:get_app --factory` (reads .env, opens viewings.db)."""
    return create_app()

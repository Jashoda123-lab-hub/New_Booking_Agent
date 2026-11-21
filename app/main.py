"""FastAPI service.

POST /message   {"sender": "...", "text": "..."}  ->  reply + action taken
GET  /bookings  all current (active) bookings with property and time
"""
from typing import List, Literal, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import sessionmaker

from .agent import ViewingAgent, make_llm
from .db import init_db, make_engine
from .tools import BookingTools

load_dotenv()


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
        agent = ViewingAgent(BookingTools(sessionmaker(bind=engine)), make_llm())

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

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


def get_app() -> FastAPI:
    """Entry point for `uvicorn app.main:get_app --factory` (reads .env, opens viewings.db)."""
    return create_app()

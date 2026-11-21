"""The viewing-booking agent, built as a LangGraph state graph.

    understand ──► lookup ──► book ─────┐
        │            │                  ├──► reply ──► END
        │            └──► (answer / no slot / clarify)
        └──► cancel ────────────────────┘

* The LLM is used for two things only: reading the message into structured
  fields (`understand`) and wording the final reply (`reply`).
* Every decision that touches data (which property, which slot, whether to
  book or cancel) is made by graph code calling the tools. The LLM cannot
  book a slot that the database did not offer.
* Conversation memory (open request, recent messages, last booking) is graph
  state, saved per sender by a LangGraph checkpointer (SQLite in the API).
* If the property is unknown or no time is given, the graph routes to a
  clarifying question instead of guessing.
"""
import os
import re
from datetime import datetime, timedelta
from typing import Dict, List, Literal, Optional, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from .tools import BookingTools, mentions

Intent = Literal["book", "cancel", "question", "other"]


class Extraction(BaseModel):
    """What the tenant is asking for, as structured data."""

    intent: Intent = Field(
        description="book = wants a viewing (including 'is X free? if not, Y works'); "
        "cancel = wants to cancel a viewing; question = only asking about availability; "
        "other = anything else or unclear."
    )
    property: Optional[str] = Field(
        None, description="Property address exactly as the tenant wrote it, or null if none given."
    )
    date: Optional[str] = Field(None, description="Requested date as YYYY-MM-DD, or null.")
    preferred_time: Optional[str] = Field(
        None, description="An exact requested start time as HH:MM (24h), or null."
    )
    earliest_time: Optional[str] = Field(
        None, description="Start of an acceptable time window as HH:MM (24h), or null."
    )
    latest_time: Optional[str] = Field(
        None, description="End of an acceptable time window as HH:MM (24h), or null."
    )
    time_of_day: Optional[Literal["morning", "afternoon", "evening"]] = Field(
        None, description="Set if the tenant names a part of the day instead of a clock time, else null."
    )
    refers_to_last_booking: bool = Field(
        False,
        description="True only if the tenant points at the booking made earlier in THIS conversation with "
        "words like 'it', 'that one', 'the one you just booked'. A general request such as "
        "'I want to cancel my viewing' is NOT such a reference: false.",
    )


# Parts of the day are mapped in code, not left to the model (it skipped this in live runs).
TIME_OF_DAY = {"morning": ("08:00", "12:00"), "afternoon": ("12:00", "17:00"), "evening": ("17:00", "21:00")}
TIME_KEYS = ("preferred_time", "earliest_time", "latest_time")

# Conversation memory settings.
HISTORY_TURNS = 6                     # messages shown to the LLM
CONVERSATION_TTL = timedelta(minutes=30)  # after this much silence, start fresh


class AgentState(TypedDict, total=False):
    # input for this turn
    sender: str
    message: str
    now: datetime
    # per-turn working values (reset at the start of every turn)
    extraction: dict
    property: Optional[dict]
    action: str  # booked | cancelled | clarify | no_availability | answered | none
    facts: dict
    booking: Optional[dict]
    reply: str
    # conversation memory — saved by the LangGraph checkpointer per sender (thread_id)
    pending: Optional[dict]        # the open request a follow-up can continue
    history: List[dict]            # recent messages [{"role": "tenant"|"agent", "text": ...}]
    last_booking: Optional[dict]   # booking made in this conversation ("cancel it")
    last_seen: Optional[datetime]  # for expiring stale conversations


# --------------------------------------------------------------------- LLM


EXTRACT_PROMPT = """You read messages sent to a property letting agency and extract the request.

Now: {now:%A %Y-%m-%d %H:%M}.
Rules:
- "tomorrow" = {tomorrow:%Y-%m-%d}. A weekday name (e.g. "Saturday") means the next such day
  strictly after today, unless they say "today" or "this <day>" and it is today.
- "morning" / "afternoon" / "evening" -> set time_of_day.
- "at 5pm" -> preferred_time 17:00. "any time after 4" -> earliest_time 16:00. "before 3" -> latest_time 15:00.
- If they ask whether a time is free AND say an alternative works, the intent is book.
- Short replies like "yes", "10:00 please", "the Oak Road one" continue the open request below:
  "yes, book 10:00" is intent book; just naming a property or time is intent other.
- Extract only what THIS message says; the system fills in the rest from the open request.
- Never invent a property, date or time. Use null instead.
{context}"""


DRAFT_PROMPT = """You polish replies for a property letting agency.
Rewrite the draft below as ONE short, friendly reply (1-3 sentences, no sign-off, no placeholders).
Keep its meaning exactly: same outcome ({action}), same addresses, and times written exactly as in
the draft (e.g. 16:00). If the draft asks a question, your reply must end with that question.
Do not add any information that is not in the draft.

Tenant wrote: {message}
Draft: {draft}"""


class LLMClient:
    """Thin wrapper so the graph can run on a real LLM or a fake one in tests."""

    def __init__(self, chat_model):
        self.chat = chat_model
        self.extractor = chat_model.with_structured_output(Extraction)

    def extract(self, message: str, now: datetime, context: str = "") -> Extraction:
        system = EXTRACT_PROMPT.format(now=now, tomorrow=now + timedelta(days=1), context=context)
        return self.extractor.invoke([("system", system), ("human", message)])

    def draft(self, message: str, action: str, template: str) -> str:
        prompt = DRAFT_PROMPT.format(message=message, action=action, draft=template)
        try:
            text = self.chat.invoke(prompt).content.strip()
            return text or template
        except Exception:  # LLM down / rate-limited: the template is still correct
            return template


def make_llm() -> LLMClient:
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    if provider == "groq":
        from langchain_groq import ChatGroq

        model = ChatGroq(model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"), temperature=0)
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI  # pip install langchain-google-genai

        model = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), temperature=0)
    elif provider == "openai":
        from langchain_openai import ChatOpenAI  # pip install langchain-openai

        model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER {provider!r}")
    return LLMClient(model)


# ----------------------------------------------------------------- helpers


def _hhmm(value: Optional[str]) -> Optional[str]:
    """Accept only well-formed HH:MM from the LLM; anything else counts as missing."""
    if not value:
        return None
    try:
        return datetime.strptime(value.strip(), "%H:%M").strftime("%H:%M")
    except ValueError:
        return None


def _date(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.strptime(value.strip(), "%Y-%m-%d")
    except ValueError:
        return None


def _at(day: datetime, hhmm: str) -> datetime:
    h, m = map(int, hhmm.split(":"))
    return day.replace(hour=h, minute=m, second=0, microsecond=0)


def human_time(starts_at: str) -> str:
    dt = datetime.strptime(starts_at, "%Y-%m-%d %H:%M")
    return f"{dt:%A} {dt.day} {dt:%b} at {dt:%H:%M}"


_TIME_RE = re.compile(r"\b\d{1,2}:\d{2}\b")


def is_faithful(action: str, template: str, text: str) -> bool:
    """Does the LLM-polished reply say the same thing as the code-drafted one?

    Cheap, strict checks: no new or missing times, address kept, outcome words
    kept, questions stay questions. If any check fails we send the template.
    """
    t, low = text.strip(), text.lower()
    if not t or len(t) > 600:
        return False
    if set(_TIME_RE.findall(t)) != set(_TIME_RE.findall(template)):
        return False
    for address in re.findall(r"\d+ [A-Z][a-z]+ (?:Road|Street|Avenue)", template):
        if address.lower() not in low:
            return False
    claims_booked = "booked" in low
    claims_cancelled = "cancelled" in low or "canceled" in low
    if action == "booked" and not claims_booked:
        return False
    if action == "cancelled" and not claims_cancelled:
        return False
    if action not in ("booked", "cancelled") and (claims_booked or claims_cancelled) and not (
        "booked" in template.lower() or "cancel" in template.lower()
    ):
        return False
    if template.rstrip().endswith("?") and not t.endswith("?"):
        return False
    return True


def template_reply(action: str, facts: dict) -> str:
    """A deterministic, factually correct reply for every outcome."""
    reason = facts.get("reason")
    if action == "booked":
        return (
            f"Hi! {human_time(facts['starts_at'])} is free at {facts['address']}. "
            "I've booked it for you. Reply here if you need to change it."
        )
    if action == "cancelled":
        return (
            f"Done. Your viewing at {facts['address']} on {human_time(facts['starts_at'])} "
            "is cancelled. Let us know if you'd like to book another time."
        )
    if action == "answered":
        times = ", ".join(human_time(s) for s in facts["free"])
        return f"Yes, {facts['address']} has free viewing slots: {times}. Would you like me to book one?"
    if action == "no_availability":
        alts = facts.get("alternatives") or []
        if alts:
            times = ", ".join(human_time(s) for s in alts)
            return (
                f"Sorry, that time isn't available at {facts['address']}. "
                f"The nearest free slots are {times}. Shall I book one of those?"
            )
        return f"Sorry, {facts['address']} has no free viewing slots in the next week."
    if action == "none" and reason == "already_booked":
        return (
            f"You're already booked for {facts['address']} on {human_time(facts['starts_at'])}. "
            "Reply here if you need to change it."
        )
    if action == "none" and reason == "no_bookings":
        return "I couldn't find an active viewing booked under your contact. Is it under a different name or number?"
    # clarify
    if reason == "unknown_property":
        known = ", ".join(facts.get("known_properties", []))
        return (
            f"Sorry, I couldn't find \"{facts.get('property_text')}\" in our listings. "
            f"We currently have viewings at {known}. Which property did you mean?"
        )
    if reason == "no_property":
        return "Happy to help. Which property would you like to view?"
    if reason == "which_booking":
        options = "; ".join(f"{b['address']} on {human_time(b['starts_at'])}" for b in facts["bookings"])
        return f"Happy to help. Which property was the viewing for? You have: {options}."
    if reason == "no_time":
        free = facts.get("free") or []
        if facts.get("day"):
            times = ", ".join(s[11:] for s in free)
            if not times:
                return f"Sorry, {facts['address']} has no free viewing slots on {facts['day']}. Would another day suit you?"
            return f"{facts['address']} has free slots on {facts['day']} at {times}. Which time would suit you?"
        hint = f" Free times include {', '.join(human_time(s) for s in free)}." if free else ""
        return f"Happy to help with {facts['address']}. What day and time would suit you?{hint}"
    return "Thanks for your message! Would you like to book a viewing or cancel one? Please tell me the property and a time."


# ------------------------------------------------------------------- agent


class ViewingAgent:
    def __init__(self, tools: BookingTools, llm, draft_with_llm: bool = True, clock=datetime.now,
                 checkpointer=None):
        self.tools = tools
        self.llm = llm
        self.draft_with_llm = draft_with_llm
        self.clock = clock  # injectable so tests can pin "today"
        # Conversation memory lives in the graph state; the checkpointer saves it per
        # sender (thread_id). Default: in memory. The API passes a SQLite one so
        # conversations survive a server restart.
        self.checkpointer = checkpointer if checkpointer is not None else InMemorySaver()
        self.graph = self._build_graph()

    # ---- nodes -----------------------------------------------------------

    def understand(self, state: AgentState) -> AgentState:
        now = state["now"]
        pending = state.get("pending")
        history = list(state.get("history") or [])
        last_booking = state.get("last_booking")
        last_seen = state.get("last_seen")
        if last_seen is not None and now - last_seen > CONVERSATION_TTL:
            pending, history, last_booking = None, [], None  # stale: start a new conversation

        message = state["message"]
        ex = self.llm.extract(message, now, self._context(pending, history, last_booking)).model_dump()
        # Ground the model's output in what THIS message says. Seen live: with history in the
        # prompt, the model copied an earlier address into "I want to cancel my viewing."
        if ex.get("property") and not mentions(message, ex["property"]):
            ex["property"] = None
        if ex.get("refers_to_last_booking") and not re.search(r"\b(it|that|this|one)\b", message.lower()):
            ex["refers_to_last_booking"] = False
        ex["preferred_time"] = _hhmm(ex.get("preferred_time"))
        ex["earliest_time"] = _hhmm(ex.get("earliest_time"))
        ex["latest_time"] = _hhmm(ex.get("latest_time"))
        part = ex.pop("time_of_day", None)
        # An exact time wins: "book at 11:00" must not become "any time this morning".
        if part in TIME_OF_DAY and not (ex["preferred_time"] or ex["earliest_time"] or ex["latest_time"]):
            ex["earliest_time"], ex["latest_time"] = TIME_OF_DAY[part]

        # Merge a follow-up ("14 Oak Road", "yes, book 10:00") into the open request.
        if pending and self._continues(pending, ex):
            merged = dict(pending)
            if any(ex.get(k) for k in TIME_KEYS):
                # A new time replaces the old one completely (no stale "morning" fallback).
                merged.update({k: None for k in TIME_KEYS})
            merged.update({k: v for k, v in ex.items() if v is not None and k != "intent"})
            if ex["intent"] == "book":  # "is it free?" -> "yes, book it"
                merged["intent"] = "book"
            ex = merged
        return {
            "extraction": ex,
            "pending": pending,
            "history": history,
            "last_booking": last_booking,
            # reset this turn's working values (the checkpointer keeps last turn's)
            "property": None, "action": "", "facts": {}, "booking": None, "reply": "",
        }

    def lookup(self, state: AgentState) -> AgentState:
        ex = state["extraction"]
        if not ex.get("property"):
            return self._clarify(ex, {"reason": "no_property"})
        prop = self.tools.find_property(ex["property"])
        if prop is None:
            return self._clarify(
                ex,
                {
                    "reason": "unknown_property",
                    "property_text": ex["property"],
                    "known_properties": self._known_properties(),
                },
            )
        ex["property"] = prop["address"]  # canonical address for any follow-up

        day = _date(ex.get("date"))
        now = state["now"]
        has_time = any(ex.get(k) for k in ("preferred_time", "earliest_time", "latest_time"))
        if day is None or not has_time:
            # Offer a few real options rather than guessing one.
            start = day or now
            free = self.tools.get_free_slots(prop["id"], start, start + timedelta(days=1 if day else 7), now)
            if ex["intent"] == "question" and day is not None and free:
                return {"property": prop, "action": "answered",
                        "facts": {"address": prop["address"], "free": [s["starts_at"] for s in free[:4]]}}
            return self._clarify(ex, {"reason": "no_time", "address": prop["address"],
                                      "day": f"{day:%A} {day.day} {day:%b}" if day else None,
                                      "free": [s["starts_at"] for s in free[:3]]}, prop)

        window_start = _at(day, ex.get("earliest_time") or ex.get("preferred_time") or "00:00")
        window_end = _at(day, ex["latest_time"]) if ex.get("latest_time") else day + timedelta(days=1)
        free = self.tools.get_free_slots(prop["id"], day, day + timedelta(days=1), now)

        chosen = None
        if ex.get("preferred_time"):
            chosen = next((s for s in free if s["starts_at"].endswith(ex["preferred_time"])), None)
            if chosen is None:
                # Not free — is that because this tenant already has it? Say so; never book another time.
                wanted = f"{day:%Y-%m-%d} {ex['preferred_time']}"
                if any(b["starts_at"] == wanted for b in self.tools.active_bookings_for(state["sender"], prop["id"])):
                    return {"property": prop, "action": "none",
                            "facts": {"reason": "already_booked", "address": prop["address"], "starts_at": wanted}}
        if chosen is None and (ex.get("earliest_time") or ex.get("latest_time")):
            # Only reached when the tenant gave a window themselves ("if not, any time after 4").
            ws, we = window_start.strftime("%Y-%m-%d %H:%M"), window_end.strftime("%Y-%m-%d %H:%M")
            chosen = next((s for s in free if ws <= s["starts_at"] < we), None)

        if chosen is None:
            alts = self.tools.get_free_slots(prop["id"], day, day + timedelta(days=7), now)
            return {"property": prop, "action": "no_availability",
                    "facts": {"address": prop["address"], "alternatives": [s["starts_at"] for s in alts[:3]]}}
        if ex["intent"] == "question":
            return {"property": prop, "action": "answered",
                    "facts": {"address": prop["address"], "free": [chosen["starts_at"]]}}
        return {"property": prop, "action": "book", "facts": {"slot": chosen}}

    def book(self, state: AgentState) -> AgentState:
        slot = state["facts"]["slot"]
        result = self.tools.book_slot(slot["slot_id"], state["sender"])
        if result["ok"]:
            return {"action": "booked", "booking": result,
                    "facts": {"address": result["address"], "starts_at": result["starts_at"]}}
        # Lost a race to another request: the DB said no. Offer what is left.
        prop = state["property"]
        day = datetime.strptime(slot["starts_at"][:10], "%Y-%m-%d")
        alts = self.tools.get_free_slots(prop["id"], day, day + timedelta(days=7), state["now"])
        return {"action": "no_availability",
                "facts": {"address": prop["address"], "alternatives": [s["starts_at"] for s in alts[:3]]}}

    def cancel(self, state: AgentState) -> AgentState:
        ex, sender = state["extraction"], state["sender"]
        last = state.get("last_booking")
        if ex.get("refers_to_last_booking") and last and not ex.get("property"):
            # "cancel it" right after we booked something: that booking, if it's still active.
            if any(b["booking_id"] == last["booking_id"] for b in self.tools.active_bookings_for(sender)):
                self.tools.cancel_booking(last["booking_id"], sender)
                return {"action": "cancelled", "booking": last,
                        "facts": {"address": last["address"], "starts_at": last["starts_at"]}}
        prop = None
        if ex.get("property"):
            prop = self.tools.find_property(ex["property"])
            if prop is None:
                # "the maple one": match against this tenant's OWN bookings only.
                own = [b for b in self.tools.active_bookings_for(sender) if mentions(ex["property"], b["address"])]
                if len({b["property_id"] for b in own}) == 1:
                    prop = {"id": own[0]["property_id"], "address": own[0]["address"]}
            if prop is None:
                return self._clarify(ex, {"reason": "unknown_property", "property_text": ex["property"],
                                          "known_properties": self._known_properties()})
        bookings = self.tools.active_bookings_for(sender, prop["id"] if prop else None)
        day = _date(ex.get("date"))
        if day is not None and len(bookings) > 1:
            bookings = [b for b in bookings if b["starts_at"].startswith(day.strftime("%Y-%m-%d"))] or bookings

        if not bookings:
            return {"action": "none", "facts": {"reason": "no_bookings"}}
        if prop is None or len(bookings) > 1:
            # Never cancel the wrong viewing: ask which one.
            return self._clarify(ex, {"reason": "which_booking", "bookings": bookings})
        b = bookings[0]
        self.tools.cancel_booking(b["booking_id"], sender)
        return {"action": "cancelled", "booking": b,
                "facts": {"address": b["address"], "starts_at": b["starts_at"]}}

    def clarify_other(self, state: AgentState) -> AgentState:
        return self._clarify(state["extraction"], {"reason": "unclear"})

    def reply(self, state: AgentState) -> AgentState:
        action, facts = state["action"], state.get("facts", {})
        template = template_reply(action, facts)
        text = template
        if self.draft_with_llm:
            polished = self.llm.draft(state["message"], action, template)
            # The LLM only gets to change the wording, never the facts.
            if is_faithful(action, template, polished):
                text = polished
        return {"reply": text, **self._remember(state, text)}

    def _remember(self, state: AgentState, reply_text: str) -> dict:
        """Update the conversation memory that the checkpointer saves for this sender."""
        action = state["action"]
        history = (state.get("history") or []) + [
            {"role": "tenant", "text": state["message"]},
            {"role": "agent", "text": reply_text},
        ]
        # Keep the open request while the reply invites a follow-up:
        # a question, free times, alternatives, a fresh booking ("can I also do 16:00?").
        keep = action in ("clarify", "answered", "no_availability", "booked", "none")
        last_booking = state.get("last_booking")
        if action == "booked":
            last_booking = state["booking"]
        elif action == "cancelled" and last_booking and state["booking"]["booking_id"] == last_booking["booking_id"]:
            last_booking = None
        return {
            "pending": state["extraction"] if keep else None,
            "history": history[-2 * HISTORY_TURNS:],
            "last_booking": last_booking,
            "last_seen": state["now"],
        }

    # ---- helpers ---------------------------------------------------------

    def _continues(self, pending: dict, ex: dict) -> bool:
        """Is this message a follow-up to the open request, rather than a new one?"""
        new, old = ex["intent"], pending["intent"]
        if new == "cancel" and old != "cancel":
            return False  # never let a booking chat decide which viewing gets cancelled
        if new == "book" and old == "cancel":
            return False
        if new not in ("other", "question", old) and not (new == "book" and old == "question"):
            return False
        if ex.get("property") and pending.get("property"):
            a, b = self.tools.find_property(ex["property"]), self.tools.find_property(pending["property"])
            if a is not None and b is not None and a["id"] != b["id"]:
                return False  # they've moved on to a different property
            # (if the earlier property was unknown, the new one is a correction: keep going)
        return True

    @staticmethod
    def _context(pending: Optional[dict], history: List[dict], last_booking: Optional[dict]) -> str:
        """What the LLM sees about the conversation so far."""
        parts = []
        if history:
            lines = "\n".join(f"  {m['role']}: {m['text']}" for m in history[-HISTORY_TURNS:])
            parts.append(f"Conversation so far:\n{lines}")
        if pending:
            open_request = {k: v for k, v in pending.items() if v}
            parts.append(f"Open request: {open_request}")
        if last_booking:
            parts.append(f"Booked in this conversation: {last_booking['address']} at {last_booking['starts_at']}")
        return "\n".join(parts)

    def _clarify(self, ex: dict, facts: dict, prop: Optional[dict] = None) -> AgentState:
        return {"action": "clarify", "facts": facts, "property": prop, "extraction": ex}

    def _known_properties(self) -> List[str]:
        from sqlalchemy import select
        from .db import Property

        with self.tools.Session() as s:
            return list(s.scalars(select(Property.address).order_by(Property.id)))

    # ---- graph -----------------------------------------------------------

    def _build_graph(self):
        g = StateGraph(AgentState)
        g.add_node("understand", self.understand)
        g.add_node("lookup", self.lookup)
        g.add_node("book", self.book)
        g.add_node("cancel", self.cancel)
        g.add_node("clarify_other", self.clarify_other)
        g.add_node("reply", self.reply)

        g.set_entry_point("understand")
        g.add_conditional_edges(
            "understand",
            lambda s: {"book": "lookup", "question": "lookup", "cancel": "cancel"}.get(
                s["extraction"]["intent"], "clarify_other"
            ),
            ["lookup", "cancel", "clarify_other"],
        )
        g.add_conditional_edges(
            "lookup", lambda s: "book" if s["action"] == "book" else "reply", ["book", "reply"]
        )
        g.add_edge("book", "reply")
        g.add_edge("cancel", "reply")
        g.add_edge("clarify_other", "reply")
        g.add_edge("reply", END)
        return g.compile(checkpointer=self.checkpointer)

    # ---- entry point -----------------------------------------------------

    def handle(self, sender: str, message: str, now: Optional[datetime] = None) -> dict:
        # One LangGraph thread per sender: the checkpointer restores their conversation.
        state = self.graph.invoke(
            {"sender": sender, "message": message, "now": now or self.clock()},
            config={"configurable": {"thread_id": f"sender:{sender}"}},
        )
        return {
            "reply": state["reply"],
            "action": state["action"],
            "booking": state.get("booking"),
        }

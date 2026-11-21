import threading

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from .conftest import NOW


def send(client, sender, text_):
    r = client.post("/message", json={"sender": sender, "text": text_})
    assert r.status_code == 200, r.text
    return r.json()


def slot_id_at(tools, address, starts_at):
    prop = tools.find_property(address)
    day = NOW.replace(day=int(starts_at[8:10]), hour=0)
    for s in tools.get_free_slots(prop["id"], day, day.replace(hour=23, minute=59)):
        if s["starts_at"] == starts_at:
            return s["slot_id"]
    raise AssertionError(f"no free slot {starts_at} at {address}")


# 1 · successful booking --------------------------------------------------------

def test_successful_booking(client, llm):
    msg = llm.when(
        "Hi, can I see the flat at 14 Oak Road on Saturday morning?",
        # what the live model returned: a part of day, no clock times
        intent="book", property="14 Oak Road", date="2026-11-28", time_of_day="morning",
    )
    out = send(client, "alice", msg)

    assert out["action"] == "booked"
    assert out["booking"]["address"] == "14 Oak Road"
    assert out["booking"]["starts_at"] == "2026-11-28 10:00"  # first free slot in the morning
    assert "booked" in out["reply"].lower()

    bookings = client.get("/bookings").json()
    assert [(b["address"], b["starts_at"], b["contact"]) for b in bookings] == [
        ("14 Oak Road", "2026-11-28 10:00", "alice")
    ]


def test_is_it_free_else_any_time_after_4(client, llm, tools):
    """The brief's second example: 5pm is taken, so the agent falls back to 'after 4'."""
    tools.book_slot(slot_id_at(tools, "22 Elm Street", "2026-11-22 17:00"), "someone-else")
    msg = llm.when(
        "Is 22 Elm Street free tomorrow at 5pm? If not, any time after 4 works.",
        intent="book", property="22 Elm Street", date="2026-11-22",
        preferred_time="17:00", earliest_time="16:00",
    )
    out = send(client, "bob", msg)
    assert out["action"] == "booked"
    assert out["booking"]["starts_at"] == "2026-11-22 16:00"


# 2 · blocked double booking ---------------------------------------------------

def test_double_booking_is_blocked_through_the_agent(client, llm):
    msg = llm.when("Book 14 Oak Road Saturday at 10am please",
                   intent="book", property="14 Oak Road", date="2026-11-28", preferred_time="10:00")
    assert send(client, "alice", msg)["action"] == "booked"

    out = send(client, "bob", msg)
    assert out["action"] == "no_availability"
    assert out["booking"] is None
    assert "11:00" in out["reply"]  # offers a real alternative
    assert len(client.get("/bookings").json()) == 1


def test_simultaneous_bookings_only_one_wins(tools):
    """Many requests for the same slot at the same moment: the DB lets exactly one through."""
    slot = slot_id_at(tools, "14 Oak Road", "2026-11-28 10:00")
    n = 8
    barrier = threading.Barrier(n)
    results = []

    def worker(i):
        barrier.wait()
        results.append(tools.book_slot(slot, f"tenant-{i}"))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sum(r["ok"] for r in results) == 1
    assert sum(r.get("reason") == "slot_taken" for r in results) == n - 1
    assert len(tools.list_bookings()) == 1


def test_database_itself_rejects_a_second_active_booking(engine):
    """Bypass all Python logic: the rule must hold in raw SQL too."""
    with engine.begin() as c:
        c.execute(text("INSERT INTO bookings (slot_id, contact) VALUES (1, 'alice')"))
    with pytest.raises(IntegrityError):
        with engine.begin() as c:
            c.execute(text("INSERT INTO bookings (slot_id, contact) VALUES (1, 'bob')"))
    # ...but once cancelled, the slot can be booked again.
    with engine.begin() as c:
        c.execute(text("UPDATE bookings SET status = 'cancelled' WHERE contact = 'alice'"))
        c.execute(text("INSERT INTO bookings (slot_id, contact) VALUES (1, 'bob')"))
        assert c.execute(text("SELECT count(*) FROM bookings WHERE status='active'")).scalar() == 1


# 3 · cancellation --------------------------------------------------------------

def test_cancellation_frees_the_slot(client, llm, tools):
    book = llm.when("Book 14 Oak Road Saturday at 10am please",
                    intent="book", property="14 Oak Road", date="2026-11-28", preferred_time="10:00")
    assert send(client, "alice", book)["action"] == "booked"

    cancel = llm.when("Please cancel my viewing at 14 Oak Road",
                      intent="cancel", property="14 Oak Road")
    out = send(client, "alice", cancel)
    assert out["action"] == "cancelled"
    assert out["booking"]["starts_at"] == "2026-11-28 10:00"
    assert client.get("/bookings").json() == []

    # The slot is bookable again by someone else.
    assert send(client, "bob", book)["action"] == "booked"


def test_cancel_without_property_asks_which_then_cancels(client, llm):
    """The brief's third example, including the tenant's follow-up answer."""
    llm.when("Book 14 Oak Road Saturday at 10am please",
             intent="book", property="14 Oak Road", date="2026-11-28", preferred_time="10:00")
    llm.when("Book 22 Elm Street tomorrow at 4pm",
             intent="book", property="22 Elm Street", date="2026-11-22", preferred_time="16:00")
    send(client, "alice", "Book 14 Oak Road Saturday at 10am please")
    send(client, "alice", "Book 22 Elm Street tomorrow at 4pm")

    out = send(client, "alice", llm.when("I want to cancel my viewing.", intent="cancel"))
    assert out["action"] == "clarify"
    assert "which property" in out["reply"].lower()
    assert len(client.get("/bookings").json()) == 2  # nothing cancelled by guessing

    out = send(client, "alice", llm.when("The Oak Road one", intent="other", property="Oak Road 14"))
    assert out["action"] == "cancelled"
    assert out["booking"]["address"] == "14 Oak Road"
    assert [b["address"] for b in client.get("/bookings").json()] == ["22 Elm Street"]


def test_cannot_cancel_someone_elses_booking(client, llm):
    msg = llm.when("Book 14 Oak Road Saturday at 10am please",
                   intent="book", property="14 Oak Road", date="2026-11-28", preferred_time="10:00")
    send(client, "alice", msg)
    out = send(client, "mallory", llm.when("Cancel the 14 Oak Road viewing",
                                           intent="cancel", property="14 Oak Road"))
    assert out["action"] == "none"
    assert len(client.get("/bookings").json()) == 1


# 4 · unknown property ----------------------------------------------------------

def test_unknown_property_gets_a_clarifying_question(client, llm):
    msg = llm.when("Can I see 99 Banana Street tomorrow at 10?",
                   intent="book", property="99 Banana Street", date="2026-11-22", preferred_time="10:00")
    out = send(client, "alice", msg)

    assert out["action"] == "clarify"
    assert out["booking"] is None
    assert "99 Banana Street" in out["reply"]
    assert "14 Oak Road" in out["reply"]  # lists what we do have
    assert client.get("/bookings").json() == []


def test_similar_but_different_address_is_not_guessed(tools):
    assert tools.find_property("14 oak rd")["address"] == "14 Oak Road"
    assert tools.find_property("the flat at 14 Oak Raod")["address"] == "14 Oak Road"  # typo
    assert tools.find_property("Oak Road 14")["address"] == "14 Oak Road"  # word order
    assert tools.find_property("41 Oak Road") is None  # wrong house number: never guess
    assert tools.find_property("14 Pine Road") is None
    assert tools.find_property("Elm") is None


# 5 · vague request -------------------------------------------------------------

def test_vague_request_gets_a_clarifying_question(client, llm):
    out = send(client, "alice", llm.when("I'd like to book a viewing", intent="book"))
    assert out["action"] == "clarify"
    assert out["reply"].endswith("?")
    assert client.get("/bookings").json() == []


def test_yes_book_it_after_an_availability_answer(client, llm):
    """Real conversation from the web page: the follow-up lost the property and day."""
    out = send(client, "guest", llm.when("Is 14 Oak Road free on Tuesday?",
                                         intent="question", property="14 Oak Road", date="2026-11-24"))
    assert out["action"] == "answered"

    out = send(client, "guest", llm.when("yes book at 10:00", intent="book", preferred_time="10:00"))
    assert out["action"] == "booked"
    assert (out["booking"]["address"], out["booking"]["starts_at"]) == ("14 Oak Road", "2026-11-24 10:00")


def test_follow_ups_after_a_booking(client, llm):
    """Real conversation from the web page: repeating 'yes book at 11:00' after it was booked."""
    send(client, "guest", llm.when("Is 14 Oak Road free on Tuesday?",
                                   intent="question", property="14 Oak Road", date="2026-11-24"))
    book_11 = llm.when("yes book at 11:00", intent="book", preferred_time="11:00")
    assert send(client, "guest", book_11)["action"] == "booked"

    out = send(client, "guest", book_11)  # asked again
    assert out["action"] == "none"
    assert "already booked" in out["reply"] and "11:00" in out["reply"]
    assert len(client.get("/bookings").json()) == 1

    out = send(client, "guest", llm.when("can I also do 16:00?", intent="book", preferred_time="16:00"))
    assert out["action"] == "booked"
    assert out["booking"]["starts_at"] == "2026-11-24 16:00"

    # someone else asking for my slot still gets "not available", not "already booked"
    out = send(client, "bob", llm.when("Book 14 Oak Road on Tuesday at 11",
                                       intent="book", property="14 Oak Road", date="2026-11-24", preferred_time="11:00"))
    assert out["action"] == "no_availability"


def test_exact_time_is_never_swapped_for_another_time(client, llm):
    """Live bug: the model tagged 'book at 11:00' as morning too, and a repeat booked 10:00."""
    send(client, "guest", llm.when("Is 22 Elm Street free on Wednesday?",
                                   intent="question", property="22 Elm Street", date="2026-11-25"))
    again = llm.when("yes book at 11:00", intent="book", preferred_time="11:00", time_of_day="morning")
    out = send(client, "guest", again)
    assert (out["action"], out["booking"]["starts_at"]) == ("booked", "2026-11-25 11:00")

    out = send(client, "guest", again)
    assert out["action"] == "none" and "already booked" in out["reply"]
    assert [b["starts_at"] for b in client.get("/bookings").json()] == ["2026-11-25 11:00"]

    # Someone else asking for 11:00 "in the morning" is offered alternatives, not silently given 10:00.
    out = send(client, "bob", llm.when("Book 22 Elm Street Wednesday at 11am", intent="book",
                                       property="22 Elm Street", date="2026-11-25",
                                       preferred_time="11:00", time_of_day="morning"))
    assert out["action"] == "no_availability"
    assert len(client.get("/bookings").json()) == 1


def test_new_time_replaces_old_window(client, llm):
    """After 'Saturday morning' was booked, 'also 16:00' (taken) must not fall back to the morning."""
    send(client, "alice", llm.when("14 Oak Road Saturday morning", intent="book",
                                   property="14 Oak Road", date="2026-11-28", time_of_day="morning"))
    send(client, "bob", llm.when("Book 14 Oak Road Saturday at 4pm", intent="book",
                                 property="14 Oak Road", date="2026-11-28", preferred_time="16:00"))
    out = send(client, "alice", llm.when("can I also do 16:00?", intent="book", preferred_time="16:00"))
    assert out["action"] == "no_availability"
    assert [b["contact"] for b in client.get("/bookings").json()] == ["alice", "bob"]


def test_taking_an_offered_alternative(client, llm):
    msg = llm.when("Book 14 Oak Road Saturday at 10am please",
                   intent="book", property="14 Oak Road", date="2026-11-28", preferred_time="10:00")
    send(client, "alice", msg)
    assert send(client, "bob", msg)["action"] == "no_availability"
    out = send(client, "bob", llm.when("11 is fine then", intent="book", preferred_time="11:00"))
    assert out["action"] == "booked"
    assert out["booking"]["starts_at"] == "2026-11-28 11:00"


def test_context_does_not_leak_into_cancel_or_other_property(client, llm):
    llm.when("Book 14 Oak Road Saturday at 10am please",
             intent="book", property="14 Oak Road", date="2026-11-28", preferred_time="10:00")
    llm.when("Book 22 Elm Street tomorrow at 4pm",
             intent="book", property="22 Elm Street", date="2026-11-22", preferred_time="16:00")
    send(client, "alice", "Book 14 Oak Road Saturday at 10am please")
    send(client, "alice", "Book 22 Elm Street tomorrow at 4pm")
    send(client, "alice", llm.when("Is 14 Oak Road free on Tuesday?",
                                   intent="question", property="14 Oak Road", date="2026-11-24"))

    # A cancel right after an Oak Road question must still ask which viewing.
    out = send(client, "alice", llm.when("I want to cancel my viewing.", intent="cancel"))
    assert out["action"] == "clarify"
    assert len(client.get("/bookings").json()) == 2

    # Asking about a different property starts fresh (no Tuesday carried over).
    send(client, "carol", llm.when("Is 14 Oak Road free on Tuesday?",
                                   intent="question", property="14 Oak Road", date="2026-11-24"))
    out = send(client, "carol", llm.when("Can I view 7 Maple Avenue?", intent="book", property="7 Maple Avenue"))
    assert out["action"] == "clarify"
    assert "Tuesday" not in out["reply"]


def test_day_but_no_time_asks_only_for_the_time(client, llm):
    out = send(client, "alice", llm.when("Can I view 7 Maple Avenue on Monday?",
                                         intent="book", property="7 Maple Avenue", date="2026-11-23"))
    assert out["action"] == "clarify"
    assert "what day" not in out["reply"].lower()
    assert "Monday 23 Nov" in out["reply"] and "10:00" in out["reply"]
    assert out["reply"].endswith("?")
    assert client.get("/bookings").json() == []


def test_property_but_no_time_asks_for_a_time(client, llm):
    out = send(client, "alice", llm.when("Can I view 7 Maple Avenue?", intent="book", property="7 Maple Avenue"))
    assert out["action"] == "clarify"
    assert "what day and time" in out["reply"].lower()
    assert client.get("/bookings").json() == []

    # The follow-up answer completes the original request.
    out = send(client, "alice", llm.when("Monday at 11 please", intent="other",
                                         date="2026-11-23", preferred_time="11:00"))
    assert out["action"] == "booked"
    assert (out["booking"]["address"], out["booking"]["starts_at"]) == ("7 Maple Avenue", "2026-11-23 11:00")


# LLM-polished replies can't change the facts -----------------------------------

def test_unfaithful_llm_reply_falls_back_to_template(client, llm, monkeypatch):
    """Real failure seen with the live model: a clarify turned into a false 'no properties'."""
    monkeypatch.setattr(llm, "draft",
                        lambda m, a, t: "Sorry, we don't have any properties available for viewing at the moment.")
    client.app.state.agent.draft_with_llm = True
    out = send(client, "erin", llm.when("I'd like to book a viewing", intent="book"))
    assert out["reply"] == "Happy to help. Which property would you like to view?"


def test_is_faithful_checks():
    from app.agent import is_faithful

    booked = "Hi! Sunday 22 Nov at 16:00 is free at 22 Elm Street. I've booked it for you. Reply here if you need to change it."
    assert is_faithful("booked", booked, "Great news, I've booked 22 Elm Street for you on Sunday 22 Nov at 16:00!")
    # seen live: booked, but the reply never says so
    assert not is_faithful("booked", booked, "The 4 pm slot tomorrow (22 Nov) is free at 22 Elm Street. Let me know if that works for you.")
    assert not is_faithful("booked", booked, "I've booked 22 Elm Street for you at 17:00.")  # changed time
    assert not is_faithful("booked", booked, "I've booked it for Sunday 22 Nov at 16:00.")  # dropped address


# API shape ---------------------------------------------------------------------

def test_web_page_and_properties_endpoint(client, llm):
    page = client.get("/")
    assert page.status_code == 200 and "Viewing Booking Agent" in page.text

    props = client.get("/properties").json()
    assert [p["address"] for p in props] == ["14 Oak Road", "22 Elm Street", "7 Maple Avenue"]
    oak_before = len(props[0]["free_slots"])
    assert oak_before == 28  # 7 days x 4 slots

    send(client, "alice", llm.when("Book 14 Oak Road Saturday at 10am please",
                                   intent="book", property="14 Oak Road", date="2026-11-28", preferred_time="10:00"))
    oak = client.get("/properties").json()[0]["free_slots"]
    assert len(oak) == oak_before - 1
    assert "2026-11-28 10:00" not in [s["starts_at"] for s in oak]


def test_message_validation(client):
    assert client.post("/message", json={"sender": "alice"}).status_code == 422
    assert client.post("/message", json={"sender": "", "text": "hi"}).status_code == 422


# conversation memory (LangGraph checkpointer) -----------------------------------

def test_conversation_survives_a_server_restart(tools, llm, tmp_path):
    """Memory is saved by a SQLite checkpointer, so a restart doesn't forget the open request."""
    import sqlite3
    from datetime import timedelta

    from langgraph.checkpoint.sqlite import SqliteSaver

    from app.agent import ViewingAgent

    def fresh_agent():  # a new process: new agent object, same conversations file
        saver = SqliteSaver(sqlite3.connect(str(tmp_path / "conv.db"), check_same_thread=False))
        saver.setup()
        return ViewingAgent(tools, llm, clock=lambda: NOW, checkpointer=saver)

    q = llm.when("Is 14 Oak Road free on Tuesday?", intent="question", property="14 Oak Road", date="2026-11-24")
    assert fresh_agent().handle("guest", q)["action"] == "answered"

    yes = llm.when("yes book at 10:00", intent="book", preferred_time="10:00")
    out = fresh_agent().handle("guest", yes, now=NOW + timedelta(minutes=2))
    assert out["action"] == "booked"
    assert (out["booking"]["address"], out["booking"]["starts_at"]) == ("14 Oak Road", "2026-11-24 10:00")


def test_stale_conversation_is_forgotten(agent, llm):
    from datetime import timedelta

    agent.handle("guest", llm.when("Is 14 Oak Road free on Tuesday?",
                                   intent="question", property="14 Oak Road", date="2026-11-24"), now=NOW)
    yes = llm.when("yes book at 10:00", intent="book", preferred_time="10:00")
    out = agent.handle("guest", yes, now=NOW + timedelta(minutes=31))
    assert out["action"] == "clarify"  # 31 minutes later: which property?
    assert agent.tools.list_bookings() == []


def test_each_sender_has_their_own_conversation(client, llm):
    send(client, "alice", llm.when("Is 14 Oak Road free on Tuesday?",
                                   intent="question", property="14 Oak Road", date="2026-11-24"))
    out = send(client, "bob", llm.when("yes book at 10:00", intent="book", preferred_time="10:00"))
    assert out["action"] == "clarify"
    assert client.get("/bookings").json() == []


def test_cancel_it_cancels_the_booking_just_made(client, llm):
    llm.when("Book 22 Elm Street tomorrow at 4pm",
             intent="book", property="22 Elm Street", date="2026-11-22", preferred_time="16:00")
    llm.when("Book 14 Oak Road Saturday at 10am please",
             intent="book", property="14 Oak Road", date="2026-11-28", preferred_time="10:00")
    send(client, "alice", "Book 22 Elm Street tomorrow at 4pm")
    send(client, "alice", "Book 14 Oak Road Saturday at 10am please")

    out = send(client, "alice", llm.when("actually, cancel it", intent="cancel", refers_to_last_booking=True))
    assert out["action"] == "cancelled"
    assert out["booking"]["address"] == "14 Oak Road"  # the one just booked, not the Elm Street one
    assert [b["address"] for b in client.get("/bookings").json()] == ["22 Elm Street"]

    # Saying it again: nothing recent left to point at, so ask instead of guessing.
    out = send(client, "alice", llm.when("cancel it", intent="cancel", refers_to_last_booking=True))
    assert out["action"] == "clarify"
    assert len(client.get("/bookings").json()) == 1


def test_llm_sees_recent_conversation(client, llm):
    send(client, "alice", llm.when("Is 14 Oak Road free on Tuesday?",
                                   intent="question", property="14 Oak Road", date="2026-11-24"))
    send(client, "alice", llm.when("yes book at 10:00", intent="book", preferred_time="10:00"))
    ctx = llm.contexts[-1]
    assert "Conversation so far" in ctx
    assert "tenant: Is 14 Oak Road free on Tuesday?" in ctx
    assert "Open request" in ctx and "14 Oak Road" in ctx


def test_address_copied_from_history_is_ignored(client, llm):
    """Live bug: the model put the last-booked address into 'I want to cancel my viewing.'"""
    llm.when("Book 22 Elm Street tomorrow at 4pm",
             intent="book", property="22 Elm Street", date="2026-11-22", preferred_time="16:00")
    llm.when("Book 7 Maple Avenue on Monday at 11",
             intent="book", property="7 Maple Avenue", date="2026-11-23", preferred_time="11:00")
    send(client, "alice", "Book 22 Elm Street tomorrow at 4pm")
    send(client, "alice", "Book 7 Maple Avenue on Monday at 11")

    out = send(client, "alice", llm.when("I want to cancel my viewing.", intent="cancel",
                                         property="7 Maple Avenue", refers_to_last_booking=True))
    assert out["action"] == "clarify"  # two bookings, nothing named: ask
    assert len(client.get("/bookings").json()) == 2

    # "the maple one" is matched against alice's own bookings.
    out = send(client, "alice", llm.when("the maple one", intent="other", property="maple"))
    assert out["action"] == "cancelled"
    assert out["booking"]["address"] == "7 Maple Avenue"
    assert [b["address"] for b in client.get("/bookings").json()] == ["22 Elm Street"]


def test_partial_name_never_matches_someone_elses_or_unknown_property(client, llm):
    llm.when("Book 22 Elm Street tomorrow at 4pm",
             intent="book", property="22 Elm Street", date="2026-11-22", preferred_time="16:00")
    send(client, "alice", "Book 22 Elm Street tomorrow at 4pm")
    out = send(client, "alice", llm.when("cancel the maple one", intent="cancel", property="maple"))
    assert out["action"] == "clarify"  # alice has no Maple booking
    assert len(client.get("/bookings").json()) == 1

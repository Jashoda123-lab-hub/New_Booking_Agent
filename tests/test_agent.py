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

def test_message_validation(client):
    assert client.post("/message", json={"sender": "alice"}).status_code == 422
    assert client.post("/message", json={"sender": "", "text": "hi"}).status_code == 422
